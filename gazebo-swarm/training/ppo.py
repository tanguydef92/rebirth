#!/usr/bin/env python3
"""Train, evaluate and export the central frontier-defense PPO actor."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import swarm
from drones.policy import PPOPolicy, SECTORS
from training.environment import make_environment
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from functools import partial
from drones.policy import observation_size

ROOT = Path(__file__).resolve().parents[1]


def make_worker(config):
    torch.set_num_threads(1)
    return Monitor(make_environment(config))


def validate_checkpoint(model,config):
    if getattr(model,'swarm_fire_model',None)!=config['fire']['model']:
        raise ValueError('Checkpoint fire model differs; start a fresh PPO training run')


def validate_learning(config):
    s = config.get('learning',{})
    required = {'sectors','seed','training_seeds','validation_seeds','test_seeds','episode_duration','steps','n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef'}
    if set(s)-{'observation_time_scale_s','fleet_backend','decision_step_s','observation_version','num_envs','validation_interval'}!=required or s['sectors']!=SECTORS: raise ValueError('learning keys/sector count: see configs/ppo-100-fast.json')
    scale=s.get('observation_time_scale_s',config['duration'])
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:
        raise ValueError('observation_time_scale_s must be positive')
    groups=[]
    for key in ('training_seeds','validation_seeds','test_seeds'):
        values=s[key]
        if not isinstance(values,list) or not values or any(type(v)is not int for v in values): raise ValueError(key+' must be nonempty integer seeds')
        groups.append(set(values))
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]): raise ValueError('Training, validation and test seeds must be disjoint')
    for key in ('steps','n_steps','batch_size','n_epochs'):
        if type(s[key])is not int or s[key]<2: raise ValueError(key+' must be an integer >=2')
    if s['n_steps']%s['batch_size']: raise ValueError('n_steps must be divisible by batch_size')
    for key in ('episode_duration','learning_rate','gamma','gae_lambda','clip_range','ent_coef'):
        if not isinstance(s[key],(int,float)) or not math.isfinite(s[key]) or s[key]<0: raise ValueError('Invalid '+key)
    if not 0<s['gamma']<=1 or not 0<s['gae_lambda']<=1 or not 0<s['clip_range']<1 or s['learning_rate']<=0 or s['episode_duration']<=0:
        raise ValueError('Invalid PPO hyperparameter')
    if not config['suppression'].get('perimeter_only'):
        raise ValueError('PPO training requires perimeter_only')
    if s.get('fleet_backend') != 'individual':
        raise ValueError('fleet_backend must be individual')
    if s.get('observation_version') != 2:raise ValueError('observation_version must be 2')
    if type(s.get('num_envs',1)) is not int or not 1<=s.get('num_envs',1)<=8:raise ValueError('num_envs must be between 1 and 8')
    if s.get('fleet_backend')=='individual':
        step=s.get('decision_step_s',1)
        if step<=0 or abs(step*config['sample_hz']-round(step*config['sample_hz']))>1e-9:
            raise ValueError('decision_step_s must contain whole navigation steps')
        intervention_end=config['duration']-config['landing_duration']-config['suppression']['return_duration']
        if config['takeoff_duration']+s['episode_duration']>intervention_end:
            raise ValueError('Training episode must finish before the return phase')


def export_actor(model, config, path):
    layers=[]
    for module in model.policy.mlp_extractor.policy_net:
        if isinstance(module,torch.nn.Linear): layers.append({'weight':module.weight.detach().cpu().tolist(),'bias':module.bias.detach().cpu().tolist(),'activation':'linear'})
        elif isinstance(module,torch.nn.Tanh): layers[-1]['activation']='tanh'
        else: raise ValueError('Portable exporter supports Linear/Tanh actor layers only')
    head=model.policy.action_net
    layers.append({'weight':head.weight.detach().cpu().tolist(),'bias':head.bias.detach().cpu().tolist(),'activation':'linear'})
    document={'format':'swarm-ppo-v1','algorithm':'PPO','sectors':SECTORS,'observation_size':observation_size(config),'layers':layers,
              'training_drone_count':config['drone_count'],'training_timesteps':model.num_timesteps,'training_seeds':config['learning']['training_seeds'],
              'fire_cell_size':config['fire']['cell_size'],'training_fleet':config['learning']['fleet_backend'],
              'fire_model':config['fire']['model'],
              'service_contract':{key:config['suppression'][key] for key in ('tank_capacity_l','refill_duration','drop_duration') if key in config['suppression']},
              'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(document,separators=(',',':'))+'\n')
    actor=PPOPolicy.load(path)
    env=make_environment(config);obs,_=env.reset(seed=987)
    for _ in range(12):
        expected,_=model.predict(obs,deterministic=True)
        if not np.allclose(actor.predict(obs),expected,atol=2e-5): raise ValueError('Exported actor disagrees with trained PPO')
        obs,_,done,truncated,_=env.step(expected)
        if done or truncated: obs,_=env.reset()
    return document


def validation_return(model,config):
    returns=[]
    for seed in config['learning']['validation_seeds']:
        env=make_environment(config,[seed]);obs,_=env.reset(options={'fire_seed':seed})
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,_,done,truncated,info=env.step(action)
            if done or truncated: break
        returns.append(info['episode_return'])
    return statistics.mean(returns)


class TrainingProgress(BaseCallback):
    def __init__(self,config,output):
        super().__init__();self.last_time=0.0;self.config=config;self.output=output
        self.history=[];self.best=-math.inf
    def _on_step(self):
        if time.monotonic()-self.last_time>30:
            print('PPO training: %d steps'%self.num_timesteps,flush=True);self.last_time=time.monotonic()
        if self.num_timesteps%self.config['learning'].get('validation_interval',2048)==0:
            result=validation_return(self.model,self.config)
            self.history.append({'steps':self.num_timesteps,'validation_return':result})
            print('PPO validation at %d steps: %.3f'%(self.num_timesteps,result),flush=True)
            if result>self.best:
                self.best=result;self.model.save(self.output/'best_validation_model')
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo-100-fast.json')
    p.add_argument('--steps',type=int)
    p.add_argument('--output',type=Path,default=ROOT/'output/ppo-100-fast-training')
    p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-only',type=Path)
    args=p.parse_args()
    config=swarm.load_config(args.config);validate_learning(config)
    s=config['learning'];torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    factories=[partial(make_worker,config) for _ in range(s.get('num_envs',1))]
    env=SubprocVecEnv(factories,start_method='spawn') if len(factories)>1 else DummyVecEnv(factories)
    check_env(make_environment(config),warn=True)
    if args.evaluate_only:
        model=PPO.load(args.evaluate_only,device='cpu')
        validate_checkpoint(model,config)
    else:
        if args.resume:
            model=PPO.load(args.resume,env=env,device='cpu')
            validate_checkpoint(model,config)
        else: model=PPO('MlpPolicy',env,seed=s['seed'],device='cpu',n_steps=s['n_steps'],batch_size=s['batch_size'],
                        n_epochs=s['n_epochs'],learning_rate=s['learning_rate'],gamma=s['gamma'],gae_lambda=s['gae_lambda'],
                        clip_range=s['clip_range'],ent_coef=s['ent_coef'],policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},verbose=0)
        model.swarm_fire_model=config['fire']['model']
        model.training_drone_count=config['drone_count']
        before=model.policy.action_net.weight.detach().clone()
        callback=TrainingProgress(config,args.output)
        model.learn(total_timesteps=args.steps or s['steps'],callback=callback,reset_num_timesteps=not bool(args.resume))
        model.save(args.output/'last_model')
        if torch.equal(before,model.policy.action_net.weight): raise ValueError('PPO actor weights did not update')
        last_validation=validation_return(model,config)
        callback.history.append({'steps':model.num_timesteps,'validation_return':last_validation})
        (args.output/'training-history.json').write_text(json.dumps(callback.history,indent=2)+'\n')
        if callback.best>last_validation:
            model=PPO.load(args.output/'best_validation_model.zip',device='cpu')
        model.save(args.output/'ppo_model')
    env.close()
    checkpoint=Path(config['suppression']['ppo_checkpoint'])
    if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint
    exported=export_actor(model,config,checkpoint)
    print('Evaluating PPO, uniform, random and no-water on held-out fire seeds...',flush=True)
    from training.evaluation import evaluate_individual
    results=evaluate_individual(exported,config,s['test_seeds'])
    report={'algorithm':'PPO','training_timesteps':exported['training_timesteps'],
            'fire_model':config['fire']['model'],
            'training_seeds':s['training_seeds'],'validation_seeds':s['validation_seeds'],'test_seeds':s['test_seeds'],
            'training_drone_count':config['drone_count'],
            'evaluation_scope':config['learning']['fleet_backend'],
            'reward':'-100*(new_burned/initial + active/initial*dt/horizon + .01*water/(initial*cell_water_threshold))',
            'no_uniformity_reward':True,'results':results}
    if not args.evaluate_only:
        report['validation_history']=callback.history
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'training-config.json').write_text(json.dumps(config,indent=2)+'\n')
    for name,result in results.items(): print('%s: return %.3f; new burned area %.1f m²'%(name,result['mean_return'],result['mean_new_burned_area_m2']),flush=True)
    print('Portable PPO checkpoint:',checkpoint)
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except (ValueError,OSError) as error: print('PPO error:',error,file=sys.stderr);sys.exit(1)
