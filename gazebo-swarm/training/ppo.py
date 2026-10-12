#!/usr/bin/env python3
"""Train, evaluate and export the central frontier-defense PPO actor."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from configuration import load_config
from drones.policy import PPOPolicy, SECTORS
from training.environment import make_environment, build_training_environment, validate_environment
from training.model import build_model, settings_from_config, model_settings, validate_rollout, DEFAULT_MODEL_SETTINGS
from drones.policy import observation_size

ROOT = Path(__file__).resolve().parents[1]


def validate_checkpoint(model,config):
    if getattr(model,'swarm_fire_model',None)!=config['fire']['model']:
        raise ValueError('Checkpoint fire model differs; start a fresh PPO training run')


def validate_learning(config):
    """Run/seed validation delegates environment and model checks to their builders."""
    settings = config['learning']
    validate_environment(config)
    allowed = set(DEFAULT_MODEL_SETTINGS) | {
        'sectors','seed','training_seeds','validation_seeds','test_seeds','steps',
        'episode_duration','decision_step_s','observation_time_scale_s','reward_scale',
        'water_cost_weight','fleet_backend','observation_version','num_envs','validation_interval'}
    if set(settings) - allowed:
        raise ValueError('Unknown learning setting')
    if type(settings['seed']) is not int or settings['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    groups = []
    for key in ('training_seeds','validation_seeds','test_seeds'):
        values = settings[key]
        if not isinstance(values,list) or not values or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(key + ' must be nonempty nonnegative integer seeds')
        groups.append(set(values))
    if any(a & b for i,a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('Training, validation and test seeds must be disjoint')
    if type(settings['steps']) is not int or settings['steps'] < 2:
        raise ValueError('steps must be an integer >=2')
    workers = settings.get('num_envs',1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('num_envs must be between 1 and 8')
    interval = settings.get('validation_interval',2048)
    if type(interval) is not int or interval < 1:
        raise ValueError('validation_interval must be a positive integer')
    validate_rollout(settings_from_config(config), workers)


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
    def __init__(self,config,output,resume=False):
        super().__init__();self.last_time=0.0;self.config=config;self.output=output
        self.history=[];self.best=-math.inf
        history_path=output/'training-history.json'
        if resume and history_path.exists():
            self.history=json.loads(history_path.read_text())
            self.best=max((row['validation_return'] for row in self.history),default=-math.inf)
            if (output/'ppo_model.zip').exists():
                shutil.copyfile(output/'ppo_model.zip',output/'best_validation_model.zip')
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


def run_training(config, output, *, steps=None, resume=None, evaluate_only=None,
                 actor_path=None, model_config=None, metadata=None):
    """One training/evaluation pipeline, called by both CLIs and experiments."""
    validate_learning(config)
    settings = config['learning']
    model_config = model_settings(model_config) if model_config is not None else settings_from_config(config)
    validate_rollout(model_config,settings.get('num_envs',1))
    if steps is not None and (type(steps) is not int or steps < 2):
        raise ValueError('steps must be an integer >=2')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    probe = make_environment(config)
    try:
        check_env(probe,warn=True)
    finally:
        probe.close()
    checkpoint = Path(actor_path or config['suppression']['ppo_checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = ROOT/checkpoint
    (output/'training-config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'model-config.json').write_text(json.dumps(model_config,indent=2)+'\n')
    if metadata:
        (output/'experiment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    callback = None
    if evaluate_only:
        model = build_model(model_config,checkpoint=evaluate_only)
        validate_checkpoint(model,config)
    else:
        environment = build_training_environment(config)
        try:
            model = build_model(model_config,environment,seed=settings['seed'],checkpoint=resume)
            if resume:
                validate_checkpoint(model,config)
            model.swarm_fire_model = config['fire']['model']
            model.training_drone_count = config['drone_count']
            before = model.policy.action_net.weight.detach().clone()
            callback = TrainingProgress(config,output,resume=bool(resume))
            model.learn(total_timesteps=steps or settings['steps'],callback=callback,
                        reset_num_timesteps=not bool(resume))
            model.save(output/'last_model')
            if torch.equal(before,model.policy.action_net.weight):
                raise ValueError('PPO actor weights did not update')
            last_validation = validation_return(model,config)
            callback.history.append({'steps':model.num_timesteps,'validation_return':last_validation})
            (output/'training-history.json').write_text(json.dumps(callback.history,indent=2)+'\n')
            if callback.best > last_validation:
                model = build_model(model_config,checkpoint=output/'best_validation_model.zip')
            model.save(output/'ppo_model')
        finally:
            environment.close()
    exported = export_actor(model,config,checkpoint)
    print('Evaluating PPO, uniform, random and no-water on held-out fire seeds...',flush=True)
    from training.evaluation import evaluate_individual
    results = evaluate_individual(exported,config,settings['test_seeds'])
    report = {
        'algorithm':'PPO','training_timesteps':exported['training_timesteps'],
        'fire_model':config['fire']['model'],'training_seeds':settings['training_seeds'],
        'validation_seeds':settings['validation_seeds'],'test_seeds':settings['test_seeds'],
        'training_drone_count':config['drone_count'],'evaluation_scope':settings['fleet_backend'],
        'reward':'-scale*(new_burned/initial + area_time/(cell_area*initial*horizon) + water_cost*water/(initial*cell_water_threshold))',
        'reward_scale':settings.get('reward_scale',100),
        'water_cost_weight':settings.get('water_cost_weight',.01),
        'no_uniformity_reward':True,'results':results}
    if callback is not None:
        report['validation_history'] = callback.history
    (output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,result in results.items():
        print('%s: return %.3f; new burned area %.1f m²' %
              (name,result['mean_return'],result['mean_new_burned_area_m2']),flush=True)
    print('Portable PPO checkpoint:',checkpoint)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ppo-100-fast.json')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'output/ppo-100-fast-training')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume',type=Path)
    mode.add_argument('--evaluate-only',type=Path)
    args = parser.parse_args()
    run_training(load_config(args.config),args.output,steps=args.steps,
                 resume=args.resume,evaluate_only=args.evaluate_only)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError) as error:
        print('PPO error:',error,file=sys.stderr)
        sys.exit(1)
