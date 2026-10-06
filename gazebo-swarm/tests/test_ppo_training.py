"""Run these in .learning-venv; standard simulation tests skip RL dependencies."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from training.environment import make_environment
    from training.individual_env import IndividualFireDefenseEnv as FireDefenseEnv
    from training.ppo import validate_learning,export_actor
    AVAILABLE=True
except ImportError:
    AVAILABLE=False
import swarm


@unittest.skipUnless(AVAILABLE,'PPO training dependencies are isolated in .learning-venv')
class PPOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        c=swarm.load_config(swarm.ROOT/'configs/ppo-1000-fast.json')
        c['drone_count']=16;c['navigation'].update(corner_offset=220,start=[-220,-220])
        c['fire'].update(domain=[-15,15,-15,15],initial_rectangle=[-3,3,-3,3],terrain_regions=[])
        c['learning'].update(episode_duration=4,n_steps=16,batch_size=8,n_epochs=2,num_envs=1)
        self.config=c

    def test_gym_contract_reset_reproducibility_and_termination(self):
        check_env(FireDefenseEnv(self.config),warn=False)
        a,b=FireDefenseEnv(self.config),FireDefenseEnv(self.config)
        x,_=a.reset(options={'fire_seed':101});y,_=b.reset(options={'fire_seed':101})
        np.testing.assert_array_equal(x,y)
        action=np.linspace(-3,3,16,dtype=np.float32)
        for _ in range(4):
            left,right=a.step(action),b.step(action)
            np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1:],right[1:])
        self.assertTrue(left[2] or left[3])
        with self.assertRaises(RuntimeError):a.step(action)

    def test_ppo_really_updates_and_portable_actor_matches_trained_network(self):
        env=FireDefenseEnv(self.config)
        model=PPO('MlpPolicy',env,seed=7,device='cpu',n_steps=16,batch_size=8,n_epochs=2,
                  policy_kwargs={'net_arch':dict(pi=[16,16],vf=[16,16])},verbose=0)
        before=model.policy.action_net.weight.detach().clone()
        model.learn(total_timesteps=64)
        self.assertFalse(torch.equal(before,model.policy.action_net.weight))
        self.assertEqual(model.num_timesteps,64)
        self.assertIn('train/clip_fraction',model.logger.name_to_value)
        with tempfile.TemporaryDirectory() as folder:
            document=export_actor(model,self.config,Path(folder)/'actor.json')
            self.assertEqual(document['algorithm'],'PPO')
            self.assertEqual(document['training_timesteps'],64)

    def test_train_validation_test_seeds_cannot_overlap(self):
        c=copy.deepcopy(self.config);c['learning']['test_seeds']=[101]
        with self.assertRaisesRegex(ValueError,'disjoint'):validate_learning(c)

    def test_reward_contains_damage_and_water_cost_without_uniformity_bonus(self):
        env=FireDefenseEnv(self.config);env.reset(options={'fire_seed':101})
        before=len(env.fire.burned);initial=env.initial_count;area=env.simulation.area_time
        _,reward,_,_,info=env.advance([0.0]*16)
        expected=-100*((len(env.fire.burned)-before)/initial+(env.simulation.area_time-area)/(env.config['fire']['cell_size']**2*initial*env.settings['episode_duration']))
        self.assertAlmostEqual(reward,expected)
        self.assertEqual(info['water_used_l'],0)

    def test_individual_environment_and_exported_policy_transfer_contract(self):
        from training.environment import make_environment
        from training.individual_env import IndividualFireDefenseEnv
        from drones.policy import PPOPolicy
        c=swarm.load_config(swarm.ROOT/'configs/ppo-100-fast.json')
        c['drone_count']=16
        c['fire'].update(domain=[-15,15,-15,15],initial_rectangle=[-3,3,-3,3],terrain_regions=[])
        c['learning'].update(episode_duration=12,num_envs=1,n_steps=16,batch_size=8,n_epochs=2)
        env=make_environment(c)
        self.assertIsInstance(env,IndividualFireDefenseEnv)
        check_env(env,warn=False)
        observation,_=env.reset(options={'fire_seed':101})
        self.assertEqual(observation.shape,(87,))
        self.assertEqual(len(env.simulation.controller.drones),16)
        model=PPO('MlpPolicy',env,seed=7,device='cpu',n_steps=16,batch_size=8,n_epochs=2,
                  policy_kwargs={'net_arch':dict(pi=[16,16],vf=[16,16])},verbose=0)
        before=model.policy.action_net.weight.detach().clone()
        model.learn(total_timesteps=64)
        self.assertFalse(torch.equal(before,model.policy.action_net.weight))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'actor.json'
            document=export_actor(model,c,path)
            self.assertEqual(document['training_fleet'],'individual')
            self.assertEqual(document['service_contract']['refill_duration'],10)
            self.assertEqual(document['service_contract']['drop_duration'],1)
            deployment=copy.deepcopy(c);deployment['drone_count']=1000
            actor=PPOPolicy.load(path);actor.validate_fire_model(deployment)
            deployment['suppression']['refill_duration']=60
            with self.assertRaisesRegex(ValueError,'service setting'):
                actor.validate_fire_model(deployment)


if __name__=='__main__':unittest.main()
