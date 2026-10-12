"""Experiment setup, artifact isolation and independently built environment/model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.experiments import (EXPERIMENTS, SETUP_FILES, initialize_experiment,
                                  load_experiment, load_run, save_setup_snapshot)
from training.model import model_settings, validate_rollout

try:
    import numpy as np
    import torch
    from stable_baselines3.common.env_checker import check_env
    from training.environment import make_environment, build_training_environment
    from training.individual_env import IndividualFireDefenseEnv
    from training.model import build_model
    from training.ppo import export_actor, validate_learning
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


class ExperimentSetupTests(unittest.TestCase):
    def test_simple_cases_support_one_drone_and_separate_world_from_model(self):
        calm = load_experiment(EXPERIMENTS/'one_drone_calm', 'trial')
        wind = load_experiment(EXPERIMENTS/'one_drone_wind', 'trial')
        self.assertEqual(calm['config']['drone_count'], 1)
        self.assertEqual(calm['model'], wind['model'])
        self.assertNotEqual(calm['config']['fire']['physics']['wind_m_s'],
                            wind['config']['fire']['physics']['wind_m_s'])
        from drones.controller import SwarmController
        ctrl = SwarmController(calm['config'])
        self.assertEqual(ctrl.drones[0].home[:2], tuple(calm['config']['navigation']['start']))
        self.assertEqual(len(ctrl.drones), 1)

    def test_new_experiment_copies_setup_without_old_runs_or_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            source = initialize_experiment(Path(temp)/'source')
            (source/'runs'/'old').mkdir(parents=True)
            (source/'runs'/'old'/'actor.json').write_text('{}')
            destination = initialize_experiment(Path(temp)/'destination', source)
            for name in SETUP_FILES:
                self.assertEqual((source/name).read_text(), (destination/name).read_text())
            self.assertFalse((destination/'runs').exists())
            self.assertFalse(list(destination.glob('*.py')))
            with self.assertRaises(FileExistsError):
                initialize_experiment(destination, source)

    def test_runs_isolate_actor_paths_and_preserve_saved_setup_after_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = initialize_experiment(Path(temp)/'case')
            first = load_experiment(directory, 'first')
            second = load_experiment(directory, 'second')
            self.assertNotEqual(first['config']['suppression']['ppo_checkpoint'],
                                second['config']['suppression']['ppo_checkpoint'])
            self.assertTrue(Path(first['config']['suppression']['ppo_checkpoint']).is_relative_to(directory))
            save_setup_snapshot(first)
            for filename, data in [('training-config.json',first['config']),('model-config.json',first['model'])]:
                (first['output']/filename).write_text(json.dumps(data))
            model = json.loads((directory/'model.json').read_text())
            model['net_arch'] = {'pi':[8], 'vf':[8]}
            (directory/'model.json').write_text(json.dumps(model))
            saved = load_run(directory, 'first')
            self.assertEqual(saved['model'], first['model'])
            self.assertNotEqual(saved['model'], load_experiment(directory, 'next')['model'])
            self.assertTrue((first['output']/'model-setup.json').exists())
            with self.assertRaises(ValueError):
                load_experiment(directory, '../escape')

    def test_model_settings_reject_invalid_architecture_and_validate_combined_rollouts(self):
        for settings in ([], {'typo':1}, {'n_steps':True}, {'net_arch':{'pi':[0],'vf':[16]}},
                         {'learning_rate':float('nan')}, {'ent_coef':-1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                model_settings(settings)
        settings = model_settings({'n_steps':16,'batch_size':32})
        validate_rollout(settings, 2)
        with self.assertRaises(ValueError):
            validate_rollout(settings, 1)


@unittest.skipUnless(RL_AVAILABLE, 'PPO dependencies live in .learning-venv')
class ExperimentBuilderTests(unittest.TestCase):
    def test_parallel_environment_builder_can_reset_and_step_two_worlds(self):
        case = load_experiment(EXPERIMENTS/'one_drone_wind', 'test')
        case['config']['learning']['num_envs'] = 2
        workers = build_training_environment(case['config'])
        try:
            observation = workers.reset()
            self.assertEqual(observation.shape, (2,87))
            following, rewards, done, info = workers.step(np.zeros((2,16),dtype=np.float32))
            self.assertEqual(following.shape, (2,87))
            self.assertTrue(np.all(np.isfinite(rewards)))
            self.assertTrue(all(row['drone_count']==1 for row in info))
        finally:
            workers.close()

    def test_single_drone_gym_model_update_export_and_resume_use_shared_classes(self):
        torch.set_num_threads(1)
        case = load_experiment(EXPERIMENTS/'one_drone_calm', 'test')
        config = case['config']
        config['learning'].update(episode_duration=8, n_steps=16, batch_size=8, n_epochs=2)
        settings = model_settings({**case['model'], 'net_arch':{'pi':[16],'vf':[8]},
                                   'n_steps':16,'batch_size':8,'n_epochs':2})
        env = make_environment(config)
        self.assertIsInstance(env, IndividualFireDefenseEnv)
        check_env(env, warn=False)
        a, b = env.reset(seed=123)[0], env.reset(seed=123)[0]
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(env.simulation.controller.drones), 1)
        workers = build_training_environment(config)
        try:
            model = build_model(settings, workers, seed=7)
            self.assertEqual(model.policy.action_net.in_features, 16)
            before = model.policy.action_net.weight.detach().clone()
            model.learn(total_timesteps=32)
            self.assertFalse(torch.equal(before, model.policy.action_net.weight))
            with tempfile.TemporaryDirectory() as temp:
                model.swarm_fire_model = 'lagrangian'
                model.save(Path(temp)/'model')
                restored = build_model(settings, workers, checkpoint=Path(temp)/'model.zip')
                self.assertEqual(restored.num_timesteps, 32)
                changed = copy.deepcopy(settings);changed['learning_rate'] *= 2
                with self.assertRaisesRegex(ValueError, 'Resume settings'):
                    build_model(changed, workers, checkpoint=Path(temp)/'model.zip')
                document = export_actor(restored, config, Path(temp)/'actor.json')
                self.assertEqual(document['training_drone_count'], 1)
                self.assertEqual(document['observation_size'], 87)
        finally:
            workers.close()
            env.close()


if __name__ == '__main__':
    unittest.main()
