"""Environment construction and episode settings; no environment class copies."""
from functools import partial
import math

from drones.policy import SECTORS

EPISODE_DEFAULTS = {
    'episode_duration': 120.0,
    'decision_step_s': 1.0,
    'observation_time_scale_s': 200.0,
    'reward_scale': 100.0,
    'water_cost_weight': .01,
}


def episode_settings(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(EPISODE_DEFAULTS):
        raise ValueError('Unknown episode setting; see training/environment.py')
    result = {**EPISODE_DEFAULTS, **settings}
    for key, value in result.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value < 0 if key == 'water_cost_weight' else value <= 0)):
            raise ValueError('Invalid episode setting: ' + key)
    return result


def validate_environment(config):
    settings = config['learning']
    if settings.get('fleet_backend') != 'individual' or settings.get('observation_version') != 2:
        raise ValueError('Use fleet_backend = individual and observation_version = 2')
    if settings['sectors'] != SECTORS or not config['suppression'].get('perimeter_only'):
        raise ValueError('Training requires 16 sectors and perimeter_only')
    episode = episode_settings({key: settings[key] for key in EPISODE_DEFAULTS if key in settings})
    step = episode['decision_step_s'] * config['sample_hz']
    if abs(step - round(step)) > 1e-9:
        raise ValueError('decision_step_s must contain whole navigation steps')
    horizon = episode['episode_duration'] * config['sample_hz']
    if abs(horizon - round(horizon)) > 1e-9:
        raise ValueError('episode_duration must contain whole navigation steps')
    takeoff = config['takeoff_duration'] * config['sample_hz']
    if abs(takeoff - round(takeoff)) > 1e-9:
        raise ValueError('takeoff_duration must contain whole navigation steps')
    end = config['duration'] - config['landing_duration'] - config['suppression']['return_duration']
    if config['takeoff_duration'] + episode['episode_duration'] > end:
        raise ValueError('Training episode must finish before the return phase')


def make_environment(config, seeds=None):
    from training.individual_env import IndividualFireDefenseEnv
    validate_environment(config)
    return IndividualFireDefenseEnv(config, seeds)


def _worker(config):
    import torch
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1)
    return Monitor(make_environment(config))


def build_training_environment(config):
    """Build rollout workers independently of the policy/network factory."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    validate_environment(config)
    workers = config['learning'].get('num_envs', 1)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('num_envs must be between 1 and 8')
    factories = [partial(_worker, config) for _ in range(workers)]
    if len(factories) > 1:
        return SubprocVecEnv(factories, start_method='spawn')
    return DummyVecEnv(factories)
