"""Construct the PPO environment using the actual drone simulation engine."""
from training.individual_env import IndividualFireDefenseEnv


def make_environment(config, seeds=None):
    if config['learning'].get('fleet_backend') != 'individual':
        raise ValueError('PPO requires fleet_backend = individual')
    return IndividualFireDefenseEnv(config, seeds)
