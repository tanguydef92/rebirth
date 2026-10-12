"""PPO construction and model settings; simulation classes stay elsewhere."""
import copy
import math

DEFAULT_MODEL_SETTINGS = {
    'algorithm': 'PPO',
    'policy': 'MlpPolicy',
    'device': 'cpu',
    'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
    'n_steps': 128,
    'batch_size': 128,
    'n_epochs': 6,
    'learning_rate': .0003,
    'gamma': .99,
    'gae_lambda': .95,
    'clip_range': .2,
    'ent_coef': .01,
}


def model_settings(settings=None):
    """Resolve defaults once, allowing experiment model.json to contain overrides."""
    settings = {} if settings is None else settings
    if not isinstance(settings, dict) or set(settings) - set(DEFAULT_MODEL_SETTINGS):
        raise ValueError('Unknown model setting; see training/model.py')
    result = {**copy.deepcopy(DEFAULT_MODEL_SETTINGS), **copy.deepcopy(settings)}
    if result['algorithm'] != 'PPO' or result['policy'] != 'MlpPolicy':
        raise ValueError('The supported model is PPO with MlpPolicy')
    if result['device'] not in ('cpu', 'auto', 'mps', 'cuda'):
        raise ValueError('device must be cpu, auto, mps or cuda')
    for key in ('n_steps', 'batch_size', 'n_epochs'):
        minimum = 1 if key == 'n_epochs' else 2
        if type(result[key]) is not int or result[key] < minimum:
            raise ValueError(key + ' must be an integer >=%d' % minimum)
    for key in ('learning_rate', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + ' must be finite and numeric')
    if (result['learning_rate'] <= 0 or not 0 < result['gamma'] <= 1 or
            not 0 < result['gae_lambda'] <= 1 or not 0 < result['clip_range'] < 1 or result['ent_coef'] < 0):
        raise ValueError('Invalid PPO hyperparameter')
    architecture = result['net_arch']
    if not isinstance(architecture, dict) or set(architecture) != {'pi', 'vf'}:
        raise ValueError('net_arch needs pi and vf lists')
    for layers in architecture.values():
        if not isinstance(layers, list) or any(type(n) is not int or n < 1 for n in layers):
            raise ValueError('Network layers must be lists of positive integers')
    return result


def settings_from_config(config):
    """Use the same factory for the existing full swarm configuration CLI."""
    return model_settings({key: value for key, value in config['learning'].items()
                           if key in DEFAULT_MODEL_SETTINGS})


def validate_rollout(settings, num_envs):
    length = settings['n_steps'] * num_envs
    if length % settings['batch_size']:
        raise ValueError('n_steps * num_envs must be divisible by batch_size')


def build_model(settings, environment=None, *, seed=7, checkpoint=None):
    """Construct or resume SB3's PPO; do not copy or subclass its implementation."""
    from stable_baselines3 import PPO
    settings = model_settings(settings)
    if checkpoint is not None:
        model = PPO.load(checkpoint, env=environment, device=settings['device'])
        if model.policy_kwargs.get('net_arch') != settings['net_arch']:
            raise ValueError('Resume network differs from model.json; start a new run to change architecture')
        saved = getattr(model, 'swarm_model_settings', None)
        if saved and any(saved[key] != settings[key] for key in settings if key != 'device'):
            raise ValueError('Resume settings differ from model.json; start a new run to change hyperparameters')
        return model
    if environment is None:
        raise ValueError('A new PPO model requires an environment')
    validate_rollout(settings, getattr(environment, 'num_envs', 1))
    parameters = {key: settings[key] for key in
                  ('n_steps','batch_size','n_epochs','learning_rate','gamma','gae_lambda','clip_range','ent_coef')}
    model = PPO(settings['policy'], environment, seed=seed, device=settings['device'],
                policy_kwargs={'net_arch': settings['net_arch']}, verbose=0, **parameters)
    model.swarm_model_settings = copy.deepcopy(settings)
    return model
