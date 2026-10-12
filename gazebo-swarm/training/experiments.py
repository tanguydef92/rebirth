"""Load per-experiment setup files and keep all generated artifacts in that folder."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from configuration import validate_config
from drones.policy import SECTORS
from training.environment import episode_settings
from training.model import model_settings

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'
SETUP_FILES = ('environment.json', 'model.json', 'experiment.json')
RUN_DEFAULTS = {
    'seed': 7,
    'training_seeds': [101, 102, 103, 104],
    'validation_seeds': [301, 302],
    'test_seeds': [9001, 9002, 9003],
    'steps': 2048,
    'num_envs': 1,
    'validation_interval': 512,
}


def experiment_path(path):
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists() or len(path.parts) > 1:
        return path.resolve()
    return (EXPERIMENTS / path).resolve()


def read_document(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(str(path) + ' must contain a JSON object')
    return data


def run_path(experiment, name):
    if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Run name must be a single folder name')
    return experiment / 'runs' / name


def load_experiment(path, run_name='preview'):
    """Assemble setup once; the existing classes receive one validated config."""
    directory = experiment_path(path)
    environment = read_document(directory / 'environment.json')
    model = model_settings(read_document(directory / 'model.json'))
    supplied_run = read_document(directory / 'experiment.json')
    if set(supplied_run) - set(RUN_DEFAULTS):
        raise ValueError('Unknown experiment.json key; see training/experiments.py')
    run = {**copy.deepcopy(RUN_DEFAULTS), **supplied_run}
    config = copy.deepcopy(environment)
    episode = episode_settings(config.pop('episode', {}))
    if 'learning' in config:
        raise ValueError('Put episode settings in environment.json and run settings in experiment.json')
    output = run_path(directory, run_name)
    config['learning'] = {**model, **episode, **run,
                          'sectors': SECTORS, 'fleet_backend': 'individual', 'observation_version': 2}
    # Training and inference share this actor location, never the global models folder.
    config['suppression']['ppo_checkpoint'] = str(output / 'actor.json')
    config['suppression']['allocation_model'] = 'uniform'
    validate_config(config)
    return {'directory': directory, 'output': output, 'config': config, 'model': model,
            'sources': {'environment': environment, 'model': model, 'run': run}}


def initialize_experiment(destination, template='one_drone_calm'):
    """Copy setup files only: no classes, checkpoints, caches or previous runs."""
    destination = experiment_path(destination)
    template = experiment_path(template)
    for name in SETUP_FILES:
        read_document(template / name)
    destination.mkdir(parents=True, exist_ok=False)
    for name in SETUP_FILES:
        shutil.copyfile(template / name, destination / name)
    if (template / 'README.md').exists():
        shutil.copyfile(template / 'README.md', destination / 'README.md')
    (destination / '.gitignore').write_text('runs/\n')
    return destination


def load_run(path, run_name):
    """Use the saved setup for old runs even after the next experiment is edited."""
    directory = experiment_path(path)
    output = run_path(directory, run_name)
    config = read_document(output / 'training-config.json')
    model = model_settings(read_document(output / 'model-config.json'))
    config['suppression']['ppo_checkpoint'] = str(output / 'actor.json')
    validate_config(config)
    return {'directory': directory, 'output': output, 'config': config, 'model': model}


def save_setup_snapshot(experiment):
    output = experiment['output']
    output.mkdir(parents=True, exist_ok=True)
    for name in SETUP_FILES:
        shutil.copyfile(experiment['directory'] / name, output / name.replace('.json', '-setup.json'))


def generated_run_name():
    return datetime.now(timezone.utc).strftime('run-%Y%m%d-%H%M%S-%f')
