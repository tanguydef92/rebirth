"""Configuration for the current thermal-fire and individual-drone simulator."""
import json
import math
from pathlib import Path


def finite_number(value, name, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite and numeric')
    if positive and value <= 0:
        raise ValueError(name + ' must be positive')


def validate_fire_config(fire, duration, world_bound):
    required = {'model','seed','cell_size','domain','initial_cells','ignition_time','update_interval','height'}
    optional = {'initial_rectangle','physics','terrain_regions'}
    if not isinstance(fire, dict) or not required <= set(fire) or set(fire) - required - optional:
        raise ValueError('Invalid fire keys; see configs/ppo-1000-fast.json')
    if fire['model'] != 'lagrangian':
        raise ValueError('fire.model must be lagrangian')
    if type(fire['seed']) is not int:
        raise ValueError('fire.seed must be an integer')
    for key in ('cell_size','update_interval','height'):
        finite_number(fire[key], 'fire.' + key)
    if not 1 <= fire['cell_size'] <= 5 or fire['update_interval'] < .5:
        raise ValueError('cell_size must be 1–5 m and update_interval >= .5 s')
    finite_number(fire['ignition_time'], 'ignition_time', positive=False)
    if not 0 <= fire['ignition_time'] < duration:
        raise ValueError('Ignition must occur within the simulation')
    domain = fire['domain']
    if (not isinstance(domain, list) or len(domain) != 4 or any(type(v) is not int for v in domain)
            or domain[0] > domain[1] or domain[2] > domain[3]):
        raise ValueError('fire.domain requires ordered integer [xmin,xmax,ymin,ymax]')
    if (domain[1]-domain[0]+1) * (domain[3]-domain[2]+1) > 262144:
        raise ValueError('Fire domain exceeds 262144 cells')
    if max(abs(v) for v in domain) * fire['cell_size'] + fire['cell_size']/2 > world_bound:
        raise ValueError('Fire domain exceeds world bounds')
    cells = fire['initial_cells']
    if not isinstance(cells, list) or not cells:
        raise ValueError('initial_cells must be nonempty')
    for cell in cells:
        if (not isinstance(cell, list) or len(cell) != 2 or any(type(v) is not int for v in cell)
                or not domain[0] <= cell[0] <= domain[1] or not domain[2] <= cell[1] <= domain[3]):
            raise ValueError('Initial cell lies outside the domain')
    if 'initial_rectangle' in fire:
        r = fire['initial_rectangle']
        if (not isinstance(r, list) or len(r) != 4 or any(type(v) is not int for v in r)
                or not domain[0] <= r[0] <= r[1] <= domain[1]
                or not domain[2] <= r[2] <= r[3] <= domain[3]):
            raise ValueError('initial_rectangle must lie inside the fire domain')
    from fire.particles import validate_particle_config
    validate_particle_config(fire)


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {'drone_count','pattern','duration','takeoff_duration','landing_duration','sample_hz',
                'spacing','minimum_separation','maximum_speed','show_paths','navigation','fire','suppression'}
    if not required <= set(config) or set(config) - required - {'learning','visualization'}:
        raise ValueError('Configuration keys must match configs/ppo-1000-fast.json')
    if type(config['drone_count']) is not int or not 2 <= config['drone_count'] <= 1000:
        raise ValueError('drone_count must be an integer from 2 to 1000')
    if config['pattern'] != 'suppress':
        raise ValueError('The supported mission is suppress')
    if type(config['show_paths']) is not bool:
        raise ValueError('show_paths must be boolean')
    for key in ('duration','takeoff_duration','landing_duration','sample_hz','spacing',
                'minimum_separation','maximum_speed'):
        finite_number(config[key], key)
    if not 1 <= config['sample_hz'] <= 100 or config['duration'] > 900:
        raise ValueError('Choose sample_hz 1–100 and duration <=900 s')
    if config['duration'] <= config['takeoff_duration'] + config['landing_duration']:
        raise ValueError('No time remains for the mission')
    if config['minimum_separation'] < 1:
        raise ValueError('minimum_separation must be >=1 m')
    from simulation.settings import validate_config
    validate_config(config)
    return config
