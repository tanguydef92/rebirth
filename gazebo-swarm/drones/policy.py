"""PPO observation/action contract and portable actor inference.

PPO chooses sector water fractions. Travel and separation remain deterministic
safety constraints in SwarmController. No future fire state is a policy input.
"""
import json
import math
from pathlib import Path

SECTORS = 16
FEATURES_PER_SECTOR = 5
GLOBAL_FEATURES = 7
OBSERVATION_SIZE = SECTORS * FEATURES_PER_SECTOR + GLOBAL_FEATURES


def observation_size(config):
    if config.get('learning',{}).get('observation_version') != 2:
        raise ValueError('PPO requires observation_version = 2')
    return OBSERVATION_SIZE


def sector_geometry(geometry):
    front = [item for item in geometry if item['edges']]
    center = ([sum(item['center'][k] for item in front)/len(front) for k in (0,1)] if front else [0,0])
    sectors = [[] for _ in range(SECTORS)]
    for item in front:
        angle = math.atan2(item['center'][1]-center[1], item['center'][0]-center[0])
        index = int((angle + math.pi) / (2*math.pi) * SECTORS) % SECTORS
        sectors[index].append(item)
    return sectors


def observation_vector(config, observation, fleet, time):
    sectors = sector_geometry(observation['geometry'])
    total_edges = sum(len(item['edges']) for sector in sectors for item in sector) or 1
    threshold = config['suppression']['water_to_extinguish_l']
    bound = config['navigation']['world_bound']
    result = []
    for sector in sectors:
        count = len(sector) or 1
        result.extend([
            sum(len(item['edges']) for item in sector)/total_edges,
            sum(item['spread_risk'] for item in sector)/(count*4),
            sum(item.get('cooling_fraction',min(1,item['water_l']/threshold)) for item in sector)/count,
            sum(item['recent_growth'] for item in sector)/(count*4),
            sum(math.hypot(*item['center']) for item in sector)/(count*bound)])
    rectangle = config['fire'].get('initial_rectangle')
    initial_count = ((rectangle[1]-rectangle[0]+1)*(rectangle[3]-rectangle[2]+1) if rectangle
                     else len({tuple(cell) for cell in config['fire']['initial_cells']}))
    result.extend([min(4,observation.get('active_count',len(observation['active']))/max(1,initial_count)),
                   min(4,observation.get('burned_count',len(observation['active']))/max(1,initial_count)),
                   fleet['water_fraction'], fleet['ready_fraction'],
                   min(4,time/config.get('learning',{}).get('observation_time_scale_s',config['duration'])),
                   min(4,total_edges*config['fire']['cell_size']/(8*bound))])
    if observation_size(config)==OBSERVATION_SIZE:
        demand=sum(item.get('water_needed_l',max(0,threshold-item['water_l'])) for sector in sectors for item in sector)
        available=fleet.get('available_water_l',fleet['ready_fraction']*config['drone_count']*config['suppression']['tank_capacity_l'])
        result.append(min(4,available/max(1,demand)))
    return result, sectors


def action_fractions(action, sectors):
    if len(action) != SECTORS or any(not math.isfinite(float(v)) for v in action):
        raise ValueError('PPO action requires 16 finite sector logits')
    logits = [max(-4.0,min(4.0,float(v))) for v in action]
    peak = max(logits)
    weights = [math.exp(logits[i]-peak) if sectors[i] else 0 for i in range(SECTORS)]
    total = sum(weights)
    return [weight/total if total else 0 for weight in weights]


def uniform_fractions(sectors):
    edges = [sum(len(item['edges']) for item in sector) for sector in sectors]
    total = sum(edges) or 1
    return [count/total for count in edges]


class PPOPolicy:
    def __init__(self, document):
        if document.get('format') != 'swarm-ppo-v1' or document.get('algorithm') != 'PPO':
            raise ValueError('Expected an exported PPO actor (swarm-ppo-v1)')
        if document.get('observation_size') != OBSERVATION_SIZE or document.get('sectors') != SECTORS:
            raise ValueError('PPO checkpoint observation/action contract mismatch')
        self.document = document
        self.layers = document['layers']
        self.observation_size = document['observation_size']
        size = self.observation_size
        for layer in self.layers:
            if len(layer['weight']) != len(layer['bias']) or any(len(row)!=size for row in layer['weight']):
                raise ValueError('Malformed PPO network dimensions')
            if any(not math.isfinite(float(v)) for row in layer['weight'] for v in row) or any(not math.isfinite(float(v)) for v in layer['bias']):
                raise ValueError('Nonfinite PPO network weights')
            if layer['activation'] not in ('tanh','linear'):
                raise ValueError('Unsupported PPO activation')
            size = len(layer['bias'])
        if size != SECTORS:
            raise ValueError('PPO actor must produce sixteen logits')
        self.numpy_layers=None
        try:
            import numpy as np
            self.numpy=np
            self.numpy_layers=[(np.asarray(layer['weight']),np.asarray(layer['bias']),layer['activation']) for layer in self.layers]
        except ImportError:
            pass  # Legacy playback still works with the standard library alone.

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, observation):
        if len(observation) != self.observation_size:
            raise ValueError('PPO observation size mismatch')
        if self.numpy_layers is not None:
            values=self.numpy.asarray(observation,dtype=float)
            for weight,bias,activation in self.numpy_layers:
                values=weight@values+bias
                if activation=='tanh':values=self.numpy.tanh(values)
            return self.numpy.clip(values,-4,4).tolist()
        values = observation
        for layer in self.layers:
            values = [sum(w*x for w,x in zip(row,values))+bias for row,bias in zip(layer['weight'],layer['bias'])]
            if layer['activation']=='tanh':
                values = [math.tanh(v) for v in values]
        return [max(-4,min(4,value)) for value in values]

    def validate_fire_model(self, config):
        trained = self.document.get('fire_model')
        selected = config['fire'].get('model')
        if trained != selected:
            raise ValueError('PPO actor was trained on %s fire, selected %s; retrain with the selected config' % (trained,selected))
        if self.observation_size!=observation_size(config):
            raise ValueError('PPO observation version differs; use the training observation_version')
        for key,value in self.document.get('service_contract',{}).items():
            if config['suppression'].get(key)!=value:raise ValueError('PPO service setting differs: '+key)

    def allocation(self, config, observation, fleet, time):
        self.validate_fire_model(config)
        vector,sectors = observation_vector(config,observation,fleet,time)
        return action_fractions(self.predict(vector),sectors), sectors
