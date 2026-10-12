"""The single Gym environment class: current fire, drone motion and episode reward.

Experiment setup is assembled by training/experiments.py; rollout workers are
built by training/environment.py. This class contains episode behavior only.
"""
import copy
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from drones.policy import SECTORS, observation_size
from fire import create_fire
from simulation.engine import SwarmSimulation


class IndividualFireDefenseEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config, seeds=None):
        self.config = copy.deepcopy(config)
        # step(action) supplies policy logits; None means the uniform baseline.
        self.config['suppression']['allocation_model'] = 'uniform'
        self.settings = self.config['learning']
        self.seeds = list(seeds if seeds is not None else self.settings['training_seeds'])
        self.random = random.Random(self.settings['seed'])
        self.action_space = spaces.Box(-4, 4, (SECTORS,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 4, (observation_size(config),), dtype=np.float32)
        self._preflight = {}
        self._initial_fire = None
        self.simulation = None

    @property
    def fire(self):
        return self.simulation.fire

    @property
    def time(self):
        return self.simulation.time

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random.seed(seed)
        fire_seed = (int(options['fire_seed']) if options and 'fire_seed' in options
                     else self.random.choice(self.seeds))
        config = copy.deepcopy(self.config)
        config['fire']['seed'] = fire_seed
        start = config['takeoff_duration']
        if fire_seed not in self._preflight:
            if self._initial_fire is None:
                self._initial_fire = create_fire(config['fire'])
            fire = self._initial_fire.clone()
            fire.config['seed'] = fire_seed
            fire.step(0)
            # Match deployment substeps. No water is released during takeoff.
            dt = 1 / config['sample_hz']
            for time in np.arange(dt, start + 1e-9, dt):
                fire.step(float(time))
            self._preflight[fire_seed] = fire
        fire = self._preflight[fire_seed].clone()
        self.simulation = SwarmSimulation(config, fire=fire, start_time=start)
        self.initial_count = max(1, len(fire.active))
        self.end_time = start + self.settings['episode_duration']
        self.return_sum = 0.0
        self.finished = False
        return self.observe(), {'fire_seed': fire_seed}

    def observe(self):
        vector, self.sectors = self.simulation.vector()
        return np.asarray(vector, dtype=np.float32)

    def advance(self, action, *, suppress=True):
        if self.finished:
            raise RuntimeError('Call reset after a terminated or truncated episode')
        before_burned = len(self.fire.burned)
        before_area = self.simulation.area_time
        before_water = self.simulation.controller.water_dropped_l
        stop = min(self.end_time, self.time + self.settings.get('decision_step_s', 1.0))
        while self.time < stop - 1e-9:
            self.simulation.step(action, suppress=suppress)
        newly_burned = (len(self.fire.burned) - before_burned) / self.initial_count
        exposure = ((self.simulation.area_time - before_area) /
                    (self.config['fire']['cell_size'] ** 2 * self.initial_count *
                     self.settings['episode_duration']))
        water_cost = (self.settings.get('water_cost_weight', .01) *
                      (self.simulation.controller.water_dropped_l - before_water) /
                      (self.initial_count * self.config['suppression']['water_to_extinguish_l']))
        reward = -self.settings.get('reward_scale', 100) * (newly_burned + exposure + water_cost)
        self.return_sum += reward
        terminated = self.fire.ignited and not self.fire.has_fire_potential()
        truncated = self.time >= self.end_time - 1e-9 and not terminated
        self.finished = terminated or truncated
        info = {**self.simulation.metrics(), 'episode_return': self.return_sum,
                'water_used_l': self.simulation.controller.water_dropped_l,
                'fleet_water_l': sum(d.water_l for d in self.simulation.controller.drones)}
        return self.observe(), float(reward), terminated, truncated, info

    def step(self, action):
        return self.advance(action)
