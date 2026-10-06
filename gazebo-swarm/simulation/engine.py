"""Causal individual-drone simulation shared by Gym training and the live viewer."""
import copy

from fire import create_fire
from drones.controller import SwarmController
from drones.state import fleet_summary
from drones.policy import observation_vector


class SwarmSimulation:
    def __init__(self, config, *, fire=None, start_time=0.0):
        self.config=copy.deepcopy(config)
        self.controller=SwarmController(self.config)
        policy=self.config['suppression']
        self.fire=fire if fire is not None else create_fire(self.config['fire'])
        self.time=float(start_time)
        self.step_size=1/self.config['sample_hz']
        self._observation=None
        self.next_fire=self.time+self.config['fire']['update_interval']
        if fire is None:self.fire.step(self.time)
        if start_time:
            for drone in self.controller.drones:
                drone.position=drone.holding
                drone.mode='attack'
        self.initial_burned=len(self.fire.burned)
        self.initial_active=len(self.fire.active)
        self.area_time=0.0
        self.drops=[]
        self.last_state=[]
        self.peak_active=len(self.fire.active)
        self.started_at=float(start_time)

    def observation(self):
        if self._observation is None:self._observation=self.fire.observe(self.time,compact=True)
        return self._observation

    def vector(self):
        fleet=fleet_summary(self.controller.drones,self.config['suppression']['tank_capacity_l'])
        return observation_vector(self.config,self.observation(),fleet,self.time)

    def step(self, action=None, *, suppress=True):
        if self.time>=self.config['duration']-1e-9:return self.last_state
        dt=min(self.step_size,self.config['duration']-self.time)
        if not suppress:
            # With no suppression, drone movement cannot affect this fire model.
            self.time=round(self.time+dt,10)
            self.fire.step(self.time)
            self._observation=None
            self.area_time+=len(self.fire.active)*self.config['fire']['cell_size']**2*dt
            self.peak_active=max(self.peak_active,len(self.fire.active))
            return self.last_state
        self.controller.external_action=action
        result=self.controller.step(self.time,dt,self.observation())
        policy=self.config['suppression']
        applied=[]
        for drop in result['drops']:
            allowed=self.fire.frontier if policy.get('perimeter_only') else None
            edges=[[x,y,dx,dy] for x,y in self.fire.footprint(drop['position'],drop['radius'])
                   if (x,y) in self.fire.frontier for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))
                   if (x+dx,y+dy) not in self.fire.active]
            effect=self.fire.apply_water(drop['position'],drop['litres'],drop['radius'],allowed_cells=allowed) if suppress else {'cells':[],'retained_l':0}
            if not effect['cells']:
                drone=self.controller.drones[drop['drone']-1]
                drone.water_l+=drop['litres']
                drone.drop_until=None
                drone.mode='attack'
                self.controller.water_dropped_l-=drop['litres']
                result['state'][drop['drone']-1]['water_l']=drone.water_l
                continue
            self.controller.covered_edges.update(self.controller.edge_key(edge) for edge in edges)
            applied.append({**drop,**effect,'perimeter_edges':edges})
        self.drops.extend(applied)
        self.time=round(self.time+dt,10)
        if getattr(self.fire,'continuous',False) or self.time>=self.next_fire-1e-9:
            self.fire.step(self.time)
            self.next_fire=self.time+self.config['fire']['update_interval']
        self._observation=None
        self.area_time+=len(self.fire.active)*self.config['fire']['cell_size']**2*dt
        self.peak_active=max(self.peak_active,len(self.fire.active))
        self.last_state=result['state']
        # Positions represent the current state. Direction is the executed,
        # collision-checked velocity of the most recent controller interval.
        for state,drone in zip(self.last_state,self.controller.drones):
            state['position']=list(drone.position)
            state['mode']=drone.mode
            state['water_l']=drone.water_l
            state['goal']=list(drone.goal) if drone.goal else None
        expected=len(self.controller.drones)*policy['tank_capacity_l']+self.controller.water_refilled_l-self.controller.water_dropped_l
        if abs(expected-sum(d.water_l for d in self.controller.drones))>1e-6:
            raise ValueError('Live simulation water conservation failed')
        return self.last_state

    def metrics(self):
        result={**self.controller.metrics(), 'elapsed':self.time-self.started_at,
                'new_burned_area_m2':(len(self.fire.burned)-self.initial_burned)*self.config['fire']['cell_size']**2,
                'active_area_time_m2_s':self.area_time,'peak_active_cells':self.peak_active,
                'final_active_cells':len(self.fire.active), 'drone_count':len(self.controller.drones),
                'remaining_tank_water_l':sum(d.water_l for d in self.controller.drones)}
        if hasattr(self.fire,'metrics'):result['fire_physics']=self.fire.metrics()
        return result
