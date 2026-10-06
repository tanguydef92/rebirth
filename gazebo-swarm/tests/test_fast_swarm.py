"""Contracts for individual PPO training, rapid service and fleet transfer."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import swarm
from drones.controller import SwarmController
from drones.policy import observation_vector,observation_size,PPOPolicy
from fire import create_fire
from simulation.engine import SwarmSimulation


class FastSwarmTests(unittest.TestCase):
    def setUp(self):
        self.config=swarm.load_config(swarm.ROOT/'configs/ppo-100-fast.json')

    def test_training_and_deployment_share_physics_and_service_with_different_fleet_sizes(self):
        other=swarm.load_config(swarm.ROOT/'configs/ppo-1000-fast.json')
        self.assertEqual(self.config['drone_count'],100)
        self.assertEqual(other['drone_count'],1000)
        self.assertEqual(self.config['fire'],other['fire'])
        for key in ('refill_duration','drop_duration','tank_capacity_l','ppo_checkpoint'):
            self.assertEqual(self.config['suppression'][key],other['suppression'][key])
        self.assertEqual(other['suppression']['refill_duration'],10)
        self.assertEqual(other['suppression']['drop_duration'],1)

    def test_full_tank_impulse_one_second_drop_and_ten_second_refill(self):
        c=copy.deepcopy(self.config)
        c['fire'].update(domain=[-3,3,-3,3],initial_rectangle=[0,0,0,0],terrain_regions=[])
        f=create_fire(c['fire']);f.step(0)
        controller=SwarmController(c)
        for d in controller.drones:d.position=d.holding;d.mode='idle'
        d=controller.drones[0];d.position=(0,0,d.lane);d.goal=d.position;d.target=(0,0);d.mode='attack'
        result=controller.step(90,.5,f.observe(90,compact=True))
        self.assertEqual(len(result['drops']),1)
        self.assertEqual(result['drops'][0]['litres'],20)
        self.assertEqual(d.drop_until,91)
        self.assertEqual(d.water_l,0)
        f.apply_water([0,0],20,.9,allowed_cells=f.frontier)
        controller.step(90.5,.5,f.observe(90.5,compact=True))
        self.assertEqual(d.mode,'drop')
        controller.step(91,.5,f.observe(91,compact=True))
        self.assertEqual(d.mode,'to_refill')
        d.position=controller.stations[d.station_index]
        controller.step(92,.5,f.observe(92,compact=True))
        self.assertEqual(d.mode,'refilling')
        controller.step(101.5,.5,f.observe(101.5,compact=True))
        self.assertEqual(d.mode,'refilling');self.assertAlmostEqual(d.water_l,19)
        controller.step(102,.5,f.observe(102,compact=True))
        self.assertEqual(d.mode,'depart_refill');self.assertAlmostEqual(d.water_l,20)
        events=[e for e in controller.refill_events if e['drone']==d.number]
        start=next(e['time'] for e in events if e['event']=='service_started')
        end=next(e['time'] for e in events if e['event']=='filled')
        self.assertEqual(end-start,10)

    def test_directions_match_actual_executed_movement(self):
        c=copy.deepcopy(self.config);c['drone_count']=8
        sim=SwarmSimulation(c)
        before=[d.position for d in sim.controller.drones]
        state=sim.step([0]*16)
        for old,row,d in zip(before,state,sim.controller.drones):
            for k in range(3):self.assertAlmostEqual(old[k]+row['direction'][k]*sim.step_size,d.position[k])

    def test_resource_feature_changes_when_fleet_capacity_changes(self):
        c=self.config;f=create_fire(c['fire']);f.step(0);obs=f.observe(0,compact=True)
        small={'water_fraction':1,'ready_fraction':1,'available_water_l':2000}
        big={**small,'available_water_l':20000}
        a,_=observation_vector(c,obs,small,90);b,_=observation_vector(c,obs,big,90)
        self.assertEqual(len(a),87);self.assertEqual(a[:-1],b[:-1]);self.assertGreater(b[-1],a[-1])

    def test_cached_fire_clone_cannot_change_its_source(self):
        f=create_fire(self.config['fire']);f.step(0);f.step(5)
        g=f.clone();cell=sorted(g.active)[0];i=g.index(cell)
        original=float(f.temperature[i]);g.apply_water([v*2 for v in cell],20,.9)
        self.assertEqual(float(f.temperature[i]),original)
        g.fuel[i]=0;self.assertGreater(float(f.fuel[i]),0)


    def test_independent_validation_catches_crossing_the_fire_between_ticks(self):
        from simulation.validation import validate_fire_clearance
        c=copy.deepcopy(self.config);c['fire']['domain']=[-3,3,-3,3]
        # Both endpoints lie outside the domain, but the low segment crosses it.
        with self.assertRaisesRegex(ValueError,'possible fire volume'):
            validate_fire_clearance(c,[[[-20,0,1]],[[20,0,1]]])
        result=validate_fire_clearance(c,[[[-20,0,20]],[[20,0,20]]])
        self.assertTrue(result['continuous_fire_clearance_validated'])

    def test_independent_validation_rejects_nonperimeter_water_records(self):
        from simulation.validation import validate_water_drops
        c=self.config
        metrics={'water_dropped_l':20,'water_refilled_l':0,'remaining_tank_water_l':1980}
        drops=[{'litres':20,'cells':[[0,0]],'perimeter_edges':[[0,0,1,0]]}]
        self.assertTrue(validate_water_drops(c,drops,metrics)['water_budget_validated'])
        drops[0]['cells']=[[1,0]]
        with self.assertRaisesRegex(ValueError,'current perimeter'):
            validate_water_drops(c,drops,metrics)


if __name__=='__main__':unittest.main()
