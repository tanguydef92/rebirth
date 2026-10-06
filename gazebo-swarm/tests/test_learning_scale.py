import copy
import math
from pathlib import Path
import random
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import swarm
from drones.policy import PPOPolicy,SECTORS,OBSERVATION_SIZE,observation_vector,action_fractions
from fire.particles import ParticleFire
from fire.grid import NEIGHBORS
from drones.spatial import SegmentIndex
from drones.controller import SwarmController


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.config=swarm.load_config(swarm.ROOT/'configs/ppo-1000-fast.json')

    def test_thousand_drone_configuration_has_four_balanced_safe_starts(self):
        ctrl=SwarmController(self.config)
        self.assertEqual(len(ctrl.drones),1000)
        self.assertEqual(len(set(ctrl.stations)),1000)
        self.assertEqual(self.config['navigation']['world_bound']*2,1000)
        self.assertEqual(self.config['fire']['cell_size'],2)
        self.assertEqual(len(ParticleFire(self.config['fire']).cells),130321)
        for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
            self.assertEqual(sum(d.home[0]*sx>0 and d.home[1]*sy>0 for d in ctrl.drones),250)
        self.assertTrue(all(d.home[:2]==s[:2] for d,s in zip(ctrl.drones,ctrl.stations)))
        paths=[[(0,*d.home,0),(1,*d.home,0)] for d in ctrl.drones]
        metrics=swarm.validate_trajectories(self.config,paths)
        self.assertTrue(metrics['minimum_separation_is_lower_bound'])
        self.assertGreaterEqual(metrics['minimum_separation_m'],5)

    def test_spatial_continuous_validator_catches_crossings_between_samples(self):
        paths=[[(0,-6,0,20,0),(1,6,0,20,0)],[(0,6,0,20,0),(1,-6,0,20,0)]]
        paths.extend([[(0,100+6*i,100,20,0),(1,100+6*i,100,20,0)] for i in range(199)])
        with self.assertRaisesRegex(ValueError,'séparation'):
            swarm.validate_trajectories(self.config,paths)
        paths[1]=[(0,6,0,26,0),(1,-6,0,26,0)]
        self.assertGreaterEqual(swarm.validate_trajectories(self.config,paths)['minimum_separation_m'],5)

    def test_swept_index_never_omits_a_potentially_unsafe_pair(self):
        rng=random.Random(7);index=SegmentIndex(5);segments=[]
        from drones.geometry import segment_separation
        for i in range(80):
            a=tuple(rng.uniform(-20,20) for _ in range(3))
            b=tuple(v+rng.uniform(-15,15) for v in a)
            found=index.query(a,b)
            for j,(c,d) in enumerate(segments):
                if segment_separation(a,b,c,d)<5:
                    self.assertIn(j,found)
            index.update(i,a,b);segments.append((a,b))
        index.update(0,(100,100,100),(100,100,100))
        self.assertIn(0,index.query((100,100,100),(100,100,100)))
        self.assertNotIn(0,index.query(*segments[0]))

    def test_sparse_fire_frontier_stays_consistent_after_spread_and_water(self):
        c={**self.config['fire'],'domain':[-12,12,-12,12],'initial_rectangle':[-3,3,-3,3],'terrain_regions':[]}
        fire=ParticleFire(c);fire.step(0)
        for t in range(1,9):
            if fire.frontier:
                cell=sorted(fire.frontier)[0]
                fire.apply_water([v*2 for v in cell],20,.9,allowed_cells=fire.frontier)
            fire.step(t*3)
            expected={cell for cell in fire.active if any((cell[0]+dx,cell[1]+dy) not in fire.active for dx,dy in NEIGHBORS)}
            self.assertEqual(fire.frontier,expected)
        obs=fire.observe(30,compact=True)
        self.assertEqual(obs['active_count'],len(fire.active))
        self.assertEqual({tuple(item['cell']) for item in obs['geometry']},fire.frontier)


    def test_portable_actor_is_causal_and_masks_absent_fronts(self):
        weights=[[0.0]*OBSERVATION_SIZE for _ in range(SECTORS)]
        weights[0][0]=2
        actor=PPOPolicy({'format':'swarm-ppo-v1','algorithm':'PPO','observation_size':OBSERVATION_SIZE,
                         'sectors':SECTORS,'layers':[{'weight':weights,'bias':[0.0]*SECTORS,'activation':'linear'}]})
        c={**self.config['fire'],'domain':[-3,3,-3,3],'initial_rectangle':[0,0,0,0],'terrain_regions':[]}
        fire=ParticleFire(c);fire.step(0)
        vector,sectors=observation_vector(self.config,fire.observe(0,compact=True),{'water_fraction':1,'ready_fraction':1},0)
        self.assertEqual(len(vector),OBSERVATION_SIZE)
        fractions=action_fractions(actor.predict(vector),sectors)
        self.assertAlmostEqual(sum(fractions),1)
        self.assertTrue(all(not fraction for sector,fraction in zip(sectors,fractions) if not sector))
        self.assertEqual(actor.predict(vector),actor.predict(list(vector)))
        with self.assertRaises(ValueError):action_fractions([float('nan')]*SECTORS,sectors)

    def test_scaled_target_scheduler_uses_observed_front_and_safe_goals(self):
        c=copy.deepcopy(self.config);c['suppression']['allocation_model']='uniform'
        ctrl=SwarmController(c);fire=ParticleFire(c['fire']);fire.step(0)
        for d in ctrl.drones:d.position=d.holding
        observation=fire.observe(90,compact=True)
        ctrl.step(90,.5,observation)
        self.assertGreater(ctrl.replans,0)
        self.assertAlmostEqual(sum(ctrl.distribution),1)
        targets=[d.target for d in ctrl.drones if d.target is not None]
        self.assertEqual(len(targets),len(set(targets)))
        self.assertTrue(all(cell in fire.frontier for cell in targets))
        goals=[[(0,*d.goal,0),(1,*d.goal,0)] for d in ctrl.drones if d.target is not None]
        swarm.validate_trajectories(c,goals)
        self.assertLess(len(observation['geometry']),observation['active_count'])

    def test_thousand_drone_sdf_has_all_shared_stations_and_tile_geometry(self):
        ctrl=SwarmController(self.config)
        tracks=[[(0,*d.home,0),(600,*d.home,0)] for d in ctrl.drones]
        scenario={'navigation':{'goal_positions':[d.home[:2] for d in ctrl.drones]},
                  'fire':{'obstacle_trajectories':[[(0,10,10,4,0),(600,10,10,4,0)]],'render_cell_size':20},
                  'suppression':{'refill_points':ctrl.stations,'stream_trajectories':[[(0,0,0,-50,0),(600,0,0,-50,0)]]*1000}}
        world=swarm.world_document(self.config,tracks,Path('/tmp/swarm-scale-unit'),scenario).find('world')
        self.assertEqual(sum(m.get('name').startswith('refill_station_') for m in world.findall('model')),1000)
        self.assertEqual(len(world.find("plugin[@name='swarm::SwarmPlayback']").findall('drone')),1000)
        self.assertEqual(world.find("plugin[@name='swarm::SwarmPlayback']/direct_pose").text,'true')
        self.assertIsNone(world.find("plugin[@name='gz::sim::systems::Physics']"))
        self.assertEqual(world.find("model[@name='ground']/link/visual/geometry/box/size").text,'1000.0 1000.0 0.1')
        self.assertEqual(world.find("model[@name='fire_000']/link/visual/geometry/box/size").text,'20 20 8.0')


if __name__=='__main__':unittest.main()
