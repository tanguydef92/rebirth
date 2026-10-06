import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swarm
from simulation import mission as suppression
from drones.policy import PPOPolicy
from fire import create_fire

try:
    import numpy as np
    from fire.particles import ParticleFire, PARCEL_FIELDS, validate_particle_config
    NUMPY = True
except ImportError:
    NUMPY = False


@unittest.skipUnless(NUMPY, "Optional Lagrangian model needs NumPy")
class ParticleTests(unittest.TestCase):
    def config(self, **physics):
        return {"model":"lagrangian", "seed":42, "cell_size":2.0,
                "domain":[-20,20,-20,20], "initial_cells":[[0,0]],
                "ignition_time":0.0, "update_interval":1.0,
                "height":8.0, "physics":physics}

    def test_water_cools_by_latent_heat_instead_of_a_fixed_litre_threshold(self):
        fire = ParticleFire(self.config(water_cooling_efficiency=.5))
        fire.step(0)
        index = fire.index((0,0))
        before = fire.temperature[index]
        capacity = fire.heat_capacity()[index]
        effect = fire.apply_water([0,0], .1, .9)
        expected = .1 * .5 * fire.physics['water_latent_heat_j_kg'] / capacity
        self.assertAlmostEqual(before-fire.temperature[index], expected)
        self.assertIn((0,0),fire.active)  # greater than the old threshold, but still hot
        dose = fire.water_needed((0,0))
        fire.apply_water([0,0],dose,.9)
        self.assertNotIn((0,0),fire.active)
        self.assertAlmostEqual(fire.temperature[index],fire.ignition-1)
        self.assertFalse(fire.observe(0)["wet"])
        self.assertAlmostEqual(effect['evaporated_l'],.05)
        self.assertLess(fire.metrics()['energy_balance_relative_error'],1e-12)

    def test_finite_fuel_burns_out_and_cannot_reignite(self):
        fire = ParticleFire(self.config(wind_m_s=[0,0],turbulence_rms_m_s=0,
                                        pyrolysis_rate_at_ignition_s=.2,maximum_pyrolysis_rate_s=.2,
                                        activation_energy_j_mol=0))
        fire.step(0)
        for time in range(1,121):fire.step(time)
        self.assertFalse(fire.active)
        self.assertFalse(fire.has_fire_potential())
        self.assertLessEqual(fire.fuel[fire.index((0,0))],fire.initial_fuel[fire.index((0,0))]*.05)
        fire.temperature[fire.index((0,0))]=2000
        fire.step(121)
        self.assertNotIn((0,0),fire.active)
        self.assertEqual(fire.burned,{(0,0)})
        self.assertTrue(np.all(fire.fuel>=0))

    def test_wind_changes_actual_transport_and_ignition_direction(self):
        results=[]
        for vx in (-1.7,1.7):
            fire=ParticleFire(self.config(wind_m_s=[vx,0],transport_factor=1,
                                         turbulence_rms_m_s=0))
            fire.step(0)
            for time in range(1,91):fire.step(time)
            results.append(fire.burned)
            self.assertGreater(len(fire.burned),1)
            self.assertTrue(all(x*vx>=0 for x,y in fire.burned))
            self.assertTrue(all(y==0 for x,y in fire.burned))
        self.assertLess(min(x for x,y in results[0]),0)
        self.assertGreater(max(x for x,y in results[1]),0)

    def test_heat_can_reignite_after_flames_were_extinguished(self):
        fire=ParticleFire(self.config(wind_m_s=[0,0],turbulence_rms_m_s=0,
                                     pyrolysis_rate_at_ignition_s=0,terrain_cooling_time_s=1e9))
        fire.step(0);fire.apply_water([0,0],100,.9)
        self.assertFalse(fire.active)
        # Controlled hot-gas packet: flame disappearance is not global extinction.
        energy=6*fire.physics['gas_specific_heat_j_kg_k']*(2000-fire.ambient)
        values=dict(x=0,y=0,vx=0,vy=0,mass=6,energy=energy,volatile=0,age=0)
        fire.parcels={key:np.array([float(values[key])]) for key in PARCEL_FIELDS}
        fire.ignition_energy_j+=energy
        self.assertTrue(fire.has_fire_potential())
        fire.step(1)
        self.assertIn((0,0),fire.active)
        self.assertGreaterEqual(fire.temperature[fire.index((0,0))],fire.ignition)
        self.assertLess(fire.metrics()['energy_balance_relative_error'],1e-12)

    def test_terrain_and_firebreak_remove_fuel_without_destroying_gas_packets(self):
        c=self.config(wind_m_s=[0,0],turbulence_rms_m_s=0)
        c['terrain_regions']=[{'bounds':[1,2,-20,20],'fuel_load_kg_m2':0}]
        c['initial_cells']=[[0,0],[1,0]]
        fire=ParticleFire(c);fire.step(0)
        self.assertNotIn((1,0),fire.active)
        self.assertEqual(fire.fuel[fire.index((1,0))],0)
        fire.step(2)
        packets=len(fire.parcels['mass'])
        removed=fire.fuel[fire.index((0,0))]
        fire.apply_firebreak([0,0,0,0])
        self.assertEqual(fire.fuel[fire.index((0,0))],0)
        self.assertFalse(fire.active)
        self.assertEqual(len(fire.parcels['mass']),packets)
        self.assertAlmostEqual(fire.fuel_removed_kg,removed)
        self.assertLess(fire.metrics()['energy_balance_relative_error'],1e-12)

    def test_mass_and_energy_are_accounted_through_transport_water_and_fuel_removal(self):
        c=self.config();c['initial_rectangle']=[-2,2,-2,2]
        fire=ParticleFire(c);fire.step(0)
        for time in range(1,91):
            fire.step(time)
            if time%7==0 and fire.frontier:
                x,y=sorted(fire.frontier)[0]
                fire.apply_water([x*2,y*2],.3,.9,allowed_cells=fire.frontier)
            if time==30:fire.apply_firebreak([5,6,-20,20])
        m=fire.metrics()
        self.assertAlmostEqual(float(fire.initial_fuel.sum()),m['remaining_fuel_kg']+m['fuel_consumed_kg']+m['fuel_removed_kg'],places=7)
        self.assertLess(m['energy_balance_relative_error'],1e-10)
        self.assertLessEqual(m['peak_particle_count'],fire.count)
        self.assertTrue(np.all(fire.temperature>=fire.ambient-1e-9))

    def test_parcel_coalescence_conserves_mass_energy_volatile_and_momentum(self):
        fire=ParticleFire(self.config())
        fire.parcels={key:np.array([1.0,2.0,3.0]) for key in PARCEL_FIELDS}
        old={key:value.copy() for key,value in fire.parcels.items()}
        fire._merge(np.array([0,0,1]))
        self.assertEqual(len(fire.parcels['mass']),2)
        for key in ('mass','energy','volatile'):
            self.assertAlmostEqual(float(old[key].sum()),float(fire.parcels[key].sum()))
        for key in ('vx','vy'):
            self.assertAlmostEqual(float((old[key]*old['mass']).sum()),float((fire.parcels[key]*fire.parcels['mass']).sum()))

    def test_keyed_randomness_reproducibility_and_observation_is_current_and_copied(self):
        a,b=ParticleFire(self.config()),ParticleFire(self.config())
        a.step(0);b.step(0)
        for time in (1,2,3,4,5):
            a.step(time);b.step(time)
            self.assertEqual(a.active,b.active)
            np.testing.assert_array_equal(a.temperature,b.temperature)
        expected=a._normal(np.array([1,2,3]),17)
        subset=a._normal(np.array([3,1]),17)
        np.testing.assert_array_equal(subset,expected[[2,0]])
        before=a.metrics()
        obs=a.observe(5,compact=True)
        self.assertEqual(a.metrics(),before)
        self.assertIn('temperature_k',obs['geometry'][0])
        self.assertIn('water_needed_l',obs['geometry'][0])
        obs['geometry'][0]['temperature_k']=0
        self.assertTrue(np.all(a.temperature>=a.ambient))
        with self.assertRaises(ValueError):a.step(4)

    def test_delayed_ignition_and_repeated_timestamp_are_causal(self):
        c=self.config();c['ignition_time']=3
        fire=ParticleFire(c)
        self.assertFalse(fire.step(2))
        fire.step(3);mass=fire.fuel.copy();temperature=fire.temperature.copy()
        fire.step(3)
        np.testing.assert_array_equal(fire.fuel,mass)
        np.testing.assert_array_equal(fire.temperature,temperature)
        self.assertEqual(fire.active,{(0,0)})

    def test_invalid_physics_and_terrain_parameters_are_rejected(self):
        for key,value in [('wind_m_s',[1]),('water_cooling_efficiency',2),('burnout_fraction',1),
                          ('fuel_load_kg_m2',-1),('ambient_temperature_k',700),('inert_load_kg_m2',0),
                          ('turbulence_rms_m_s',True),('maximum_substep_s',2),('unknown',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_particle_config(self.config(**{key:value}))
        c=self.config();c['terrain_regions']=[{'bounds':[-21,0,0,0],'fuel_load_kg_m2':0}]
        with self.assertRaises(ValueError):validate_particle_config(c)


    def test_factory_and_checkpoint_reject_unsupported_model(self):
        self.assertIsInstance(create_fire(self.config()),ParticleFire)
        c=self.config();c['model']='unsupported'
        with self.assertRaises(ValueError):create_fire(c)
        actor=PPOPolicy.load(swarm.ROOT/'models/ppo-100-fast.json')
        with self.assertRaisesRegex(ValueError,'retrain'):
            actor.validate_fire_model({'fire':c})

    def test_complete_individual_mission_uses_the_thermal_backend(self):
        c=swarm.load_config(swarm.ROOT/'configs/ppo-100-fast.json')
        c.update(drone_count=8,duration=180,takeoff_duration=30,landing_duration=30)
        c['navigation'].update(start=[-80,-80],corner_offset=80)
        c['suppression'].update(return_duration=80,altitude_lanes=[20,26],allocation_model='uniform')
        c['fire'].update(domain=[-15,15,-15,15],initial_rectangle=[-3,3,-3,3],terrain_regions=[])
        c['fire']['physics'].update(wind_m_s=[.6,0])
        mission=suppression.generate_mission(c)
        swarm.validate_trajectories(c,mission['trajectories'])
        suppression.validate_paths(c,mission)
        m=mission['suppression']['metrics']
        self.assertEqual(m['fire_model'],'lagrangian')
        self.assertGreater(m['water_dropped_l'],0)
        self.assertGreater(m['fire_physics']['water_cooling_j'],0)
        self.assertLess(m['fire_physics']['energy_balance_relative_error'],1e-10)


if __name__=='__main__':unittest.main()
