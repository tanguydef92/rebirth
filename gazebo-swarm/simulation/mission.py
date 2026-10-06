"""Export Gazebo playback using the same online engine as PPO and the live viewer."""
import math
import numpy as np

from simulation.engine import SwarmSimulation
from simulation.settings import validate_config
from simulation.fire_export import fire_data
from simulation.validation import validate_fire_clearance, validate_water_drops


def generate_mission(config):
    validate_config(config)
    sim = SwarmSimulation(config)
    tracks = [[] for _ in sim.controller.drones]
    snapshots, states = [], []
    previous_active = set()
    next_state = 0.0
    extinction_time = None

    def record():
        nonlocal previous_active, next_state, extinction_time
        for track, drone in zip(tracks, sim.controller.drones):
            track.append((sim.time, *drone.position, 0.0))
        observation = sim.observation()
        fire = sim.fire
        if (not snapshots or fire.active != previous_active or
                sim.time - snapshots[-1]['time'] >= config['suppression']['decision_interval'] - 1e-9 or
                sim.time >= config['duration'] - 1e-9):
            tile = config.get('visualization', {}).get('fire_tile_size', config['fire']['cell_size'])
            size = config['fire']['cell_size']
            snapshots.append({'time':sim.time,'active':[],
                              'active_added':[list(cell) for cell in sorted(fire.active-previous_active)],
                              'active_removed':[list(cell) for cell in sorted(previous_active-fire.active)],
                              'active_count':len(fire.active),
                              'render_active':[list(cell) for cell in sorted({(math.floor(x*size/tile),math.floor(y*size/tile)) for x,y in fire.active})],
                              'render_cell_size':tile,'wet':[],
                              'perimeter_m':observation['perimeter_m'],
                              'fire_potential':fire.has_fire_potential(),
                              'particle_count':len(fire.parcels['mass']), 'physics_metrics':fire.metrics()})
            previous_active = set(fire.active)
        if sim.time >= next_state-1e-9 or sim.time >= config['duration']-1e-9:
            drones = sim.last_state or [
                {'drone':d.number,'mode':d.mode,'water_l':d.water_l,'lane':d.lane,
                 'position':list(d.position),'goal':None,'target_cell':None,'blocked':False,
                 'station':d.station_index,'direction':[0,0,0]} for d in sim.controller.drones]
            states.append({'time':sim.time,'drones':drones,
                           'station_owner':sim.controller.station_owner,
                           'station_owners':list(sim.controller.station_owners),
                           'refill_queue':list(sim.controller.refill_queue),
                           'water_dropped_l':sim.controller.water_dropped_l,
                           'water_refilled_l':sim.controller.water_refilled_l,
                           'water_fractions':list(sim.controller.distribution),
                           'perimeter_wetted_m':len(sim.controller.covered_edges)*config['fire']['cell_size']})
            next_state = sim.time + config['suppression'].get('state_interval', 1)
        if fire.ignited and not fire.has_fire_potential() and extinction_time is None:
            extinction_time = sim.time

    record()
    while sim.time < config['duration']-1e-9:
        sim.step()
        record()
    streams = [[(0,*d.home[:2],-50.0,0.0)] for d in sim.controller.drones]
    for drop in sim.drops:
        stream = streams[drop['drone']-1]
        x,y,z = drop['position']
        stream.append((drop['time'],x,y,z/2,0.0))
        stream.append((min(config['duration'],drop['time']+config['suppression']['drop_duration']),x,y,-50.0,0.0))
    for stream in streams:
        if stream[-1][0] < config['duration']:
            stream.append((config['duration'],*stream[-1][1:]))
    events = []
    for state in states:
        visible = [d['drone'] for d in state['drones'] if d['mode']=='drop']
        if not events or visible != events[-1]['drones']:
            events.append({'time':state['time'],'drones':visible})
    metrics = {**sim.metrics(), 'fire_model':'lagrangian',
               'unique_burned_area_m2':len(sim.fire.burned)*config['fire']['cell_size']**2,
               'extinguished_by_water':sim.fire.extinguished_by_water,
               'fire_extinguished_time':extinction_time,'retained_water_l':sim.fire.water_received}
    return {'trajectories':tracks,'fire':fire_data(config,sim.fire,snapshots),
            'navigation':{'mode':'suppression','goal_positions':[list(d.home[:2]) for d in sim.controller.drones],
                          'plans':[],'replans':sim.controller.replans,'arrival_time':None},
            'suppression':{'states':states,'drops':sim.drops,'decisions':sim.controller.decisions,
                           'refill_points':[list(s) for s in sim.controller.stations],
                           'refill_events':sim.controller.refill_events,'metrics':metrics,
                           'stream_events':events,'stream_trajectories':streams}}


def validate_paths(config, mission):
    """Independently verify fire clearance, tanks, services and return-home poses."""
    positions = np.asarray(mission['trajectories'])[:,:,1:4].transpose(1,0,2)
    data = mission['suppression']
    metrics = data['metrics']
    metrics.update(validate_fire_clearance(config,positions))
    metrics.update(validate_water_drops(config,data['drops'],metrics))
    homes = np.asarray(mission['navigation']['goal_positions'])
    if not np.all(np.linalg.norm(positions[-1,:,:2]-homes,axis=1)<1e-6) or not np.all(np.abs(positions[-1,:,2]-.3)<1e-6):
        raise ValueError('A drone did not land at its home')
    tracks = mission['trajectories']
    times = np.asarray(tracks[0])[:,0]
    services = {}
    for event in data['refill_events']:
        if event['event'] not in ('service_started','filled'):
            continue
        station = data['refill_points'][event['station']]
        index = int(np.searchsorted(times,event['time']))
        if math.dist(tracks[event['drone']-1][index][1:4],station)>1e-6:
            raise ValueError('Refill occurred away from the reserved station')
        if event['event']=='service_started':
            services[event['drone']] = event
        else:
            start = services.pop(event['drone'],None)
            duration = config['suppression']['refill_duration']
            if start is None or start['station'] != event['station'] or abs(event['time']-start['time']-duration)>1e-8:
                raise ValueError('Refill duration or station changed during service')
    for state in data['states']:
        owners = [v for v in state['station_owners'] if v is not None]
        if len(owners)!=len(set(owners)):
            raise ValueError('A drone reserved multiple stations')
        for drone in state['drones']:
            if not -1e-8<=drone['water_l']<=config['suppression']['tank_capacity_l']+1e-8:
                raise ValueError('Tank capacity exceeded')
            if drone['mode']=='refilling' and state['station_owners'][drone['station']]!=drone['drone']:
                raise ValueError('Refilling without an exclusive reservation')
