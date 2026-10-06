"""Local live simulation server: compute control online and stream heading arrows."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import webbrowser

import numpy as np

import swarm
from simulation.engine import SwarmSimulation
from simulation.fire_only import encode_grid

ROOT=Path(__file__).resolve().parents[1]
MODES=['takeoff','attack','drop','queued','to_refill','refilling','depart_refill','idle','returning','landing']


class LiveRuntime:
    def __init__(self,config,output,speed=8.0):
        self.config=config;self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=True)
        self.condition=threading.Condition()
        self.speed=speed;self.paused=True;self.finished=False;self.restart=False;self.error=None
        self.control_dirty=False
        self.sequence=0;self.epoch=0;self.wall_seconds=0.0
        self.engine=SwarmSimulation(config)
        self.trained_drone_count=None
        if config['suppression']['allocation_model']=='ppo':
            from drones.policy import PPOPolicy
            path=Path(config['suppression']['ppo_checkpoint'])
            path=path if path.is_absolute() else ROOT/path
            actor=PPOPolicy.load(path)
            actor.validate_fire_model(config)
            self.engine.controller.defense_policy=actor
            self.trained_drone_count=actor.document['training_drone_count']
        self.positions=[];self.times=[];self.last_png='';self.last_png_time=-math.inf
        self.packet=None
        self.publish(force=True)
        threading.Thread(target=self.run,daemon=True).start()

    def metadata(self):
        f=self.engine.fire
        return {'config':self.config,'grid':{'nx':f.nx,'ny':f.ny,'maximum_temperature_k':2000.0,
                                           'maximum_fuel_load_kg_m2':float(f.initial_fuel.max())/f.size**2},
                'modes':MODES,'trained_drone_count':self.trained_drone_count,
                'columns':['id','x','y','z','vx','vy','vz','water_l','mode','goal_x','goal_y','goal_z','station']}

    def publish(self,force=False):
        sim=self.engine
        if force or sim.time-self.last_png_time>=1-1e-9:
            self.last_png=encode_grid(sim.fire);self.last_png_time=sim.time
        directions={s['drone']:s['direction'] for s in sim.last_state}
        drones=[];counts={mode:0 for mode in MODES}
        for d in sim.controller.drones:
            velocity=directions.get(d.number,[0,0,0]);goal=d.goal or (None,None,None)
            drones.append([d.number,*[round(v,3) for v in d.position],*[round(v,3) for v in velocity],
                           round(d.water_l,2),MODES.index(d.mode),*[round(v,3) if v is not None else None for v in goal],d.station_index])
            counts[d.mode]+=1
        metrics=sim.metrics()
        fields={'time':sim.time,'epoch':self.epoch,'paused':self.paused,'finished':self.finished,
                'error':self.error,'speed_target':self.speed,'speed_actual':sim.time/max(.001,self.wall_seconds),
                'grid':self.last_png,'grid_time':self.last_png_time,'drones':drones,'mode_counts':counts,
                'active_cells':len(sim.fire.active),'burned_cells':len(sim.fire.burned),
                'allocation':sim.controller.distribution,'metrics':metrics}
        with self.condition:
            self.sequence+=1;fields['sequence']=self.sequence
            self.packet=json.dumps(fields,separators=(',',':'),allow_nan=False)
            self.condition.notify_all()

    def control(self,action):
        with self.condition:
            if action.get('command')=='restart':self.restart=True;self.paused=False
            elif action.get('command')=='pause':self.paused=True
            elif action.get('command')=='resume' and not self.finished:self.paused=False
            if 'speed' in action:
                speed=float(action['speed'])
                if not math.isfinite(speed) or not 0<speed<=50:raise ValueError('speed must be between 0 and 50')
                self.speed=speed
            self.control_dirty=True
            self.condition.notify_all()

    def finish(self):
        self.finished=True;self.paused=True
        metrics=self.engine.metrics()
        # Independent continuous checks for all recorded motion segments. The
        # renderer uses rounded positions; validation retains full precision.
        if self.positions:
            positions=np.asarray(self.positions,dtype=np.float64)
            dt=np.diff(np.asarray(self.times))
            velocities=np.diff(positions,axis=0)/dt[:,None,None]
            maximum_speed=float(np.linalg.norm(velocities,axis=2).max())
            if maximum_speed>self.config['maximum_speed']+1e-8:
                raise ValueError('Maximum drone speed exceeded')
            if np.abs(positions[:,:,:2]).max()>self.config['navigation']['world_bound']+1e-8:
                raise ValueError('A drone left the world')
            from drones.spatial import SegmentIndex
            from drones.geometry import segment_separation
            limit=self.config['minimum_separation'];closest=math.inf;checks=0
            for before,after in zip(positions,positions[1:]):
                index=SegmentIndex(limit)
                for i,(a,b) in enumerate(zip(before,after)):
                    for j in index.query(a,b):
                        distance=segment_separation(a,b,before[j],after[j]);checks+=1;closest=min(closest,distance)
                        if distance<limit-1e-8:raise ValueError('Continuous separation violation')
                    index.update(i,a,b)
            metrics.update(minimum_separation_m=limit,minimum_separation_is_lower_bound=True,
                           closest_checked_pair_m=closest if math.isfinite(closest) else None,
                           continuous_pair_checks=checks,continuous_motion_validated=True)
            metrics['maximum_speed_m_s']=maximum_speed
            homes=np.asarray([d.home for d in self.engine.controller.drones])
            metrics['all_drones_landed']=bool(np.all(np.linalg.norm(positions[-1]-homes,axis=1)<1e-6))
            if not metrics['all_drones_landed']:raise ValueError('A drone did not land at its home')
            from simulation.validation import validate_fire_clearance
            metrics.update(validate_fire_clearance(self.config,positions))
            np.savez_compressed(self.output/'trajectories.npz',time=np.asarray(self.times),position=positions)
        from simulation.validation import validate_water_drops
        metrics.update(validate_water_drops(self.config,self.engine.drops,metrics))
        events=self.engine.controller.refill_events
        services={};durations=[]
        for event in events:
            key=event['drone']
            if event['event']=='service_started':services[key]=event['time']
            elif event['event']=='filled':
                duration=event['time']-services.pop(key)
                if abs(duration-self.config['suppression']['refill_duration'])>1e-8:
                    raise ValueError('Refill duration did not match the configured service time')
                durations.append(duration)
        metrics['completed_refill_duration_s']=sorted(set(durations))
        metrics['trained_drone_count']=self.trained_drone_count
        (self.output/'refill-events.json').write_text(json.dumps(events,separators=(',',':'))+'\n')
        (self.output/'water-drops.json').write_text(json.dumps(self.engine.drops,separators=(',',':'))+'\n')
        self.final_metrics=metrics

    def run(self):
        self.times=[0.0];self.positions=[[d.position for d in self.engine.controller.drones]]
        last_publish=0.0
        while True:
            with self.condition:
                self.condition.wait_for(lambda:not self.paused or self.restart or self.control_dirty,timeout=1.0)
                if self.restart:
                    self.engine=SwarmSimulation(self.config);self.positions=[[d.position for d in self.engine.controller.drones]]
                    self.times=[0.0];self.wall_seconds=0.0;self.finished=False;self.error=None;self.restart=False;self.epoch+=1
                    self.last_png_time=-math.inf
                    self.publish(force=True)
                if self.control_dirty:
                    self.publish();self.control_dirty=False
                if self.paused:continue
            started=time.monotonic()
            try:
                self.engine.step()
                self.times.append(self.engine.time);self.positions.append([d.position for d in self.engine.controller.drones])
                if self.engine.time>=self.config['duration']-1e-9:self.finish()
            except Exception as error:
                self.error=str(error);self.paused=True
                print('Live simulation error:',error,flush=True)
            elapsed=time.monotonic()-started
            delay=max(0,self.engine.step_size/self.speed-elapsed)
            if delay:time.sleep(delay)
            self.wall_seconds+=time.monotonic()-started
            if time.monotonic()-last_publish>=.1 or self.finished or self.error:
                published_at=time.monotonic();self.publish(force=self.finished)
                self.wall_seconds+=time.monotonic()-published_at;last_publish=time.monotonic()
            if self.finished and not self.error:
                self.final_metrics['simulation_wall_seconds']=self.wall_seconds
                self.final_metrics['mean_real_time_factor']=self.engine.time/max(.001,self.wall_seconds)
                (self.output/'report.json').write_text(json.dumps({'config':self.config,'metrics':self.final_metrics},indent=2)+'\n')
                print('Live simulation complete:',json.dumps(self.final_metrics,separators=(',',':')),flush=True)


def serve(config,output,port=0,speed=8,open_browser=False):
    runtime=LiveRuntime(config,output,speed)
    template=(ROOT/'simulation/live.html').read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def json_response(self,value,status=200):
            body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/':
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(template)
            elif self.path=='/meta':self.json_response(runtime.metadata())
            elif self.path=='/state':
                self.json_response(json.loads(runtime.packet))
            elif self.path=='/events':
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.end_headers()
                sequence=-1
                try:
                    while True:
                        with runtime.condition:
                            runtime.condition.wait_for(lambda:runtime.sequence!=sequence,timeout=1)
                            sequence=runtime.sequence;packet=runtime.packet
                        self.wfile.write(('data: '+packet+'\n\n').encode());self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
            else:self.send_error(404)
        def do_POST(self):
            if self.path!='/control':self.send_error(404);return
            # This local viewer accepts controls from its own origin only.
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):self.send_error(403);return
            length=int(self.headers.get('Content-Length','0'))
            if length>2048:self.send_error(413);return
            try:
                action=json.loads(self.rfile.read(length));runtime.control(action);self.json_response({'ok':True})
            except (ValueError,TypeError):self.json_response({'error':'Invalid control'},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    endpoint={'url':'http://127.0.0.1:%d/'%server.server_port,'pid':__import__('os').getpid(),
              'config':str(config.get('_config_path',''))}
    (Path(output)/'endpoint.json').write_text(json.dumps(endpoint,indent=2)+'\n')
    print('Live swarm viewer:',endpoint['url'],flush=True)
    if open_browser:webbrowser.open(endpoint['url'])
    server.serve_forever()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/ppo-1000-fast.json')
    p.add_argument('--output',type=Path,default=ROOT/'output/live-1000-fast')
    p.add_argument('--port',type=int,default=0);p.add_argument('--speed',type=float,default=8)
    p.add_argument('--open',action='store_true');p.add_argument('--daemon',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.daemon:
        command=[sys.executable,str(ROOT/'live_sim.py'),'--config',str(args.config.resolve()),'--output',str(args.output.resolve()),'--port',str(args.port),'--speed',str(args.speed)]
        if args.open:command.append('--open')
        with (args.output/'server.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
        print('Live server PID:',process.pid,flush=True);return
    config=swarm.load_config(args.config)
    if config['suppression']['allocation_model']=='ppo':
        from drones.policy import PPOPolicy
        path=Path(config['suppression']['ppo_checkpoint']);path=path if path.is_absolute() else ROOT/path
        PPOPolicy.load(path).validate_fire_model(config)
    config['_config_path']=str(args.config.resolve())
    serve(config,args.output,args.port,args.speed,args.open)
