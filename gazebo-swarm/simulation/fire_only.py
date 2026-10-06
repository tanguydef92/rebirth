#!/usr/bin/env python3
"""Run only the thermal fire and open an offline animated preview: no drones."""
import argparse
import base64
import copy
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib

import numpy as np

from fire import create_fire
from configuration import validate_fire_config

ROOT = Path(__file__).resolve().parents[1]


def load_settings(path):
    source = json.loads(Path(path).read_text())
    settings = {"duration":source["duration"], "world_bound":source.get("world_bound",source.get("navigation",{}).get("world_bound",500)),
                "snapshot_interval":source.get("snapshot_interval",2.0), "fire":copy.deepcopy(source["fire"])}
    for key in ("duration", "world_bound", "snapshot_interval"):
        value = settings[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(key+" must be positive and finite")
    if settings["duration"]>3600 or settings["snapshot_interval"]>settings["duration"]:
        raise ValueError("Choose duration <=3600 s and snapshot_interval <= duration")
    if settings["fire"].get("model")!="lagrangian":
        raise ValueError("This thermal preview needs fire.model = lagrangian")
    validate_fire_config(settings['fire'],settings['duration'],settings['world_bound'])
    return settings


def encode_grid(fire, maximum_temperature=2000.0):
    """Lossless RGB PNG packing of quantized temperature, fuel and cell state.

    These are simulation fields, not a rendered scene. The browser selects its
    palette after decoding. Rows run north to south for the top-down map.
    """
    channels = np.empty((fire.count,3),dtype=np.uint8)
    channels[:,0] = np.rint(np.clip((fire.temperature-fire.ambient)/(maximum_temperature-fire.ambient),0,1)*255).astype(np.uint8)
    channels[:,1] = np.rint(np.clip(fire.fuel/max(1e-12,float(fire.initial_fuel.max())),0,1)*255).astype(np.uint8)
    channels[:,2] = np.where(fire.initial_fuel==0,3,np.where(fire.flames,1,np.where(fire.ever_burning,2,0)))
    rows = channels.reshape(fire.ny,fire.nx*3)[::-1].copy()
    # PNG Sub prediction preserves all bytes while compressing uniform terrain.
    filtered = rows.copy()
    filtered[:,3:] = rows[:,3:]-rows[:,:-3]
    packed = np.empty((fire.ny,fire.nx*3+1),dtype=np.uint8)
    packed[:,0],packed[:,1:] = 1,filtered
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',fire.nx,fire.ny,8,2,0,0,0))
           +chunk(b'IDAT',zlib.compress(packed.tobytes(),6))+chunk(b'IEND',b''))
    return base64.b64encode(png).decode('ascii')


def generate(settings, output):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    fire = create_fire(settings["fire"])
    frame_times = list(np.arange(0,settings["duration"],settings["snapshot_interval"])) + [float(settings["duration"])]
    frames = []
    started = last_progress = time.monotonic()
    for stamp in frame_times:
        fire.step(float(stamp))
        p = fire.parcels
        stride = max(1,math.ceil(len(p['mass'])/256))
        particles = [[round(float(x),2),round(float(y),2)] for x,y in zip(p['x'][::stride],p['y'][::stride])]
        metrics = fire.metrics()
        if metrics['energy_balance_relative_error']>1e-9:
            raise ValueError('Thermal energy accounting failed at %.1f s'%stamp)
        frames.append({"time":float(stamp),"grid":encode_grid(fire),"active_cells":len(fire.active),
                       "burned_cells":len(fire.burned),"particles":particles,"fire_potential":fire.has_fire_potential(),
                       "metrics":metrics})
        if time.monotonic()-last_progress>20:
            print('Fire only: %.0f / %.0f s simulated'%(stamp,settings['duration']),flush=True)
            last_progress = time.monotonic()
    data = {"config":settings,"grid":{"nx":fire.nx,"ny":fire.ny,"maximum_temperature_k":2000.0,
                                     "maximum_fuel_load_kg_m2":float(fire.initial_fuel.max())/fire.size**2},
            "frames":frames}
    text = json.dumps(data,separators=(',',':')).replace('<','\\u003c')
    template = (ROOT/'fire-preview.html').read_text()
    (output/'preview.html').write_text(template.replace('__FIRE_DATA__',text))
    report = {"config":settings,"drone_count":0,"water_dropped_l":0,"frame_count":len(frames),
              "active_cells":len(fire.active),"burned_area_m2":len(fire.burned)*fire.size**2,
              "generation_seconds":time.monotonic()-started,"physics":fire.metrics()}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Fire-only simulation ready:',output/'preview.html',flush=True)
    print('%d frames; no drones or water; %.0f m² burned'%(len(frames),report['burned_area_m2']),flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config-fire-only.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/fire-only')
    args = parser.parse_args()
    generate(load_settings(args.config),args.output)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as error:
        print('Fire-only error:',error,file=sys.stderr)
        sys.exit(1)
