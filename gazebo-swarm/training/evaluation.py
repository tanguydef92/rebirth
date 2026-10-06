"""Independent fire-seed evaluation, parallelized separately from PPO training."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import statistics

import numpy as np

from drones.policy import PPOPolicy, SECTORS
from training.environment import make_environment


def run_episode(task):
    config,seed,name,document=task
    env=make_environment(config,[seed]);obs,_=env.reset(options={'fire_seed':seed})
    actor=PPOPolicy(document) if name=='ppo' else None
    rng=np.random.default_rng(seed)
    while True:
        if name=='ppo':transition=env.step(actor.predict(obs))
        elif name=='uniform':transition=env.advance(None)
        elif name=='random':transition=env.step(rng.uniform(-4,4,SECTORS).astype(np.float32))
        else:transition=env.advance(None,suppress=False)
        obs,_,terminated,truncated,info=transition
        if terminated or truncated:return name,{'seed':seed,**info,'policy':name}


def evaluate_individual(document,config,seeds):
    names=('ppo','uniform','random','no_water')
    results={name:[] for name in names}
    tasks=[(config,seed,name,document) for seed in seeds for name in names]
    workers=config['learning'].get('num_envs',1)
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows=list(pool.map(run_episode,tasks))
    else:rows=[run_episode(task) for task in tasks]
    for name,row in rows:results[name].append(row)
    return {name:{'episodes':rows,'mean_return':statistics.mean(r['episode_return'] for r in rows),
                  'return_std':statistics.stdev(r['episode_return'] for r in rows) if len(rows)>1 else 0,
                  'mean_new_burned_area_m2':statistics.mean(r['new_burned_area_m2'] for r in rows),
                  'mean_area_time_m2_s':statistics.mean(r['active_area_time_m2_s'] for r in rows)}
            for name,rows in results.items()}
