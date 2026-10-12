# One-drone fire-defense experiment

This folder contains setup only. Edit `environment.json` for the world, drone,
water services, fire physics and episode/reward settings; `model.json` for PPO
and network architecture; `experiment.json` for seeds, steps and worker count.
The environment has a 100 m × 100 m world, a 25 × 25 fire grid (2 m cells),
one drone starting at (-36, 0) m and its refill station at the same location.
The initial 5 × 5 fire has multiple frontier targets, so the allocation action
can select between them. The only difference between the two examples is wind.

From the repository root, replace CASE with this folder's name:

```bash
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py check CASE
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py train CASE --run trial-01
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py simulate CASE --run trial-01 --open
```

Generated artifacts belong in `runs/trial-01/` and are ignored by Git. The
implementation is shared across experiments; copy no environment or PPO classes.
See [the experiment guide](../README.md) for creating cases and resuming runs.
