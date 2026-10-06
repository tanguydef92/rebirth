# Rebirth simulation prototypes

- [Gazebo swarm](gazebo-swarm/README.md): thermal wildfire simulation, PPO training with 100 individual drones, and deployment with 1,000 drones.
- [ForFiS prototype](forfis-prototype/README.md): forest-fire simulation.

From the repository root, launch the current live simulation:

```bash
bash gazebo-swarm/install-learning.sh
bash gazebo-swarm/run-live.sh --config gazebo-swarm/configs/ppo-1000-fast.json --speed 8 --open
```

The browser shows current drone directions, selected targets, altitude, tank water
and PPO sector allocations. Each drone carries 20 L, refills in 10 seconds after
arrival, and releases its tank immediately with a one-second drop interval.

Train or resume the policy:

```bash
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/train_ppo.py --steps 8192
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/train_ppo.py --resume gazebo-swarm/output/ppo-100-fast-training/last_model.zip --steps 100000
```

For native Gazebo playback on macOS Apple Silicon:

```bash
bash gazebo-swarm/install-local.sh
bash gazebo-swarm/run.sh
```

Gazebo exports and replays the same simulation engine used by PPO and the live
viewer. Its poses are prescribed kinematically; this is not an autopilot or motor
simulation. The included PPO actor is a short beta run and did not outperform
the random allocation baseline.

| Change | File |
| --- | --- |
| Training fleet, services and hyperparameters | [configs/ppo-100-fast.json](gazebo-swarm/configs/ppo-100-fast.json) |
| Deployment fleet, speeds, altitudes and fire parameters | [configs/ppo-1000-fast.json](gazebo-swarm/configs/ppo-1000-fast.json) |
| PPO training and validation selection | [training/ppo.py](gazebo-swarm/training/ppo.py) |
| Reward and individual-drone training environment | [training/individual_env.py](gazebo-swarm/training/individual_env.py) |
| Observations and sector allocations | [drones/policy.py](gazebo-swarm/drones/policy.py) |
| Assign targets and reserve refill stations | [drones/controller.py](gazebo-swarm/drones/controller.py) |
| Updated directions and collision avoidance | [drones/navigation.py](gazebo-swarm/drones/navigation.py) |
| Fire propagation and water cooling | [fire/particles.py](gazebo-swarm/fire/particles.py) |
| Online simulation and plotting | [simulation/engine.py](gazebo-swarm/simulation/engine.py), [live.html](gazebo-swarm/simulation/live.html) |
| Gazebo playback | [swarm.py](gazebo-swarm/swarm.py), [SwarmPlayback.cc](gazebo-swarm/src/SwarmPlayback.cc) |

See the [swarm guide](gazebo-swarm/README.md) and [validation results](gazebo-swarm/VALIDATION.md).
