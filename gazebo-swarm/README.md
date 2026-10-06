# Thermal fire and PPO drone swarm

Train with **100 individual drones**, then deploy the same central policy with
**1,000 drones**. Training, live simulation and Gazebo exports use the same
causal engine, thermal fire, drone motion and water-service logic.

## Launch

From the repository root:

```bash
bash gazebo-swarm/install-learning.sh
bash gazebo-swarm/run-live.sh --config gazebo-swarm/configs/ppo-1000-fast.json --speed 8 --open
```

The server prints a local URL. The browser plots executed direction arrows and
the selected drone's target, altitude, velocity, tank and station. Controls:
pause, restart, speed, and fire/temperature/remaining-fuel layers. The view is in
plan; vertical motion appears in the inspector. States are computed online,
with 0.5-second navigation steps, one-second policy decisions and up to 10 Hz
of browser updates. `--daemon` keeps the server in the background; `endpoint.json`
records its URL and PID.

A full tank is **20 L / 20 kg**. Refilling takes **10 seconds after arrival**
at an exclusively reserved station. A drop releases the tank immediately and
holds position for **one second**. Water applies only to the current perimeter,
rechecked after each earlier drone's dose. Stations are shared and placed at the
1,000 home positions. Starts are balanced across the four corners of the 1 km
world. Speed: 10 m/s horizontal, 3 m/s climbing, 2 m/s descending; lanes: 20–86 m;
minimum separation: 5 m. These rapid service times are experiment parameters.

Completed runs write `report.json`, `trajectories.npz`, `refill-events.json` and
`water-drops.json` under `output/live-1000-fast/`.

## Train and evaluate

```bash
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/train_ppo.py --config gazebo-swarm/configs/ppo-100-fast.json --steps 8192 --output gazebo-swarm/output/ppo-100-fast-training
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/train_ppo.py --resume gazebo-swarm/output/ppo-100-fast-training/last_model.zip --steps 100000
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/train_ppo.py --evaluate-only gazebo-swarm/output/ppo-100-fast-training/ppo_model.zip --output gazebo-swarm/output/evaluation
```

Stable-Baselines3 PPO uses four parallel environments of 100 drones, including
real travel, altitude, collision reservations, refills and drops. It observes
87 current-state features and produces 16 sector logits, converted into water
fractions over observed frontier sectors. One feature measures available
operational water relative to frontier cooling demand. The controller turns
fractions into individual targets. Reward penalizes newly burned area, fire
persistence and water consumption; it imposes no uniformity bonus.

The included actor completed **8,192 PPO steps**. Training seeds: 101–108;
validation selection: 301–302; held-out tests: 9001–9003. Each test episode lasts
240 seconds after takeoff. Higher return is better:

| Policy | Mean return | Mean newly burned area |
| --- | ---: | ---: |
| PPO | −257.769 | 21,463 m² |
| Uniform per exposed edge | −257.936 | 21,551 m² |
| Random concentrated allocation | −255.508 | 21,199 m² |
| No water | −257.763 | 21,561 m² |

This short run does **not** establish superior PPO performance. Random allocation
performed better; the PPO/uniform difference is small relative to variation
between seeds. The 1,000-drone run verifies deployment, rather than policy
superiority at that scale. See [actor provenance](models/README.md) and
[complete evaluation](models/ppo-100-fast-evaluation.json).

The JSON actor is inference-only. `last_model.zip` includes optimizer/critic
state for resuming; `ppo_model.zip` is the selected checkpoint. Outputs and
Python environments are ignored by Git. The installed versions are recorded
in the requirements files; the learning environment isolates PyTorch from
Gazebo's conda/OpenMP runtime.

## Gazebo

```bash
bash gazebo-swarm/install-local.sh
bash gazebo-swarm/run.sh
```

Gazebo Harmonic / Sim 8 uses the generated SDF and CSV poses. The launcher uses
the current 1,000-drone configuration by default. Generation needs NumPy; the
launcher uses `.learning-venv` when available. For a prepared mission:

```bash
bash gazebo-swarm/run.sh --replay --output gazebo-swarm/output
```

`--headless --iterations 1500` runs 15 seconds without the GUI. Verify recorded
poses with `python3 gazebo-swarm/verify_telemetry.py --output gazebo-swarm/output`.
For more than 200 drones, the plugin applies poses directly without a Gazebo
physics solver. Fire visualization uses 20 m tiles while propagation remains
on 2 m cells. Poses are prescribed; motors, aerodynamics, batteries and PX4/ROS 2
are not simulated. The standalone `output/preview.html` replays the same export
with rotation, zoom and time controls.

## Fire alone

```bash
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/fire_only.py
open gazebo-swarm/output/fire-only/preview.html
```

[config-fire-only.json](config-fire-only.json) simulates 600 seconds without
water or drones. The thermal domain has 361 × 361 cells of 2 m inside the 1 km
world, wind at 2 m/s eastward, different fuel loads and a fuel-free strip.
The preview offers fire, temperature, remaining fuel and hot-gas particles.

[ParticleFire](fire/particles.py) adapts ideas from
[Petersen et al. (2023)](https://www.repository.cam.ac.uk/handle/1810/350891):
finite fuel, Arrhenius pyrolysis, transported hot gas, reaction/entrainment,
conservative heat exchange and latent-heat water cooling. Particles in one cell
are merged conservatively. A dose removes `efficiency × water mass × latent heat`
from the terrain; remaining hot gases can reignite a cooled cell.

Coefficients and terrain are uncalibrated demonstration assumptions. This is
a reduced model, not an exact reproduction of the paper. Radiation, embers,
relief, detailed chemistry and three-dimensional gas flow are omitted.

## Code and parameters

| Component | Main files |
| --- | --- |
| Configuration and safety bounds | [configuration.py](configuration.py), [simulation/settings.py](simulation/settings.py) |
| Training, reward and evaluation | [training/ppo.py](training/ppo.py), [individual_env.py](training/individual_env.py), [evaluation.py](training/evaluation.py) |
| Shared simulation and exports | [simulation/engine.py](simulation/engine.py), [mission.py](simulation/mission.py), [validation.py](simulation/validation.py) |
| Live viewer | [simulation/live.py](simulation/live.py), [live.html](simulation/live.html) |
| Targets, water and navigation | [drones/controller.py](drones/controller.py), [navigation.py](drones/navigation.py), [geometry.py](drones/geometry.py), [spatial.py](drones/spatial.py) |
| Drone state and PPO contract | [drones/state.py](drones/state.py), [policy.py](drones/policy.py) |
| Fire physics and grid geometry | [fire/particles.py](fire/particles.py), [grid.py](fire/grid.py) |
| Training/deployment settings | [configs/ppo-100-fast.json](configs/ppo-100-fast.json), [ppo-1000-fast.json](configs/ppo-1000-fast.json) |
| Gazebo models and playback | [swarm.py](swarm.py), [src/SwarmPlayback.cc](src/SwarmPlayback.cc), [Trajectory.hh](src/Trajectory.hh) |

Root scripts `train_ppo.py`, `live_sim.py` and `fire_only.py` are executable
entry points. Implementation changes belong in the packages above. Only thermal
fire, individual-drone training and the current 87-feature actor are supported.

Performance improvements include batched NumPy observations, cached preflight
states, spatial collision indexes, vectorized distances, NumPy actor inference
and parallel environments. The thermal substep remains 0.25 seconds. A 30-step
benchmark fell from 2.75 to 1.57 seconds; the first complete live 1,000-drone run
simulated 690 seconds in about 115 seconds, including validation.

## Checks

```bash
python3 -m unittest discover -s gazebo-swarm/tests
gazebo-swarm/.learning-venv/bin/python -m unittest discover -s gazebo-swarm/tests -p 'test_ppo_training.py'
bash gazebo-swarm/build.sh
```

[VALIDATION.md](VALIDATION.md) records measured simulation, learning and playback checks.
