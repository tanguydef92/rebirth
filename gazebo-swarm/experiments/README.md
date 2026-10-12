# Folder-based experiments

Each experiment contains setup files. The environment, fire, drone controller
and PPO implementation are shared across all folders.

```text
experiments/
  one_drone_calm/
    environment.json   # World, drone, fire, water, episode and reward
    model.json         # PPO hyperparameters and actor/critic layers
    experiment.json    # Seeds, training steps, workers and validation frequency
    README.md
    runs/              # Generated; ignored by Git
      trial-01/
        *-setup.json          # Exact input files
        training-config.json  # Resolved environment/run setup
        model-config.json     # Resolved model settings
        experiment.json       # Run identity and requested step count
        last_model.zip        # Optimizer/critic checkpoint for resuming
        ppo_model.zip         # Selected checkpoint
        actor.json            # Portable inference weights
        training-history.json
        evaluation.json
        simulation/           # Live run outputs
  one_drone_wind/
    ...
```

The provided cases use **one drone**, a **100 m × 100 m world**, and a
**25 × 25 grid of 2 m fire cells**. The initial 5 × 5 fire gives the policy
multiple perimeter targets to choose from. The calm case has zero mean wind;
the wind case has an eastward wind of 1.5 m/s. Both share mild turbulence,
20 L tanks, 10-second refills, one-second drops and a 120-second training episode.
Full simulations include 20-second takeoff, 40-second return and 20-second landing,
for a total of 200 seconds. Training starts after takeoff.

From the repository root:

```bash
# Inspect setup, then run a small first trial.
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py check one_drone_calm
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py train one_drone_calm --run trial-01 --steps 2048

# Evaluate or inspect that run in the live viewer.
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py evaluate one_drone_calm --run trial-01
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py simulate one_drone_calm --run trial-01 --open

# Continue training from the last optimizer checkpoint.
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py train one_drone_calm --run trial-01 --resume --steps 2048
```

Install the isolated environment once with `bash gazebo-swarm/install-learning.sh`.
Names such as `one_drone_calm` refer to folders under `gazebo-swarm/experiments/`.
You can also supply a relative or absolute folder path. Omitting `--run` during
training creates a unique timestamped run folder. A named run is not replaced
by another fresh training command; use a new name or `--resume`.

## Create your next case

```bash
gazebo-swarm/.learning-venv/bin/python gazebo-swarm/experiment.py init my_case --template one_drone_calm
```

This copies only the three setup files and README. It copies no model classes,
old runs, checkpoints or caches. Edit the files and train `my_case`.

| File | What to edit |
| --- | --- |
| `environment.json` | Drone count and start, domain and initial fire, wind/fuel/thermal parameters, stations, speeds, water services. Its `episode` section controls duration, decision step, time normalization, reward scale and water cost. |
| `model.json` | `net_arch.pi` (actor layers), `net_arch.vf` (critic layers), learning rate, rollout length, batch size, epochs, discount, GAE, PPO clipping, entropy and device. |
| `experiment.json` | Training/validation/test seeds, steps, parallel worlds and validation interval. Seed groups must be disjoint. |

The policy still chooses allocations across **16 frontier sectors**; with one
drone, the controller selects a target from that allocation. It is not a direct
motor or velocity policy. Reward measures new burned area, fire persistence and
water consumption. It does not reward a particular distribution.

Each run saves the exact input files and resolved settings. Evaluation,
simulation and resuming use those saved settings, even if you subsequently
edit the experiment's setup files. Editing setup and training with a new run
name produces a separate trial. `--resume --steps N` adds N steps; PPO rounds
the requested count to complete rollouts (`n_steps × num_envs`). Held-out
comparisons against uniform, random and no-water baselines are saved automatically.

## Shared implementation

| Purpose | File |
| --- | --- |
| Load setup files and organize run artifacts | [training/experiments.py](../training/experiments.py) |
| Validate episode settings and build environment/workers | [training/environment.py](../training/environment.py) |
| The one Gym environment class and reward behavior | [training/individual_env.py](../training/individual_env.py) |
| Validate model settings and construct/resume SB3 PPO | [training/model.py](../training/model.py) |
| Shared train/select/export pipeline | [training/ppo.py](../training/ppo.py) |
| Held-out policy comparisons | [training/evaluation.py](../training/evaluation.py) |
| Observations and inference contract | [drones/policy.py](../drones/policy.py) |
| Current fire/drone dynamics | [simulation/engine.py](../simulation/engine.py) |

Changing a setup needs only JSON edits. Changing observation/action/reward
behavior belongs in the shared implementation, followed by a fresh run. PPO
itself is imported from Stable-Baselines3; experiment folders contain no copied
Gym or PPO classes. The existing `train_ppo.py --config ...` command also uses
these same builders and training pipeline.
