# PPO frontier-defense actors

## Current beta: train 100, deploy 1,000

`ppo-100-fast.json` is the selected actor from a genuine **8,192-step PPO run**
with **100 individual drones** and thermal/particle fire. It uses the same
navigation and water-service engine as the live 1,000-drone simulation:
20 L tanks, 10-second refills after arrival, instantaneous water doses with
one-second drop intervals, and collision reservations.

The network is **87 → 64 → 64 → 16**, tanh hidden layers and clipped mean
logits converted to sector water fractions. The added observation measures
operational water relative to frontier cooling demand. Deployment preserves
the observation and service contract; changing fleet count changes capacity
and congestion and requires separate performance evaluation.

Config: `configs/ppo-100-fast.json`; deployment: `configs/ppo-1000-fast.json`.
Seeds: training 101–108, validation 301–302, held-out tests 9001–9003.
Four parallel individual environments use the isolated package versions below.
The actor's config SHA256 records the normalized training configuration.
`ppo-100-fast-evaluation.json` preserves all held-out results and checkpoint
selection. PPO return −257.769 versus uniform −257.936, random −255.508 and
no-water −257.763; this short run does **not** establish superior performance.

Full checkpoints: `output/ppo-100-fast-training/last_model.zip` for resuming,
`ppo_model.zip` for the deployed selection. The JSON contains only inference
weights. Full 1,000-drone deployment validation is in
`output/live-1000-fast/report.json`. Use `bash gazebo-swarm/run-live.sh --open`
from the repository root to launch the latest policy.

Training dependencies: Python 3.12.14, Stable-Baselines3 2.9.0, PyTorch 2.14.1,
Gymnasium 1.4.0 and NumPy 2.5.3 on macOS arm64. The config SHA256 refers to the
configuration used for the original run, before unused configuration fields
were removed; the trained thermal parameters and observation values are preserved.
