# Rebirth simulation prototypes

This repository contains two independent experiments:

- [Gazebo swarm beta](gazebo-swarm/README.md): ten drones following automatically
  generated trajectories, with a standalone 3D preview and CSV telemetry.
- [ForFiS prototype](forfis-prototype/README.md): forest-fire simulation.

## Run the drone universe

On macOS Apple Silicon, from this repository's root:

```bash
bash gazebo-swarm/install-local.sh
bash gazebo-swarm/run.sh
```

The local Pixi environment installs Gazebo Harmonic and builds the C++ playback
plugin. After installation, only the second command is needed. For an immediate
preview without Gazebo:

```bash
python3 gazebo-swarm/swarm.py generate
open gazebo-swarm/output/preview.html
```

## Change trajectories and targets

| File | What to change |
| --- | --- |
| [`gazebo-swarm/config.json`](gazebo-swarm/config.json) | Drone count, `pattern` (`circle`, `helix`, `sweep`), altitude, radius, grid spacing, mission timing, speed and separation limits. |
| [`gazebo-swarm/swarm.py`](gazebo-swarm/swarm.py) | `position(config, index, time)` defines each drone's `(x, y, z, yaw)` over time. Edit it for custom destinations or waypoints. `world_document()` defines the environment and drone geometry. |
| [`gazebo-swarm/src/Trajectory.hh`](gazebo-swarm/src/Trajectory.hh) | Reads CSV trajectories and interpolates poses between samples. Change only to alter playback/interpolation. |
| [`gazebo-swarm/src/SwarmPlayback.cc`](gazebo-swarm/src/SwarmPlayback.cc) | Applies poses using Gazebo's simulation clock and records actual positions. Change to alter the runtime controller. |

The default missions take off, follow a path, return to their starting positions,
and land. **There is currently no configurable target or waypoint list.** Define
destination coordinates and approach/landing behavior in `position()` to change
the target. For a simple translation of the entire mission, add the same x/y
offsets to the coordinates returned by that function. For a new end destination,
vary the offset smoothly during the mission and hold the final offset during
landing. Keep the drones separated throughout the trip.

After changing configuration or `swarm.py`, run `bash gazebo-swarm/run.sh` again:
it regenerates the world and CSV files automatically. To preview the new mission,
rerun `python3 gazebo-swarm/swarm.py generate`. After editing C++ code, first run
`bash gazebo-swarm/build.sh` to rebuild the plugin.

The flow is: **configuration → Python trajectory generator → CSV + SDF world →
C++ playback in Gazebo**. The HTML preview uses the same generated samples.
Generated results, installed packages, and build artifacts stay outside Git.
Do not edit `output/trajectories/*.csv` as the primary configuration: generation
overwrites them.

This beta is kinematic: poses are prescribed, with no motor or autopilot dynamics.
The ten-drone circle mission was validated for 100.91 simulated seconds. The
native macOS view still emits rendering errors; the standalone HTML preview is
available as a fallback. See [validation details](gazebo-swarm/VALIDATION.md).
