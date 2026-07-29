LOCAL SETUP

Python version used: Python 3.9.6

Installation commands:
.venv/bin/python -m venv .venv
.venv/bin/pip install -r requirements-local.txt

Run command (non-GUI):
export FORFIS_CONFIG=config_first_simulation.yml
.venv/bin/python main.py

Configuration used (config_first_simulation.yml):
- size: 41
- agents: 0
- wind_x: 1.0, wind_y: 0.5
- gui: false
- timesteps: 10

Files created/modified:
- requirements-local.txt (created)
- main.py (modified; added FORFIS_CONFIG env override and non-GUI fixes)
- Forest.py (modified; numpy compatibility shim)
- config_first_simulation.yml (created)
- run_first_simulation.sh (created)
- run_first_simulation.ps1 (created)
- results/first_simulation/run.log (generated)

Errors encountered and fixes applied:
1) pip failed building numpy 1.19.2 -> updated requirements-local.txt to use numpy>=1.24.0 and newer matplotlib to use prebuilt wheels. (minimal correction: update requirements file)
2) AttributeError: 'Simulation' object has no attribute 'memoryless' in non-GUI path -> fixed by setting self.memoryless = memoryless in Simulation.__init__ (main.py).
3) numpy removed aliases np.float/np.int used by hexalattice -> added a small compatibility shim in Forest.py to set np.float/int to builtin types when missing.
4) non-GUI run relied on self.fig/self.a for plotting -> created a headless matplotlib.figure.Figure() and axes in Simulation.__init__ when GUI disabled.

Results location: results/first_simulation/run.log
