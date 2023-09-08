# gym-auv-SB3
gym-auv repository upgraded to Stable-Baselines 3


## Installation
Tested in Ubuntu 20.04.
Python version 3.8.x

### Create a virtual environment (pip)
```
python3 -m venv /path/to/new/virtual/environment
source /path/to/venv/bin/activate
```
```
python3 -m pip install -r requirements.txt
```

### Install Acados
```
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
```
```
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4
```
```
pip install -e <acados_root>/interfaces/acados_template
```

#### Set environment variables
```
export LD_LIBRARY_PATH="<absolute_acados_root>/lib"
export ACADOS_SOURCE_DIR="<absolute_acados_root>"
```

Download tera renderer executable from https://github.com/acados/tera_renderer/releases and place them in <acados_root>/bin (please strip the version and platform from the binaries (e.g.t_renderer-v0.0.34 -> t_renderer). Notice that you might need to make t_renderer executable by right clicking on the file -> Properties -> Permissions -> Allow executing file as program.

#### Misc fixes:
- Updated SB3 from 1.1.0 -> 1.8.0: Change ```run.py (line 451)```: ```agent = PPO("MlpPolicy", ...``` to ```agent = PPO("MultiInputPolicy", ...```
