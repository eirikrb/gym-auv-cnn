# gym-auv-CNN
DRL feature extraction via VAE-enoder.

## Installation
Tested in Ubuntu 22.04.
Python version: 3.10.x

### Install Acados
Assumed you have git, make and cmake installed on your system
Do not clone into you local copy of this repo, but rather next to it for more stable results
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
#### Set Acados environment variables
```
export LD_LIBRARY_PATH="<absolute_acados_root>/lib" (Broken???)
export ACADOS_SOURCE_DIR="<absolute_acados_root>"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/acados/lib/

```
### Create virtual environment (pip)
```
python3 -m venv /path/to/new/virtual/environment
source /path/to/venv/bin/activate
```
```
python3 -m pip install -r requirements.txt
pip install -e <acados_root>/interfaces/acados_template
```

#### Additional 
Download the tera renderer binaries from https://github.com/acados/tera_renderer/releases and place them in <acados_root>/bin (strip the version and platform from the binaries (e.g.t_renderer-v0.0.34 -> t_renderer). Notice that you might need to make "t_renderer" executable by right clicking on the file -> Properties -> Permissions -> Allow executing file as program.

