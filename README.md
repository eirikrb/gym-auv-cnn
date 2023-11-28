# gym-auv-CNN
DRL feature extraction via VAE-enoder.

## Installation
Tested on:<br/>
Ubuntu 22.04.<br/>
Python 3.10.x

### Install Acados
``Acados`` is used for solving nonlinear optimization problems in the PSF module. <br/>
<br/>
Assuming you already have ``git``, ``make`` and ``cmake`` installed on your system. <br/>
(Do not clone into your local copy of this repo, but rather next to it for more stable results)
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
export LD_LIBRARY_PATH="<absolute_acados_root>/lib"
export ACADOS_SOURCE_DIR="<absolute_acados_root>"
```
(If the former export command does not work, try: ``export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/acados/lib/``)

### Create a virtual environment (pip)
```
python3 -m venv /path/to/new/virtual/environment
source /path/to/venv/bin/activate
```
```
python3 -m pip install -r requirements.txt
pip install -e <acados_root>/interfaces/acados_template
```

#### Additional 
Download the [https://github.com/acados/tera_renderer/releases](Tera renderer binaries) and place them in <acados_root>/bin.<br/>
Strip the version and platform from the binaries (e.g.: t_renderer-v0.0.34 -> t_renderer). <br/>
Notice that you might need to make "t_renderer" executable (right-click on the file -> Properties -> Permissions -> Allow executing file as program).

