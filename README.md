# ML-server
*ML-server* is a Python script that enables sending energies and gradients to a molecular dynamics (MD) engine to perform *ML/MM simulations*. 

The supported MD engines are `sander` (Amber) and `sire` (OpenMM).

Several examples of models implemented in Python are included, such as the GPX models of 3-hydroxyflavone (published in https://doi.org/10.1039/D4DD00295D), uracil, N-methylacetamide, and alanine dipeptide.

The interface with `sire` only requires providing a `callback` function, and simulations can be run on both CPU and GPU.

The interface with `sander` can be:
*   `filebased`, which uses the file-based interface between Sander and Orca. Pro: the installation is easier because you don't need to recompile Amber. Con: the communication is slow because it's limited by reading/writing on disk.
*   `direct`, where the data is directly exchanged via python-fortran socket. Pro: faster communication. Con: more involved installation, because you have to add some files to the Amber source code and recompile it. 
In this case, simulations can be run only on CPU.

## Installation

### Quick installation
In order to use the interface with `sire` or the file-based interface with `sander`, you can simply install the python package with e.g.
```shell
git clone https://github.com/Molecolab-Pisa/ML-server
cd ML-server
pip install .
```
Instructions for installing `sire` can be found at https://sire.openbiosim.org/install.html. 

### Complete installation (`direct` interface with `sander`)
To use the faster `direct` interface you need access to the AmberTools source code files. 
1.   Copy the files in `ML-server/fortran` inside the Sander folder (`$AMBERTOOLSHOME/src/sander`).
2.   Add the instruction for compilation to `$AMBERTOOLSHOME/src/sander/CMakeLists.txt`. To do this, add the following line inside the `set` of `QM_SOURCE`, `QM_API_SOURCE`, `LES_SOURCE` (together with the other `qm2_extern_*` files).
``` shell
qm2_extern_socket_module.F90 sockets.c fsockets.f90
```
3. Add the instruction for compilation to `$AMBERTOOLSHOME/src/sander/Makefile`. To do this, add the following line to `QMOBJ`, `QMAPIOBJ`, `LESOBJ` (together with the other `qm2_extern_*` files).
``` shell
qm2_extern_socket_module.o sockets.o fsockets.o
```
4. Compile AmberTools in the usual way, with e.g. MPI support, or the other options that you may need.

Then you can install the python package as described above.

## How to use the interface with `sire`

As described in the `sire` documentation, running a ML/MM (QM/MM) simulation requires preparing a topology file and an initial coordinate file. These can be standard Amber input files, which you can load as follows:

``` python
import sire as sr
mols = sr.load("aqueous_uracil.crd", "aqueous_uracil.prmtop")
```
Next, import and load the ML models. If you want to run simulations using the models listed above, you can simply do:
```python
from ml_server.models import available_models

model = available_models['modelvacgs_ura'](workdir=path_to_workdir).load()
model = available_models['modelenvgs_ura'](workdir=path_to_workdir, model_vac=model).load()
```
With this approach, the model parameters will be downloaded automatically from https://doi.org/10.5281/zenodo.17601122. This download is required only the first time; after that, the models can be directly imported from the corresponding file. 
If you want to apply, for instance, the $\Delta$-learning correction only to the vacuum model, you should load the models as follows:
```python
from ml_server.models.models_ura import ModelVacGS, ModelVacGSDelta, ModelEnvGS

basemodel = ModelVacGS(workdir=path_to_workdir).load()
model = ModelVacGS(workdir=path_to_workdir, basemodel=model).load()
model = ModelEnvGS(workdir=path_to_workdir, model_vac=model).load()
```
Then, define the QM and MM regions and configure the simulation engine:
```python
callback = model._sire_callback

qm_mols, engine = sr.qm.create_engine(
    mols,
    mols[0],
    callback,
    cutoff="999A",
)

d = qm_mols.dynamics(
    timestep="0.5fs",
    constraint="none",
    qm_engine=engine,
    platform="gpu",
)
```
You can now run a minimization:
```python
d.minimise()
```
or a molecular dynamics simulation:
```python
d.run("100ps", energy_frequency="0.05ps", frame_frequency="0.05ps")
```

## How to use the interface with `sander`

In order to run a simulation, you need to prepare your Amber simulation files as for QM/MM dynamics with `qm_theory='EXTERN'`. 


If you want to use the `filebased` interface, you simply need to specify orca as extern method in the Amber input file:
```shell
 &orc
 /
```
Before running the simulation, you must activate the Python server, specifying the keyword `--filebased`
```shell
ml-server --filebased --model_vac MY_MODEL_VAC --model_env MY_MODEL_ENV &
```
where `MY_MODEL_VAC` and `MY_MODEL_ENV` are strings that identify the models you have implemented (see below).

To specify that you want to use the `direct` interface, select `socket` as extern method by adding the following lines to the Amber input file:
```shell
 &socket
 port = 2024
 /
```
where the port is an available AF_INET port on localhost. 
Also in this case, before running the simulation, you must activate the Python server, specifying the same port
```shell
export ML_SERVER_PORT=2024
ml-server --model_vac MY_MODEL_VAC --model_env MY_MODEL_ENV &
```

Note that it's not required to indicate an environment model: you can also run the dynamics of a molecule in vacuum. 

Then run `sander` as usual.

## How to add your model

In order to add your own model, you should:
1. create a `MYMODELS.py` file inside `ml_server/models/`. This is the file where you can implement the models.
2. Create a class for your models that inherits either from `BaseModelVac` or `BaseModelEnv`, if the model is a vacuum one or an environment one, respectively. You can find all the details in the documentation inside those two classes. All you have to do is to implement a load function (where you load whatever you may need during the prediction) and a prediction function, predicting energy and gradients. For example, for a vacuum model:
```python
from .basemodels import BaseModelVac

class MyModelVac(BaseModelVac):
    def __init__(self, workdir, *args, **kwargs):
        super(MyModelVac).__init__(workdir)
        # do something with args and kwargs

    def load(self):
        # load what you need for the prediction
        ...
        return self

    def predict(self, coords_qm, **kwargs):
        # perform the prediction
        ...
        return energy, grads_qm
```
3. Expose your models so that they can be selected from the command line interface. To do this, add a unique string identifying your model along with your model's class to the `available_models` dictionary inside `ml_server/models/__init__.py`, e.g.:
```python
available_models = {
    # ...
    "my_model_vac": MyModelVac,
    # ...
}
```
At this point, you can select your model from the command line with:
```shell
ml-server --model_vac my_model_vac &
```
Note: if you have installed this package with `pip install .`, then you should re-install it after adding your model. Otherwise, simply install in editable mode with `pip install -e .`.

## Citing ML-server
In order to cite ML-server you can use the following bibtex entry:
```
@software{ml-server2024github,
  author = {Patrizia Mazzeo and Edoardo Cignoni and Lorenzo Cupellini and Benedetta Mennucci},
  title = {ML-server},
  url = {https://github.com/Molecolab-Pisa/ML-server},
  version = {0.1.0},
  year = {2024},
}
```