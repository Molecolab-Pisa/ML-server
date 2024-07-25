# ML-server
ML-server is a Python script that allows sending energies and gradients to Sander (Amber) to perform QM/MM simulations. 

As an example of models implemented in Python, we have the GPX models of 3-hydroxyflavone published in __INSERT_PUBLICATION__.

We provide two kinds of communication between Python and Sander:
*   `filebased`, where we exploit the file-based interface between Sander and Orca. Pro: the installation is easier because you don't have to recompile Amber. Con: the communication is slow because it's limited by reading/writing on disk.
*   `direct`, where the data is directly exchanged via python-fortran socket. Pro: faster communication. Con: more involved installation, because you have to add some files to the Amber source code and recompile it. 

## Installation

### Quick installation (`filebased` only)
In order to use the file-based interface you can simply install the python package with e.g.
```shell
git clone https://github.com/Molecolab-Pisa/ML-server
cd ML-server
pip install .
```
### Complete installation (`direct` and  `filebased`)
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

## How to use

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