import os
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from gpx.bijectors import Softplus
from gpx.kernels import Linear, Matern52, Prod
from gpx.mean_functions import zero_mean
from gpx.models import GPR
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from jax import Array, jit
from jax.typing import ArrayLike

from .io import read_inpfile, read_ptchrg, write_engrad, write_pcgrad

class BaseModel:
    """base model class

    Every model should inherit from this class, as here
    we implement the basic functions needed for the interface.
    """

    def __init__(self, workdir: str) -> None:
        """
        Args:
            workdir: path to the working directory
        """
        self.workdir = workdir
        # set the paths now
        self.inpfile = os.path.join(workdir, "inpfile.xyz")
        self.ptchrg = os.path.join(workdir, "ptchrg.xyz")
        self.engrad = os.path.join(workdir, "orc_job.engrad")
        self.pcgrad = os.path.join(workdir, "orc_job.pcgrad")

    def read_sander_xyz(self):
        "reads the inpfile.xyz and ptchrg.xyz"
        # try reading the two files, if the file is not there
        # (e.g., maybe there are no atoms in the mm part)
        # handle that case and return None
        try:
            num_qm, coords_qm, elems_qm = read_inpfile(self.inpfile)
        except FileNotFoundError:
            num_qm, coords_qm, elems_qm = None, None, None

        try:
            num_mm, coords_mm, charges_mm = read_ptchrg(self.ptchrg)
        except FileNotFoundError:
            num_mm, coords_mm, charges_mm = None, None, None

        return num_qm, coords_qm, elems_qm, num_mm, coords_mm, charges_mm

    def write_engrad_pcgrad(self, e_tot=None, grads_qm=None, grads_mm=None):
        "writes the engrad and pcgrad files"
        if e_tot is not None and grads_qm is not None:
            write_engrad(self.engrad, e_tot=e_tot, grads_qm=grads_qm)

        if grads_mm is not None:
            write_pcgrad(self.pcgrad, grads_mm=grads_mm)

    def load(self):
        raise NotImplementedError


class BaseModelVac(BaseModel):
    def __init__(self, workdir):
        super().__init__(workdir)

    def predict(self, coords_qm, **kwargs):
        raise NotImplementedError

    def run(self, coords_qm=None, filebased=True):
        if filebased:
            # read input
            _, coords_qm, _, _, _, _ = self.read_sander_xyz()

        # predict energy and grads
        energies_vac, grads_vac = self.predict(coords_qm)

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies_vac, grads_qm=grads_vac, grads_mm=None
            )
        else:
            return energies_vac, grads_vac

class BaseModelEnv(BaseModel):
    def __init__(self, workdir, model_vac):
        super().__init__(workdir)
        self.model_vac = model_vac

    def predict(self, coords_qm, coords_mm, charges_mm, **kwargs):
        raise NotImplementedError

    def run(self, coords_qm=None, coords_mm=None, charges_mm=None, filebased=True):
        if filebased:
            # read input
            _, coords_qm, _, _, coords_mm, charges_mm = self.read_sander_xyz()

        # predict QM/MM interaction energy and grads
        energies, grads_qm, grads_mm = self.predict(
            coords_qm, coords_mm, charges_mm
        )

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies, grads_qm=grads_qm, grads_mm=grads_mm
            )
        else:
            return energies, grads_qm, grads_mm
