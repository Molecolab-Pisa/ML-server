#  ML-server, Python script that allows sending energies and gradients 
#  to Sander (Amber) to perform QM/MM simulations.
#
#  Copyright (C) 2024, Patrizia Mazzeo and Edoardo Cignoni and 
#  Lorenzo Cupellini and Benedetta Mennucci
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

#  Contributed by Miguel de la Puente

"""
Models for ground-state ML/MM simulations on the radical intermediate of CvRAP1. 
Copy this file in ml_server/models/ and expose this classes in the
available_models dictionary in ml_server/models/__init__.py.
Also, copy the model files (model*.npz, *.model) in ml_server/avail_models.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from gpx.bijectors import Softplus, Identity
from gpx.kernels import Polynomial, Matern52, Prod
from gpx.mean_functions import zero_mean
from gpx.models import GPR, GPR_TD
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from jax import Array, jit
from jax.typing import ArrayLike

from ase.io import read
import numpy as np


from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator
from scipy.constants import e, Avogadro, calorie

from .basemodels import BaseModelEnv, BaseModelVac
from .models_utils import inv_dist, inv_dist_jac, elec_pot, elec_pot_jac, squared_distances

from time import perf_counter

ParameterDict = Dict[str, Parameter]
Kernel = Any

# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models/FAPmace")

# Conversion constants
eV2kcal = e*Avogadro*(1e-3)/calorie
H2kcal = 627.5094740631
eV2H = eV2kcal / H2kcal
Bohr2Ang = 0.529177210903


class ModelVacGS(BaseModelVac):
    """Model for the QM part only (vacuum), ground state, decarboxylated radical substrate of FAP.

    Predicts the QM energies and QM gradients with a MACE MLP:
    L = 0, r_cut = 10.0, num_channels=128
    See XXX.
    """

    def __init__(self, workdir: str) -> None:
        super().__init__(workdir)

    
    def load(self) -> ModelVacGS:
        import torch
        try:
            try:
                from mace.tools import utils, to_one_hot, atomic_numbers_to_indices, torch_tools
                from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol
            except ImportError as e:
                raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")
            try:
                from e3nn.util import jit
            except ImportError as e:
                raise ImportError(f"Failed to import e3nn with error: {e}. Install e3nn with 'pip install e3nn'.")
            
            # Set torch params
            torch_tools.set_default_dtype("float64")
            device = torch_tools.init_device("cuda")
            self._device = device
            
            # Load model
            model = torch.load( f=os.path.join(AVAIL_MODELS_DIR, "model_vac.model"), map_location="cuda" )
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
    
            self._model = model
            # Initialize the atomic numbers/symbols once and for all
            atom_template = read( os.path.join(AVAIL_MODELS_DIR, "template.xyz" ) )
            self._atomic_numbers = atom_template.get_atomic_numbers()
            
            z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
            self._z_table = z_table
        except:
            print("Falling back to CPU version")
            print("This is default behavior with Amber (if using OpenMM, check your setup)")

            self._device = "cpu"
            
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            

            # Use the MACE-ASE interface to load the model as a calculator
            model = MACECalculator( 
                model_paths = os.path.join(AVAIL_MODELS_DIR, "model_vac.model"),
                device = "cpu",
                default_dtype="float64"
            )
            self._model = model
            # Initialize the atomic numbers/symbols once and for all
            # -> provide one xyz file of the vacuum system in the AVAIL_MODELS_DIR folder
            atom_template = read( os.path.join(AVAIL_MODELS_DIR, "template.xyz" ) )
            self._atomic_numbers = atom_template.get_atomic_numbers()
            self._z_table = []
        
        return self

        
    def predict(
        self,
        coords_qm: Array,
        **kwargs,
    ) -> Tuple[float, Array]:
        """predicts the QM energy and QM gradients

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        """

        energy, grads = predict_vac(self._model, coords_qm, self._atomic_numbers, self._z_table, self._device)
        energy = energy * eV2H
        grads = grads * eV2H * Bohr2Ang
        
        return jnp.array(energy), grads


   

class ModelEnvGS(BaseModelEnv):
    """Environment model, ground state, CvRAP1.

    Predicts the energy of the QM part plus the QM/MM interaction, and the
    gradients of that energy w.r.t. the QM and MM atoms.
    The prediction is done with Gaussian process regression using a product kernel,
    where a linear kernel operates on the electrostatic potential of the MM part
    acting on the QM atoms, and a Matern(5/2) kernel operates on the inverse
    distances of the QM atoms.
    See https://doi.org/10.1039/D4DD00295D.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac) -> None:
        super().__init__(workdir, model_vac=model_vac)
        self._max_mm_atoms = 0

    def load(self) -> ModelEnvGS:
        sigma_energies = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_grads = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs2 = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                10.0,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(loc=10.0, scale=10.),
            )
        )

        k1_params = {
            'degree': Parameter(value=2.0, trainable=False, bijector=Softplus, prior=NormalPrior(1.0, 1.0)),
            'offset': Parameter(value=1.0, trainable=False, bijector=Identity, prior=NormalPrior(0.0, 1.0))
        }
        
        kernel_params = {"kernel1": k1_params, "kernel2": k2_lengthscale}

        ### Hard-coded: indicates the size of the ID description of the ML region (N^2) and that of the
        ### total descriptor (including the electrostatic potential, of size N)
        ind_dim = 406
        n_feat = 406 + 29
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Polynomial(no_intercept=True,active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        k = Prod(k1, k2)

        model = GPR_TD(
            kernel=k,
            kernel_params=kernel_params,
            sigma_targets=sigma_energies,
            sigma_derivs=sigma_grads,
            sigma_derivs2=sigma_grads, # Not used here but required by GPR_TD
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "model_env.npz"))
        model.print()
        self._model = model
        return self

    def get_input(
        self, ind_descr: Array, ind_jac: Array, pot_descr: Array, pot_jac_qm: Array
    ) -> Tuple[Array, Array]:
        """concatenates the inverse distances and the electrostatic potential
        descriptors/jacobians.
        """
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm

    def predict(
        self, coords_qm: Array, coords_mm: Array, charges_mm: Array, **kwargs,
    ) -> Tuple[float, Array, Array]:
        """predicts the QM/MM shift energy and QM and MM shift gradients

        Predicts the energy of the QM part plus the QM/MM interaction, and
        the gradients of that energy w.r.t. the QM and MM atoms.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part.
        coords_mm: array, shape (num_MM_atoms, 3)
            coordinates of the MM part.
        charges_mm: array, shape (num_MM_atoms,)
            charges of the MM part.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        grads_mm: array, shape (num_MM_atoms, 3)
            gradients of the energy w.r.t. the MM atoms, [hartree/bohr].
        """
        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)
        
        # concatenate descriptors
        descr, jacobian_qm = self.get_input(ind, ind_jac, pot, pot_jac_qm)

        # predict energy and grads in vacuum
        energy_vac, grads_vac = self.model_vac.predict(coords_qm)
        
        # predict QM/MM interaction energy and grads
        energy_env, grads_env_qm, grads_env_mm = predict_env(
            self._model, descr, jacobian_qm, pot_jac_mm
        )
        energy_env = energy_env.squeeze() / H2kcal
        grads_env_qm = grads_env_qm / H2kcal* Bohr2Ang
        grads_env_mm = grads_env_mm / H2kcal * Bohr2Ang

        # combine QM vacuum and QM/MM contributions
        energy = energy_vac + energy_env
        grads_qm = grads_vac + grads_env_qm
        grads_mm = grads_env_mm

        return energy, grads_qm, grads_mm


class ModelShiftGS(BaseModelEnv):
    """Environment shift model, ground state, CvRAP1.

    Predicts the QM/MM interaction, and the gradients of that energy w.r.t. the QM
    and MM atoms.
    The prediction is done with Gaussian process regression using a product kernel,
    where a linear kernel operates on the electrostatic potential of the MM part
    acting on the QM atoms, and a Matern(5/2) kernel operates on the inverse
    distances of the QM atoms.
    See https://doi.org/10.1039/D4DD00295D.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac) -> None:
        super().__init__(workdir, model_vac=model_vac)
        self._max_mm_atoms = 0

    def load(self) -> ModelEnvGS:
        sigma_energies = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_grads = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs2 = Parameter(
            1e-3,
            trainable=False,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                10.0,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(loc=10.0, scale=10.),
            )
        )

        k1_params = {
            'degree': Parameter(value=2.0, trainable=False, bijector=Softplus, prior=NormalPrior(1.0, 1.0)),
            'offset': Parameter(value=1.0, trainable=False, bijector=Identity, prior=NormalPrior(0.0, 1.0))
        }
        
        kernel_params = {"kernel1": k1_params, "kernel2": k2_lengthscale}

        ### Hard-coded: indicates the size of the ID description of the ML region (N^2) and that of the
        ### total descriptor (including the electrostatic potential, of size N)
        ind_dim = 406
        n_feat = 406 + 29
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Polynomial(no_intercept=True,active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        k = Prod(k1, k2)

        model = GPR_TD(
            kernel=k,
            kernel_params=kernel_params,
            sigma_targets=sigma_energies,
            sigma_derivs=sigma_grads,
            sigma_derivs2=sigma_grads, # Not used here but required by GPR_TD
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "model_env.npz"))
        model.print()
        self._model = model
        return self

    def get_input(
        self, ind_descr: Array, ind_jac: Array, pot_descr: Array, pot_jac_qm: Array
    ) -> Tuple[Array, Array]:
        """concatenates the inverse distances and the electrostatic potential
        descriptors/jacobians.
        """
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm

    def predict(
        self, coords_qm: Array, coords_mm: Array, charges_mm: Array, **kwargs,
    ) -> Tuple[float, Array, Array]:
        """predicts the QM/MM shift energy and QM and MM shift gradients

        Predicts the energy of the QM part plus the QM/MM interaction, and
        the gradients of that energy w.r.t. the QM and MM atoms.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part.
        coords_mm: array, shape (num_MM_atoms, 3)
            coordinates of the MM part.
        charges_mm: array, shape (num_MM_atoms,)
            charges of the MM part.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        grads_mm: array, shape (num_MM_atoms, 3)
            gradients of the energy w.r.t. the MM atoms, [hartree/bohr].
        """
        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)
        
        # concatenate descriptors
        descr, jacobian_qm = self.get_input(ind, ind_jac, pot, pot_jac_qm)

        # predict QM/MM interaction energy and grads
        energy_env, grads_env_qm, grads_env_mm = predict_env(
            self._model, descr, jacobian_qm, pot_jac_mm
        )
        energy_env = energy_env.squeeze() / H2kcal
        grads_env_qm = grads_env_qm / H2kcal* Bohr2Ang
        grads_env_mm = grads_env_mm / H2kcal * Bohr2Ang

        return energy_env, grads_env_qm, grads_env_mm


# ============================================================
# Auxiliary functions
# ============================================================

def predict_vac(
    model,
    x_qm: ArrayLike,
    atomic_numbers: ArrayLike,
    z_table: AtomicNumberTable,
    device
) -> Array:

    if device == "cpu":
        atom = Atoms(numbers=atomic_numbers, positions=x_qm, pbc = False)
        atom.calc = model
        energy, gradients = atom.get_potential_energy(), -atom.get_forces()
        return energy, gradients
    
    else:
        pos = np.ascontiguousarray(x_qm, dtype=np.float32)
    
        config = data.Configuration(
            atomic_numbers=atomic_numbers,
            positions=pos,
            properties={},
            property_weights={}
        )
    
        adata = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(model.r_max),
            heads=None
        )
    
        batch = Batch.from_data_list([adata]).to(device)
    
        output = model(batch)
        energy = output["energy"]
    
        # avoid cpu conversion unless needed
        forces = output["forces"]
    
        return energy.item(), -forces.cpu().numpy()


def _predict_env(
    x1: ArrayLike,
    x2: ArrayLike,
    params: Dict[str, Parameter],
    jaccoef1: ArrayLike,
    jaccoef2: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
    c_energies: ArrayLike,
    mu: ArrayLike,
    active_dims_m: ArrayLike,
    active_dims_p: ArrayLike,
    dipole: bool,
    jacobian_chg: ArrayLike,
) -> Array:

    ns1, nf1 = x1.shape
    ns2, nf2 = x2.shape
    nact_p = active_dims_p.shape[0]

    z1_p = x1[:, active_dims_p]
    z2_p = x2[:, active_dims_p]
    jaccoef1_p = jaccoef1[:, active_dims_p]
    jacobian_qm_p = jacobian_qm[:, active_dims_p]

    offset = params["kernel1"]["offset"].value
    degree = params["kernel1"]["degree"].value
    poly = ((offset + z1_p @ z2_p.T) ** degree) - (offset**degree)
    const1 = degree * (offset + z1_p @ z2_p.T) ** (degree - 1)
    d0k_p = jnp.einsum("st,ft->sft", const1, z2_p.T)
    d0k_jc1_p = jnp.einsum("sf,sft->st", jaccoef1_p, d0k_p)
    d1k_p = jnp.einsum("st,se->ste", const1, z1_p)
    d1k_jtqm_p = jnp.einsum("ste,tev->stv", d1k_p, jacobian_qm_p)
    tmp1 = jnp.einsum(
        "st,sfte->sfte",
        const1,
        jnp.tile(jnp.eye((nact_p)), (ns1, ns2)).reshape(ns1, nact_p, ns2, nact_p),
    )
    const2 = (degree * (degree - 1) * (offset + z1_p @ z2_p.T) ** (degree - 2))
    tmp2 = jnp.einsum("sf,st,te->setf", z1_p, const2, z2_p)
    d01k_jc1_p = jnp.einsum("sf,sfte->ste",jaccoef1_p, tmp1 + tmp2)
    d01k_jc1_jtqm_p = jnp.einsum("ste,tev->stv", d01k_jc1_p, jacobian_qm_p)


    lengthscale = params["kernel2"]["lengthscale"].value

    nact_m = active_dims_m.shape[0]
    z1_m = x1[:, active_dims_m] / lengthscale
    z2_m = x2[:, active_dims_m] / lengthscale
    jaccoef1_m = jaccoef1[:, active_dims_m]
    jacobian_qm_m = jacobian_qm[:, active_dims_m]
    diff_m = jnp.sqrt(5.0) * (z1_m[:, jnp.newaxis] - z2_m)
    d2_m = squared_distances(z1_m, z2_m)
    d_m = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2_m, 1e-36))
    expd_m = jnp.exp(-d_m)

    const = (jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d_m) * expd_m
    d01const = (5.0 / (3.0 * lengthscale**2)) * expd_m

    mat52 = (1.0 + d_m + d_m**2 / 3.0) * expd_m

    diff_jc1 = jnp.einsum("stf,sf->st", diff_m, jaccoef1_m)
    diff_jtqm = jnp.einsum("stf,tfv->stv", diff_m, jacobian_qm_m)

    d0k_jc1_m = -const * diff_jc1
    d1k_jtqm_m = -jnp.einsum("st,stv->stv", -const, diff_jtqm)

    d01k_jc1_jtqm_m = jnp.einsum("st,st,stv->stv", -d01const, diff_jc1, diff_jtqm)
    diagonal = d01const * (1.0 + d_m)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nact_m, axis=2)
    d01k_jc1_jtqm_m += jnp.einsum("sf,stf,tfv->stv", jaccoef1_m, diagonal, jacobian_qm_m)

    energy = mu + jnp.einsum("st,st,s->t", poly, mat52, c_energies)
    energy += jnp.einsum("st,st->t", poly, d0k_jc1_m)
    energy += jnp.einsum("st,st->t", d0k_jc1_p, mat52)

    grads_qm = jnp.einsum("st,stv,s->tv", poly, d1k_jtqm_m, c_energies)
    grads_qm += jnp.einsum("stv,st,s->tv", d1k_jtqm_p, mat52, c_energies)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc1_jtqm_p, mat52)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc1_p, d1k_jtqm_m)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc1_m, d1k_jtqm_p)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc1_jtqm_m, poly)

    tmp = jnp.einsum("stf,st,s->tf", d1k_p, mat52, c_energies)
    tmp += jnp.einsum("stf,st->tf", d01k_jc1_p, mat52)
    tmp += jnp.einsum("st,stf->tf", d0k_jc1_m, d1k_p)

    grads_mm = jnp.einsum("tf,tfv->tv", tmp, jacobian_mm)

    if dipole:
        d1k_jtchg_p = jnp.einsum("ste,tev->stv", d1k_p, jacobian_chg)
        d01k_jc1_jtchg_p = jnp.einsum("ste,tev->stv", d01k_jc1_p, jacobian_chg)

        dipole = jnp.einsum('stv,st,s->tv', d1k_jtchg_p, mat52, c_energies)
        dipole += jnp.einsum('stv,st->tv', d1k_jtchg_p, d0k_jc1_m)
        dipole += jnp.einsum('stv,st->tv', d01k_jc1_jtchg_p, mat52)
        dipole += jnp.einsum('stv,st->tv', d01k_jc2_jtchg_p, mat52)

        return energy, grads_qm.reshape(-1, 3), grads_mm.reshape(-1, 3), dipole.reshape(-1, 3)
        
    else:
        
        return energy, grads_qm.reshape(-1, 3), grads_mm.reshape(-1, 3)



@partial(jit, static_argnums=(0,4))
def predict_env(
    model: ModelState,
    x: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
    dipole: bool = False,
    jacobian_chg: ArrayLike = None,
):
    ind_dim = 406
    n_feat = 406 + 29
    ind_active_dims = jnp.arange(0, ind_dim)
    pot_active_dims = jnp.arange(ind_dim, n_feat)
    kernel_params = model.state.params["kernel_params"]
    return _predict_env(
        x1=model.state.x_train,
        x2=x,
        params=kernel_params,
        jaccoef1=model.state.jaccoef,
        jaccoef2=model.state.jaccoef_2,
        jacobian_qm=jacobian_qm,
        jacobian_mm=jacobian_mm,
        c_energies=model.state.c_targets,
        mu=model.state.mu,
        active_dims_m=ind_active_dims,
        active_dims_p=pot_active_dims,
        dipole=dipole,
        jacobian_chg=jacobian_chg,
    )
