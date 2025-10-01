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
"""
Uracil models for ground-state ML/MM simulations.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

import jax.numpy as jnp
from gpx.bijectors import Softplus
from gpx.kernels import Polynomial, Matern52, Prod
from gpx.mean_functions import zero_mean
from gpx.models import GPR, GPR_TD
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from jax import Array, jit
from jax.typing import ArrayLike

from .basemodels import BaseModelEnv, BaseModelVac
from .models_utils import inv_dist, inv_dist_jac, elec_pot, elec_pot_jac, squared_distances

ParameterDict = Dict[str, Parameter]
Kernel = Any

# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models/Uracil")

# Conversion constants
H2kcal = 627.5094740631
Bohr2Ang = 0.529177210903
conversion_charge = 1. / H2kcal / Bohr2Ang
conversion_dipole = conversion_charge / Bohr2Ang


class ModelVacGS(BaseModelVac):
    """Model for the QM part only (vacuum), ground state, uracil.

    Predicts the QM energies and QM gradients with Gaussian process regression
    using a Matern(5/2) kernel on the inverse distances descriptor.
    """

    def __init__(self, workdir: str) -> None:
        super().__init__(workdir)

    def load(self) -> ModelVacGS:
        lengthscale = Parameter(
            1.0, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        sigma = Parameter(
            0.1, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        kernel_params = dict(lengthscale=lengthscale)

        model = GPR(
            kernel=Matern52(),
            kernel_params=kernel_params,
            mean_function=zero_mean,
            sigma=sigma,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelvacgs.npz"))
        model.print()
        self._model = model
        self.constant = model.state.constant
        return self

    def predict(
        self,
        coords_qm: Array,
        ind: Optional[Array] = None,
        ind_jac: Optional[Array] = None,
        dipole: Optional[bool] = False,
        model_env: Optional[ModelState] = None,
        **kwargs,
    ) -> Tuple[float, Array]:
        """predicts the QM energy and QM gradients

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part.
        ind: array, shape (num_ID, 3)
            inverse distances descriptor
            num_ID = num_QM_atoms * (num_QM_atoms - 1) / 2
        ind_jac: array, shape (num_ID, 3 * num_QM_atoms)
            jacobian of the inverse distances descriptor.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        """
        if ind is None:
            ind = inv_dist(coords_qm)
        if ind_jac is None:
            ind_jac = inv_dist_jac(coords_qm)
        energy, grads = predict_vac(self._model, ind, ind_jac)

        if dipole:
            na, _ = coords_qm.shape
            pot = jnp.zeros((1,na))
            x = jnp.concatenate((ind, pot), axis=1)
            dipole_vac = predict_dipole(model_env._model, x, jnp.expand_dims(coords_qm, axis=0))
            dipole_vac = dipole_vac.squeeze() * conversion_dipole
            with open(os.path.join(self.workdir,'dipole.dat'),'a') as f:
                f.write('%14.10f   %14.10f   %14.10f\n' %(dipole_vac[0],dipole_vac[1],dipole_vac[2]))

        energy = (energy.squeeze() + self.constant) / H2kcal
        grads = grads / H2kcal * Bohr2Ang
        return energy, grads


class ModelEnvGS(BaseModelEnv):
    """Environment model, ground state, uracil.

    Predicts the energy of the QM part plus the QM/MM interaction, and the
    gradients of that energy w.r.t. the QM and MM atoms.
    The prediction is done with Gaussian process regression using a product kernel,
    where a Polynomial kernel operates on the electrostatic potential of the MM part
    acting on the QM atoms, and a Matern(5/2) kernel operates on the inverse
    distances of the QM atoms.
    See __ADD_PAPER_LINK__.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac) -> None:
        super().__init__(workdir, model_vac=model_vac)
        self._max_mm_atoms = 0

    def load(self) -> ModelEnvGS:
        sigma_targets = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs2 = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                5.0,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(),
            )
        )

        ind_dim = 66
        n_feat = 66 + 12
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Polynomial(no_intercept=True, active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        kernel_params = {"kernel1": k1.default_params(), "kernel2": k2_lengthscale}


        k = Prod(k1, k2)

        model = GPR_TD(
            kernel=k,
            kernel_params=kernel_params,
            sigma_targets=sigma_targets,
            sigma_derivs=sigma_derivs,
            sigma_derivs2=sigma_derivs2,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenvgs.npz"))
        model.print()
        self._model = model
        return self

    def get_input(
        self, ind_descr: Array, ind_jac: Array, pot_descr: Array, pot_jac_qm: Array, pot_jac_mm: Array
    ) -> Tuple[Array, Array]:
        """concatenates the inverse distances and the electrostatic potential
        descriptors/jacobians.
        """
        n, f, _ = ind_jac.shape
        _, _, v = pot_jac_mm.shape
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm, pot_jac_mm 

    def predict(
            self, coords_qm: Array, coords_mm: Array, charges_mm: Array, dipole: bool
    ) -> Tuple[float, Array, Array]:
        """predicts the QM + QM/MM energy and QM and MM gradients

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
        descr, jacobian_qm, jacobian_mm = self.get_input(ind, ind_jac, pot, pot_jac_qm, pot_jac_mm)

        # predict energy and grads in vacuum
        # note: we send ind and ind_jac to avoid recomputing the descriptor
        energy_vac, grads_vac = self.model_vac.predict(
            coords_qm, ind=ind, ind_jac=ind_jac
        )

        # predict QM/MM interaction energy and grads
        if dipole:
            energy_env, grads_env_qm, grads_env_mm, dipole_env = predict_env(
                self._model, descr, jacobian_qm, jacobian_mm, dipole, jnp.expand_dims(coords_qm, axis=0)
            )
            dipole_env = dipole_env.squeeze() * conversion_dipole
            with open(os.path.join(self.workdir,'dipole.dat'),'a') as f:
                f.write('%14.10f   %14.10f   %14.10f\n' %(dipole_env[0],dipole_env[1],dipole_env[2]))
        else:
            energy_env, grads_env_qm, grads_env_mm = predict_env(
                self._model, descr, jacobian_qm, jacobian_mm
            )

        energy_env = energy_env.squeeze() / H2kcal
        grads_env_qm = grads_env_qm / H2kcal * Bohr2Ang
        grads_env_mm = grads_env_mm / H2kcal * Bohr2Ang

        # combine QM vacuum and QM/MM contributions
        energy = energy_vac + energy_env
        grads_qm = grads_vac + grads_env_qm
        grads_mm = grads_env_mm

        return energy, grads_qm, grads_mm


class ModelVacGSDelta(BaseModelVac):
    """Model for the QM part only (vacuum), ground state, uracil.

    Predicts the QM energies and QM gradients with Gaussian process regression
    using a Matern(5/2) kernel on the inverse distances descriptor.
    See __ADD_PAPER_LINK__.
    """

    def __init__(self, workdir: str, basemodel: ModelVacGS) -> None:
        super().__init__(workdir)
        self._basemodel = basemodel

    def load(self) -> ModelVacGS:
        lengthscale = Parameter(
            1.0, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        sigma = Parameter(
            0.1, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        kernel_params = dict(lengthscale=lengthscale)

        model = GPR(
            kernel=Matern52(),
            kernel_params=kernel_params,
            mean_function=zero_mean,
            sigma=sigma,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelvacgsdelta.npz"))
        model.print()
        self._model = model
        self.constant = model.state.constant
        return self

    def predict(
        self,
        coords_qm: Array,
        ind: Optional[Array] = None,
        ind_jac: Optional[Array] = None,
        dipole: Optional[bool] = False,
        model_env: Optional[ModelState] = None,
        **kwargs,
    ) -> Tuple[float, Array]:
        """predicts the QM energy and QM gradients

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part.
        ind: array, shape (num_ID, 3)
            inverse distances descriptor
            num_ID = num_QM_atoms * (num_QM_atoms - 1) / 2
        ind_jac: array, shape (num_ID, 3 * num_QM_atoms)
            jacobian of the inverse distances descriptor.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        """
        if ind is None:
            ind = inv_dist(coords_qm)
        if ind_jac is None:
            ind_jac = inv_dist_jac(coords_qm)
        energy, grads = self._basemodel.predict(coords_qm, ind, ind_jac)
        delta_energy, delta_grads = predict_vac(self._model, ind, ind_jac)
        delta_energy = (delta_energy + self.constant)/ H2kcal
        delta_grads = delta_grads / H2kcal * Bohr2Ang

        if dipole:
            na, _ = coords_qm.shape
            pot = jnp.zeros((1,na))
            x = jnp.concatenate((ind, pot), axis=1)
            if hasattr(model_env,'_basemodel'):
                dipole_vac = predict_dipole(model_env._basemodel._model, x, jnp.expand_dims(coords_qm, axis=0))
                delta_dipole_vac = predict_dipole(model_env._model, x, jnp.expand_dims(coords_qm, axis=0))
                dipole_vac = (dipole_vac + delta_dipole_vac).squeeze() * conversion_dipole
            else:
                dipole_vac = predict_dipole(model_env._model, x, jnp.expand_dims(coords_qm, axis=0))
                dipole_vac = dipole_vac.squeeze() * conversion_dipole
            with open(os.path.join(self.workdir,'dipole.dat'),'a') as f:
                f.write('%14.10f   %14.10f   %14.10f\n' %(dipole_vac[0],dipole_vac[1],dipole_vac[2]))

        energy = (energy + delta_energy).squeeze() 
        grads = grads + delta_grads
        return energy, grads


class ModelEnvGSDelta(BaseModelEnv):
    """Environment model, ground state, uracil 

    Predicts the energy of the QM part plus the QM/MM interaction, and the
    gradients of that energy w.r.t. the QM and MM atoms.
    The prediction is done with Gaussian process regression using a product kernel,
    where a polynomial kernel operates on the electrostatic potential of the MM part
    acting on the QM atoms, and a Matern(5/2) kernel operates on the inverse
    distances of the QM atoms.
    See __ADD_PAPER_LINK__.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac, basemodel: ModelEnvGS) -> None:
        super().__init__(workdir, model_vac=model_vac)
        self._basemodel = basemodel
        self._max_mm_atoms = 0

    def load(self) -> ModelEnvGS:
        sigma_targets = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_derivs2 = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                5.0,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(),
            )
        )

        ind_dim = 66
        n_feat = 66 + 12
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Polynomial(no_intercept=True, active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        kernel_params = {"kernel1": k1.default_params(), "kernel2": k2_lengthscale}


        k = Prod(k1, k2)

        model = GPR_TD(
            kernel=k,
            kernel_params=kernel_params,
            sigma_targets=sigma_targets,
            sigma_derivs=sigma_derivs,
            sigma_derivs2=sigma_derivs2,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenvgsdelta.npz"))
        model.print()
        self._model = model
        return self

    def get_input(
        self, ind_descr: Array, ind_jac: Array, pot_descr: Array, pot_jac_qm: Array, pot_jac_mm: Array
    ) -> Tuple[Array, Array]:
        """concatenates the inverse distances and the electrostatic potential
        descriptors/jacobians.
        """
        n, f, _ = ind_jac.shape
        _, _, v = pot_jac_mm.shape
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm, pot_jac_mm 

    def predict(
            self, coords_qm: Array, coords_mm: Array, charges_mm: Array, dipole: bool
    ) -> Tuple[float, Array, Array]:
        """predicts the QM + QM/MM energy and QM and MM gradients

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
        descr, jacobian_qm, jacobian_mm = self.get_input(ind, ind_jac, pot, pot_jac_qm, pot_jac_mm)

        # predict energy and grads in vacuum
        # note: we send ind and ind_jac to avoid recomputing the descriptor
        energy_vac, grads_vac = self.model_vac.predict(
            coords_qm, ind=ind, ind_jac=ind_jac
        )

        if dipole:
            # predict QM/MM interaction energy and grads
            energy_env, grads_env_qm, grads_env_mm, dipole_env = predict_env( 
                    self._basemodel._model, descr, jacobian_qm, jacobian_mm, dipole, jnp.expand_dims(coords_qm, axis=0)
            )
    
            # predict contribution of the delta model 
            delta_energy_env, delta_grads_env_qm, delta_grads_env_mm, delta_dipole_env = predict_env(
                    self._model, descr, jacobian_qm, jacobian_mm, dipole, jnp.expand_dims(coords_qm, axis=0)
            )
            dipole_env = (dipole_env + delta_dipole_env).squeeze() * conversion_dipole
            with open(os.path.join(self.workdir,'dipole.dat'),'a') as f:
                f.write('%14.10f   %14.10f   %14.10f\n' %(dipole_env[0],dipole_env[1],dipole_env[2]))
        else:
            # predict QM/MM interaction energy and grads
            energy_env, grads_env_qm, grads_env_mm = predict_env( self._basemodel._model,
                descr, jacobian_qm, jacobian_mm
            )
    
            # predict contribution of the delta model 
            delta_energy_env, delta_grads_env_qm, delta_grads_env_mm = predict_env(
                self._model, descr, jacobian_qm, jacobian_mm
            )


        energy_env = (energy_env + delta_energy_env).squeeze() / H2kcal
        grads_env_qm = (grads_env_qm + delta_grads_env_qm) / H2kcal * Bohr2Ang
        grads_env_mm = (grads_env_mm + delta_grads_env_mm) / H2kcal * Bohr2Ang

        # combine QM vacuum and QM/MM contributions
        energy = energy_vac + energy_env
        grads_qm = grads_vac + grads_env_qm
        grads_mm = grads_env_mm

        return energy, grads_qm, grads_mm


# ============================================================
# Auxiliary functions for prediction
# ============================================================


def _predict_vac(
    x1: ArrayLike,
    x2: ArrayLike,
    params: Dict[str, Parameter],
    jaccoef: ArrayLike,
    jacobian: ArrayLike,
    mu: ArrayLike,
) -> Array:
    lengthscale = params["lengthscale"].value

    nf = x1.shape[1]
    z1 = x1 / lengthscale
    z2 = x2 / lengthscale
    diff = jnp.sqrt(5.0) * (z1[:, jnp.newaxis] - z2)
    d2 = squared_distances(z1, z2)
    d = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2, 1e-36))
    expd = jnp.exp(-d)

    const = (jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d) * expd
    d01const = (5.0 / (3.0 * lengthscale**2)) * expd

    diff_jc = jnp.einsum("stf,sf->st", diff, jaccoef)
    diff_jt = jnp.einsum("stf,tfv->stv", diff, jacobian)

    d0k_jc = -const * diff_jc

    d01k_jc_jt = jnp.einsum("st,st,stv->stv", -d01const, diff_jc, diff_jt)
    diagonal = d01const * (1.0 + d)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nf, axis=2)
    d01k_jc_jt += jnp.einsum("sf,stf,tfv->stv", jaccoef, diagonal, jacobian)

    energy = mu + jnp.einsum("st->t", d0k_jc)

    grads = jnp.einsum("stv->tv", d01k_jc_jt)

    return energy, grads.reshape(-1, 3)


@partial(jit, static_argnums=0)
def predict_vac(model: ModelState, x: ArrayLike, jacobian: ArrayLike):
    kernel_params = model.state.params["kernel_params"]
    return _predict_vac(
        x1=model.state.x_train,
        x2=x,
        params=kernel_params,
        jaccoef=model.state.jaccoef,
        jacobian=jacobian,
        mu=model.state.mu,
    )

def _predict_dipole(
    x1: ArrayLike,
    x2: ArrayLike,
    params: Dict[str, Parameter],
    jaccoef1: ArrayLike,
    jaccoef2: ArrayLike,
    c_energies: ArrayLike,
    active_dims_m: ArrayLike,
    active_dims_p: ArrayLike,
    jacobian_chg: ArrayLike,
) -> Array:
    ns1, nf1 = x1.shape
    ns2, nf2 = x2.shape
    nact_p = active_dims_p.shape[0]

    z1_p = x1[:, active_dims_p]
    z2_p = x2[:, active_dims_p]
    jaccoef1_p = jaccoef1[:, active_dims_p]
    jaccoef2_p = jaccoef2[:, active_dims_p]

    offset = params["kernel1"]["offset"].value
    degree = params["kernel1"]["degree"].value

    const1 = degree * (offset + z1_p @ z2_p.T) ** (degree - 1)
    d1k_p = jnp.einsum("st,se->ste", const1, z1_p)
    tmp1 = jnp.einsum(
        "st,sfte->sfte",
        const1,
        jnp.tile(jnp.eye((nact_p)), (ns1, ns2)).reshape(ns1, nact_p, ns2, nact_p),
    )
    const2 = (degree * (degree - 1) * (offset + z1_p @ z2_p.T) ** (degree - 2))
    tmp2 = jnp.einsum("sf,st,te->setf", z1_p, const2, z2_p)
    d01k_jc1_p = jnp.einsum("sf,sfte->ste",jaccoef1_p, tmp1 + tmp2)
    d01k_jc2_p = jnp.einsum("sf,sfte->ste",jaccoef2_p, tmp1 + tmp2)

    d1k_jtchg_p = jnp.einsum("ste,tev->stv", d1k_p, jacobian_chg)
    d01k_jc1_jtchg_p = jnp.einsum("ste,tev->stv", d01k_jc1_p, jacobian_chg)
    d01k_jc2_jtchg_p = jnp.einsum("ste,tev->stv", d01k_jc2_p, jacobian_chg)

    lengthscale = params["kernel2"]["lengthscale"].value

    nact_m = active_dims_m.shape[0]
    z1_m = x1[:, active_dims_m] / lengthscale
    z2_m = x2[:, active_dims_m] / lengthscale
    jaccoef1_m = jaccoef1[:, active_dims_m]
    diff_m = jnp.sqrt(5.0) * (z1_m[:, jnp.newaxis] - z2_m)
    d2_m = squared_distances(z1_m, z2_m)
    d_m = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2_m, 1e-36))
    expd_m = jnp.exp(-d_m)

    const = (jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d_m) * expd_m

    mat52 = (1.0 + d_m + d_m**2 / 3.0) * expd_m
    diff_jc1 = jnp.einsum("stf,sf->st", diff_m, jaccoef1_m)
    d0k_jc1_m = -const * diff_jc1
    
    dipole = jnp.einsum('stv,st,s->tv', d1k_jtchg_p, mat52, c_energies)
    dipole += jnp.einsum('stv,st->tv', d1k_jtchg_p, d0k_jc1_m)
    dipole += jnp.einsum('stv,st->tv', d01k_jc1_jtchg_p, mat52)
    dipole += jnp.einsum('stv,st->tv', d01k_jc2_jtchg_p, mat52)
    
    return dipole.reshape(-1, 3)

@partial(jit, static_argnums=0)
def predict_dipole(
    model: ModelState,
    x: ArrayLike,
    jacobian_chg: ArrayLike = None,
):
    ind_dim = 66
    n_feat = 66 + 12
    ind_active_dims = jnp.arange(0, ind_dim)
    pot_active_dims = jnp.arange(ind_dim, n_feat)
    kernel_params = model.state.params["kernel_params"]
    return _predict_dipole(
        x1=model.state.x_train,
        x2=x,
        params=kernel_params,
        jaccoef1=model.state.jaccoef,
        jaccoef2=model.state.jaccoef_2,
        c_energies=model.state.c_targets,
        active_dims_m=ind_active_dims,
        active_dims_p=pot_active_dims,
        jacobian_chg=jacobian_chg,
    )

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
    jaccoef2_p = jaccoef2[:, active_dims_p]
    jacobian_qm_p = jacobian_qm[:, active_dims_p]

    offset = params["kernel1"]["offset"].value
    degree = params["kernel1"]["degree"].value
    poly = ((offset + z1_p @ z2_p.T) ** degree) - (offset**degree)
    const1 = degree * (offset + z1_p @ z2_p.T) ** (degree - 1)
    d0k_p = jnp.einsum("st,ft->sft", const1, z2_p.T)
    d0k_jc1_p = jnp.einsum("sf,sft->st", jaccoef1_p, d0k_p)
    d0k_jc2_p = jnp.einsum("sf,sft->st", jaccoef2_p, d0k_p)
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
    d01k_jc2_p = jnp.einsum("sf,sfte->ste",jaccoef2_p, tmp1 + tmp2)
    d01k_jc1_jtqm_p = jnp.einsum("ste,tev->stv", d01k_jc1_p, jacobian_qm_p)
    d01k_jc2_jtqm_p = jnp.einsum("ste,tev->stv", d01k_jc2_p, jacobian_qm_p)


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
    energy += jnp.einsum("st,st->t", d0k_jc2_p, mat52)

    grads_qm = jnp.einsum("st,stv,s->tv", poly, d1k_jtqm_m, c_energies)
    grads_qm += jnp.einsum("stv,st,s->tv", d1k_jtqm_p, mat52, c_energies)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc1_jtqm_p, mat52)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc1_p, d1k_jtqm_m)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc1_m, d1k_jtqm_p)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc1_jtqm_m, poly)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc2_p, d1k_jtqm_m)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc2_jtqm_p, mat52)

    tmp = jnp.einsum("stf,st,s->tf", d1k_p, mat52, c_energies)
    tmp += jnp.einsum("stf,st->tf", d01k_jc1_p, mat52)
    tmp += jnp.einsum("stf,st->tf", d01k_jc2_p, mat52)
    tmp += jnp.einsum("st,stf->tf", d0k_jc1_m, d1k_p)

    grads_mm = jnp.einsum("tf,tfv->tv", tmp, jacobian_mm)

    if dipole:
        d1k_jtchg_p = jnp.einsum("ste,tev->stv", d1k_p, jacobian_chg)
        d01k_jc1_jtchg_p = jnp.einsum("ste,tev->stv", d01k_jc1_p, jacobian_chg)
        d01k_jc2_jtchg_p = jnp.einsum("ste,tev->stv", d01k_jc2_p, jacobian_chg)

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
    ind_dim = 66
    n_feat = 66 + 12
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
