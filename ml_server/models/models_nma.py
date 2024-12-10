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
from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from gpx.bijectors import Softplus
from gpx.kernels import Polynomial, Matern52, Prod
from gpx.mean_functions import zero_mean
from gpx.models import GPR
from gpx.models.targetderivs import TargetsDerivs
from gpx.parameters import ModelState, Parameter
from gpx.priors import NormalPrior
from jax import Array, jit
from jax.typing import ArrayLike

from .basemodels import BaseModelEnv, BaseModelVac

ParameterDict = Dict[str, Parameter]
Kernel = Any

# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models")

# Conversion constants
H2kcal = 627.5094740631
Bohr2Ang = 0.529177210903
#from emle._units import (
#    _NANOMETER_TO_BOHR,
#    _BOHR_TO_ANGSTROM,
#    _HARTREE_TO_KJ_MOL,
#    _NANOMETER_TO_ANGSTROM,
#)

class ModelVacGSNMA(BaseModelVac):
    """Model for the QM part only (vacuum), ground state, N-methylacetamide.

    Predicts the QM energies and QM gradients with Gaussian process regression
    using a Matern(5/2) kernel on the inverse distances descriptor.
    See __ADD_PAPER_LINK__.
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
            kernel=Matern52(nperms=9),
            kernel_params=kernel_params,
            mean_function=zero_mean,
            sigma=sigma,
        )

#        model.load(os.path.join(AVAIL_MODELS_DIR, "modelvacgs.npz"))
        model.load("/home/p.mazzeo/IR/nma/vac/model_vac_jaccoef.npz")
        model.print()
        self._model = model
        self.constant = model.state.constant
        return self

    def predict(
        self,
        coords_qm: Array,
        ind: Optional[Array] = None,
        ind_jac: Optional[Array] = None,
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
#        grads = self._model.predict_derivs(ind, ind_jac).reshape(-1,3)
#        energy = self._model.predict_y_derivs(ind)
        energy, grads = predict_vac(self._model, ind, ind_jac)
        energy = energy.squeeze() / Bohr2Ang + self.constant
        return energy, grads


class ModelEnvGSNMA(BaseModelEnv):
    """Environment model, ground state, N-methylacetamide.

    Predicts the energy of the QM part plus the QM/MM interaction, and the
    gradients of that energy w.r.t. the QM and MM atoms.
    The prediction is done with Gaussian process regression using a product kernel,
    where a linear kernel operates on the electrostatic potential of the MM part
    acting on the QM atoms, and a Matern(5/2) kernel operates on the inverse
    distances of the QM atoms.
    See __ADD_PAPER_LINK__.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac) -> None:
        super().__init__(workdir, model_vac=model_vac)
        self._max_mm_atoms = 0

    def load(self) -> ModelEnvGS:
        sigma_energies = Parameter(
            1e-3,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_grads = Parameter(
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

        k1 = Polynomial(no_intercept=True, active_dims=pot_active_dims, nperms=9)
        k2 = Matern52(active_dims=ind_active_dims, nperms=9)

        kernel_params = {"kernel1": k1.default_params(), "kernel2": k2_lengthscale}


        k = Prod(k1, k2, nperms=9)

        model = TargetsDerivs(
            kernel=k,
            kernel_params=kernel_params,
            sigma_targets=sigma_energies,
            sigma_derivs=sigma_grads,
            mean_function=zero_mean,
        )

        model.load("/home/p.mazzeo/IR/nma/env/model_env_jaccoef.npz")
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
        jacobian_mm = jnp.concatenate((jnp.zeros((n,f,v)),pot_jac_mm),axis=1)
        return descr, jacobian_qm, jacobian_mm

    def predict(
        self, coords_qm: Array, coords_mm: Array, charges_mm: Array
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
#        energy_env, grads_env_qm = self._model.predict(
#            descr, jacobian_qm)
#        grads_env_qm = grads_env_qm.reshape(-1,3)
#        _, grads_env_mm = self._model.predict(
#            descr, jacobian_mm)
#        grads_env_mm = grads_env_mm.reshape(-1,3)
        energy_env, grads_env_qm, grads_env_mm = predict_env_poly(
            self._model, descr, jacobian_qm, pot_jac_mm
        )
        energy_env = energy_env.squeeze() / H2kcal
        grads_env_qm = grads_env_qm / H2kcal * Bohr2Ang
        grads_env_mm = grads_env_mm / H2kcal * Bohr2Ang

        # combine QM vacuum and QM/MM contributions
        energy = energy_vac + energy_env
        grads_qm = grads_vac + grads_env_qm
        grads_mm = grads_env_mm

        return energy, grads_qm, grads_mm

    def _sire_callback(
        self,
        numbers_qm: List[int],
        charges_mm: List[float],
        xyz_qm: List[List[float]],
        xyz_mm: List[List[float]],
        idx_mm: Optional[List[int]] = None,
    ) -> Tuple[float, List[List[float]], List[List[float]]]:
    
        num_mm_atoms = len(charges_mm)
        if num_mm_atoms > self._max_mm_atoms:
            self._max_mm_atoms = num_mm_atoms
    
        # Pad the MM coordinates and charges arrays to avoid re-jitting.
        if self._max_mm_atoms > num_mm_atoms:
            num_pad = self._max_mm_atoms - num_mm_atoms
            xyz_mm_pad = num_pad * [[0.0, 0.0, 0.0]]
            charges_mm_pad = num_pad * [0.0]
            xyz_mm = np.append(xyz_mm, xyz_mm_pad, axis=0)
            charges_mm = np.append(charges_mm, charges_mm_pad)
    
        
        numbers_qm = np.array(numbers_qm)
        charges_mm = np.array(charges_mm)
        xyz_qm = np.array(xyz_qm)
        xyz_mm = np.array(xyz_mm)
    
        charge = 0
        
        outputs = self.predict(xyz_qm, xyz_mm, charges_mm)
        return ( outputs[0].item() * _HARTREE_TO_KJ_MOL , 
                (- outputs[1] * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_BOHR).tolist(), 
                (- outputs[2][:num_mm_atoms] * _HARTREE_TO_KJ_MOL * _NANOMETER_TO_BOHR).tolist(),
               )
# ============================================================
# Auxiliary functions
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
    nperms = 9

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

    energy = mu + jnp.einsum("st->t", d0k_jc) / nperms

    grads = jnp.einsum("stv->tv", d01k_jc_jt) / nperms

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


def _predict_env(
    x1: ArrayLike,
    x2: ArrayLike,
    params: Dict[str, Parameter],
    jaccoef: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
    c_energies: ArrayLike,
    mu: ArrayLike,
    active_dims_m: ArrayLike,
    active_dims_l: ArrayLike,
) -> Array:
    ns1, nf1 = x1.shape
    ns2, nf2 = x2.shape

    z1_l = x1[:, active_dims_l]
    z2_l = x2[:, active_dims_l]
    jaccoef_l = jaccoef[:, active_dims_l]
    jacobian_qm_l = jacobian_qm[:, active_dims_l]

    lin = z1_l @ z2_l.T
    d0k_jc_l = jnp.einsum("sf,tf->st", jaccoef_l, z2_l)
    d1k_jtqm_l = jnp.einsum("sf,tfv->stv", z1_l, jacobian_qm_l)
    d1k_l = z1_l
    d01k_jc_jtqm_l = jnp.einsum("sf,tfv->stv", jaccoef_l, jacobian_qm_l)
    d01k_jc_l = jaccoef_l

    lengthscale = params["kernel2"]["lengthscale"].value

    nact_m = active_dims_m.shape[0]
    z1_m = x1[:, active_dims_m] / lengthscale
    z2_m = x2[:, active_dims_m] / lengthscale
    jaccoef_m = jaccoef[:, active_dims_m]
    jacobian_qm_m = jacobian_qm[:, active_dims_m]
    diff_m = jnp.sqrt(5.0) * (z1_m[:, jnp.newaxis] - z2_m)
    d2_m = squared_distances(z1_m, z2_m)
    d_m = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2_m, 1e-36))
    expd_m = jnp.exp(-d_m)

    const = (jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d_m) * expd_m
    d01const = (5.0 / (3.0 * lengthscale**2)) * expd_m

    mat52 = (1.0 + d_m + d_m**2 / 3.0) * expd_m

    diff_jc = jnp.einsum("stf,sf->st", diff_m, jaccoef_m)
    diff_jtqm = jnp.einsum("stf,tfv->stv", diff_m, jacobian_qm_m)

    d0k_jc_m = -const * diff_jc
    d1k_jtqm_m = -jnp.einsum("st,stv->stv", -const, diff_jtqm)

    d01k_jc_jtqm_m = jnp.einsum("st,st,stv->stv", -d01const, diff_jc, diff_jtqm)
    diagonal = d01const * (1.0 + d_m)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nact_m, axis=2)
    d01k_jc_jtqm_m += jnp.einsum("sf,stf,tfv->stv", jaccoef_m, diagonal, jacobian_qm_m)

    energy = mu + jnp.einsum("st,st,s->t", lin, mat52, c_energies)
    energy += jnp.einsum("st,st->t", lin, d0k_jc_m)
    energy += jnp.einsum("st,st->t", d0k_jc_l, mat52)

    grads_qm = jnp.einsum("st,stv,s->tv", lin, d1k_jtqm_m, c_energies)
    grads_qm += jnp.einsum("stv,st,s->tv", d1k_jtqm_l, mat52, c_energies)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc_jtqm_l, mat52)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc_l, d1k_jtqm_m)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc_m, d1k_jtqm_l)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc_jtqm_m, lin)

    tmp = jnp.einsum("sf,st,s->tf", d1k_l, mat52, c_energies)
    tmp += jnp.einsum("sf,st->tf", d01k_jc_l, mat52)
    tmp += jnp.einsum("st,sf->tf", d0k_jc_m, d1k_l)

    grads_mm = jnp.einsum("tf,tfv->tv", tmp, jacobian_mm)

    return energy, grads_qm.reshape(-1, 3), grads_mm.reshape(-1, 3)


@partial(jit, static_argnums=0)
def predict_env(
    model: ModelState,
    x: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
):
    ind_dim = 378
    n_feat = 378 + 28
    ind_active_dims = jnp.arange(0, ind_dim)
    pot_active_dims = jnp.arange(ind_dim, n_feat)
    kernel_params = model.state.params["kernel_params"]
    return _predict_env(
        x1=model.state.x_train,
        x2=x,
        params=kernel_params,
        jaccoef=model.state.jaccoef,
        jacobian_qm=jacobian_qm,
        jacobian_mm=jacobian_mm,
        c_energies=model.state.c_energies,
        mu=model.state.mu,
        active_dims_m=ind_active_dims,
        active_dims_l=pot_active_dims,
    )


def _predict_env_poly(
    x1: ArrayLike,
    x2: ArrayLike,
    params: Dict[str, Parameter],
    jaccoef: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
    c_energies: ArrayLike,
    mu: ArrayLike,
    active_dims_m: ArrayLike,
    active_dims_p: ArrayLike,
) -> Array:
    nperms = 9
    nsp1, nf1 = x1.shape
    ns1 = int(nsp1/nperms)
    ns2, nf2 = x2.shape
    nact_p = active_dims_p.shape[0]

    z1_p = x1[:, active_dims_p]
    z2_p = x2[:, active_dims_p]
    jaccoef_p = jaccoef[:, active_dims_p]
    jacobian_qm_p = jacobian_qm[:, active_dims_p]

    offset = params["kernel1"]["offset"].value
    degree = params["kernel1"]["degree"].value

    poly = ((offset + z1_p @ z2_p.T) ** degree) - (offset**degree)
    poly = poly.reshape(ns1,nperms,ns2).sum(axis=1) / nperms 
    const1 = degree * (offset + z1_p @ z2_p.T) ** (degree - 1)
    d0k_p = jnp.einsum("st,ft->sft", const1, z2_p.T)
    d0k_jc_p = jnp.einsum("sf,sft->st", jaccoef_p, d0k_p)
    d0k_jc_p = d0k_jc_p.reshape(ns1,nperms,ns2).sum(axis=1) / nperms
    d1k_p = jnp.einsum("st,se->ste", const1, z1_p)
    d1k_p = d1k_p.reshape(ns1,nperms,ns2,-1).sum(axis=1) / nperms
    d1k_jtqm_p = jnp.einsum("ste,tev->stv", d1k_p, jacobian_qm_p)
    tmp1 = jnp.einsum(
        "st,sfte->sfte",
        const1,
        jnp.tile(jnp.eye((nact_p)), (nsp1, ns2)).reshape(nsp1, nact_p, ns2, nact_p),
    )
    const2 = (degree * (degree - 1) * (offset + z1_p @ z2_p.T) ** (degree - 2))
    tmp2 = jnp.einsum("sf,st,te->setf", z1_p, const2, z2_p)
    d01k_jc_jtqm_p = jnp.einsum("sf,sfte,tev->stv", jaccoef_p, tmp1 + tmp2, jacobian_qm_p)
    d01k_jc_jtqm_p = d01k_jc_jtqm_p.reshape(ns1,nperms,ns2,-1).sum(axis=1) / nperms
    d01k_jc_p = jnp.einsum("sf,sfte->ste",jaccoef_p, tmp1 + tmp2)
    d01k_jc_p = d01k_jc_p.reshape(ns1,nperms,ns2,-1).sum(axis=1) / nperms

    lengthscale = params["kernel2"]["lengthscale"].value

    nact_m = active_dims_m.shape[0]
    z1_m = x1[:, active_dims_m] / lengthscale
    z2_m = x2[:, active_dims_m] / lengthscale
    jaccoef_m = jaccoef[:, active_dims_m]
    jacobian_qm_m = jacobian_qm[:, active_dims_m]
    diff_m = jnp.sqrt(5.0) * (z1_m[:, jnp.newaxis] - z2_m)
    d2_m = squared_distances(z1_m, z2_m)
    d_m = jnp.sqrt(5.0) * jnp.sqrt(jnp.maximum(d2_m, 1e-36))
    expd_m = jnp.exp(-d_m)

    const = (jnp.sqrt(5.0) / (3.0 * lengthscale)) * (1 + d_m) * expd_m
    d01const = (5.0 / (3.0 * lengthscale**2)) * expd_m

    mat52 = (1.0 + d_m + d_m**2 / 3.0) * expd_m
    mat52 = mat52.reshape(ns1,nperms,ns2).sum(axis=1) / nperms 

    diff_jc = jnp.einsum("stf,sf->st", diff_m, jaccoef_m)
    diff_jtqm = jnp.einsum("stf,tfv->stv", diff_m, jacobian_qm_m)

    d0k_jc_m = -const * diff_jc
    d0k_jc_m = d0k_jc_m.reshape(ns1,nperms,ns2).sum(axis=1) / nperms
    d1k_jtqm_m = -jnp.einsum("st,stv->stv", -const, diff_jtqm)
    d1k_jtqm_m = d1k_jtqm_m.reshape(ns1,nperms,ns2,-1).sum(axis=1) / nperms 

    d01k_jc_jtqm_m = jnp.einsum("st,st,stv->stv", -d01const, diff_jc, diff_jtqm)
    diagonal = d01const * (1.0 + d_m)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nact_m, axis=2)
    d01k_jc_jtqm_m += jnp.einsum("sf,stf,tfv->stv", jaccoef_m, diagonal, jacobian_qm_m)
    d01k_jc_jtqm_m = d01k_jc_jtqm_m.reshape(ns1,nperms,ns2,-1).sum(axis=1) / nperms 

    energy = mu + jnp.einsum("st,st,s->t", poly, mat52, c_energies)
    energy += jnp.einsum("st,st->t", poly, d0k_jc_m)
    energy += jnp.einsum("st,st->t", d0k_jc_p, mat52)

    grads_qm = jnp.einsum("st,stv,s->tv", poly, d1k_jtqm_m, c_energies)
    grads_qm += jnp.einsum("stv,st,s->tv", d1k_jtqm_p, mat52, c_energies)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc_jtqm_p, mat52)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc_p, d1k_jtqm_m)
    grads_qm += jnp.einsum("st,stv->tv", d0k_jc_m, d1k_jtqm_p)
    grads_qm += jnp.einsum("stv,st->tv", d01k_jc_jtqm_m, poly)

    tmp = jnp.einsum("stf,st,s->tf", d1k_p, mat52, c_energies)
    tmp += jnp.einsum("stf,st->tf", d01k_jc_p, mat52)
    tmp += jnp.einsum("st,stf->tf", d0k_jc_m, d1k_p)

    grads_mm = jnp.einsum("tf,tfv->tv", tmp, jacobian_mm)

    return energy, grads_qm.reshape(-1, 3), grads_mm.reshape(-1, 3)


@partial(jit, static_argnums=0)
def predict_env_poly(
    model: ModelState,
    x: ArrayLike,
    jacobian_qm: ArrayLike,
    jacobian_mm: ArrayLike,
):
    ind_dim = 66
    n_feat = 66 + 12
    ind_active_dims = jnp.arange(0, ind_dim)
    pot_active_dims = jnp.arange(ind_dim, n_feat)
    kernel_params = model.state.params["kernel_params"]
    return _predict_env_poly(
        x1=model.state.x_train,
        x2=x,
        params=kernel_params,
        jaccoef=model.state.jaccoef,
        jacobian_qm=jacobian_qm,
        jacobian_mm=jacobian_mm,
        c_energies=model.state.c_targets,
        mu=model.state.mu,
        active_dims_m=ind_active_dims,
        active_dims_p=pot_active_dims,
    )


class EnergiesGrads:
    def __init__(
        self,
        kernel: Callable,
        mean_function: Callable,
        kernel_params: Dict[str, Parameter] = None,
        sigma_energies: Parameter = None,
        sigma_grads: Parameter = None,
    ):
        params = {
            "kernel_params": kernel_params,
            "sigma_energies": sigma_energies,
            "sigma_grads": sigma_grads,
        }
        opt = {
            "x_train": None,
            "jacobian_train": None,
            "jaccoef": None,
            "y_train": None,
            "y_derivs_train": None,
            "is_fitted": False,
            "is_fitted_derivs": False,
            "c": None,
            "c_energies": None,
            "mu": None,
        }

        self.state = ModelState(kernel, mean_function, params, **opt)

    @partial(jit,static_argnums=(0,9))
    def _predict(self,
         params,
         x_train,
         jacobian_train,
         x,
         jacobian_qm,
         jacobian_mm,
         c,
         mu,
         kernel
        ):
        
        kernel_params = params["kernel_params"]
        ns, nf = x.shape
        
        K = kernel.k(
            x1=x,
            x2=x_train,
            params=kernel_params,
        )
        
        
        D1kj = kernel.d1kj(
            x1=x,
            x2=x_train,
            params=kernel_params,
            jacobian=jacobian_train,
        )
    
        K_mn = jnp.concatenate((K,D1kj),axis=1)
        y_pred = jnp.dot(K_mn,c) + mu

        D0k = kernel.d0k(
            x1=x,
            x2=x_train,
            params=kernel_params,
        ).reshape(ns,nf,-1)

        D01k = kernel.d01k(
            x1=x,
            x2=x_train,
            params=kernel_params,
        ).reshape(ns,nf,-1,nf)

        D01kj1 = jnp.einsum('ifje,jev->ifjv',D01k,jacobian_train).reshape(ns,nf,-1)
        K_mn = jnp.concatenate((D0k,D01kj1),axis=-1)
        K_mnc = jnp.dot(K_mn,c).reshape(ns,nf)
        forces_qm = jnp.einsum('ifu,if->iu',jacobian_qm,K_mnc)
        forces_mm = jnp.einsum('ifu,if->iu',jacobian_mm,K_mnc)
        
        return y_pred, forces_qm.reshape(-1,3), forces_mm.reshape(-1,3)
    
    def predict(self,
        x,
        jacobian_qm,
        jacobian_mm,
        ):
        
        if not self.state.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Run `fit` to fit the model before prediction."
            )
        return self._predict(
        params=self.state.params,
        x_train=self.state.x_train,
        jacobian_train=self.state.jacobian_train,
        x=x,
        jacobian_qm=jacobian_qm,
        jacobian_mm=jacobian_mm,
        c=self.state.c,
        mu=self.state.mu,
        kernel=self.state.kernel
        )

    def print(self) -> None:
        return self.state.print_params()

    def load(self, state_file):
        self.state = self.state.load(state_file)
        return self


# ============================================================
# Descriptors
# ============================================================


def squared_distances(x1: ArrayLike, x2: ArrayLike) -> Array:
    """squared euclidean distances

    This is a memory-efficient implementation of the calculation of
    squared euclidean distances. Euclidean distances between `x1`
    of shape (n_atoma_1, 3) and `x2` of shape (n_atoms_2, 3)
    is evaluated by using the "euclidean distances trick":

        dist = X1 @ X1.T - 2 X1 @ X2.T + X2 @ X2.T

    Note: this function evaluates distances between batches of points

    Args:
        x1: shape (n_atoms_1, 3)
        x2: shape (n_atoms_2, 3)
    Returns:
        squared distances: shape (n_atoms_1, n_atoms_2)
    """
    jitter = 1e-12
    x1s = jnp.sum(jnp.square(x1), axis=-1)
    x2s = jnp.sum(jnp.square(x2), axis=-1)
    dist = x1s[:, jnp.newaxis] - 2 * jnp.dot(x1, x2.T) + x2s + jitter
    return dist


@jit
def inv_dist(coords_qm: ArrayLike) -> Array:
    """inverse distances descriptor

        This function takes the off-diagonal part of the
        inverse distances matrix.

    Args:
        coords_qm: shape (n_atoms_qm, 3)
    Returns:
        inverse distances: shape (1, n_atoms_qm (n_atoms_qm - 1)/2)
    """
    n_qm, _ = coords_qm.shape
    dist_sq = squared_distances(coords_qm, coords_qm)
    inv_dist = 1 / (jnp.sqrt(dist_sq[jnp.triu_indices(n_qm, k=1)]))
    return jnp.expand_dims(inv_dist, axis=0)


@jit
def inv_dist_jac(coords_qm: ArrayLike) -> Array:
    """Jacobian of the inverse distances descriptor

    Args:
        coords_qm: shape (n_atoms_qm, 3)
    Returns:
        jacobian: shape (1, n_atoms_qm (n_atoms_qm - 1)/2, n_atoms_qm*3)
    """

    n_qm, _ = coords_qm.shape
    n_feat = int(n_qm * (n_qm - 1) / 2)
    jac_invdist = jnp.zeros((n_feat, n_qm, 3))

    def row_scan(i, jac_invdist):
        def inner_func(j, jac_invdist):
            diff = coords_qm[i] - coords_qm[j]
            d = jnp.sqrt(jnp.sum((coords_qm[i] - coords_qm[j]) ** 2))
            k = (n_qm * (n_qm - 1) / 2) - (n_qm - i) * ((n_qm - i) - 1) / 2 + j - i - 1

            def select(atom, jac_invdist):
                return jac_invdist.at[k.astype(int), atom].set(
                    jnp.where(
                        atom == i,
                        -diff / d**3,
                        jnp.where(atom == j, diff / d**3, 0.0),
                    )
                )

            return jax.lax.fori_loop(0, n_qm, select, jac_invdist)

        return jax.lax.fori_loop(i + 1, n_qm, inner_func, jac_invdist)

    jac_invdist = jax.lax.fori_loop(0, n_qm - 1, row_scan, jac_invdist)

    return jnp.expand_dims(jac_invdist.reshape(n_feat, n_qm * 3), axis=0)


@jit
def elec_pot(
    coords_qm: ArrayLike,
    coords_mm: ArrayLike,
    charges_mm: ArrayLike,
) -> Array:
    """Electrostatic potential generated from MM atoms on QM atoms

    Args:
        coords_qm: shape (n_atoms_qm, 3)
        coords_mm: shape (n_atoms_mm, 3)
        charges_mm: shape (n_atoms_mm,)
    Returns:
        potential: shape (1, n_atoms_qm)
    """

    dd = squared_distances(coords_qm, coords_mm) ** 0.5
    pot = compute_potential(charges_mm, dd)

    return jnp.expand_dims(pot, axis=0)


@jit
def elec_pot_jac(
    coords_qm: ArrayLike,
    coords_mm: ArrayLike,
    charges_mm: ArrayLike,
) -> Array:
    """Jacobian of electrostatic potential QM coordinates

    Args:
        coords_qm: shape (n_atoms_qm, 3)
        coords_mm: shape (n_atoms_mm, 3)
        charges_mm: shape (n_atoms_mm,)
    Returns:
        jacobian qm: shape (1, n_atoms_qm, n_atoms_qm*3)
        jacobian mm: shape (1, n_atoms_qm, n_atoms_mm*3)
    """
    n_qm, _ = coords_qm.shape
    n_mm, _ = coords_mm.shape
    jac_pot_qm = jnp.zeros((n_qm, n_qm, 3))

    diff = coords_qm[:, jnp.newaxis, :] - coords_mm

    d = jnp.sqrt(jnp.sum((coords_qm[:, jnp.newaxis, :] - coords_mm) ** 2, axis=-1))

    jac_pot_mm = (
        charges_mm[jnp.newaxis, :, jnp.newaxis] * diff / d[:, :, jnp.newaxis] ** 3
    )

    def row_scan(i, jac_pot_qm):
        deriv = -jnp.sum(
            (charges_mm[:, jnp.newaxis] * diff[i] / d[i, :, jnp.newaxis] ** 3), axis=0
        )

        def select(atom, jac_pot_qm):
            return jac_pot_qm.at[i, atom].set(jnp.where(atom == i, deriv, 0.0))

        return jax.lax.fori_loop(0, n_qm, select, jac_pot_qm)

    jac_pot_qm = jax.lax.fori_loop(0, n_qm, row_scan, jac_pot_qm)

    return jnp.expand_dims(jac_pot_qm.reshape(n_qm, n_qm * 3), axis=0), jnp.expand_dims(
        jac_pot_mm.reshape(n_qm, n_mm * 3), axis=0
    )


def compute_potential(charges_mm: ArrayLike, dd: ArrayLike) -> Array:
    """Electrostatic potential

    Args:
        charges_mm: set of charges, shape (n_atoms_mm,)
        dd: pairwise distances between qm and mm, shape (n_atoms_qm, n_atoms_mm)
    Returns:
        potential: electrostatic potential on the atoms of 1
                   shape (n_atoms_qm,)
    """
    return jnp.sum(charges_mm / dd, axis=1)
