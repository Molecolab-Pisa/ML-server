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

from .energiesgrads import EnergiesGrads

from .basemodels import BaseModelVac, BaseModelEnv

# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models")
H2kcal = 627.5094740631
Bohr2Ang = 0.529177210903

class ModelVacGS(BaseModelVac):
    def __init__(self, workdir):
        super().__init__(workdir)

    def load(self):
        l = 1.0
        s = 0.1

        lengthscale = Parameter(
            l, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        sigma = Parameter(s, trainable=False, bijector=Softplus(), prior=NormalPrior())

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

    def predict(self, coords_qm, ind=None, ind_jac=None, **kwargs):
        if ind is None:
            ind = inv_dist(coords_qm)
        if ind_jac is None:
            ind_jac = inv_dist_jac(coords_qm)
        energy, grads = predict_vac(self._model, ind, ind_jac)
        energy = energy.squeeze() / Bohr2Ang + self.constant
        return energy, grads


class ModelVacES(BaseModelVac):
    def __init__(self, workdir):
        super().__init__(workdir)

    def load(self):
        l = 1.0
        s = 0.1

        lengthscale = Parameter(
            l, trainable=False, bijector=Softplus(), prior=NormalPrior()
        )

        sigma = Parameter(s, trainable=False, bijector=Softplus(), prior=NormalPrior())

        kernel_params = dict(lengthscale=lengthscale)

        model = GPR(
            kernel=Matern52(),
            kernel_params=kernel_params,
            mean_function=zero_mean,
            sigma=sigma,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelvaces.npz"))  # pure model
        model.print()
        self._model = model
        self.constant = model.state.constant
        return self

    def predict(self, coords_qm, ind=None, ind_jac=None, **kwargs):
        if ind is None:
            ind = inv_dist(coords_qm)
        if ind_jac is None:
            ind_jac = inv_dist_jac(coords_qm)
        energy, grads = predict_vac(self._model, ind, ind_jac)
        energy = energy.squeeze() / Bohr2Ang + self.constant
        return energy, grads


class ModelEnvGS(BaseModelEnv):
    def __init__(self, workdir, model_vac):
        super().__init__(workdir, model_vac=model_vac)

    def load(self):
        s_energies = 1e-3
        s_grads = 1e-3
        k2_l = 5.0

        sigma_energies = Parameter(
            s_energies,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_grads = Parameter(
            s_grads,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                k2_l,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(),
            )
        )

        kernel_params = {"kernel1": {}, "kernel2": k2_lengthscale}

        ind_dim = 378
        n_feat = 378 + 28
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Linear(active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        k = Prod(k1, k2)

        model = EnergiesGrads(
            kernel=k,
            kernel_params=kernel_params,
            sigma_energies=sigma_energies,
            sigma_grads=sigma_grads,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenvgs.npz"))
        model.print()
        self._model = model
        return self

    def get_input(self, ind_descr, ind_jac, pot_descr, pot_jac_qm):
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm

    def predict(self, coords_qm, coords_mm, charges_mm):

        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)

        # concatenate descriptors
        descr, jacobian_qm = self.get_input(
            ind, ind_jac, pot, pot_jac_qm
        )

        # predict energy and grads in vacuum
        energy_vac, grads_vac = self.model_vac.predict(coords_qm, ind=ind, ind_jac=ind_jac)

        # predict QM/MM interaction energy and grads
        energy_env, grads_env_qm, grads_env_mm = predict_env(
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



class ModelEnvES(BaseModelEnv):
    def __init__(self, workdir, model_vac):
        super().__init__(workdir, model_vac)
        self.model_vac = model_vac

    def load(self):
        s_energies = 1e-3
        s_grads = 1e-3
        k2_l = 5.0

        sigma_energies = Parameter(
            s_energies,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        sigma_grads = Parameter(
            s_grads,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(),
        )

        k2_lengthscale = dict(
            lengthscale=Parameter(
                k2_l,
                trainable=True,
                bijector=Softplus(),
                prior=NormalPrior(loc=k2_l, scale=10.0),
            )
        )

        kernel_params = {"kernel1": {}, "kernel2": k2_lengthscale}

        ind_dim = 378
        n_feat = 378 + 28
        ind_active_dims = jnp.arange(0, ind_dim)
        pot_active_dims = jnp.arange(ind_dim, n_feat)

        k1 = Linear(active_dims=pot_active_dims)
        k2 = Matern52(active_dims=ind_active_dims)

        k = Prod(k1, k2)

        model = EnergiesGrads(
            kernel=k,
            kernel_params=kernel_params,
            sigma_energies=sigma_energies,
            sigma_grads=sigma_grads,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenves.npz"))  # pure
        model.print()
        self._model = model
        return self

    def get_input(self, ind_descr, ind_jac, pot_descr, pot_jac_qm):
        descr = jnp.concatenate((ind_descr, pot_descr), axis=-1)
        jacobian_qm = jnp.concatenate((ind_jac, pot_jac_qm), axis=1)
        return descr, jacobian_qm

    def predict(self, coords_qm, coords_mm, charges_mm):
        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)

        # concatenate descriptors
        descr, jacobian_qm = self.get_input(
            ind, ind_jac, pot, pot_jac_qm
        )

        # predict energy and grads in vacuum
        energy_vac, grads_vac = self.model_vac.predict(coords_qm, ind=ind, ind_jac=ind_jac)

        # predict QM/MM interaction energy and grads
        energy_env, grads_env_qm, grads_env_mm = predict_env(
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


# ============================================================
# Expose the available models
# ============================================================

available_models = {
    # models: vacuum
    "model_vac_gs": ModelVacGS,
    "model_vac_es": ModelVacES,
    # models: environment
    "model_env_gs": ModelEnvGS,
    "model_env_es": ModelEnvES,
}


def list_available_models():
    print("Available models:")
    for model in available_models:
        print(f"\t{model}")


# ============================================================
# Useful functions
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
