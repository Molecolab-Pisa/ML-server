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

from .energiesforces import EnergiesForces
from .io import read_inpfile, read_ptchrg, write_engrad, write_pcgrad

# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models")
H2kcal = 627.5094740631
Bohr2Ang = 0.529177210903


class Model:
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

    def predict_energies(self, x):
        return self._model.predict_y_derivs(x)

    def predict_forces(self, x, jacobian):
        return self._model.predict_derivs(x, jacobian)


# ============================================================
# Available models
# ============================================================


class ModelVacGS(Model):
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

    def predict(self, ind, ind_jac):
        energy, forces = predict_vac(self._model, ind, ind_jac)
        energy = energy.squeeze() / Bohr2Ang + self.constant
        return energy, forces

    #    def predict_energies(self, x):
    #        pred = super().predict_energies(x) / Bohr2Ang + self.constant
    #        return pred.squeeze()
    #
    #    def predict_forces(self, x, jacobian):
    #        pred = super().predict_forces(x, jacobian)
    #        forces_vac = pred.reshape(1, -1, 3)
    #        return forces_vac.squeeze()

    def run(self, coords_qm=None, filebased=True):
        if filebased:
            # read input
            _, coords_qm, _, _, _, _ = self.read_sander_xyz()

        # descriptor
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # predict energy and forces
        energies_vac, forces_vac = self.predict(ind, ind_jac)

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies_vac, grads_qm=forces_vac, grads_mm=None
            )
        else:
            return energies_vac, forces_vac


class ModelVacES(Model):
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

    #    def predict_energies(self, x):
    #        pred = super().predict_energies(x) / Bohr2Ang + self.constant
    #        return pred.squeeze()
    #
    #    def predict_forces(self, x, jacobian):
    #        pred = super().predict_forces(x, jacobian)
    #        forces_vac = pred.reshape(1, -1, 3)
    #        return forces_vac.squeeze()

    def predict(self, ind, ind_jac):
        energy, forces = predict_vac(self._model, ind, ind_jac)
        energy = energy.squeeze() / Bohr2Ang + self.constant
        return energy, forces

    def run(self, coords_qm=None, filebased=True):

        if filebased:
            # read input
            _, coords_qm, _, _, _, _ = self.read_sander_xyz()

        # descriptor
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # predict energy and forces
        energies_vac, forces_vac = self.predict(ind, ind_jac)

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies_vac, grads_qm=forces_vac, grads_mm=None
            )
        else:
            return energies_vac, forces_vac


class ModelEnvGS(Model):
    def __init__(self, workdir, model_vac=None):
        super().__init__(workdir)
        if model_vac is None:
            raise ValueError("You need to specify the vacuum model.")
        else:
            self.model_vac = model_vac

    def load(self):
        s_energies = 1e-3
        s_forces = 1e-3
        k2_l = 5.0

        sigma_energies = Parameter(
            s_energies,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(loc=s_energies, scale=0.01),
        )

        sigma_forces = Parameter(
            s_forces,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(loc=s_forces, scale=0.01),
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

        model = EnergiesForces(
            kernel=k,
            kernel_params=kernel_params,
            sigma_energies=sigma_energies,
            sigma_forces=sigma_forces,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenvgs.npz"))
        model.print()
        self._model = model
        return self

    def predict(self, x, jacobian_qm, jacobian_mm):
        energy, forces_qm, forces_mm = predict_env(
            self._model, x, jacobian_qm, jacobian_mm
        )
        energy = energy.squeeze() / H2kcal
        forces_qm = forces_qm / H2kcal * Bohr2Ang
        forces_mm = forces_mm / H2kcal * Bohr2Ang
        return energy, forces_qm, forces_mm

    def get_input(self, qm_descr, qm_jac, mm_descr, mm_jac_qm, mm_jac_mm):
        descr = jnp.concatenate((qm_descr, mm_descr), axis=-1)
        jacobian_qm = jnp.concatenate((qm_jac, mm_jac_qm), axis=1)
        jacobian_mm = mm_jac_mm
        return descr, jacobian_qm, jacobian_mm

    def run(self, coords_qm=None, coords_mm=None, charges_mm=None, filebased=True):
        if filebased:
            # read input
            _, coords_qm, _, _, coords_mm, charges_mm = self.read_sander_xyz()

        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)

        # concatenate descriptors
        descr, jacobian_qm, jacobian_mm = self.get_input(
            ind, ind_jac, pot, pot_jac_qm, pot_jac_mm
        )

        # predict energy and forces in vacuum
        energies_vac, forces_vac = self.model_vac.predict(ind, ind_jac)

        # predict QM/MM interaction energy and forces
        energies_env, forces_env_qm, forces_env_mm = self.predict(
            descr, jacobian_qm, jacobian_mm
        )

        # combine QM vacuum and QM/MM contributions
        energies = energies_vac + energies_env
        forces_qm = forces_vac + forces_env_qm
        forces_mm = forces_env_mm

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies, grads_qm=forces_qm, grads_mm=forces_mm
            )
        else:
            return energies, forces_qm, forces_mm


class ModelEnvES(Model):
    def __init__(self, workdir, model_vac=None):
        super().__init__(workdir)
        if model_vac is None:
            raise ValueError("You need to specify the vacuum model.")
        else:
            self.model_vac = model_vac

    def load(self):
        s_energies = 1e-3
        s_forces = 1e-3
        k2_l = 5.0

        sigma_energies = Parameter(
            s_energies,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(loc=s_energies, scale=0.01),
        )

        sigma_forces = Parameter(
            s_forces,
            trainable=True,
            bijector=Softplus(),
            prior=NormalPrior(loc=s_forces, scale=0.01),
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

        model = EnergiesForces(
            kernel=k,
            kernel_params=kernel_params,
            sigma_energies=sigma_energies,
            sigma_forces=sigma_forces,
            mean_function=zero_mean,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelenves.npz"))  # pure
        model.print()
        self._model = model
        return self

    def predict(self, x, jacobian_qm, jacobian_mm):
        energy, forces_qm, forces_mm = predict_env(
            self._model, x, jacobian_qm, jacobian_mm
        )
        energy = energy.squeeze() / H2kcal
        forces_qm = forces_qm / H2kcal * Bohr2Ang
        forces_mm = forces_mm / H2kcal * Bohr2Ang
        return energy, forces_qm, forces_mm

    def get_input(self, qm_descr, qm_jac, mm_descr, mm_jac_qm, mm_jac_mm):
        descr = jnp.concatenate((qm_descr, mm_descr), axis=-1)
        jacobian_qm = jnp.concatenate((qm_jac, mm_jac_qm), axis=1)
        jacobian_mm = mm_jac_mm
        return descr, jacobian_qm, jacobian_mm

    def run(self, coords_qm=None, coords_mm=None, charges_mm=None, filebased=True):

        if filebased:
            # read input
            _, coords_qm, _, _, coords_mm, charges_mm = self.read_sander_xyz()

        # descriptor for the QM part
        ind = inv_dist(coords_qm)
        ind_jac = inv_dist_jac(coords_qm)

        # descriptor for the environment
        pot = elec_pot(coords_qm, coords_mm, charges_mm)
        pot_jac_qm, pot_jac_mm = elec_pot_jac(coords_qm, coords_mm, charges_mm)

        # concatenate descriptors
        descr, jacobian_qm, jacobian_mm = self.get_input(
            ind, ind_jac, pot, pot_jac_qm, pot_jac_mm
        )

        # predict energy and forces in vacuum
        energies_vac, forces_vac = self.model_vac.predict(ind, ind_jac)

        # predict QM/MM interaction energy and forces
        energies_env, forces_env_qm, forces_env_mm = self.predict(
            descr, jacobian_qm, jacobian_mm
        )

        # combine QM vacuum and QM/MM contributions
        energies = energies_vac + energies_env
        forces_qm = forces_vac + forces_env_qm
        forces_mm = forces_env_mm

        if filebased:
            # write to file
            self.write_engrad_pcgrad(
                e_tot=energies, grads_qm=forces_qm, grads_mm=forces_mm
            )
        else:
            return energies, forces_qm, forces_mm


# ============================================================
# Dummy models, sometimes useful for testing
# ============================================================


class DummyModelZeroGrads(Model):
    "Model that always outputs zero gradients for the qm and mm part"

    def load(self):
        return self

    def run(self):
        # read input
        (
            num_qm,
            coords_qm,
            elems_qm,
            num_mm,
            coords_mm,
            charges_mm,
        ) = self.read_sander_xyz()

        # compute energy and gradients
        e_tot = 0.0
        grads_qm = np.zeros((num_qm, 3)) if num_qm is not None else None
        grads_mm = np.zeros((num_mm, 3)) if num_mm is not None else None

        # write to file
        self.write_engrad_pcgrad(e_tot=e_tot, grads_qm=grads_qm, grads_mm=grads_mm)


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
    # dummy models:
    "dummy_zerograd": DummyModelZeroGrads,
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

    diff_ja = jnp.einsum("stf,sf->st", diff, jaccoef)
    diff_t = jnp.einsum("stf,tfv->stv", diff, jacobian)

    d0kjam = -const * diff_ja

    tmp1 = jnp.einsum("st,st->st", -d01const, diff_ja)
    d01kjam = jnp.einsum("st,stv->stv", tmp1, diff_t)
    diagonal = d01const * (1.0 + d)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nf, axis=2)
    tmp2 = jnp.einsum("sf,stf->stf", jaccoef, diagonal)
    d01kjam += jnp.einsum("stf,tfv->stv", tmp2, jacobian)

    energy = mu + jnp.einsum("st->t", d0kjam)

    forces = jnp.einsum("stv->tv", d01kjam)

    return energy, forces.reshape(-1, 3)


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
    d0kjal = jnp.einsum("sf,tf->st", jaccoef_l, z2_l)
    d1kjl_qm = jnp.einsum("sf,tfv->stv", z1_l, jacobian_qm_l)
    d1kl = z1_l
    d01kjal_qm = jnp.einsum("sf,tfv->stv", jaccoef_l, jacobian_qm_l)
    d0j1kal = jaccoef_l

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

    diff_ja = jnp.einsum("stf,sf->st", diff_m, jaccoef_m)
    diff_t_qm = jnp.einsum("stf,tfv->stv", diff_m, jacobian_qm_m)

    d0kjam = -const * diff_ja
    d1kjm_qm = -jnp.einsum("st,stv->stv", -const, diff_t_qm)

    tmp1 = jnp.einsum("st,st->st", -d01const, diff_ja)
    d01kjam_qm = jnp.einsum("st,stv->stv", tmp1, diff_t_qm)
    diagonal = d01const * (1.0 + d_m)
    diagonal = diagonal[:, :, jnp.newaxis].repeat(nact_m, axis=2)
    tmp2 = jnp.einsum("sf,stf->stf", jaccoef_m, diagonal)
    d01kjam_qm += jnp.einsum("stf,tfv->stv", tmp2, jacobian_qm_m)

    energies = mu + jnp.einsum("st,st,s->t", lin, mat52, c_energies)
    energies += jnp.einsum("st,st->t", lin, d0kjam)
    energies += jnp.einsum("st,st->t", d0kjal, mat52)

    forces_qm = jnp.einsum("st,stv,s->tv", lin, d1kjm_qm, c_energies)
    forces_qm += jnp.einsum("stv,st,s->tv", d1kjl_qm, mat52, c_energies)
    forces_qm += jnp.einsum("stv,st->tv", d01kjal_qm, mat52)
    forces_qm += jnp.einsum("st,stv->tv", d0kjal, d1kjm_qm)
    forces_qm += jnp.einsum("st,stv->tv", d0kjam, d1kjl_qm)
    forces_qm += jnp.einsum("stv,st->tv", d01kjam_qm, lin)

    forces = jnp.einsum("sf,st,s->tf", d1kl, mat52, c_energies)
    forces += jnp.einsum("sf,st->tf", d0j1kal, mat52)
    forces += jnp.einsum("st,sf->tf", d0kjam, d1kl)

    forces_mm = jnp.einsum("tf,tfv->tv", forces, jacobian_mm)

    return energies, forces_qm.reshape(-1, 3), forces_mm.reshape(-1, 3)


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
def sq_dist(coords_qm: ArrayLike) -> Array:
    """squared distances descriptor

        This function takes the off-diagonal part of the
        squared distances matrix.

    Args:
        coords_qm: shape (n_atoms_qm, 3)
    Returns:
        squared distances: shape (1, n_atoms_qm (n_atoms_qm - 1)/2)
    """
    n_qm, _ = coords_qm.shape
    dist = squared_distances(coords_qm, coords_qm)
    return jnp.expand_dims(dist[jnp.triu_indices(n_qm, k=1)], axis=0)


@jit
def sq_dist_jac(coords_qm: ArrayLike):
    """Jacobian of the squared distances descriptor

    Args:
        coords_qm: shape (n_atoms_qm, 3)
    Returns:
        jacobian: shape (1, n_atoms_qm (n_atoms_qm - 1)/2, n_atoms_qm*3)
    """

    n_qm, _ = coords_qm.shape
    n_feat = int(n_qm * (n_qm - 1) / 2)
    jac_dist = jnp.zeros((n_feat, n_qm, 3))

    def row_scan(i, jac_dist):
        def inner_func(j, jac_dist):
            diff = coords_qm[i] - coords_qm[j]
            k = (n_qm * (n_qm - 1) / 2) - (n_qm - i) * ((n_qm - i) - 1) / 2 + j - i - 1

            def select(atom, jac_dist):
                return jac_dist.at[k.astype(int), atom].set(
                    jnp.where(atom == i, 2 * diff, jnp.where(atom == j, -2 * diff, 0.0))
                )

            return jax.lax.fori_loop(0, n_qm, select, jac_dist)

        return jax.lax.fori_loop(i + 1, n_qm, inner_func, jac_dist)

    jac_dist = jax.lax.fori_loop(0, n_qm - 1, row_scan, jac_dist)

    return jnp.expand_dims(jac_dist.reshape(n_feat, n_qm * 3), axis=0)


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
