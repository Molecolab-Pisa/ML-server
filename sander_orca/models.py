import os
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import numpy as np

# import gpx
# from gpx.parameters import Parameter
# from gpx.bijectors import Softplus
# from gpx.kernels import SquaredExponential
# from gpx.priors import NormalPrior
# from gpx.models import GPR
# from gpx.models.gpr import neg_log_posterior_derivs
# from gpx.mean_functions import zero_mean

from .io import read_inpfile, read_ptchrg, write_engrad, write_pcgrad


# Folder to the parameters of the available models
AVAIL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "avail_models")


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


class ModelVac1000pt(Model):
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
            kernel=SquaredExponential(),
            kernel_params=kernel_params,
            mean_function=zero_mean,
            sigma=sigma,
        )

        model.load(os.path.join(AVAIL_MODELS_DIR, "modelvac1000.npz"))
        model.print()
        self._model = model
        return self

    def predict_energies(self, x):
        pred = super().predict_energies(x)
        return pred.squeeze()

    def predict_forces(self, x, jacobian):
        pred = super().predict_forces(x, jacobian)
        forces_vac = pred.reshape(1, -1, 3)
        return forces_vac.squeeze()

    def run(self):
        # read input
        _, coords_qm, _, _, _, _ = self.read_sander_xyz()

        # descriptor
        sqd = sq_dist(coords_qm)
        sqd_jac = sq_dist_jac(coords_qm)

        # predict energy and forces
        energies_vac = self.predict_energies(sqd)
        forces_vac = self.predict_forces(sqd, sqd_jac)

        # write to file
        self.write_engrad_pcgrad(e_tot=energies_vac, grads_qm=forces_vac, grads_mm=None)


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
    "model_vac_1000": ModelVac1000pt,
    # models: environment
    #
    # dummy models:
    "dummy_zerograd": DummyModelZeroGrads,
}


def list_available_models():
    print("Available models:")
    for model in available_models:
        print(f"\t{model}")


# ============================================================
# Descriptors
# ============================================================


def squared_distances(x1: ArrayLike, x2: ArrayLike) -> Array:
    """squared euclidean distances

    This is a memory-efficient implementation of the calculation of
    squared euclidean distances. Euclidean distances between `x1`
    of shape (n_samples1, n_feats) and `x2` of shape (n_samples2, n_feats)
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


@jax.jit
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


@jax.jit
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


@jax.jit
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
        potential: shape (n_atoms_qm,)
    """

    dd = squared_distances(coords_qm, coords_mm) ** 0.5
    pot = compute_potential(charges_mm, dd)

    return pot


@jax.jit
def elec_pot_jac_qm(
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
        jacobian: shape (n_atoms_qm,n_atoms_qm*3)
    """
    n_qm, _ = coords_qm.shape
    jac = jax.jacrev(elec_pot, argnums=0)(coords_qm, coords_mm, charges_mm)

    return jac.reshape(n_qm, n_qm * 3)


@jax.jit
def elec_pot_jac_mm(
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
        jacobian: shape (n_atoms_qm,)
    """
    n_mm, _ = coords_mm.shape
    jac = jax.jacrev(elec_pot, argnums=1)(coords_qm, coords_mm, charges_mm)

    return jac.reshape(n_atoms_qm, n_mm * 3)


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
