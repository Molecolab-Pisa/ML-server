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
Useful functions for the available models.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

import jax.numpy as jnp
from gpx.parameters import ModelState, Parameter
from jax import Array, jit
from jax.typing import ArrayLike

ParameterDict = Dict[str, Parameter]
Kernel = Any
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

    n_atoms = coords_qm.shape[0]
    idx_i, idx_j = jnp.triu_indices(n_atoms, k=1) 

    diffs = coords_qm[idx_i] - coords_qm[idx_j]       
    dists = jnp.linalg.norm(diffs, axis=1, keepdims=True) 

    derivs_i = -diffs / dists**3  
    derivs_j = -derivs_i          

    jac = jnp.zeros((idx_i.shape[0], n_atoms, 3))
    jac = jac.at[jnp.arange(idx_i.shape[0]), idx_i].set(derivs_i)
    jac = jac.at[jnp.arange(idx_i.shape[0]), idx_j].set(derivs_j)

    return jnp.expand_dims(jac.reshape(jac.shape[0], -1), axis=0)


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

    d2 = jnp.sum(diff ** 2, axis=-1)
    d3 = d2 * jnp.sqrt(d2)

    jac_pot_mm = (
        charges_mm[jnp.newaxis, :, jnp.newaxis] * diff / d3[:, :, jnp.newaxis] 
    )

    deriv = -jnp.sum(charges_mm[jnp.newaxis, :, jnp.newaxis] * diff / d3[:, :, jnp.newaxis], axis=1)

    mask = jnp.eye(n_qm)[:, :, jnp.newaxis]
    jac_pot_qm = mask * deriv[:, jnp.newaxis, :]

    jac_pot_qm = jnp.expand_dims(jac_pot_qm.reshape(n_qm, n_qm * 3), axis=0)
    jac_pot_mm = jnp.expand_dims(jac_pot_mm.reshape(n_qm, n_mm * 3), axis=0)
    return jac_pot_qm, jac_pot_mm


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

