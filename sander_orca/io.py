"Here we define functions needed for IO between sander and fake ORCA"
from __future__ import annotations
from typing import Tuple

import numpy as np


# ============================================================
# Function to read the sander input files
# ============================================================


def read_ptchrg(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """reads the ptchrg.xyz file provided by sander

    The ptchrg file contains the number of mm atoms, the
    coordinates of the atoms, and the charges.

    Args:
        path: path to ptchrg.xyz file

    Returns:
        num_atoms: number of atoms
        xyz: coordinates, shape (num_atoms, 3)
        q: point charges, shape (num_atoms,)
    """
    # read the number of atoms
    with open(path, "r") as handle:
        line = handle.readline()
        num_atoms = int(line.strip())

    # read the coordinates and the charges
    # skip two lines (the num_atoms and an empty line)
    xyz = np.loadtxt(path, skiprows=2)
    if xyz.shape[0] != num_atoms:
        raise RuntimeError(
            f"Number of atoms ({num_atoms}) does not match coordinates ({xyz.shape[0]})"
        )
    # charges are in the first column
    q = xyz[:, 0]
    xyz = xyz[:, 1:]

    return num_atoms, xyz, q


def read_inpfile(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """reads the inpfile.xyz file provided by sander

    The inpfile contains the number of qm atoms, the element of each atom
    (e.g., H, C, O, ...), and the coordinates.

    Args:
        path: path to inpfile.xyz file

    Returns:
        num_atoms: number of atoms
        xyz: atom coordinates, shape (num_atoms, 3)
        elems: atom elements, shape (num_atoms)
    """
    # read the number of atoms
    with open(path, "r") as handle:
        line = handle.readline()
        num_atoms = int(line.strip())

    # read the coordinates and the atom types
    # skip two lines (the num_atoms and an empty line)
    xyz = np.loadtxt(path, skiprows=2, dtype=str)
    elems = xyz[:, 0]
    xyz = xyz[:, 1:].astype(np.float64)

    return num_atoms, xyz, elems
