"Here we define functions needed for IO between sander and fake ORCA"
from __future__ import annotations
from typing import Tuple

import numpy as np


# ============================================================
# Function to read the sander input files
# ============================================================


def read_ptchrg(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """reads the ptchrg.xyz file provided by SANDER

    The ptchrg file contains the number of atoms, the
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
    xyz = xyz[:, 1]

    return num_atoms, xyz, q
