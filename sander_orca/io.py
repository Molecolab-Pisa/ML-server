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


# ============================================================
# Function to write the "QM" output files
# ============================================================


def write_engrad(path: str, e_tot: float, grads_qm: np.ndarray) -> None:
    """writes the .engrad file read by sander

    The .engrad file contains the total energy (e_qm + e_mm + e_qm/mm)
    and the gradients w.r.t. the qm atoms.

    Args:
        path: path to the .engrad file
        e_tot: total energy
        grads_qm: gradients w.r.t. the qm atoms, shape (num_qm_atoms, 3)
    """
    header = "# The current total energy in Eh\n#\n{:22.12f}\n# The current gradient in Eh/bohr\n#".format(
        e_tot
    )
    np.savetxt(path, grads_qm.reshape(-1), fmt="%16.10f", header=header, comments="")


def write_pcgrad(path: str, grads_mm: np.ndarray) -> None:
    """writes the .pcgrad file read by sander

    The .pcgrad file contains the number of mm atoms and the gradients
    w.r.t. the mm atoms.

    Args:
        path: path to the .pcgrad file
        grads_mm: gradients w.r.t. the mm atoms, shape (num_mm_atoms, 3)
    """
    num_mm = grads_mm.shape[0]
    np.savetxt(path, grads_mm, fmt="%17.12f", header="%d" % num_mm, comments="")
