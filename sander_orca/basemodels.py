from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from jax import Array

from .io import read_inpfile, read_ptchrg, write_engrad, write_pcgrad


class BaseModel(ABC):
    """base model class

    Every model should inherit from this class, as here
    we implement the basic functions needed for the interface.
    """

    def __init__(self, workdir: str) -> None:
        """
        Arguments
        ---------
        workdir: str
            path to the working directory.
        """
        self.workdir = workdir
        # these paths are only used in the "filebased" interface
        # where we exploit the existing Sander-ORCA interface
        self.inpfile = os.path.join(workdir, "inpfile.xyz")
        self.ptchrg = os.path.join(workdir, "ptchrg.xyz")
        self.engrad = os.path.join(workdir, "orc_job.engrad")
        self.pcgrad = os.path.join(workdir, "orc_job.pcgrad")

    def read_sander_xyz(self) -> Tuple[int, Array, Array, int, Array, Array]:
        """filebased: reads the inpfile.xyz and ptchrg.xyz

        Only used in the filebased interface. Reads the files containing
        the QM and MM coordinates and charges. Handles the case where
        the files are not written (e.g. when there are no atoms in the MM
        part).
        """
        try:
            num_qm, coords_qm, elems_qm = read_inpfile(self.inpfile)
        except FileNotFoundError:
            num_qm, coords_qm, elems_qm = None, None, None

        try:
            num_mm, coords_mm, charges_mm = read_ptchrg(self.ptchrg)
        except FileNotFoundError:
            num_mm, coords_mm, charges_mm = None, None, None

        return num_qm, coords_qm, elems_qm, num_mm, coords_mm, charges_mm

    def write_engrad_pcgrad(
        self, e_tot: float = None, grads_qm: Array = None, grads_mm: Array = None
    ) -> None:
        """filebased: writes the engrad and pcgrad files

        Only used in the filebased interface. Writes the files containing
        the energy and the QM and MM gradients.

        Arguments
        ---------
        e_tot: float
            total energy, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the QM atoms, [hartree/bohr].
        grads_mm: array, shape (num_MM_atoms, 3)
            gradients of the MM atoms, [hartree/bohr].
        """
        if e_tot is not None and grads_qm is not None:
            write_engrad(self.engrad, e_tot=e_tot, grads_qm=grads_qm)

        if grads_mm is not None:
            write_pcgrad(self.pcgrad, grads_mm=grads_mm)

    @abstractmethod
    def load(self) -> BaseModel:
        """loads whatever is useful for the model.

        Loads whatever quantity is useful to perform a prediction with the model.
        You can do whatever you want at this stage, setting attributes, precomputing
        variables, etc. This method will be called with no arguments the first time
        the ML server is set up. Then, the model is used by calling exclusively the
        `run` method. That is, load here whatever you need to call model.run() and
        perform the prediction.
        """
        pass


class BaseModelVac(BaseModel):
    """base class for a vacuum model.

    Base class for a vacuum model. Every vacuum model should
    inherit from this abstract class.
    To use the interface, you have to implement just two methods:
    the `load` and the `predict`.

    >>> class MyModelVac(BaseModelVac):
    >>>     def __init__(self, workdir, *args, **kwargs):
    >>>         super(MyModelVac).__init__(workdir)
    >>>         # do something with args and kwargs
    >>>
    >>>     def load(self):
    >>>         # load what you need for the prediction
    >>>         ...
    >>>         return self
    >>>
    >>>     def predict(self, coords_qm, **kwargs):
    >>>         # perform the prediction
    >>>         ...
    >>>         return energy, grads_qm

    Since the predict method of the vacuum model is used in principle
    inside the predict method of the environment model, there is the
    possibility that e.g. the descriptor you compute for the environment
    is used also in the vacuum model. You can avoid computing the descriptor
    twice by providing that descriptor as an optional argument, i.e.:

    >>> class MyModelVac(BaseModelVac):
    >>>     def __init__(self, workdir, *args, **kwargs):
    >>>         super(MyModelVac).__init__(workdir)
    >>>         # do something with args and kwargs
    >>>
    >>>     def load(self):
    >>>         # load what you need for the prediction
    >>>         ...
    >>>         return self
    >>>
    >>>     def predict(self, coords_qm, coulomb_matrix=None, **kwargs):
    >>>         if coulomb_matrix is None:
    >>>             coulomb_matrix = ... # compute the coulomb matrix
    >>>         # perform the prediction
    >>>         ...
    >>>         return energy, grads_qm
    """

    def __init__(self, workdir: str) -> None:
        """
        Arguments
        ---------
        workdir: str
            path to the working directory.
        """
        super().__init__(workdir)

    @abstractmethod
    def predict(self, coords_qm: Array, **kwargs) -> Tuple[float, Array]:
        """predicts the QM energy and QM gradients.

        Predicts the energy for the QM part and its gradients.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part, [angstrom].

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the QM energy w.r.t. the QM atoms, [hartree/bohr].

        Note
        ----
        The **kwargs are there because the environment model
        may compute some quantity/descriptor that is useful also for
        the vacuum model. In that case, you can pass those quantities
        as keyword arguments, and use them in your implementation of
        `predict` in vacuum without recomputing them. Note that if you do
        this you have to pay attention to avoid name clashes between the
        different vacuum models (i.e., if both models want a descriptor
        as `desc`, but one model uses the coulomb matrix and the other
        one uses the inverse distances, then this trick completely breaks).
        """
        pass

    def run(
        self, coords_qm: Array = None, filebased: Optional[bool] = True
    ) -> Optional[Tuple[Array, Array]]:
        """runs the prediction

        This is the higher level wrapper around self.predict that is
        called during the simulation.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part, [angstrom].
        filebased: bool
            if True, switches to the file-based interface.

        Returns
        -------
        energy: float
            energy of the QM part, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the QM energy w.r.t. the QM atoms, [hartree/bohr].
        """
        if filebased:
            # read input
            _, coords_qm, _, _, _, _ = self.read_sander_xyz()

        # predict energy and grads
        energy, grads_qm = self.predict(coords_qm)

        if filebased:
            # write to file
            self.write_engrad_pcgrad(e_tot=energy, grads_qm=grads_qm, grads_mm=None)
        else:
            return energy, grads_qm


class BaseModelEnv(BaseModel):
    """base class for an environment model.

    Base class for an environment model. Every environment model should
    inherit from this abstract class.
    To use the interface, you have to implement just two methods:
    the `load` and the `predict`.

    >>> class MyModelEnv(BaseModelEnv):
    >>>     def __init__(self, workdir, model_vac, *args, **kwargs):
    >>>         super(MyModelEnv).__init__(workdir, model_vac)
    >>>         # do something with args and kwargs
    >>>
    >>>     def load(self):
    >>>         # load what you need for the prediction
    >>>         ...
    >>>         return self
    >>>
    >>>     def predict(self, coords_qm, coords_mm, charges_mm, **kwargs):
    >>>         # perform the prediction
    >>>         ...
    >>>         return energy, grads_qm, grads_mm

    Note that the vacuum model is stored inside the environment
    model and can be accessed with self.model_vac, if needed, in the prediction.
    """

    def __init__(self, workdir: str, model_vac: BaseModelVac) -> None:
        """
        Arguments
        ---------
        workdir: str
            path to the working directory.
        model_vac: BaseModelVac
            vacuum model.
        """
        super().__init__(workdir)
        self.model_vac = model_vac

    @abstractmethod
    def predict(
        self,
        coords_qm: Array,
        coords_mm: Array,
        charges_mm: Array,
    ) -> Optional[Array, Array, Array]:
        """predicts the QM + QM/MM energy and QM and MM gradients.

        Predicts the energy for the QM part plus the QM/MM interaction,
        and the gradients of that energy w.r.t. the QM atoms and the MM
        atoms.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part, [angstrom].
        coords_mm: array, shape (num_MM_atoms, 3)
            coordinates of the MM part, [angstrom].
        charges_mm: array, shape (num_MM_atoms,)
            charges of the MM part, [a.u.].

        Returns
        -------
        energy: float
            energy of the QM part + the QM/MM interaction, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        grads_mm: array, shape (num_MM_atoms, 3)
            gradients of the energy w.r.t. the MM atoms, [hartree/bohr].
        """
        pass

    def run(
        self,
        coords_qm: Array = None,
        coords_mm: Array = None,
        charges_mm: Array = None,
        filebased: Optional[bool] = True,
    ) -> Optional[Tuple[Array, Array, Array]]:
        """runs the prediction

        This is the higher level wrapper around self.predict that is
        called during the simulation.

        Arguments
        ---------
        coords_qm: array, shape (num_QM_atoms, 3)
            coordinates of the QM part, [angstrom].
        coords_mm: array, shape (num_MM_atoms, 3)
            coordinates of the MM part, [angstrom].
        charges_mm: arrary, shape (num_MM_atoms,)
            charges of the MM part, [a.u.].
        filebased: bool
            if True, switches to the file-based interface.

        Returns
        -------
        energy: float
            energy of the QM part plus the QM/MM interaction, [hartree].
        grads_qm: array, shape (num_QM_atoms, 3)
            gradients of the energy w.r.t. the QM atoms, [hartree/bohr].
        grads_mm: array, shape (num_MM_atoms, 3)
            gradients of the energy w.r.t. the MM atoms, [hartree/bohr].
        """
        if filebased:
            # read input
            _, coords_qm, _, _, coords_mm, charges_mm = self.read_sander_xyz()

        # predict QM/MM interaction energy and grads
        energy, grads_qm, grads_mm = self.predict(coords_qm, coords_mm, charges_mm)

        if filebased:
            # write to file
            self.write_engrad_pcgrad(e_tot=energy, grads_qm=grads_qm, grads_mm=grads_mm)
        else:
            return energy, grads_qm, grads_mm
