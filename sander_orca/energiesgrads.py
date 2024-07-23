from __future__ import annotations

from typing import Any, Callable, Dict

from gpx.parameters import ModelState, Parameter

ParameterDict = Dict[str, Parameter]
Kernel = Any


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

    def print(self) -> None:
        return self.state.print_params()

    def load(self, state_file):
        self.state = self.state.load(state_file)
        return self
