from jax import jit, Array
from functools import partial
from typing import Any, Dict, Callable
from gpx.parameters import ModelState
from gpx.models._gpr import _A_lhs, _A_derivs_lhs

ParameterDict = Dict[str, Parameter]
Kernel = Any

class EnergiesForces:
    
    def __init__(self,
            kernel: Callable,
            mean_function: Callable,
            kernel_params: Dict[str, Parameter] = None,
            sigma: Parameter = None,
    ):
        params = {"kernel_params": kernel_params, "sigma": sigma}
        opt = {
            "x_train": None,
            "jacobian_train": None,
            "y_train": None,
            "y_derivs_train": None,
            "is_fitted": False,
            "is_fitted_derivs": False,
            "c": None,
            "mu": None,
        }

        self.state = ModelState(kernel, mean_function, params, **opt)
    
    @partial(jit,static_argnums=(0,9))
    def _predict(self,
            params: ParameterDict,
            x_train: ArrayLike,
            jacobian_train: ArrayLike,
            x: ArrayLike,
            jacobian_qm: ArrayLike,
            jacobian_mm: ArrayLike,
            c: ArrayLike,
            mu: ArrayLike,
            kernel: Kernel
        ) -> Tuple[Array, Array, Array]:
        
        kernel_params=params["kernel_params"]
        ns, nf = x.shape
        
        K = _A_lhs(
            x1=x,
            x2=x_train,
            params=params,
            kernel=kernel,
            noise=False,
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
            x: ArrayLike,
            jacobian_qm: ArrayLike,
            jacobian_mm: ArrayLike,
        ) -> Tuple[Array, Array, Array]:
        
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
