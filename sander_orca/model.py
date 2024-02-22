import gpx
from gpx.parameters import Parameter
from gpx.bijectors import Softplus
from gpx.kernels import SquaredExponential
from gpx.priors import NormalPrior
from gpx.models import GPR
from gpx.models.gpr import neg_log_posterior_derivs
from gpx.mean_functions import zero_mean





class Model:

    def __init__(self):
        pass

    def load(self):
        raise NotImplementedError

    def predict(self, x):
        return self._model.predict(x)



class ModelEneForces200pt(Model):

    def load_model(self):
    
        l = 1.0
        s = 0.1
    
        lengthscale = Parameter(l,
                                trainable=False,
                                bijector=Softplus(),
                                prior=NormalPrior()
                                )
        
        sigma = Parameter(s,
                          trainable=False,
                          bijector=Softplus(),
                          prior=NormalPrior())
    
        kernel_params = dict(lengthscale=lengthscale)
    
        model = GPR(kernel=SquaredExponential(),
                    kernel_params=kernel_params,
                    mean_function=zero_mean,
                    sigma=sigma,
                    )
    
        model.load(model_file)
        self._model = model 
        return self

#    def load(self):
#        # la tua roba sopra
#        # store inside self._model 

    def predict(self):
        pred = super().predict()
        # manipulate shape / units
        return pred


available_models = {
        'model_ene_forces_200pt': ModelEneForces200pt
}

# nel server
# passi come arg --model='model_ene_forces_200pt'
from .models import available_models

model = available_models[model_string]
