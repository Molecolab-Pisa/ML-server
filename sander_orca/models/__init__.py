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
from .models_3HF import ModelEnvES, ModelEnvGS, ModelVacES, ModelVacGS

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
}


def list_available_models():
    print("Available models:")
    for model in available_models:
        print(f"\t{model}")
