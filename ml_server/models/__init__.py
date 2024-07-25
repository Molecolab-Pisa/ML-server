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
#
# ============================================================
# Expose the available models
# ============================================================
from collections.abc import Mapping
import importlib
import tarfile
import os

def _download_3HF_models(model_name):
    try:
        module = importlib.import_module(".models_3HF", package=__name__)
        cls = getattr(module, model_name)
    except ImportError as e:
        import urllib.request
        print("Downloading the models implementation in ml_server/models/models_3HF.py...")
        urllib.request.urlretrieve("https://zenodo.org/records/12819135/files/models_3HF.py?download=1", "models_3HF.py")
        module = importlib.import_module(".models_3HF", package=__name__)
        cls = getattr(module, model_name)
        print("Downloading the models' data in ml_server/models/avail_models/")
        urllib.request.urlretrieve("https://zenodo.org/records/12819135/files/avail_models.tar.gz?download=1", "avail_models.tar.gz")
        print("Decompressing...")
        tar = tarfile.open("avail_models.tar.gz")
        tar.extractall()
        tar.close()
        os.remove("avail_models.tar.gz")
    return cls


class LazyDict(Mapping):
    "https://stackoverflow.com/questions/16669367/setup-dictionary-lazily"
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        try:
            func, arg = self._raw_dict.__getitem__(key)
            val = func(arg)
        except Exception:
            val = self._raw_dict.__getitem__(key)
        return val

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


available_models = LazyDict({
    # models: vacuum
    "model_vac_gs": (_download_3HF_models, "ModelVacGS"),
    "model_vac_es": (_download_3HF_models, "ModelVacES"),
    # models: environment
    "model_env_gs": (_download_3HF_models, "ModelEnvGS"),
    "model_env_es": (_download_3HF_models, "ModelEnvES"),
    # You can also add your model via simple key value:
    # "my_model": MyModel,
})


def list_available_models():
    print("Available models:")
    for model in available_models:
        print(f"\t{model}")
