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
import urllib.request

path = os.path.dirname(__file__)

def _download_3HF_models(model_name):
    module = importlib.import_module(".models_3HF", package=__name__)
    cls = getattr(module, model_name)

    if not os.path.exists(os.path.join(path,"avail_models/3HF")):
        print("Downloading the models' data in ml_server/models/avail_models/3HF/")
        urllib.request.urlretrieve("https://zenodo.org/records/13739507/files/avail_models.tar.gz?download=1", os.path.join(path,"avail_models.tar.gz"))
        print("Decompressing...")
        tar = tarfile.open(os.path.join(path,"avail_models.tar.gz"))
        tar.extractall(path=os.path.join(path,"avail_models"))
        tar.close()
        os.remove(os.path.join(path,"avail_models.tar.gz"))
        os.rename(os.path.join(path,"avail_models/avail_models"),os.path.join(path,"avail_models/3HF"))
    return cls


def _download_ura_models(model_name):
    module = importlib.import_module(".models_ura", package=__name__)
    cls = getattr(module, model_name)

    if not os.path.exists(os.path.join(path,"avail_models/Uracil")):
        print("Downloading the models' data in ml_server/models/avail_models/Uracil/")
        urllib.request.urlretrieve("https://zenodo.org/records/17601122/files/models_uracil.tar.gz?download=1", os.path.join(path, "models_uracil.tar.gz"))
        print("Decompressing...")
        tar = tarfile.open(os.path.join(path,"models_uracil.tar.gz"))
        tar.extractall(path=os.path.join(path,"avail_models"))
        tar.close()
        os.remove(os.path.join(path,"models_uracil.tar.gz"))
        os.rename(os.path.join(path,"avail_models/models"), os.path.join(path,"avail_models/Uracil"))
    return cls


def _download_nma_models(model_name):
    module = importlib.import_module(".models_nma", package=__name__)
    cls = getattr(module, model_name)

    if not os.path.exists(os.path.join(path,"avail_models/NMA")):
        print("Downloading the models' data in ml_server/models/avail_models/NMA/")
        urllib.request.urlretrieve("https://zenodo.org/records/17601122/files/models_nmethylacetamide.tar.gz?download=1", os.path.join(path, "models_nmethylacetamide.tar.gz"))
        print("Decompressing...")
        tar = tarfile.open(os.path.join(path,"models_nmethylacetamide.tar.gz"))
        tar.extractall(path=os.path.join(path,"avail_models"))
        tar.close()
        os.remove(os.path.join(path,"models_nmethylacetamide.tar.gz"))
        os.rename(os.path.join(path,"avail_models/models"), os.path.join(path,"avail_models/NMA"))
    return cls


def _download_ala2_models(model_name):
    module = importlib.import_module(".models_ala2", package=__name__)
    cls = getattr(module, model_name)

    if not os.path.exists(os.path.join(path,"avail_models/Ala2")):
        print("Downloading the models' data in ml_server/models/avail_models/Ala2/")
        urllib.request.urlretrieve("https://zenodo.org/records/17601122/files/models_alanine_dipeptide.tar.gz?download=1", os.path.join(path, "models_alanine_dipeptide.tar.gz"))
        print("Decompressing...")
        tar = tarfile.open(os.path.join(path,"models_alanine_dipeptide.tar.gz"))
        tar.extractall(path=os.path.join(path,"avail_models"))
        tar.close()
        os.remove(os.path.join(path,"models_alanine_dipeptide.tar.gz"))
        os.rename(os.path.join(path,"avail_models/models"), os.path.join(path,"avail_models/Ala2"))
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
    "modelvacgs_3HF": (_download_3HF_models, "ModelVacGS"),
    "modelvaces_3HF": (_download_3HF_models, "ModelVacES"),
    "modelenvgs_3HF": (_download_3HF_models, "ModelEnvGS"),
    "modelenves_3HF": (_download_3HF_models, "ModelEnvES"),

    "modelvacgs_ura": (_download_ura_models, "ModelVacGS"),
    "modelenvgs_ura": (_download_ura_models, "ModelEnvGS"),
    "modelvacgsdelta_ura": (_download_ura_models, "ModelVacGSDelta"),
    "modelenvgsdelta_ura": (_download_ura_models, "ModelEnvGSDelta"),

    "modelvacgs_nma": (_download_nma_models, "ModelVacGS"),
    "modelenvgs_nma": (_download_nma_models, "ModelEnvGS"),
    "modelvacgsdelta_nma": (_download_nma_models, "ModelVacGSDelta"),

    "modelvacgs_ala2": (_download_ala2_models, "ModelVacGS"),
    "modelenvgs_ala2": (_download_ala2_models, "ModelEnvGS"),
    "modelvacgsdelta_ala2": (_download_ala2_models, "ModelVacGSDelta"),
})


def list_available_models():
    print("Available models:")
    for model in available_models:
        print(f"\t{model}")
