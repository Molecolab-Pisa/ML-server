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
