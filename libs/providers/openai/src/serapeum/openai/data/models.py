"""OpenAI model metadata and capability queries.

Data is loaded from models.yaml at import time. All public constants and
query functions remain the same as before so callers need no changes.
"""

from __future__ import annotations

from importlib.resources import files

import yaml

# Load model registry from YAML (once, at import time)
_yaml_text = (files("serapeum.openai.data") / "models.yaml").read_text(encoding="utf-8")
_registry: dict = yaml.safe_load(_yaml_text)


# Public constants — same names and types as before
O1_MODELS: dict[str, int] = _registry["o1_models"]
RESPONSES_API_ONLY_MODELS: dict[str, int] = _registry["responses_api_only_models"]
O1_MODELS_WITHOUT_FUNCTION_CALLING: set[str] = set(
    _registry["o1_models_without_function_calling"]
)
GPT4_MODELS: dict[str, int] = _registry["gpt4_models"]
AZURE_TURBO_MODELS: dict[str, int] = _registry["azure_turbo_models"]
TURBO_MODELS: dict[str, int] = _registry["turbo_models"]
GPT3_5_MODELS: dict[str, int] = _registry["gpt3_5_models"]
GPT3_MODELS: dict[str, int] = _registry["gpt3_models"]
DISCONTINUED_MODELS: dict[str, int] = _registry["discontinued_models"]
JSON_SCHEMA_MODELS: list[str] = _registry["json_schema_models"]

ALL_AVAILABLE_MODELS: dict[str, int] = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
}

CHAT_MODELS: dict[str, int] = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
}


def is_chatcomp_api_supported(model: str) -> bool:
    return model not in RESPONSES_API_ONLY_MODELS


def is_json_schema_supported(model: str) -> bool:
    try:
        from openai.resources.chat.completions import completions

        if not hasattr(completions, "_type_to_response_format"):
            result = False
        else:
            result = not model.startswith("o1-mini") and any(
                model.startswith(m) for m in JSON_SCHEMA_MODELS
            )
    except ImportError:
        result = False

    return result


def openai_modelname_to_contextsize(modelname: str) -> int:
    """Calculate the maximum number of tokens possible to generate for a model.

    Args:
        modelname: The modelname we want to know the context size for.

    Returns:
        The maximum context size

    Example:
        .. code-block:: python

            max_tokens = openai.modelname_to_contextsize("text-davinci-003")

    Modified from:
        https://github.com/hwchase17/langchain/blob/master/langchain/llms/openai.py
    """
    # handling finetuned models
    if modelname.startswith("ft:"):
        modelname = modelname.split(":")[1]
    elif ":ft-" in modelname:  # legacy fine-tuning
        modelname = modelname.split(":")[0]

    if modelname in DISCONTINUED_MODELS:
        raise ValueError(
            f"OpenAI model {modelname} has been discontinued. "
            "Please choose another model."
        )
    if modelname not in ALL_AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {modelname!r}. Please provide a valid OpenAI model name in:"
            f" {', '.join(ALL_AVAILABLE_MODELS.keys())}"
        )
    return ALL_AVAILABLE_MODELS[modelname]


def is_chat_model(model: str) -> bool:
    return model in CHAT_MODELS


def is_function_calling_model(model: str) -> bool:
    # default to True for models that are not in the ALL_AVAILABLE_MODELS dict
    if model not in ALL_AVAILABLE_MODELS:
        is_fc = True
    else:
        # checking whether the model is fine-tuned or not.
        # ft:gpt-3.5-turbo:acemeco:suffix:abc123
        if model.startswith("ft-"):  # legacy fine-tuning
            model = model.split(":")[0]
        elif model.startswith("ft:"):
            model = model.split(":")[1]

        is_chat_model_ = is_chat_model(model)
        is_old = "0314" in model or "0301" in model
        is_o1_beta = model in O1_MODELS_WITHOUT_FUNCTION_CALLING

        is_fc = is_chat_model_ and not is_old and not is_o1_beta

    return is_fc
