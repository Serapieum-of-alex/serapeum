"""OpenAI model metadata and capability queries."""

from __future__ import annotations

O1_MODELS: dict[str, int] = {
    "o1": 200000,
    "o1-2024-12-17": 200000,
    "o1-pro": 200000,
    "o1-pro-2025-03-19": 200000,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "o3-mini": 200000,
    "o3-mini-2025-01-31": 200000,
    "o3": 200000,
    "o3-2025-04-16": 200000,
    "o3-pro": 200000,
    "o3-pro-2025-06-10": 200000,
    "o4-mini": 200000,
    "o4-mini-2025-04-16": 200000,
    # gpt-5 is a reasoning model, putting it in the o models list
    "gpt-5": 400000,
    "gpt-5-2025-08-07": 400000,
    "gpt-5-chat": 128000,
    "gpt-5-chat-latest": 128000,
    "gpt-5-mini": 400000,
    "gpt-5-mini-2025-08-07": 400000,
    "gpt-5-nano": 400000,
    "gpt-5-nano-2025-08-07": 400000,
    "gpt-5-pro": 400000,
    "gpt-5-pro-2025-10-06": 400000,
    "gpt-5.1": 400000,
    "gpt-5.1-2025-11-13": 400000,
    "gpt-5.1-chat-latest": 128000,
    "gpt-5.2": 400000,
    "gpt-5.2-2025-12-11": 400000,
    "gpt-5.2-chat-latest": 128000,
}

RESPONSES_API_ONLY_MODELS = {
    "gpt-5.2-pro": 400000,
    "gpt-5.2-pro-2025-12-11": 400000,
}

O1_MODELS_WITHOUT_FUNCTION_CALLING = {
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
}

GPT4_MODELS: dict[str, int] = {
    # stable model names:
    #   resolves to gpt-4-0314 before 2023-06-27,
    #   resolves to gpt-4-0613 after
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    # turbo models (Turbo, JSON mode)
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-turbo-preview": 128000,
    # multimodal model
    "gpt-4-vision-preview": 128000,
    "gpt-4-1106-vision-preview": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-audio-preview": 128000,
    "gpt-4o-audio-preview-2024-12-17": 128000,
    "gpt-4o-audio-preview-2024-10-01": 128000,
    "gpt-4o-mini-audio-preview": 128000,
    "gpt-4o-mini-audio-preview-2024-12-17": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    "gpt-4.5-preview": 128000,
    "gpt-4.5-preview-2025-02-27": 128000,
    # Intended for research and evaluation
    "chatgpt-4o-latest": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    # 0314 models
    "gpt-4-0314": 8192,
    "gpt-4-32k-0314": 32768,
    # GPT 4.1 Models
    "gpt-4.1": 1047576,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1-nano": 1047576,
    "gpt-4.1-2025-04-14": 1047576,
    "gpt-4.1-mini-2025-04-14": 1047576,
    "gpt-4.1-nano-2025-04-14": 1047576,
    # Latest GPT-5-chat supports setting temperature, so putting it here
    "gpt-5-chat-latest": 128000,
}

AZURE_TURBO_MODELS: dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-35-turbo-16k": 16384,
    "gpt-35-turbo": 4096,
    # 0125 (2024) model (JSON mode)
    "gpt-35-turbo-0125": 16384,
    # 1106 model (JSON mode)
    "gpt-35-turbo-1106": 16384,
    # 0613 models (function calling):
    "gpt-35-turbo-0613": 4096,
    "gpt-35-turbo-16k-0613": 16384,
}

TURBO_MODELS: dict[str, int] = {
    # stable model names:
    #   resolves to gpt-3.5-turbo-0125 as of 2024-04-29.
    "gpt-3.5-turbo": 16384,
    # resolves to gpt-3.5-turbo-16k-0613 until 2023-12-11
    # resolves to gpt-3.5-turbo-1106 after
    "gpt-3.5-turbo-16k": 16384,
    # 0125 (2024) model (JSON mode)
    "gpt-3.5-turbo-0125": 16384,
    # 1106 model (JSON mode)
    "gpt-3.5-turbo-1106": 16384,
    # 0613 models (function calling):
    #   https://openai.com/blog/function-calling-and-other-api-updates
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    # 0301 models
    "gpt-3.5-turbo-0301": 4096,
}

GPT3_5_MODELS: dict[str, int] = {
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    # instruct models
    "gpt-3.5-turbo-instruct": 4096,
}

GPT3_MODELS: dict[str, int] = {
    "text-ada-001": 2049,
    "text-babbage-001": 2040,
    "text-curie-001": 2049,
    "ada": 2049,
    "babbage": 2049,
    "curie": 2049,
    "davinci": 2049,
}

ALL_AVAILABLE_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
}

CHAT_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
}


DISCONTINUED_MODELS = {
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}

JSON_SCHEMA_MODELS = [
    "o4-mini",
    "o1",
    "o1-pro",
    "o3",
    "o3-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "gpt-5.2",
]


def is_chatcomp_api_supported(model: str) -> bool:
    return model not in RESPONSES_API_ONLY_MODELS


def is_json_schema_supported(model: str) -> bool:
    try:
        from openai.resources.chat.completions import completions

        if not hasattr(completions, "_type_to_response_format"):
            return False

        return not model.startswith("o1-mini") and any(
            model.startswith(m) for m in JSON_SCHEMA_MODELS
        )
    except ImportError:
        return False


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
        return True

    # checking whether the model is fine-tuned or not.
    # fine-tuned model names these days look like:
    # ft:gpt-3.5-turbo:acemeco:suffix:abc123
    if model.startswith("ft-"):  # legacy fine-tuning
        model = model.split(":")[0]
    elif model.startswith("ft:"):
        model = model.split(":")[1]

    is_chat_model_ = is_chat_model(model)
    is_old = "0314" in model or "0301" in model
    is_o1_beta = model in O1_MODELS_WITHOUT_FUNCTION_CALLING

    return is_chat_model_ and not is_old and not is_o1_beta
