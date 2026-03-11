"""OpenAI model metadata registry and capability query functions.

This module loads the model registry from ``models.yaml`` at import time and
exposes it as a set of public dictionaries (keyed by model name, valued by
context-window size in tokens) and predicate functions for querying model
capabilities.

Public constants:

- :data:`O1_MODELS` -- o1-series reasoning models.
- :data:`RESPONSES_API_ONLY_MODELS` -- models only available via the Responses API.
- :data:`O1_MODELS_WITHOUT_FUNCTION_CALLING` -- o1-series models that lack
  function-calling support.
- :data:`GPT4_MODELS` -- GPT-4 family models.
- :data:`AZURE_TURBO_MODELS` -- Azure-specific turbo model identifiers.
- :data:`TURBO_MODELS` -- GPT-3.5/4 turbo models.
- :data:`GPT3_5_MODELS` -- GPT-3.5 family models.
- :data:`GPT3_MODELS` -- Legacy GPT-3 models.
- :data:`DISCONTINUED_MODELS` -- Models that have been retired by OpenAI.
- :data:`JSON_SCHEMA_MODELS` -- Model prefixes that support native JSON-schema
  ``response_format``.
- :data:`ALL_AVAILABLE_MODELS` -- Union of all non-discontinued model groups.
- :data:`CHAT_MODELS` -- Subset of models that support the chat-completions API.

Public functions:

- :func:`is_chatcomp_api_supported` -- check Chat Completions API support.
- :func:`is_json_schema_supported` -- check native JSON-schema structured output.
- :func:`openai_modelname_to_contextsize` -- look up a model's context window.
- :func:`is_chat_model` -- check whether a model is a chat model.
- :func:`is_function_calling_model` -- check function-calling capability.
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
    """Check whether a model supports the Chat Completions API.

    Some newer models (e.g. ``gpt-5.2-pro``) are only available through the
    Responses API and cannot be used with the ``/chat/completions`` endpoint.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``, ``"gpt-5.2-pro"``).

    Returns:
        ``True`` if the model can be used with the Chat Completions API,
        ``False`` if it is restricted to the Responses API.

    Examples:
        - Standard models support Chat Completions:
            ```python
            >>> from serapeum.openai.data.models import is_chatcomp_api_supported
            >>> is_chatcomp_api_supported("gpt-4o")
            True

            ```

        - Responses-API-only models do not:
            ```python
            >>> from serapeum.openai.data.models import is_chatcomp_api_supported
            >>> is_chatcomp_api_supported("gpt-5.2-pro")
            False

            ```

    See Also:
        :data:`RESPONSES_API_ONLY_MODELS`: The set of models excluded from
            Chat Completions support.
    """
    return model not in RESPONSES_API_ONLY_MODELS


def is_json_schema_supported(model: str) -> bool:
    """Check whether a model supports native JSON-schema structured output.

    Native JSON-schema mode uses the ``response_format`` parameter with a
    ``json_schema`` payload, allowing the API to constrain generation to
    match a Pydantic model's schema directly -- without function-calling.

    The check has three requirements:

    1. The ``openai`` SDK must be installed and expose the internal
       ``_type_to_response_format`` helper.
    2. The model name must **not** start with ``"o1-mini"`` (which lacks
       JSON-schema support).
    3. The model name must start with one of the prefixes listed in
       :data:`JSON_SCHEMA_MODELS`.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``, ``"o1-mini"``).

    Returns:
        ``True`` if the model supports native JSON-schema structured output,
        ``False`` otherwise (including when the ``openai`` SDK is unavailable).

    Examples:
        - GPT-4o supports JSON schema:
            ```python
            >>> from serapeum.openai.data.models import is_json_schema_supported
            >>> is_json_schema_supported("gpt-4o")  # doctest: +SKIP
            True

            ```

        - o1-mini does not support JSON schema:
            ```python
            >>> from serapeum.openai.data.models import is_json_schema_supported
            >>> is_json_schema_supported("o1-mini")  # doctest: +SKIP
            False

            ```

    See Also:
        :data:`JSON_SCHEMA_MODELS`: List of model-name prefixes that have
            JSON-schema support.
        :class:`~serapeum.openai.llm.base.structured.StructuredOutput`:
            Uses this predicate to decide between native and function-calling
            structured output.
    """
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
    """Look up the maximum context-window size (in tokens) for an OpenAI model.

    Fine-tuned model identifiers (both legacy ``<base>:ft-*`` and modern
    ``ft:<base>:*`` formats) are resolved to their base model before lookup.

    Args:
        modelname: OpenAI model identifier, e.g. ``"gpt-4o"``,
            ``"ft:gpt-4o:my-org:custom:id"``, or ``"gpt-3.5-turbo"``.

    Returns:
        The context-window size in tokens for the given model.

    Raises:
        ValueError: If *modelname* refers to a discontinued model or is not
            found in :data:`ALL_AVAILABLE_MODELS`.

    Examples:
        - Look up context size for a standard model:
            ```python
            >>> from serapeum.openai.data.models import openai_modelname_to_contextsize
            >>> openai_modelname_to_contextsize("gpt-4o")
            128000

            ```

        - Fine-tuned model identifiers resolve to their base:
            ```python
            >>> from serapeum.openai.data.models import openai_modelname_to_contextsize
            >>> openai_modelname_to_contextsize("ft:gpt-4o:acme:suffix:abc123")
            128000

            ```

        - Discontinued models raise an error:
            ```python
            >>> from serapeum.openai.data.models import openai_modelname_to_contextsize
            >>> openai_modelname_to_contextsize("code-davinci-002")  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: OpenAI model code-davinci-002 has been discontinued. ...

            ```

    See Also:
        :data:`ALL_AVAILABLE_MODELS`: The full mapping of model names to
            context-window sizes.
        :data:`DISCONTINUED_MODELS`: Models that are no longer available.
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
    """Check whether a model is a chat-completions model.

    Chat models support the multi-turn ``/chat/completions`` endpoint as
    opposed to the legacy single-turn ``/completions`` endpoint.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``, ``"davinci-002"``).

    Returns:
        ``True`` if *model* appears in :data:`CHAT_MODELS`, ``False``
        otherwise.

    Examples:
        - GPT-4o is a chat model:
            ```python
            >>> from serapeum.openai.data.models import is_chat_model
            >>> is_chat_model("gpt-4o")
            True

            ```

        - Legacy completion-only models are not:
            ```python
            >>> from serapeum.openai.data.models import is_chat_model
            >>> is_chat_model("davinci-002")
            False

            ```

    See Also:
        :data:`CHAT_MODELS`: The authoritative set of chat model identifiers.
        :func:`is_function_calling_model`: A stricter check that also excludes
            old snapshots and certain o1 models.
    """
    return model in CHAT_MODELS


def is_function_calling_model(model: str) -> bool:
    """Check whether a model supports function calling (tool use).

    Function calling is available for chat models that are **not** early
    snapshots (``0301`` / ``0314`` date suffixes) and **not** in the
    :data:`O1_MODELS_WITHOUT_FUNCTION_CALLING` set.

    For unknown models (not in :data:`ALL_AVAILABLE_MODELS`), the function
    optimistically returns ``True`` to allow custom or fine-tuned model names.

    Fine-tuned identifiers (``ft:<base>:*`` and legacy ``ft-<base>:*``) are
    resolved to their base model name before the check.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``,
            ``"gpt-3.5-turbo-0301"``, ``"ft:gpt-4o:org:tag:id"``).

    Returns:
        ``True`` if the model supports function calling, ``False`` otherwise.

    Examples:
        - Modern chat models support function calling:
            ```python
            >>> from serapeum.openai.data.models import is_function_calling_model
            >>> is_function_calling_model("gpt-4o")
            True

            ```

        - Early snapshots do not:
            ```python
            >>> from serapeum.openai.data.models import is_function_calling_model
            >>> is_function_calling_model("gpt-4-0314")
            False

            ```

        - Unknown models are assumed capable:
            ```python
            >>> from serapeum.openai.data.models import is_function_calling_model
            >>> is_function_calling_model("my-custom-model")
            True

            ```

    See Also:
        :func:`is_chat_model`: Checks only whether the model is a chat model,
            without the snapshot and o1 exclusions.
        :data:`O1_MODELS_WITHOUT_FUNCTION_CALLING`: The set of o1 models
            excluded from function-calling support.
    """
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
