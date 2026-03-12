"""OpenAI utility functions for credential resolution, API key validation, and tool choice mapping.

Provides helper functions consumed by the OpenAI provider classes:

* :func:`resolve_openai_credentials` -- multi-source credential resolution
  (explicit parameter > environment variable > ``openai`` module global >
  built-in default).
* :func:`validate_openai_api_key` -- guard that raises when no API key can
  be found.
* :func:`resolve_tool_choice` -- normalises the ``tool_choice`` parameter
  into the dict / string format expected by the OpenAI chat-completions API.

Module-level constants define the default API type, base URL, and version.
"""

from __future__ import annotations

import os

import openai

from serapeum.core.base.llms.utils import get_from_param_or_env

DEFAULT_OPENAI_API_TYPE = "open_ai"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"
DEFAULT_OPENAI_API_VERSION = ""

MISSING_API_KEY_ERROR_MESSAGE = """No API key found for OpenAI.
Please set either the OPENAI_API_KEY environment variable or \
openai.api_key prior to initialization.
API keys can be found or created at \
https://platform.openai.com/account/api-keys
"""


def resolve_openai_credentials(
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
) -> tuple[str | None, str, str]:
    """Resolve OpenAI API credentials from multiple sources.

    Each credential is resolved independently using the following
    precedence order (first non-empty value wins):

    1. Explicit *parameter* passed to this function.
    2. **Environment variable** (``OPENAI_API_KEY``, ``OPENAI_API_BASE``,
       ``OPENAI_API_VERSION``).
    3. **``openai`` module global** (``openai.api_key``,
       ``openai.base_url``, ``openai.api_version``).
    4. **Built-in default** defined in this module
       (``DEFAULT_OPENAI_API_BASE``, ``DEFAULT_OPENAI_API_VERSION``).

    Args:
        api_key: Explicit API key.  When ``None`` or empty, falls through
            to environment / module / default resolution.
        api_base: Explicit base URL for the API.  When ``None`` or empty,
            defaults to ``https://api.openai.com/v1``.
        api_version: Explicit API version string.  When ``None`` or empty,
            defaults to an empty string (standard OpenAI does not require
            a version; Azure OpenAI does).

    Returns:
        A three-element tuple ``(api_key, api_base, api_version)`` where
        ``api_key`` may be an empty string if no key was found anywhere,
        ``api_base`` is always a non-empty URL string, and
        ``api_version`` is a string (possibly empty).

    Examples:
        - Explicit parameters take highest precedence:
            ```python
            >>> from serapeum.openai.utils import resolve_openai_credentials
            >>> key, base, version = resolve_openai_credentials(
            ...     api_key="sk-explicit",
            ...     api_base="https://custom.api.com/v1",
            ...     api_version="2024-02-01",
            ... )
            >>> key
            'sk-explicit'
            >>> base
            'https://custom.api.com/v1'
            >>> version
            '2024-02-01'

            ```
        - Omitted values fall back to defaults:
            ```python
            >>> from serapeum.openai.utils import resolve_openai_credentials
            >>> _, base, version = resolve_openai_credentials()
            >>> base
            'https://api.openai.com/v1'
            >>> version
            ''

            ```

    See Also:
        validate_openai_api_key: Raises ``ValueError`` when no key exists.
        serapeum.core.base.llms.utils.get_from_param_or_env: Low-level
            param / env / default resolver.
    """
    # resolve from param or env
    api_key = get_from_param_or_env("api_key", api_key, "OPENAI_API_KEY", "")
    api_base = get_from_param_or_env("api_base", api_base, "OPENAI_API_BASE", "")
    api_version = get_from_param_or_env(
        "api_version", api_version, "OPENAI_API_VERSION", ""
    )

    # resolve from openai module or default
    final_api_key = api_key or openai.api_key or ""
    final_api_base = api_base or openai.base_url or DEFAULT_OPENAI_API_BASE
    final_api_version = api_version or openai.api_version or DEFAULT_OPENAI_API_VERSION

    return final_api_key, str(final_api_base), final_api_version


def validate_openai_api_key(api_key: str | None = None) -> None:
    """Validate that an OpenAI API key is available.

    Checks the explicit *api_key* parameter first, then falls back to the
    ``OPENAI_API_KEY`` environment variable.  Raises immediately if
    neither source provides a non-empty value.

    Args:
        api_key: Explicit API key to validate.  When ``None`` or empty,
            the ``OPENAI_API_KEY`` environment variable is checked.

    Raises:
        ValueError: If no API key can be found from either the parameter
            or the environment variable.

    Examples:
        - A non-empty key passes validation silently:
            ```python
            >>> from serapeum.openai.utils import validate_openai_api_key
            >>> validate_openai_api_key("sk-valid-key")

            ```
        - An empty key with no environment variable raises:
            ```python
            >>> import os
            >>> from serapeum.openai.utils import validate_openai_api_key
            >>> saved = os.environ.pop("OPENAI_API_KEY", None)
            >>> try:
            ...     validate_openai_api_key("")
            ... except ValueError as e:
            ...     "No API key found" in str(e)
            ... finally:
            ...     if saved is not None:
            ...         os.environ["OPENAI_API_KEY"] = saved
            True

            ```

    See Also:
        resolve_openai_credentials: Full credential resolution with
            fallback chain.
    """
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)


def resolve_tool_choice(
    tool_choice: str | dict | None, tool_required: bool = False
) -> str | dict:
    """Resolve a tool-choice value into the format expected by the OpenAI API.

    Normalises the *tool_choice* parameter so callers can pass a simple
    function name string, a raw dict, one of the special keywords
    (``"none"``, ``"auto"``, ``"required"``), or ``None`` and always get
    back a value the OpenAI chat-completions endpoint accepts.

    When *tool_choice* is ``None`` the default is ``"auto"`` unless
    *tool_required* is ``True``, in which case ``"required"`` is used.

    Args:
        tool_choice: Desired tool-choice value.  Accepted forms:

            * ``None`` -- use the default (``"auto"`` or ``"required"``).
            * ``"none"`` / ``"auto"`` / ``"required"`` -- OpenAI keywords
              passed through unchanged.
            * A function name string (e.g. ``"get_weather"``) -- converted
              to ``{"type": "function", "function": {"name": ...}}``.
            * A raw ``dict`` -- passed through unchanged (caller is
              responsible for the correct shape).
        tool_required: When ``True`` and *tool_choice* is ``None``, the
            resolved value is ``"required"`` instead of ``"auto"``.
            Defaults to ``False``.

    Returns:
        Either a keyword string (``"none"``, ``"auto"``, or
        ``"required"``), or a dict in the
        ``{"type": "function", "function": {"name": "<name>"}}`` shape.

    Examples:
        - Default resolution returns ``"auto"``:
            ```python
            >>> from serapeum.openai.utils import resolve_tool_choice
            >>> resolve_tool_choice(None)
            'auto'

            ```
        - Force tool use when tools are required:
            ```python
            >>> from serapeum.openai.utils import resolve_tool_choice
            >>> resolve_tool_choice(None, tool_required=True)
            'required'

            ```
        - A function name is converted to the structured dict format:
            ```python
            >>> from serapeum.openai.utils import resolve_tool_choice
            >>> result = resolve_tool_choice("get_weather")
            >>> result["type"]
            'function'
            >>> result["function"]["name"]
            'get_weather'

            ```
        - OpenAI keywords are passed through unchanged:
            ```python
            >>> from serapeum.openai.utils import resolve_tool_choice
            >>> resolve_tool_choice("none")
            'none'
            >>> resolve_tool_choice("required")
            'required'

            ```
        - A raw dict is returned as-is:
            ```python
            >>> from serapeum.openai.utils import resolve_tool_choice
            >>> custom = {"type": "function", "function": {"name": "my_fn"}}
            >>> resolve_tool_choice(custom) == custom
            True

            ```

    See Also:
        serapeum.core.tools.CallableTool: Tool abstraction whose schema
            feeds into the ``tools`` parameter alongside ``tool_choice``.
    """
    if tool_choice is None:
        tool_choice = "required" if tool_required else "auto"

    if isinstance(tool_choice, dict):
        result: str | dict = tool_choice
    elif tool_choice not in ["none", "auto", "required"]:
        result = {"type": "function", "function": {"name": tool_choice}}
    else:
        result = tool_choice

    return result
