"""OpenAI utility functions: credentials, tool choice, and constants."""

from __future__ import annotations

import os

import openai

from serapeum.core.base.llms.utils import get_from_param_or_env
from serapeum.core.llms import ChatResponse, ToolCallBlock

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
    """Resolve OpenAI credentials.

    The order of precedence is:
    1. param
    2. env
    3. openai module
    4. default
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
    openai_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    if not openai_api_key:
        raise ValueError(MISSING_API_KEY_ERROR_MESSAGE)


def resolve_tool_choice(
    tool_choice: str | dict | None, tool_required: bool = False
) -> str | dict:
    """Resolve tool choice.

    If tool_choice is a function name string, return the appropriate dict.
    """
    if tool_choice is None:
        tool_choice = "required" if tool_required else "auto"
    if isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice not in ["none", "auto", "required"]:
        return {"type": "function", "function": {"name": tool_choice}}

    return tool_choice


def force_single_tool_call(response: ChatResponse) -> None:
    """Keep only the first tool call in a response, discarding extras."""
    tool_calls = [
        block for block in response.message.chunks if isinstance(block, ToolCallBlock)
    ]
    if len(tool_calls) > 1:
        response.message.chunks = [
            block
            for block in response.message.chunks
            if not isinstance(block, ToolCallBlock)
        ] + [tool_calls[0]]
