"""OpenAI utility functions: retry, credentials, tool choice, and constants."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Tuple

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)
from tenacity.stop import stop_base

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

logger = logging.getLogger(__name__)


def create_retry_decorator(
    max_retries: int,
    random_exponential: bool = False,
    stop_after_delay_seconds: float | None = None,
    min_seconds: float = 4,
    max_seconds: float = 60,
) -> Callable[[Any], Any]:
    wait_strategy = (
        wait_random_exponential(min=min_seconds, max=max_seconds)
        if random_exponential
        else wait_exponential(multiplier=1, min=min_seconds, max=max_seconds)
    )

    stop_strategy: stop_base = stop_after_attempt(max_retries)
    if stop_after_delay_seconds is not None:
        stop_strategy = stop_strategy | stop_after_delay(stop_after_delay_seconds)

    return retry(
        reraise=True,
        stop=stop_strategy,
        wait=wait_strategy,
        retry=(
            retry_if_exception_type(
                (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                )
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def resolve_openai_credentials(
    api_key: str | None = None,
    api_base: str | None = None,
    api_version: str | None = None,
) -> Tuple[str | None, str, str]:
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
