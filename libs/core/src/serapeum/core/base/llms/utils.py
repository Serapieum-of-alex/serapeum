"""Helper adapters to bridge chat and completion interfaces for LLM backends."""

import os

__all__ = [
    "get_from_param_or_env",
]


def get_from_param_or_env(
    key: str,
    param: str | None = None,
    env_key: str | None = None,
    default: str | None = None,
) -> str:
    """Return value from explicit param, environment, or default.

    Resolution order: ``param`` > environment variable ``env_key`` > ``default``.
    Raises ``ValueError`` when no value can be resolved.
    """
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )
