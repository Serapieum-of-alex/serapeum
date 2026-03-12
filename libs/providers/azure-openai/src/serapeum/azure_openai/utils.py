"""Azure OpenAI utility functions for token management and alias resolution.

Provides helper functions consumed by the Azure OpenAI provider classes:

* :func:`refresh_openai_azure_ad_token` -- refresh or acquire a Microsoft
  Entra ID (Azure AD) bearer token for the Azure Cognitive Services scope.
* :func:`resolve_from_aliases` -- pick the first non-``None`` value from
  a sequence of alias candidates.
"""

from __future__ import annotations

import time
from typing import Any

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential


def refresh_openai_azure_ad_token(
    azure_ad_token: Any = None,
) -> Any:
    """Refresh or acquire a Microsoft Entra ID token for Azure OpenAI.

    Checks whether *azure_ad_token* is still valid (at least 60 seconds
    until expiry).  When no token is provided or the existing token is
    about to expire, a new one is acquired via
    :class:`~azure.identity.DefaultAzureCredential` for the
    ``https://cognitiveservices.azure.com/.default`` scope.

    Args:
        azure_ad_token: An existing Azure AD access-token object with
            ``token`` and ``expires_on`` attributes.  When ``None`` or
            expired, a fresh token is acquired.

    Returns:
        An access-token object with ``.token`` (the bearer string) and
        ``.expires_on`` (epoch timestamp).  Either the original
        *azure_ad_token* if still valid, or a newly acquired one.

    Raises:
        ValueError: If
            :class:`~azure.identity.DefaultAzureCredential` cannot
            authenticate (wraps the underlying
            :class:`~azure.core.exceptions.ClientAuthenticationError`).

    Examples:
        - Acquire a fresh token (requires Azure credentials):
            ```python
            from serapeum.azure_openai.utils import refresh_openai_azure_ad_token

            token = refresh_openai_azure_ad_token()
            token.token[:5]   # e.g. 'eyJ0a'

            ```
        - Reuse a still-valid token (returned unchanged):
            ```python
            import time
            from serapeum.azure_openai.utils import refresh_openai_azure_ad_token

            existing = refresh_openai_azure_ad_token()
            same = refresh_openai_azure_ad_token(existing)
            same.token == existing.token  # True

            ```

    See Also:
        serapeum.azure_openai.llm.AzureClient._resolve_api_key: Calls
            this function when ``use_azure_ad=True`` and no custom
            token provider is configured.
    """
    if not azure_ad_token or azure_ad_token.expires_on < time.time() + 60:
        try:
            credential = DefaultAzureCredential()
            azure_ad_token = credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
        except ClientAuthenticationError as err:
            raise ValueError(
                "Unable to acquire a valid Microsoft Entra ID (former Azure AD) token for "
                f"the resource due to the following error: {err.message}"
            ) from err
    return azure_ad_token


def resolve_from_aliases(*args: str | None) -> str | None:
    """Return the first non-``None`` value among positional arguments.

    Used by :class:`~serapeum.azure_openai.llm.AzureClient` to resolve
    the Azure deployment name from several legacy and current field
    aliases (``engine``, ``deployment_name``, ``deployment_id``,
    ``deployment``, ``azure_deployment``).

    Args:
        *args: Candidate values, checked in order.  The first
            non-``None`` value wins.

    Returns:
        The first non-``None`` string, or ``None`` if every argument
        is ``None``.

    Examples:
        - First non-None value is returned:
            ```python
            >>> from serapeum.azure_openai.utils import resolve_from_aliases
            >>> resolve_from_aliases(None, None, "my-deployment")
            'my-deployment'

            ```
        - When multiple non-None values exist, the earliest wins:
            ```python
            >>> from serapeum.azure_openai.utils import resolve_from_aliases
            >>> resolve_from_aliases("first", "second", "third")
            'first'

            ```
        - All None returns None:
            ```python
            >>> from serapeum.azure_openai.utils import resolve_from_aliases
            >>> resolve_from_aliases(None, None, None) is None
            True

            ```

    See Also:
        serapeum.azure_openai.llm.AzureClient._validate_azure_env:
            Consumes this function to resolve the ``engine`` field
            from multiple alias names.
    """
    result = next((arg for arg in args if arg is not None), None)
    return result
