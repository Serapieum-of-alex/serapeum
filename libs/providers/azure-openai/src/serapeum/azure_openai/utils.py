from __future__ import annotations

import time
from typing import Any

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential


def refresh_openai_azure_ad_token(
    azure_ad_token: Any = None,
) -> Any:
    """
    Checks the validity of the associated token, if any, and tries to refresh it
    using the credentials available in the current context. Different authentication
    methods are tried, in order, until a successful one is found as defined at the
    package `azure-identity`.
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
    """Return the first non-``None`` value among *args*, or ``None``."""
    result = next((arg for arg in args if arg is not None), None)
    return result
