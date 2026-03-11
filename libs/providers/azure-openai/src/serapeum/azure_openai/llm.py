from __future__ import annotations

import os
from typing import Any

from openai import AsyncAzureOpenAI
from openai import AzureOpenAI as SyncAzureOpenAI
from openai.lib.azure import AzureADTokenProvider
from pydantic import Field, PrivateAttr, model_validator

from serapeum.azure_openai.utils import (
    refresh_openai_azure_ad_token,
    resolve_from_aliases,
)
from serapeum.core.base.llms.utils import get_from_param_or_env
from serapeum.openai import Completions
from serapeum.openai.utils import DEFAULT_OPENAI_API_BASE

__all__ = ["AzureOpenAI", "SyncAzureOpenAI", "AsyncAzureOpenAI"]


class AzureOpenAI(Completions):
    """Azure OpenAI.

    To use this, you must first deploy a model on Azure OpenAI.
    Unlike OpenAI, you need to specify an ``engine`` parameter to identify
    your deployment (called "model deployment name" in Azure portal).

    - ``model``: Name of the model (e.g. ``gpt-4o``).
      This is only used to decide completion vs. chat endpoint.
    - ``engine``: This will correspond to the custom name you chose
      for your deployment when you deployed a model.

    You must have the following environment variables set:

    - ``OPENAI_API_VERSION``: set this to ``2024-02-01`` or newer.
    - ``AZURE_OPENAI_ENDPOINT``: your endpoint should look like
      ``https://YOUR_RESOURCE_NAME.openai.azure.com/``
    - ``AZURE_OPENAI_API_KEY``: your API key if the api type is ``azure``.
      Or pass through ``azure_ad_token_provider`` and set ``use_azure_ad=True``
      to use managed identity with Azure Entra ID.

    More information can be found here:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart

    Examples:
        ```python
        from serapeum.azure_openai import AzureOpenAI

        llm = AzureOpenAI(
            engine="my-deployment",
            model="gpt-4o",
            api_key="YOUR_AZURE_OPENAI_API_KEY",
            azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
            api_version="2024-02-01",
        )
        ```

        Using managed identity (passing a token provider instead of an API key):

        ```python
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from serapeum.azure_openai import AzureOpenAI

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )

        llm = AzureOpenAI(
            engine="my-deployment",
            model="gpt-4o",
            azure_ad_token_provider=token_provider,
            use_azure_ad=True,
            azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
            api_version="2024-02-01",
        )
        ```
    """

    model: str = Field(default="gpt-35-turbo", description="The OpenAI model name.")
    engine: str = Field(description="The name of the deployed azure engine.")
    azure_endpoint: str | None = Field(
        default=None, description="The Azure endpoint to use."
    )
    azure_deployment: str | None = Field(
        default=None, description="The Azure deployment to use."
    )
    use_azure_ad: bool = Field(
        default=False,
        description="Indicates if Microsoft Entra ID (former Azure AD) is used for token authentication.",
    )
    azure_ad_token_provider: AzureADTokenProvider | None = Field(
        default=None, description="Callback function to provide Azure Entra ID token."
    )

    _azure_ad_token: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _validate_azure_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve engine aliases, validate required Azure fields, and resolve endpoint."""
        # --- engine alias resolution ---
        engine = resolve_from_aliases(
            values.get("engine"),
            values.pop("deployment_name", None),
            values.pop("deployment_id", None),
            values.pop("deployment", None),
            values.get("azure_deployment"),
        )
        if engine is None:
            raise ValueError("You must specify an `engine` parameter.")
        values["engine"] = engine

        # --- api_version validation ---
        if values.get("api_version") is None:
            raise ValueError("You must set OPENAI_API_VERSION for Azure OpenAI.")

        # --- azure_endpoint resolution from env ---
        api_base = values.get("api_base")
        azure_endpoint = values.get("azure_endpoint")

        if api_base is None and azure_endpoint is None:
            values["azure_endpoint"] = get_from_param_or_env(
                "azure_endpoint", None, "AZURE_OPENAI_ENDPOINT", ""
            )

        if (
            api_base == DEFAULT_OPENAI_API_BASE
            and azure_endpoint is None
        ):
            raise ValueError(
                "You must set OPENAI_API_BASE to your Azure endpoint. "
                "It should look like https://YOUR_RESOURCE_NAME.openai.azure.com/"
            )

        return values

    @model_validator(mode="after")
    def _reset_api_base_for_azure(self) -> AzureOpenAI:
        """Reset api_base to None when it is the OpenAI default or azure_endpoint is set.

        Runs after the parent ``_resolve_credentials`` validator which may set
        ``api_base`` to the OpenAI default URL.
        """
        if self.api_base == DEFAULT_OPENAI_API_BASE or self.azure_endpoint:
            self.api_base = None
        return self

    def _resolve_api_key(self) -> str:
        """Resolve the API key from Azure AD, token provider, or environment.

        Returns:
            The resolved API key string.

        Raises:
            ValueError: If no API key can be resolved from any source.
        """
        if self.use_azure_ad:
            if self.azure_ad_token_provider:
                api_key = self.azure_ad_token_provider()
            else:
                self._azure_ad_token = refresh_openai_azure_ad_token(
                    self._azure_ad_token
                )
                api_key = self._azure_ad_token.token
        else:
            api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(
                "You must set an `api_key` parameter. "
                "Alternatively, you can set the AZURE_OPENAI_API_KEY env var "
                "OR set `use_azure_ad=True`."
            )
        self.api_key = api_key
        return api_key

    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        """Build kwargs for the Azure OpenAI SDK client constructor."""
        self._resolve_api_key()
        return {
            "api_key": self.api_key,
            "max_retries": 0,  # SDK retries disabled; handled by @retry decorator
            "timeout": self.timeout,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment": self.azure_deployment,
            "base_url": self.api_base,
            "azure_ad_token_provider": self.azure_ad_token_provider,
            "api_version": self.api_version,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _build_sync_client(self, **kwargs: Any) -> SyncAzureOpenAI:
        """Create a synchronous Azure OpenAI SDK client."""
        return SyncAzureOpenAI(**kwargs)

    def _build_async_client(self, **kwargs: Any) -> AsyncAzureOpenAI:
        """Create an asynchronous Azure OpenAI SDK client."""
        return AsyncAzureOpenAI(**kwargs)

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs["model"] = self.engine
        return model_kwargs

    @classmethod
    def class_name(cls) -> str:
        return "azure_openai_llm"
