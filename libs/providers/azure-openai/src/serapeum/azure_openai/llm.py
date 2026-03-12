"""Azure OpenAI provider classes for Serapeum.

Provides :class:`Completions` and :class:`Responses`, which are
Azure-specific subclasses of the corresponding OpenAI provider classes.
Both inherit connection management from the :class:`AzureClient` mixin,
which handles Azure AD / Microsoft Entra ID authentication,
deployment-name (engine) aliasing, and Azure SDK client construction.

Public classes:

- :class:`AzureClient` -- Mixin that adds Azure-specific fields
  (``engine``, ``azure_endpoint``, ``use_azure_ad``) and overrides SDK
  client construction to target Azure OpenAI endpoints.
- :class:`Completions` -- Azure Chat Completions API provider
  (``/chat/completions``).
- :class:`Responses` -- Azure Responses API provider
  (``/v1/responses``).
"""

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
from serapeum.openai import Completions as OpenAICompletions
from serapeum.openai import Responses as OpenAIResponses
from serapeum.openai.utils import DEFAULT_OPENAI_API_BASE

__all__ = [
    "AzureClient",
    "Completions",
    "Responses",
    "SyncAzureOpenAI",
    "AsyncAzureOpenAI",
]


class AzureClient:
    """Mixin that adds Azure-specific connection fields and client construction.

    Overrides the credential resolution and SDK client construction
    methods from :class:`~serapeum.openai.llm.base.Client` to target
    Azure OpenAI endpoints using Azure AD or API-key authentication.

    This mixin is designed to be composed with an API-specific class
    (``OpenAICompletions`` or ``OpenAIResponses``) via multiple
    inheritance::

        class Completions(AzureClient, OpenAICompletions): ...
        class Responses(AzureClient, OpenAIResponses): ...

    Args:
        model: OpenAI model name (e.g. ``"gpt-4o"``).  Used for routing
            and capability checks; the actual deployment is identified
            by ``engine``.  Defaults to ``"gpt-35-turbo"``.
        engine: Azure deployment name (called "model deployment name"
            in the Azure portal).  Aliases ``deployment_name``,
            ``deployment_id``, and ``deployment`` are also accepted.
        azure_endpoint: Azure OpenAI resource URL, e.g.
            ``https://YOUR_RESOURCE.openai.azure.com/``.  Falls back to
            the ``AZURE_OPENAI_ENDPOINT`` environment variable.
        azure_deployment: Optional Azure deployment identifier (passed
            directly to the SDK).
        use_azure_ad: When ``True``, authenticate via Microsoft Entra
            ID instead of an API key.  Defaults to ``False``.
        azure_ad_token_provider: Callback that returns a bearer token
            for Azure AD authentication.

    Examples:
        - Concrete classes expose the Azure-specific fields from this
          mixin:
            ```python
            >>> from serapeum.azure_openai.llm import Completions
            >>> sorted(
            ...     f for f in Completions.model_fields
            ...     if f.startswith("azure")
            ... )
            ['azure_ad_token_provider', 'azure_deployment', 'azure_endpoint']

            ```
        - The ``engine`` field identifies the Azure deployment:
            ```python
            >>> from serapeum.azure_openai.llm import Completions
            >>> Completions.model_fields["engine"].description
            'The name of the deployed azure engine.'

            ```
        - Instantiate with explicit credentials (requires a live
          endpoint):
            ```python
            from serapeum.azure_openai import Completions

            llm = Completions(
                engine="my-gpt4o-deployment",
                model="gpt-4o",
                api_key="sk-...",
                azure_endpoint="https://myresource.openai.azure.com/",
                api_version="2024-02-01",
            )

            ```

    See Also:
        serapeum.openai.llm.base.Client: Parent mixin providing shared
            connection fields (``api_key``, ``timeout``,
            ``max_retries``) and lazy SDK client lifecycle.
        Completions: Concrete Azure Chat Completions provider.
        Responses: Concrete Azure Responses API provider.
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
    def _reset_api_base_for_azure(self) -> AzureClient:
        """Reset api_base when it is the OpenAI default or azure_endpoint is set.

        Runs after the parent ``_resolve_credentials`` validator which
        may set ``api_base`` to the OpenAI default URL.
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
        """Swap ``model`` for the Azure ``engine`` deployment name."""
        model_kwargs = super()._get_model_kwargs(**kwargs)
        model_kwargs["model"] = self.engine
        return model_kwargs


class Completions(AzureClient, OpenAICompletions):
    """Azure OpenAI Chat Completions API provider.

    Combines Azure-specific connection management from
    :class:`AzureClient` with the Chat Completions API logic from
    :class:`~serapeum.openai.llm.chat_completions.Completions`.

    To use this provider you must first deploy a model on Azure OpenAI.
    Unlike direct OpenAI, you need to specify an ``engine`` parameter
    to identify your deployment (called "model deployment name" in the
    Azure portal):

    - ``model`` -- the OpenAI model name (e.g. ``gpt-4o``).  Used
      internally for capability checks; not sent to the API.
    - ``engine`` -- the Azure deployment name you chose when deploying
      the model.

    Required environment variables (or equivalent constructor args):

    - ``OPENAI_API_VERSION`` -- set to ``2024-02-01`` or newer.
    - ``AZURE_OPENAI_ENDPOINT`` -- your resource URL, e.g.
      ``https://YOUR_RESOURCE_NAME.openai.azure.com/``.
    - ``AZURE_OPENAI_API_KEY`` -- your API key, **or** pass
      ``azure_ad_token_provider`` and ``use_azure_ad=True`` for
      managed identity with Microsoft Entra ID.

    More information:
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/quickstart

    Examples:
        - Query the canonical provider name:
            ```python
            >>> from serapeum.azure_openai import Completions
            >>> Completions.class_name()
            'azure_openai_completions'

            ```
        - Inspect Azure-specific fields available on this class:
            ```python
            >>> from serapeum.azure_openai import Completions
            >>> sorted(
            ...     f for f in Completions.model_fields
            ...     if f.startswith("azure")
            ... )
            ['azure_ad_token_provider', 'azure_deployment', 'azure_endpoint']

            ```
        - Instantiate with API-key authentication:
            ```python
            from serapeum.azure_openai import Completions

            llm = Completions(
                engine="my-deployment",
                model="gpt-4o",
                api_key="YOUR_AZURE_OPENAI_API_KEY",
                azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
                api_version="2024-02-01",
            )
            response = llm.chat(messages=[...])

            ```
        - Instantiate with Microsoft Entra ID (managed identity):
            ```python
            from azure.identity import (
                DefaultAzureCredential,
                get_bearer_token_provider,
            )
            from serapeum.azure_openai import Completions

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )
            llm = Completions(
                engine="my-deployment",
                model="gpt-4o",
                azure_ad_token_provider=token_provider,
                use_azure_ad=True,
                azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
                api_version="2024-02-01",
            )

            ```

    See Also:
        AzureClient: Mixin providing Azure-specific fields and client
            construction.
        serapeum.openai.llm.chat_completions.Completions: The OpenAI
            Chat Completions base class that this class extends.
        Responses: Azure Responses API provider for reasoning models.
    """

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical provider identifier for this class.

        Returns:
            The string ``"azure_openai_completions"``.

        Examples:
            - Retrieve the provider name for logging or dispatch:
                ```python
                >>> from serapeum.azure_openai import Completions
                >>> Completions.class_name()
                'azure_openai_completions'

                ```

        See Also:
            Responses.class_name: Returns
                ``"azure_openai_responses"``.
        """
        return "azure_openai_completions"


class Responses(AzureClient, OpenAIResponses):
    """Azure OpenAI Responses API provider.

    Combines Azure-specific connection management from
    :class:`AzureClient` with the Responses API logic from
    :class:`~serapeum.openai.llm.responses.Responses`.

    Supports streaming, built-in tools (web search, file search, code
    interpreter), stateful conversation continuation via
    ``track_previous_responses``, structured output via tool-call
    forcing, and reasoning-effort control -- all through an Azure
    OpenAI deployment.

    Required environment variables (or equivalent constructor args):

    - ``OPENAI_API_VERSION`` -- set to ``2024-02-01`` or newer.
    - ``AZURE_OPENAI_ENDPOINT`` -- your resource URL, e.g.
      ``https://YOUR_RESOURCE_NAME.openai.azure.com/``.
    - ``AZURE_OPENAI_API_KEY`` -- your API key, **or** pass
      ``azure_ad_token_provider`` and ``use_azure_ad=True`` for
      managed identity with Microsoft Entra ID.

    Examples:
        - Query the canonical provider name:
            ```python
            >>> from serapeum.azure_openai import Responses
            >>> Responses.class_name()
            'azure_openai_responses'

            ```
        - Inspect Azure-specific fields available on this class:
            ```python
            >>> from serapeum.azure_openai import Responses
            >>> sorted(
            ...     f for f in Responses.model_fields
            ...     if f.startswith("azure")
            ... )
            ['azure_ad_token_provider', 'azure_deployment', 'azure_endpoint']

            ```
        - The ``engine`` field identifies the Azure deployment:
            ```python
            >>> from serapeum.azure_openai import Responses
            >>> Responses.model_fields["engine"].description
            'The name of the deployed azure engine.'

            ```
        - Instantiate with an o3 deployment:
            ```python
            from serapeum.azure_openai import Responses

            llm = Responses(
                engine="my-o3-deployment",
                model="o3",
                api_key="YOUR_AZURE_OPENAI_API_KEY",
                azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
                api_version="2024-02-01",
            )
            response = llm.chat(messages=[...])

            ```

    See Also:
        AzureClient: Mixin providing Azure-specific fields and client
            construction.
        serapeum.openai.llm.responses.Responses: The OpenAI Responses
            base class that this class extends.
        Completions: Azure Chat Completions API provider for standard
            chat models.
    """

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical provider identifier for this class.

        Returns:
            The string ``"azure_openai_responses"``.

        Examples:
            - Retrieve the provider name for logging or dispatch:
                ```python
                >>> from serapeum.azure_openai import Responses
                >>> Responses.class_name()
                'azure_openai_responses'

                ```

        See Also:
            Completions.class_name: Returns
                ``"azure_openai_completions"``.
        """
        return "azure_openai_responses"