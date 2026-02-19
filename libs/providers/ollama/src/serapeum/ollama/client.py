"""Shared Ollama connection configuration and client mixin."""
from __future__ import annotations

from typing import Any

import ollama as ollama_sdk  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_CLOUD_BASE_URL = "https://api.ollama.com"


__all__ = ["DEFAULT_BASE_URL", "OLLAMA_CLOUD_BASE_URL", "OllamaClientMixin"]


class OllamaClientMixin(BaseModel):
    """Shared connection fields and client injection for Ollama provider classes.

    Owns the server connection configuration (base_url, api_key), handles
    automatic URL resolution to Ollama Cloud when an api_key is provided, and
    supports injecting pre-built SDK clients via constructor kwargs for testing.

    Subclasses override ``_build_client_kwargs`` to add provider-specific
    options (e.g. ``timeout`` for the LLM class, or extra ``client_kwargs``
    for the embedding class).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description="Base URL where the Ollama server is hosted.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "API key for authenticated Ollama endpoints (e.g. Ollama Cloud). "
            "When set and base_url is the local default, base_url is automatically "
            "switched to the Ollama Cloud endpoint."
        ),
    )

    _client: ollama_sdk.Client | None = PrivateAttr(default=None)
    _async_client: ollama_sdk.AsyncClient | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _inject_clients(cls, data: Any, handler: Any) -> "OllamaClientMixin":
        """Intercept client/async_client kwargs before Pydantic validation.

        Pops ``client`` and ``async_client`` from the raw input dict so Pydantic
        never sees them (they cannot be declared as regular fields), runs normal
        validation via ``handler``, then writes the objects into private attributes.
        When no client is injected the private attrs remain ``None`` for lazy or
        eager initialization by the subclass.
        """
        client = None
        async_client = None
        if isinstance(data, dict):
            client = data.pop("client", None)
            async_client = data.pop("async_client", None)

        instance = handler(data)

        if client is not None:
            instance._client = client
        if async_client is not None:
            instance._async_client = async_client

        return instance

    @model_validator(mode="after")
    def _resolve_base_url(self) -> "OllamaClientMixin":
        """Switch base_url to Ollama Cloud when api_key is provided with the default base_url.

        An explicit non-default base_url is always preserved so custom remote
        deployments are not affected.
        """
        if self.api_key and self.base_url == DEFAULT_BASE_URL:
            self.base_url = OLLAMA_CLOUD_BASE_URL
        return self

    def _build_client_kwargs(self) -> dict[str, Any]:
        """Build base kwargs for Ollama client initialization.

        Returns ``host`` and, when an ``api_key`` is set, an ``Authorization``
        header. Subclasses override this to add provider-specific options.
        """
        kwargs: dict[str, Any] = {"host": self.base_url}
        if self.api_key:
            kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
        return kwargs

    @property
    def client(self) -> ollama_sdk.Client:
        """Synchronous Ollama client, lazily created on first access."""
        if self._client is None:
            self._client = ollama_sdk.Client(**self._build_client_kwargs())
        return self._client

    @property
    def async_client(self) -> ollama_sdk.AsyncClient:
        """Asynchronous Ollama client, lazily created on first access."""
        if self._async_client is None:
            self._async_client = ollama_sdk.AsyncClient(**self._build_client_kwargs())
        return self._async_client

    def list_models(self) -> list[str]:
        """Return the names of all models available on the Ollama server.

        Returns:
            list[str]: Model name strings (e.g. ``["llama3.1:latest", "mistral:latest"]``).

        Examples:
            - List available models
                ```python
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="llama3.1")
                >>> models = llm.list_models()  # doctest: +SKIP
                >>> isinstance(models, list)    # doctest: +SKIP
                True

                ```
        """
        return [m.model for m in self.client.list().models if m.model is not None]

    async def alist_models(self) -> list[str]:
        """Asynchronously return the names of all models available on the Ollama server.

        Returns:
            list[str]: Model name strings (e.g. ``["llama3.1:latest", "mistral:latest"]``).

        Examples:
            - Async list available models
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="llama3.1")
                >>> async def get_models():         # doctest: +SKIP
                ...     return await llm.alist_models()
                >>> asyncio.run(get_models())       # doctest: +SKIP
                ['llama3.1:latest', 'mistral:latest']

                ```
        """
        response = await self.async_client.list()
        return [m.model for m in response.models if m.model is not None]
