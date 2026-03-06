"""Shared OpenAI connection configuration and client mixin."""

from __future__ import annotations

from typing import Any

import httpx
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from serapeum.openai.utils import resolve_openai_credentials


__all__ = ["OpenAIClientMixin"]


class OpenAIClientMixin(BaseModel):
    """Shared connection fields and client management for OpenAI provider classes.

    Owns the API credential configuration (api_key, api_base, api_version),
    connection parameters (timeout, max_retries, default_headers), and SDK
    client lifecycle. Supports injecting pre-built SDK clients via constructor
    kwargs for testing.

    Subclasses may override the ``client`` / ``async_client`` properties to
    customise client lifecycle (e.g. ``OpenAI`` adds ``reuse_client`` support).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str | None = Field(
        default=None, exclude=True, description="The OpenAI API key."
    )
    api_base: str | None = Field(
        default=None, description="The base URL for OpenAI API."
    )
    api_version: str | None = Field(
        default=None, description="The API version for OpenAI API."
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of API retries.",
        ge=0,
    )
    timeout: float = Field(
        default=60.0,
        description="The timeout, in seconds, for API requests.",
        ge=0,
    )
    default_headers: dict[str, str] | None = Field(
        default=None, description="The default headers for API requests."
    )

    _client: SyncOpenAI | None = PrivateAttr(default=None)
    _async_client: AsyncOpenAI | None = PrivateAttr(default=None)
    _http_client: httpx.Client | None = PrivateAttr(default=None)
    _async_http_client: httpx.AsyncClient | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _inject_clients(
        cls, data: Any, handler: Any
    ) -> OpenAIClientMixin:
        """Intercept client kwargs before Pydantic validation.

        Pops ``openai_client``, ``async_openai_client``, ``http_client``, and
        ``async_http_client`` from the raw input dict so Pydantic never sees
        them, runs normal validation via ``handler``, then writes the objects
        into private attributes. When no client is injected the private attrs
        remain ``None`` for lazy initialization via properties.
        """
        openai_client = None
        async_openai_client = None
        http_client = None
        async_http_client = None

        if isinstance(data, dict):
            openai_client = data.pop("openai_client", None)
            async_openai_client = data.pop("async_openai_client", None)
            http_client = data.pop("http_client", None)
            async_http_client = data.pop("async_http_client", None)

        instance = handler(data)

        if openai_client is not None:
            instance._client = openai_client
        if async_openai_client is not None:
            instance._async_client = async_openai_client
        if http_client is not None:
            instance._http_client = http_client
        if async_http_client is not None:
            instance._async_http_client = async_http_client

        return instance

    @model_validator(mode="after")
    def _resolve_credentials(self) -> OpenAIClientMixin:
        """Resolve API credentials from environment variables and defaults."""
        api_key, api_base, api_version = resolve_openai_credentials(
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
        )
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        return self

    def _get_credential_kwargs(self, is_async: bool = False) -> dict[str, Any]:
        """Build kwargs for the OpenAI SDK client constructor."""
        return {
            "api_key": self.api_key,
            "base_url": self.api_base,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    @property
    def client(self) -> SyncOpenAI:
        """Synchronous OpenAI client, lazily created on first access."""
        if self._client is None:
            self._client = SyncOpenAI(**self._get_credential_kwargs())
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Asynchronous OpenAI client, lazily created on first access."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                **self._get_credential_kwargs(is_async=True)
            )
        return self._async_client
