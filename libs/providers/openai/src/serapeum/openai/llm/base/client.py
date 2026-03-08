"""Shared OpenAI connection configuration and client mixin."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai import OpenAI as SyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from serapeum.core.retry import Retry
from serapeum.openai.utils import resolve_openai_credentials


__all__ = ["Client"]


class Client(Retry, BaseModel):
    """Shared connection fields and client management for OpenAI provider classes.

    Owns the API credential configuration (api_key, api_base, api_version),
    connection parameters (timeout, max_retries, default_headers), and SDK
    client lifecycle. Supports injecting pre-built SDK clients via constructor
    kwargs for testing.

    The ``async_client`` property tracks the asyncio event loop and automatically
    recreates the client when the loop has been closed (e.g. across pytest-asyncio
    tests).
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
    _async_client_loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _inject_clients(
        cls, data: Any, handler: Any
    ) -> Client:
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
    def _resolve_credentials(self) -> Client:
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
            "max_retries": 0,  # SDK retries disabled; handled by @retry decorator
            "timeout": self.timeout,
            "default_headers": self.default_headers,
            "http_client": self._async_http_client if is_async else self._http_client,
        }

    def _build_sync_client(self, **kwargs: Any) -> SyncOpenAI:
        """Create a synchronous SDK client.

        Subclasses override this to return a provider-specific client
        (e.g. ``SyncAzureOpenAI``) while inheriting the lazy-init and
        event-loop safety logic from the ``client`` / ``async_client``
        properties.
        """
        return SyncOpenAI(**kwargs)

    def _build_async_client(self, **kwargs: Any) -> AsyncOpenAI:
        """Create an asynchronous SDK client.

        Subclasses override this to return a provider-specific client
        (e.g. ``AsyncAzureOpenAI``).
        """
        return AsyncOpenAI(**kwargs)

    def _needs_async_client_recreation(self) -> bool:
        """Return ``True`` when the cached async client must be recreated.

        Checks whether the event loop the client was created on has been
        closed (e.g. between pytest-asyncio test functions).
        """
        cached = self._async_client_loop
        if cached is None:
            result = False
        elif hasattr(cached, "is_closed") and cached.is_closed():
            result = True
        else:
            result = False
        return result

    @property
    def client(self) -> SyncOpenAI:
        """Synchronous OpenAI client, lazily created on first access."""
        if self._client is None:
            self._client = self._build_sync_client(**self._get_credential_kwargs())
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Asynchronous OpenAI client with event-loop safety.

        Lazily creates the client on first access and tracks the asyncio event
        loop it was created on. If the loop has been closed (e.g. between
        pytest-asyncio test functions), the client is automatically recreated.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._async_client is None or self._needs_async_client_recreation():
            self._async_client = self._build_async_client(
                **self._get_credential_kwargs(is_async=True)
            )

        self._async_client_loop = current_loop
        return self._async_client
