"""Shared OpenAI connection configuration and client lifecycle management.

Provides :class:`Client`, a Pydantic mixin that owns API credentials
(``api_key``, ``api_base``, ``api_version``), connection parameters
(``timeout``, ``max_retries``, ``default_headers``), and lazy-initialised
synchronous / asynchronous OpenAI SDK clients.

The mixin is designed for multiple inheritance: concrete provider classes
such as ``Completions`` and ``Responses`` inherit from ``Client`` to gain
connection management without duplicating configuration fields.
"""

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

    Owns the API credential configuration (``api_key``, ``api_base``,
    ``api_version``), connection parameters (``timeout``, ``max_retries``,
    ``default_headers``), and the OpenAI SDK client lifecycle.  Both the
    synchronous and asynchronous SDK clients are lazily initialised on first
    access via :pyattr:`client` and :pyattr:`async_client`.

    Pre-built SDK clients can be injected through the constructor keyword
    arguments ``openai_client``, ``async_openai_client``, ``http_client``,
    and ``async_http_client``.  This is useful for testing or for sharing a
    single HTTP connection pool across multiple provider instances.

    The :pyattr:`async_client` property tracks the asyncio event loop and
    automatically recreates the SDK client when the loop has been closed
    (e.g. between ``pytest-asyncio`` test functions).

    Inherits ``max_retries`` from :class:`~serapeum.core.retry.Retry`.

    Args:
        api_key: OpenAI API key.  Resolved from the ``OPENAI_API_KEY``
            environment variable when not provided explicitly.
        api_base: Base URL for the OpenAI API.  Defaults to
            ``https://api.openai.com/v1`` when not set.
        api_version: API version string.  Typically left empty for the
            standard OpenAI endpoint; required for Azure OpenAI.
        timeout: Timeout in seconds for each HTTP request.  Defaults to
            ``60.0``.  Must be >= 0.
        default_headers: Optional mapping of HTTP headers sent with every
            request.
        max_retries: Maximum number of retry attempts for transient errors.
            Inherited from :class:`~serapeum.core.retry.Retry`.
            Defaults to ``3``.

    Examples:
        - Create a client with explicit credentials and inspect defaults:
            ```python
            >>> from serapeum.openai.llm.base import Client  # doctest: +SKIP
            >>> c = Client(api_key="sk-test")  # doctest: +SKIP
            >>> c.timeout  # doctest: +SKIP
            60.0
            >>> c.api_base  # doctest: +SKIP
            'https://api.openai.com/v1'

            ```
        - Override the timeout and add custom headers:
            ```python
            >>> from serapeum.openai.llm.base import Client  # doctest: +SKIP
            >>> c = Client(  # doctest: +SKIP
            ...     api_key="sk-test",
            ...     timeout=120.0,
            ...     default_headers={"X-Custom": "value"},
            ... )
            >>> c.timeout  # doctest: +SKIP
            120.0
            >>> c.default_headers["X-Custom"]  # doctest: +SKIP
            'value'

            ```
        - Inject a pre-built synchronous client for testing:
            ```python
            >>> from unittest.mock import MagicMock  # doctest: +SKIP
            >>> from serapeum.openai.llm.base import Client  # doctest: +SKIP
            >>> mock_sdk = MagicMock()  # doctest: +SKIP
            >>> c = Client(api_key="sk-test", openai_client=mock_sdk)  # doctest: +SKIP
            >>> c.client is mock_sdk  # doctest: +SKIP
            True

            ```

    See Also:
        serapeum.core.retry.Retry: Mixin providing ``max_retries``.
        serapeum.openai.utils.resolve_openai_credentials: Credential
            resolution logic used during validation.
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
        """Synchronous OpenAI SDK client, lazily created on first access.

        On the first call the client is built via :meth:`_build_sync_client`
        using the credentials and connection settings from this instance.
        Subsequent calls return the cached client.  If a pre-built client was
        injected through the ``openai_client`` constructor kwarg, that object
        is returned directly.

        Returns:
            The synchronous ``openai.OpenAI`` SDK client configured for this
            instance.

        Examples:
            - Access the lazily-initialised synchronous client:
                ```python
                >>> from serapeum.openai.llm.base import Client  # doctest: +SKIP
                >>> c = Client(api_key="sk-test")  # doctest: +SKIP
                >>> sdk = c.client  # doctest: +SKIP
                >>> sdk.base_url  # doctest: +SKIP
                URL('https://api.openai.com/v1/')

                ```

        See Also:
            async_client: Asynchronous counterpart with event-loop safety.
        """
        if self._client is None:
            self._client = self._build_sync_client(**self._get_credential_kwargs())
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """Asynchronous OpenAI SDK client with event-loop safety.

        Lazily creates the client on first access via
        :meth:`_build_async_client` and records the running asyncio event
        loop.  On subsequent accesses the cached client is returned unless
        the event loop it was created on has been closed, in which case a
        fresh client is built automatically.  This prevents
        ``RuntimeError: Event loop is closed`` errors that arise when
        reusing a client across ``pytest-asyncio`` test functions.

        Returns:
            The asynchronous ``openai.AsyncOpenAI`` SDK client configured
            for this instance.

        Examples:
            - Access the lazily-initialised async client inside a coroutine:
                ```python
                >>> import asyncio  # doctest: +SKIP
                >>> from serapeum.openai.llm.base import Client  # doctest: +SKIP
                >>> c = Client(api_key="sk-test")  # doctest: +SKIP
                >>> async def demo():  # doctest: +SKIP
                ...     sdk = c.async_client
                ...     return str(sdk.base_url)
                >>> asyncio.run(demo())  # doctest: +SKIP
                'https://api.openai.com/v1/'

                ```

        See Also:
            client: Synchronous counterpart.
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
