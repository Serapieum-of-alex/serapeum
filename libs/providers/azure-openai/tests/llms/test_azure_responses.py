from __future__ import annotations

from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncAzureOpenAI
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)

from serapeum.azure_openai import Responses
from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.llms import Message, TextChunk

AZURE_DEFAULTS: dict[str, Any] = {
    "engine": "my-deployment",
    "api_key": "test-key",
    "api_version": "2024-02-01",
}


def _make_responses_llm(**overrides: Any) -> Responses:
    """Create a Responses instance with sensible Azure defaults."""
    params = {**AZURE_DEFAULTS, **overrides}
    return Responses(**params)


def _make_mock_response(
    *,
    response_id: str = "resp_abc123",
    text: str = "Hello world",
) -> MagicMock:
    """Build a mock ``openai.types.responses.Response`` object."""
    mock_response = MagicMock()
    mock_response.id = response_id

    output_msg = ResponseOutputMessage(
        type="message",
        content=[{"type": "output_text", "text": text, "annotations": []}],
        role="assistant",
        id="msg_001",
        status="completed",
    )
    mock_response.output = [output_msg]

    usage = MagicMock()
    usage.output_tokens = 10
    usage.input_tokens = 5
    usage.output_tokens_details = MagicMock()
    usage.output_tokens_details.reasoning_tokens = 0
    mock_response.usage = usage

    return mock_response


def _make_stream_events() -> list:
    """Build a sequence of mock streaming events for the Responses API."""
    return [
        ResponseTextDeltaEvent(
            content_index=0,
            item_id="item_001",
            output_index=0,
            delta="Hello from",
            type="response.output_text.delta",
            sequence_number=1,
            logprobs=[],
        ),
        ResponseTextDeltaEvent(
            content_index=0,
            item_id="item_001",
            output_index=0,
            delta=" Azure",
            type="response.output_text.delta",
            sequence_number=2,
            logprobs=[],
        ),
    ]


async def _async_stream_events(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[ResponseTextDeltaEvent, None]:
    """Return an async generator yielding mock streaming events."""

    async def gen() -> AsyncGenerator[ResponseTextDeltaEvent, None]:
        for event in _make_stream_events():
            yield event

    return gen()


@pytest.mark.unit
class TestResponses:
    """Unit tests for the azure_openai.Responses class."""

    def test_mro_includes_base_llm(self) -> None:
        """Verify Responses inherits from BaseLLM.

        Test scenario:
            The MRO of Responses should include BaseLLM to ensure it satisfies
            the framework's LLM interface contract.
        """
        mro_names = [b.__name__ for b in Responses.__mro__]
        assert BaseLLM.__name__ in mro_names, f"BaseLLM not found in MRO: {mro_names}"

    def test_class_name(self) -> None:
        """Verify class_name() returns the expected provider identifier.

        Test scenario:
            class_name() must return 'azure_openai_responses' to distinguish
            this class from the Completions variant and from plain OpenAI.
        """
        assert (
            Responses.class_name() == "azure_openai_responses"
        ), f"Expected 'azure_openai_responses', got '{Responses.class_name()}'"

    def test_missing_engine_raises(self) -> None:
        """Instantiation without engine raises ValueError.

        Test scenario:
            Omitting the required 'engine' field triggers the _validate_azure_env
            validator which raises with a message containing 'engine'.
        """
        with pytest.raises(ValueError, match="engine"):
            Responses(api_key="k", api_version="2024-02-01")

    def test_missing_api_version_raises(self) -> None:
        """Instantiation without api_version raises ValueError.

        Test scenario:
            Omitting api_version triggers validation requiring OPENAI_API_VERSION.
        """
        with pytest.raises(ValueError, match="OPENAI_API_VERSION"):
            Responses(engine="dep", api_key="k")

    def test_default_openai_base_without_azure_endpoint_raises(self) -> None:
        """Using default OpenAI base URL without azure_endpoint raises.

        Test scenario:
            Setting api_base to the default OpenAI URL without providing an
            azure_endpoint should raise because it's clearly misconfigured.
        """
        with pytest.raises(ValueError, match="OPENAI_API_BASE"):
            Responses(
                engine="dep",
                api_key="k",
                api_version="2024-02-01",
                api_base="https://api.openai.com/v1",
            )

    def test_engine_alias_deployment_name(self) -> None:
        """Engine can be set via the deployment_name alias.

        Test scenario:
            Passing deployment_name instead of engine should resolve correctly
            through resolve_from_aliases.
        """
        llm = Responses(
            deployment_name="my-dep",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "my-dep", f"Expected engine 'my-dep', got '{llm.engine}'"

    def test_engine_alias_deployment_id(self) -> None:
        """Engine can be set via the deployment_id alias.

        Test scenario:
            deployment_id is an alternative alias for the engine field.
        """
        llm = Responses(
            deployment_id="my-dep-id",
            api_key="k",
            api_version="2024-02-01",
        )
        assert (
            llm.engine == "my-dep-id"
        ), f"Expected engine 'my-dep-id', got '{llm.engine}'"

    def test_engine_alias_deployment(self) -> None:
        """Engine can be set via the deployment alias.

        Test scenario:
            deployment is yet another alias for the engine field.
        """
        llm = Responses(
            deployment="my-dep",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "my-dep", f"Expected engine 'my-dep', got '{llm.engine}'"

    def test_engine_alias_azure_deployment(self) -> None:
        """Engine can be set via the azure_deployment field.

        Test scenario:
            azure_deployment is also accepted as an alias for engine when
            engine itself is not provided.
        """
        llm = Responses(
            azure_deployment="az-dep",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "az-dep", f"Expected engine 'az-dep', got '{llm.engine}'"

    def test_api_base_reset_when_azure_endpoint_set(self) -> None:
        """api_base is reset to None when azure_endpoint is provided.

        Test scenario:
            The _reset_api_base_for_azure validator nullifies api_base
            when an azure_endpoint is explicitly set.
        """
        llm = Responses(
            engine="dep",
            api_key="k",
            api_version="2024-02-01",
            azure_endpoint="https://myres.openai.azure.com/",
        )
        assert llm.api_base is None, f"Expected api_base None, got '{llm.api_base}'"
        assert llm.azure_endpoint == "https://myres.openai.azure.com/"

    def test_default_model(self) -> None:
        """Default model is gpt-35-turbo (inherited from AzureClient).

        Test scenario:
            When no model is specified, the AzureClient default of
            'gpt-35-turbo' takes precedence.
        """
        llm = _make_responses_llm()
        assert (
            llm.model == "gpt-35-turbo"
        ), f"Expected 'gpt-35-turbo', got '{llm.model}'"

    def test_custom_model(self) -> None:
        """A custom model name is preserved.

        Test scenario:
            Explicitly passing model='o3' should override the default.
        """
        llm = _make_responses_llm(model="o3")
        assert llm.model == "o3", f"Expected 'o3', got '{llm.model}'"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_get_model_kwargs_uses_engine(self, sync_mock: MagicMock) -> None:
        """_get_model_kwargs substitutes engine for model.

        Test scenario:
            The Azure override swaps the 'model' key with the deployment
            engine name so the SDK targets the correct Azure deployment.
        """
        llm = _make_responses_llm(engine="my-responses-deployment")
        kwargs = llm._get_model_kwargs()
        assert (
            kwargs["model"] == "my-responses-deployment"
        ), f"Expected model='my-responses-deployment', got '{kwargs['model']}'"

    def test_azure_deployment_field_preserved(self) -> None:
        """azure_deployment field is stored separately from engine.

        Test scenario:
            Both engine and azure_deployment can be set independently.
        """
        llm = Responses(
            engine="dep",
            api_key="k",
            api_version="2024-02-01",
            azure_deployment="az-dep-id",
        )
        assert (
            llm.azure_deployment == "az-dep-id"
        ), f"Expected 'az-dep-id', got '{llm.azure_deployment}'"

    def test_use_azure_ad_default_false(self) -> None:
        """use_azure_ad defaults to False.

        Test scenario:
            Without explicit opt-in, Azure AD authentication is disabled.
        """
        llm = _make_responses_llm()
        assert (
            llm.use_azure_ad is False
        ), f"Expected use_azure_ad=False, got {llm.use_azure_ad}"


@pytest.mark.unit
class TestResponsesCredentials:
    """Tests for _resolve_api_key and _get_credential_kwargs on Responses."""

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_api_key_from_env(
        self, sync_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API key is resolved from AZURE_OPENAI_API_KEY env var.

        Test scenario:
            When no api_key is passed explicitly, _resolve_api_key should
            fall back to the AZURE_OPENAI_API_KEY environment variable.
        """
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        llm = Responses(engine="dep", api_version="2024-02-01")
        key = llm._resolve_api_key()
        assert key == "env-key", f"Expected 'env-key', got '{key}'"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_api_key_missing_raises(
        self, sync_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key raises ValueError.

        Test scenario:
            With no api_key, no env var, and no Azure AD, _resolve_api_key
            must raise a clear error.
        """
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = Responses(engine="dep", api_version="2024-02-01")
        llm.api_key = None
        with pytest.raises(ValueError, match="api_key"):
            llm._resolve_api_key()

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_azure_ad_token_provider_sets_key(self, sync_mock: MagicMock) -> None:
        """Azure AD token provider callback is invoked to set api_key.

        Test scenario:
            With use_azure_ad=True and a token provider callback, the
            callback's return value becomes the api_key.
        """
        llm = Responses(
            engine="dep",
            api_version="2024-02-01",
            use_azure_ad=True,
            azure_ad_token_provider=lambda: "ad-token",
        )
        key = llm._resolve_api_key()
        assert key == "ad-token", f"Expected 'ad-token', got '{key}'"
        assert llm.api_key == "ad-token"

    @patch("serapeum.azure_openai.llm.refresh_openai_azure_ad_token")
    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_azure_ad_refresh_fallback(
        self, sync_mock: MagicMock, refresh_mock: MagicMock
    ) -> None:
        """Without token provider, falls back to refresh_openai_azure_ad_token.

        Test scenario:
            With use_azure_ad=True but no provider callback, the framework
            uses refresh_openai_azure_ad_token to obtain a token.
        """
        mock_token = MagicMock()
        mock_token.token = "refreshed-token"  # nosec B105
        refresh_mock.return_value = mock_token

        llm = Responses(
            engine="dep",
            api_version="2024-02-01",
            use_azure_ad=True,
        )
        key = llm._resolve_api_key()
        assert key == "refreshed-token", f"Expected 'refreshed-token', got '{key}'"
        refresh_mock.assert_called_once()

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_get_credential_kwargs_contains_azure_fields(
        self, sync_mock: MagicMock
    ) -> None:
        """_get_credential_kwargs includes Azure-specific fields.

        Test scenario:
            The kwargs dict must include azure_endpoint, azure_deployment,
            api_version, api_key, and azure_ad_token_provider for the SDK.
        """
        llm = Responses(
            engine="dep",
            api_key="k",
            api_version="2024-02-01",
            azure_endpoint="https://myres.openai.azure.com/",
            azure_deployment="dep",
        )
        kwargs = llm._get_credential_kwargs()
        assert (
            kwargs["azure_endpoint"] == "https://myres.openai.azure.com/"
        ), f"Unexpected azure_endpoint: {kwargs['azure_endpoint']}"
        assert kwargs["azure_deployment"] == "dep"
        assert kwargs["api_version"] == "2024-02-01"
        assert kwargs["api_key"] == "k"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_get_credential_kwargs_async_uses_async_http_client(
        self, sync_mock: MagicMock
    ) -> None:
        """_get_credential_kwargs with is_async=True uses async http client.

        Test scenario:
            Passing is_async=True should select _async_http_client instead
            of _http_client for the http_client kwarg.
        """
        llm = _make_responses_llm()
        sync_kwargs = llm._get_credential_kwargs(is_async=False)
        async_kwargs = llm._get_credential_kwargs(is_async=True)
        assert sync_kwargs["http_client"] is not async_kwargs["http_client"] or (
            sync_kwargs["http_client"] is None and async_kwargs["http_client"] is None
        )

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_client_property_returns_azure_client(self, sync_mock: MagicMock) -> None:
        """client property creates a SyncAzureOpenAI instance.

        Test scenario:
            Accessing .client should call _build_sync_client which
            constructs a SyncAzureOpenAI (mocked here).
        """
        llm = _make_responses_llm()
        _ = llm.client
        sync_mock.assert_called_once()

    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_async_client_property_returns_azure_async_client(
        self, sync_mock: MagicMock, async_mock: MagicMock
    ) -> None:
        """async_client property creates an AsyncAzureOpenAI instance.

        Test scenario:
            Accessing .async_client should call _build_async_client which
            constructs an AsyncAzureOpenAI (mocked here).
        """
        llm = _make_responses_llm()
        _ = llm.async_client
        async_mock.assert_called_once()

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_build_sync_client_returns_sync_azure(self, sync_mock: MagicMock) -> None:
        """_build_sync_client returns a SyncAzureOpenAI instance.

        Test scenario:
            The method delegates to the SyncAzureOpenAI constructor
            with the provided kwargs.
        """
        llm = _make_responses_llm()
        result = llm._build_sync_client(api_key="k", azure_endpoint="https://e.com/")
        sync_mock.assert_called_once_with(api_key="k", azure_endpoint="https://e.com/")
        assert result == sync_mock.return_value

    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    def test_build_async_client_returns_async_azure(
        self, async_mock: MagicMock
    ) -> None:
        """_build_async_client returns an AsyncAzureOpenAI instance.

        Test scenario:
            The method delegates to the AsyncAzureOpenAI constructor
            with the provided kwargs.
        """
        llm = _make_responses_llm()
        result = llm._build_async_client(api_key="k", azure_endpoint="https://e.com/")
        async_mock.assert_called_once_with(api_key="k", azure_endpoint="https://e.com/")
        assert result == async_mock.return_value


@pytest.mark.mock
class TestResponsesWithMocks:
    """Tests for Responses that require mocking the Azure SDK clients."""

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_custom_http_client(self, sync_azure_mock: MagicMock) -> None:
        """Verify that a custom http_client is passed to the Azure SDK client.

        Test scenario:
            A user-supplied httpx.Client should be forwarded to the
            SyncAzureOpenAI constructor via _get_credential_kwargs.
        """
        custom_http_client = httpx.Client()
        mock_instance = sync_azure_mock.return_value
        mock_resp = _make_mock_response(text="test output")
        mock_instance.responses.create.return_value = mock_resp

        llm = Responses(
            engine="foo-bar",
            http_client=custom_http_client,
            api_key="mock",
            api_version="2024-02-01",
        )
        llm.complete("test prompt")
        sync_azure_mock.assert_called()
        kwargs = sync_azure_mock.call_args.kwargs
        assert (
            "http_client" in kwargs
        ), f"http_client not in SDK kwargs: {list(kwargs.keys())}"
        assert kwargs["http_client"] == custom_http_client

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_custom_azure_ad_token_provider(self, sync_azure_mock: MagicMock) -> None:
        """Verify that a custom azure ad token provider is used for authentication.

        Test scenario:
            With use_azure_ad=True and a token provider, the provider's
            return value should become the api_key used for requests.
        """

        def custom_provider() -> str:
            return "mock_api_key"

        mock_instance = sync_azure_mock.return_value
        mock_resp = _make_mock_response(text="test output")
        mock_instance.responses.create.return_value = mock_resp

        llm = Responses(
            engine="foo-bar",
            use_azure_ad=True,
            azure_ad_token_provider=custom_provider,
            api_version="2024-02-01",
        )
        llm.complete("test prompt")
        assert (
            llm.api_key == "mock_api_key"
        ), f"Expected api_key='mock_api_key', got '{llm.api_key}'"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_non_streaming_complete(self, sync_azure_mock: MagicMock) -> None:
        """Non-streaming complete returns a CompletionResponse.

        Test scenario:
            A non-streaming call to complete() should return a single
            CompletionResponse with the response text.
        """
        mock_instance = sync_azure_mock.return_value
        mock_resp = _make_mock_response(text="Hello from Azure Responses")
        mock_instance.responses.create.return_value = mock_resp

        llm = _make_responses_llm()
        response = llm.complete("test prompt")
        assert (
            response.text == "Hello from Azure Responses"
        ), f"Expected 'Hello from Azure Responses', got '{response.text}'"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_non_streaming_chat(self, sync_azure_mock: MagicMock) -> None:
        """Non-streaming chat returns a ChatResponse.

        Test scenario:
            A non-streaming call to chat() should return a ChatResponse
            with the assistant's message content.
        """
        mock_instance = sync_azure_mock.return_value
        mock_resp = _make_mock_response(text="Chat reply from Azure")
        mock_instance.responses.create.return_value = mock_resp

        llm = _make_responses_llm()
        message = Message(role="user", chunks=[TextChunk(content="hello")])
        response = llm.chat([message])
        assert (
            response.message.content == "Chat reply from Azure"
        ), f"Expected 'Chat reply from Azure', got '{response.message.content}'"
        assert response.message.role == "assistant"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_streaming_complete(self, sync_azure_mock: MagicMock) -> None:
        """Streaming complete yields CompletionResponse chunks.

        Test scenario:
            A streaming call to complete(stream=True) should yield incremental
            text chunks that accumulate the full response.
        """
        mock_instance = sync_azure_mock.return_value
        mock_instance.responses.create.return_value = _make_stream_events()

        llm = _make_responses_llm()
        response_gen = llm.complete("test prompt", stream=True)
        responses = list(response_gen)
        assert len(responses) > 0, "Expected at least one streaming response"
        assert (
            "Azure" in responses[-1].text
        ), f"Expected 'Azure' in final text, got '{responses[-1].text}'"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_streaming_chat(self, sync_azure_mock: MagicMock) -> None:
        """Streaming chat yields ChatResponse chunks.

        Test scenario:
            A streaming call to chat(stream=True) should yield incremental
            ChatResponse chunks with accumulating message content.
        """
        mock_instance = sync_azure_mock.return_value
        mock_instance.responses.create.return_value = _make_stream_events()

        llm = _make_responses_llm()
        message = Message(role="user", chunks=[TextChunk(content="hello")])
        chat_gen = llm.chat([message], stream=True)
        chat_responses = list(chat_gen)
        assert len(chat_responses) > 0, "Expected at least one streaming response"
        final_content = chat_responses[-1].message.content
        assert (
            "Azure" in final_content
        ), f"Expected 'Azure' in final content, got '{final_content}'"

    @pytest.mark.asyncio
    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    async def test_async_non_streaming_complete(
        self, async_azure_mock: MagicMock
    ) -> None:
        """Async non-streaming complete returns a CompletionResponse.

        Test scenario:
            acomplete() without streaming returns a single CompletionResponse.
        """
        mock_instance = MagicMock(spec=AsyncAzureOpenAI)
        async_azure_mock.return_value = mock_instance

        mock_resp = _make_mock_response(text="Async Azure reply")
        create_fn = AsyncMock(return_value=mock_resp)
        mock_instance.responses.create = create_fn

        llm = _make_responses_llm()
        response = await llm.acomplete("test prompt")
        assert (
            response.text == "Async Azure reply"
        ), f"Expected 'Async Azure reply', got '{response.text}'"

    @pytest.mark.asyncio
    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    async def test_async_non_streaming_chat(self, async_azure_mock: MagicMock) -> None:
        """Async non-streaming chat returns a ChatResponse.

        Test scenario:
            achat() without streaming returns a ChatResponse with content.
        """
        mock_instance = MagicMock(spec=AsyncAzureOpenAI)
        async_azure_mock.return_value = mock_instance

        mock_resp = _make_mock_response(text="Async chat reply")
        create_fn = AsyncMock(return_value=mock_resp)
        mock_instance.responses.create = create_fn

        llm = _make_responses_llm()
        message = Message(role="user", chunks=[TextChunk(content="hello")])
        response = await llm.achat([message])
        assert (
            response.message.content == "Async chat reply"
        ), f"Expected 'Async chat reply', got '{response.message.content}'"
        assert response.message.role == "assistant"

    @pytest.mark.asyncio
    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    async def test_async_streaming_complete(self, async_azure_mock: MagicMock) -> None:
        """Async streaming complete yields CompletionResponse chunks.

        Test scenario:
            acomplete(stream=True) should yield incremental response chunks.
        """
        mock_instance = MagicMock(spec=AsyncAzureOpenAI)
        async_azure_mock.return_value = mock_instance

        create_fn = AsyncMock(side_effect=_async_stream_events)
        mock_instance.responses.create = create_fn

        llm = _make_responses_llm()
        response_gen = await llm.acomplete("test prompt", stream=True)
        responses = [item async for item in response_gen]
        assert len(responses) > 0, "Expected at least one streaming response"
        assert (
            "Azure" in responses[-1].text
        ), f"Expected 'Azure' in final text, got '{responses[-1].text}'"

    @pytest.mark.asyncio
    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    async def test_async_streaming_chat(self, async_azure_mock: MagicMock) -> None:
        """Async streaming chat yields ChatResponse chunks.

        Test scenario:
            achat(stream=True) should yield incremental ChatResponse chunks.
        """
        mock_instance = MagicMock(spec=AsyncAzureOpenAI)
        async_azure_mock.return_value = mock_instance

        create_fn = AsyncMock(side_effect=_async_stream_events)
        mock_instance.responses.create = create_fn

        llm = _make_responses_llm()
        message = Message(role="user", chunks=[TextChunk(content="hello")])
        chat_gen = await llm.achat([message], stream=True)
        chat_responses = [item async for item in chat_gen]
        assert len(chat_responses) > 0, "Expected at least one streaming response"
        final_content = chat_responses[-1].message.content
        assert (
            "Azure" in final_content
        ), f"Expected 'Azure' in final content, got '{final_content}'"
