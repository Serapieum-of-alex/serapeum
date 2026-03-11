from __future__ import annotations

from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import AsyncAzureOpenAI
from openai import AzureOpenAI as SyncAzureOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion import CompletionUsage

from serapeum.azure_openai import Completions
from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.llms import Message, TextChunk


def mock_chat_completion_v1(*args: Any, **kwargs: Any) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-abc123",
        object="chat.completion",
        created=1677858242,
        model="gpt-3.5-turbo-0301",
        usage=CompletionUsage(prompt_tokens=13, completion_tokens=7, total_tokens=20),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant", content="\n\nThis is a test!"
                ),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def mock_chat_completion_stream_with_filter_results(
    *args: Any, **kwargs: Any
) -> Generator[ChatCompletionChunk, None, None]:
    """Azure sends a chunk without text content (empty `choices` attribute) as the first chunk.

    It only contains prompt filter results. Documentation on this can be found here:
    https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
    """
    responses = [
        ChatCompletionChunk.model_construct(
            id="",
            object="",
            created=0,
            model="",
            prompt_filter_results=[
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"},
                    },
                }
            ],
            choices=[],
            usage=None,
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="Hello from\n"),
                    finish_reason=None,
                    index=0,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="Azure"), finish_reason=None, index=0
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
            object="chat.completion.chunk",
            created=1677825464,
            model="gpt-3.5-turbo-0301",
            choices=[ChunkChoice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        ),
    ]
    yield from responses


async def mock_async_chat_completion_stream_with_filter_results(
    *args: Any, **kwargs: Any
) -> AsyncGenerator[ChatCompletionChunk, None]:
    async def gen() -> AsyncGenerator[ChatCompletionChunk, None]:
        for response in mock_chat_completion_stream_with_filter_results(
            *args, **kwargs
        ):
            yield response

    return gen()


@pytest.mark.unit
class TestCompletions:
    """Tests for Completions class."""

    def test_mro_includes_base_llm(self) -> None:
        """Verify Completions inherits from BaseLLM."""
        names_of_base_classes = [b.__name__ for b in Completions.__mro__]
        assert BaseLLM.__name__ in names_of_base_classes

    def test_class_name(self) -> None:
        """Verify class_name() returns the expected string."""
        assert Completions.class_name() == "azure_openai_completions"

    def test_missing_engine_raises(self) -> None:
        """Instantiation without engine raises ValueError."""
        with pytest.raises(ValueError, match="engine"):
            Completions(api_key="k", api_version="2024-02-01")

    def test_missing_api_version_raises(self) -> None:
        """Instantiation without api_version raises ValueError."""
        with pytest.raises(ValueError, match="OPENAI_API_VERSION"):
            Completions(engine="dep", api_key="k")

    def test_default_openai_base_without_azure_endpoint_raises(self) -> None:
        """Using default OpenAI base URL without azure_endpoint raises."""
        with pytest.raises(ValueError, match="OPENAI_API_BASE"):
            Completions(
                engine="dep",
                api_key="k",
                api_version="2024-02-01",
                api_base="https://api.openai.com/v1",
            )

    def test_engine_alias_deployment_name(self) -> None:
        """Engine can be set via deployment_name alias."""
        llm = Completions(
            deployment_name="my-dep",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "my-dep"

    def test_engine_alias_deployment_id(self) -> None:
        """Engine can be set via deployment_id alias."""
        llm = Completions(
            deployment_id="my-dep-id",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "my-dep-id"

    def test_engine_alias_deployment(self) -> None:
        """Engine can be set via deployment alias."""
        llm = Completions(
            deployment="my-dep",
            api_key="k",
            api_version="2024-02-01",
        )
        assert llm.engine == "my-dep"

    def test_api_base_reset_when_azure_endpoint_set(self) -> None:
        """api_base is reset to None when azure_endpoint is provided."""
        llm = Completions(
            engine="dep",
            api_key="k",
            api_version="2024-02-01",
            azure_endpoint="https://myres.openai.azure.com/",
        )
        assert llm.api_base is None
        assert llm.azure_endpoint == "https://myres.openai.azure.com/"

    def test_default_model(self) -> None:
        """Default model is gpt-35-turbo."""
        llm = Completions(
            engine="dep", api_key="k", api_version="2024-02-01"
        )
        assert llm.model == "gpt-35-turbo"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_get_model_kwargs_uses_engine(
        self, sync_mock: MagicMock
    ) -> None:
        """_get_model_kwargs substitutes engine for model."""
        llm = Completions(
            engine="my-deployment",
            api_key="k",
            api_version="2024-02-01",
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["model"] == "my-deployment"


@pytest.mark.unit
class TestCompletionsCredentials:
    """Tests for _resolve_api_key and _get_credential_kwargs."""

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_api_key_from_env(
        self, sync_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API key is resolved from AZURE_OPENAI_API_KEY env var."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        llm = Completions(
            engine="dep", api_version="2024-02-01"
        )
        key = llm._resolve_api_key()
        assert key == "env-key"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_api_key_missing_raises(
        self, sync_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing API key raises ValueError."""
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = Completions(
            engine="dep", api_version="2024-02-01"
        )
        # Clear any key resolved during init
        llm.api_key = None
        with pytest.raises(ValueError, match="api_key"):
            llm._resolve_api_key()

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_azure_ad_token_provider_sets_key(
        self, sync_mock: MagicMock
    ) -> None:
        """Azure AD token provider callback is invoked to set api_key."""
        llm = Completions(
            engine="dep",
            api_version="2024-02-01",
            use_azure_ad=True,
            azure_ad_token_provider=lambda: "ad-token",
        )
        key = llm._resolve_api_key()
        assert key == "ad-token"
        assert llm.api_key == "ad-token"

    @patch("serapeum.azure_openai.llm.refresh_openai_azuread_token")
    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_azure_ad_refresh_fallback(
        self, sync_mock: MagicMock, refresh_mock: MagicMock
    ) -> None:
        """Without token provider, falls back to refresh_openai_azuread_token."""
        mock_token = MagicMock()
        mock_token.token = "refreshed-token"
        refresh_mock.return_value = mock_token

        llm = Completions(
            engine="dep",
            api_version="2024-02-01",
            use_azure_ad=True,
        )
        key = llm._resolve_api_key()
        assert key == "refreshed-token"
        refresh_mock.assert_called_once()

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_get_credential_kwargs_contains_azure_fields(
        self, sync_mock: MagicMock
    ) -> None:
        """_get_credential_kwargs includes Azure-specific fields."""
        llm = Completions(
            engine="dep",
            api_key="k",
            api_version="2024-02-01",
            azure_endpoint="https://myres.openai.azure.com/",
            azure_deployment="dep",
        )
        kwargs = llm._get_credential_kwargs()
        assert kwargs["azure_endpoint"] == "https://myres.openai.azure.com/"
        assert kwargs["azure_deployment"] == "dep"
        assert kwargs["api_version"] == "2024-02-01"
        assert kwargs["api_key"] == "k"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_client_property_returns_azure_client(
        self, sync_mock: MagicMock
    ) -> None:
        """client property creates a SyncAzureOpenAI instance."""
        llm = Completions(
            engine="dep", api_key="k", api_version="2024-02-01"
        )
        _ = llm.client
        sync_mock.assert_called_once()


@pytest.mark.mock
class TestCompletionsWithMocks:
    """Tests for Completions that require mocking the Azure SDK clients."""

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_custom_http_client(self, sync_azure_openai_mock: MagicMock) -> None:
        """Verify that a custom http_client is passed to the Azure SDK client."""
        custom_http_client = httpx.Client()
        mock_instance = sync_azure_openai_mock.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()
        azure_openai = Completions(
            engine="foo bar",
            http_client=custom_http_client,
            api_key="mock",
            api_version="2024-02-01",
        )
        azure_openai.complete("test prompt")
        sync_azure_openai_mock.assert_called()
        kwargs = sync_azure_openai_mock.call_args.kwargs
        assert "http_client" in kwargs
        assert kwargs["http_client"] == custom_http_client

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_custom_azure_ad_token_provider(
        self, sync_azure_openai_mock: MagicMock
    ) -> None:
        """Verify that a custom azure ad token provider is used for authentication."""

        def custom_azure_ad_token_provider() -> str:
            return "mock_api_key"

        mock_instance = sync_azure_openai_mock.return_value
        mock_instance.chat.completions.create.return_value = mock_chat_completion_v1()
        azure_openai = Completions(
            engine="foo bar",
            use_azure_ad=True,
            azure_ad_token_provider=custom_azure_ad_token_provider,
            api_version="2024-02-01",
        )
        azure_openai.complete("test prompt")
        assert azure_openai.api_key == "mock_api_key"

    @patch("serapeum.azure_openai.llm.SyncAzureOpenAI")
    def test_chat_completion_with_filter_results(
        self, sync_azure_openai_mock: MagicMock
    ) -> None:
        """Test sync chat completions work when first chunk has prompt filter results."""
        mock_instance = MagicMock(spec=SyncAzureOpenAI)
        sync_azure_openai_mock.return_value = mock_instance

        chat_mock = MagicMock()
        chat_mock.completions.create.return_value = (
            mock_chat_completion_stream_with_filter_results()
        )
        mock_instance.chat = chat_mock

        llm = Completions(engine="foo bar", api_key="mock", api_version="2024-02-01")
        prompt = "test prompt"
        message = Message(role="user", chunks=[TextChunk(content="test message")])

        response_gen = llm.complete(prompt, stream=True)
        responses = list(response_gen)
        assert responses[-1].text == "Hello from\nAzure"

        mock_instance.chat.completions.create.return_value = (
            mock_chat_completion_stream_with_filter_results()
        )
        chat_response_gen = llm.chat([message], stream=True)
        chat_responses = list(chat_response_gen)
        assert chat_responses[-1].message.content == "Hello from\nAzure"
        assert chat_responses[-1].message.role == "assistant"

    @pytest.mark.asyncio
    @patch("serapeum.azure_openai.llm.AsyncAzureOpenAI")
    async def test_async_chat_completion_with_filter_results(
        self,
        async_azure_openai_mock: MagicMock,
    ) -> None:
        """Test async chat completions work when first chunk has prompt filter results."""
        mock_instance = MagicMock(spec=AsyncAzureOpenAI)
        async_azure_openai_mock.return_value = mock_instance
        create_fn = AsyncMock()
        create_fn.side_effect = mock_async_chat_completion_stream_with_filter_results
        chat_mock = MagicMock()
        chat_mock.completions.create = create_fn
        mock_instance.chat = chat_mock

        llm = Completions(engine="foo bar", api_key="mock", api_version="2024-02-01")
        prompt = "test prompt"
        message = Message(role="user", chunks=[TextChunk(content="test message")])

        response_gen = await llm.acomplete(prompt, stream=True)
        responses = [item async for item in response_gen]
        assert responses[-1].text == "Hello from\nAzure"

        chat_response_gen = await llm.achat([message], stream=True)
        chat_responses = [item async for item in chat_response_gen]
        assert chat_responses[-1].message.content == "Hello from\nAzure"
