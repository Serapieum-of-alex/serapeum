"""Comprehensive unit tests for the OpenAI (Chat Completions) provider class.

Targets ``serapeum.openai.llm.chat_completions.OpenAI`` and covers model
validation, routing logic, ``_get_model_kwargs``, response token extraction,
max-token inference, tool handling, metadata properties, and async methods.
"""

from __future__ import annotations

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion import Completion, CompletionChoice, CompletionUsage

from serapeum.core.base.llms.types import (
    ChatResponse,
    Message,
    MessageRole,
    TextChunk,
    ToolCallBlock,
)
from serapeum.core.tools import CallableTool
from serapeum.openai import OpenAI
from serapeum.openai.data.models import O1_MODELS, RESPONSES_API_ONLY_MODELS


@pytest.fixture()
def llm() -> OpenAI:
    """Create a default OpenAI instance for unit tests.

    Returns:
        OpenAI: Instance configured with a fake API key and gpt-4o-mini model.
    """
    return OpenAI(model="gpt-4o-mini", api_key="sk-test-key")


@pytest.fixture()
def completion_llm() -> OpenAI:
    """Create an OpenAI instance configured for legacy completion models.

    Returns:
        OpenAI: Instance using text-davinci-003 with a fake API key.
    """
    return OpenAI(model="text-davinci-003", api_key="sk-test-key")


@pytest.fixture()
def o1_llm() -> OpenAI:
    """Create an OpenAI instance configured for an O1 reasoning model.

    Returns:
        OpenAI: Instance using o1-mini with a fake API key.
    """
    return OpenAI(model="o1-mini", api_key="sk-test-key")


def _make_search_tool() -> CallableTool:
    """Create a simple search tool for testing."""
    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"

    return CallableTool.from_function(
        func=search, name="search_tool", description="Search tool"
    )


def _make_chat_completion(
    content: str = "Hello!", role: str = "assistant"
) -> ChatCompletion:
    """Build a mock ChatCompletion object."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        usage=CompletionUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
        choices=[
            Choice(
                message=ChatCompletionMessage(role=role, content=content),
                finish_reason="stop",
                index=0,
            )
        ],
    )


def _make_completion(text: str = "Hello!") -> Completion:
    """Build a mock Completion object."""
    return Completion(
        id="cmpl-test",
        object="text_completion",
        created=1700000000,
        model="text-davinci-003",
        usage=CompletionUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
        choices=[
            CompletionChoice(
                text=text, finish_reason="stop", index=0
            )
        ],
    )


def _make_stream_chunks() -> Generator[ChatCompletionChunk, None, None]:
    """Build a sequence of mock streaming chunks."""
    chunks = [
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant", content="Hel"),
                    finish_reason=None,
                    index=0,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content="lo!"),
                    finish_reason=None,
                    index=0,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                    index=0,
                )
            ],
        ),
    ]
    yield from chunks


def _make_tool_call_stream_chunks() -> Generator[ChatCompletionChunk, None, None]:
    """Build streaming chunks that contain tool calls."""
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    chunks = [
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_123",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="search_tool",
                                    arguments='{"query":',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments=' "test"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                    index=0,
                )
            ],
        ),
    ]
    yield from chunks


def _make_chat_completion_with_tool_calls(
    tool_name: str = "search_tool",
    arguments: str = '{"query": "test"}',
    tool_type: str = "function",
) -> ChatCompletion:
    """Build a mock ChatCompletion with tool calls."""
    return ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        usage=CompletionUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type=tool_type,
                            function=Function(
                                name=tool_name, arguments=arguments
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
                index=0,
            )
        ],
    )


@pytest.mark.unit
class TestValidateModel:
    """Tests for OpenAI._validate_model."""

    def test_o1_model_forces_temperature(self):
        """Test that O1 models force temperature to 1.0.

        Test scenario:
            Creating an OpenAI instance with an O1 model and custom temperature
            should override the temperature to 1.0.
        """
        llm = OpenAI(model="o1-mini", api_key="sk-test", temperature=0.5)
        assert llm.temperature == 1.0, (
            f"O1 model should force temperature to 1.0, got {llm.temperature}"
        )

    def test_non_o1_model_preserves_temperature(self):
        """Test that non-O1 models preserve custom temperature.

        Test scenario:
            Creating an OpenAI instance with a non-O1 model and custom
            temperature should keep the custom value.
        """
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test", temperature=0.7)
        assert llm.temperature == 0.7, (
            f"Non-O1 model should preserve temperature, got {llm.temperature}"
        )

    def test_responses_only_model_raises_valueerror(self):
        """Test that models exclusive to the Responses API are rejected.

        Test scenario:
            Creating an OpenAI instance with a Responses API-only model
            should raise ValueError directing the user to OpenAIResponses.
        """
        responses_only = next(iter(RESPONSES_API_ONLY_MODELS))
        with pytest.raises(ValueError, match="Responses API") as exc_info:
            OpenAI(model=responses_only, api_key="sk-test")
        assert responses_only in str(exc_info.value), (
            f"Error should mention the model name, got: {exc_info.value}"
        )


@pytest.mark.unit
class TestClassName:
    """Tests for OpenAI.class_name."""

    def test_returns_openai(self):
        """Test that class_name returns 'openai'.

        Test scenario:
            Calling class_name() should return the canonical provider string.
        """
        assert OpenAI.class_name() == "openai", (
            f"Expected 'openai', got '{OpenAI.class_name()}'"
        )


@pytest.mark.unit
class TestMetadata:
    """Tests for the OpenAI.metadata property."""

    def test_chat_model_metadata(self, llm: OpenAI):
        """Test metadata for a standard chat model.

        Test scenario:
            gpt-4o-mini should be identified as a chat model with function calling
            support and the correct context window.
        """
        meta = llm.metadata
        assert meta.model_name == "gpt-4o-mini", (
            f"Expected model_name 'gpt-4o-mini', got '{meta.model_name}'"
        )
        assert meta.is_chat_model is True, "gpt-4o-mini should be a chat model"
        assert meta.is_function_calling_model is True, (
            "gpt-4o-mini should support function calling"
        )
        assert meta.context_window > 0, "Context window should be positive"
        assert meta.system_role == MessageRole.SYSTEM, (
            "Non-O1 models should use SYSTEM role"
        )

    def test_o1_model_metadata_system_role(self, o1_llm: OpenAI):
        """Test that O1 models report system_role as USER.

        Test scenario:
            O1 models do not accept dedicated system messages, so system_role
            should be set to MessageRole.USER.
        """
        meta = o1_llm.metadata
        assert meta.system_role == MessageRole.USER, (
            f"O1 models should use USER system_role, got {meta.system_role}"
        )

    def test_metadata_num_output_with_max_tokens(self):
        """Test that num_output reflects max_tokens when set.

        Test scenario:
            Setting max_tokens=100 should be reflected in metadata.num_output.
        """
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test", max_tokens=100)
        assert llm.metadata.num_output == 100, (
            f"Expected num_output=100, got {llm.metadata.num_output}"
        )

    def test_metadata_num_output_default(self, llm: OpenAI):
        """Test that num_output is -1 when max_tokens is None.

        Test scenario:
            When max_tokens is not set, num_output defaults to -1.
        """
        assert llm.metadata.num_output == -1, (
            f"Expected num_output=-1, got {llm.metadata.num_output}"
        )

    def test_completion_model_metadata(self, completion_llm: OpenAI):
        """Test metadata for a legacy completion model.

        Test scenario:
            text-davinci-003 should not be identified as a chat model.
        """
        meta = completion_llm.metadata
        assert meta.is_chat_model is False, (
            "text-davinci-003 should not be a chat model"
        )


@pytest.mark.unit
class TestUseChatCompletions:
    """Tests for OpenAI._use_chat_completions."""

    def test_override_true_in_kwargs(self, llm: OpenAI):
        """Test explicit use_chat_completions=True override.

        Test scenario:
            Passing use_chat_completions=True should force chat completions
            even if the model metadata says otherwise.
        """
        result = llm._use_chat_completions({"use_chat_completions": True})
        assert result is True, "Explicit True override should return True"

    def test_override_false_in_kwargs(self, llm: OpenAI):
        """Test explicit use_chat_completions=False override.

        Test scenario:
            Passing use_chat_completions=False should force legacy completions
            even for a chat model.
        """
        result = llm._use_chat_completions({"use_chat_completions": False})
        assert result is False, "Explicit False override should return False"

    def test_fallback_to_metadata_chat_model(self, llm: OpenAI):
        """Test that a chat model defaults to chat completions.

        Test scenario:
            Without the override key, gpt-4o-mini should use chat completions.
        """
        result = llm._use_chat_completions({})
        assert result is True, "Chat model should default to chat completions"

    def test_fallback_to_metadata_completion_model(self, completion_llm: OpenAI):
        """Test that a completion model defaults to legacy completions.

        Test scenario:
            Without the override key, text-davinci-003 should use legacy completions.
        """
        result = completion_llm._use_chat_completions({})
        assert result is False, "Completion model should not use chat completions"


@pytest.mark.unit
class TestGetModelKwargs:
    """Tests for OpenAI._get_model_kwargs."""

    def test_basic_kwargs(self, llm: OpenAI):
        """Test basic kwargs include model and temperature.

        Test scenario:
            Default kwargs should contain model and temperature keys.
        """
        kwargs = llm._get_model_kwargs()
        assert kwargs["model"] == "gpt-4o-mini", (
            f"Expected model 'gpt-4o-mini', got {kwargs['model']}"
        )
        assert "temperature" in kwargs, "temperature should be present"

    def test_max_tokens_included_when_set(self):
        """Test that max_tokens is included when explicitly set.

        Test scenario:
            Setting max_tokens=256 should include it in the output kwargs.
        """
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test", max_tokens=256)
        kwargs = llm._get_model_kwargs()
        assert kwargs["max_tokens"] == 256, (
            f"Expected max_tokens=256, got {kwargs.get('max_tokens')}"
        )

    def test_max_tokens_omitted_when_none(self, llm: OpenAI):
        """Test that max_tokens is omitted when None.

        Test scenario:
            When max_tokens is None (default), it should not appear in kwargs.
        """
        kwargs = llm._get_model_kwargs()
        assert "max_tokens" not in kwargs, (
            "max_tokens should not be in kwargs when None"
        )

    def test_logprobs_for_chat_model(self):
        """Test logprobs injection for chat models.

        Test scenario:
            Setting logprobs=True and top_logprobs=5 on a chat model should
            inject both keys.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            logprobs=True, top_logprobs=5
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["logprobs"] is True, "logprobs should be True"
        assert kwargs["top_logprobs"] == 5, (
            f"Expected top_logprobs=5, got {kwargs['top_logprobs']}"
        )

    def test_logprobs_for_completion_model(self):
        """Test logprobs injection for legacy completion models.

        Test scenario:
            For non-chat models, logprobs should be set to the integer
            top_logprobs value (not a boolean).
        """
        llm = OpenAI(
            model="text-davinci-003", api_key="sk-test",
            logprobs=True, top_logprobs=3
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["logprobs"] == 3, (
            f"For completion models logprobs should be int, got {kwargs['logprobs']}"
        )

    def test_logprobs_not_set_when_false(self, llm: OpenAI):
        """Test that logprobs is not injected when set to False.

        Test scenario:
            logprobs=False (or None default) should not add logprobs to kwargs.
        """
        kwargs = llm._get_model_kwargs()
        assert "logprobs" not in kwargs, (
            "logprobs should not be in kwargs when not enabled"
        )

    def test_stream_options_removed_when_not_streaming(self):
        """Test that stream_options is removed when stream is not in kwargs.

        Test scenario:
            If additional_kwargs contains stream_options but stream is not in
            the final kwargs, stream_options should be removed.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            additional_kwargs={"stream_options": {"include_usage": True}},
        )
        kwargs = llm._get_model_kwargs()
        assert "stream_options" not in kwargs, (
            "stream_options should be removed when not streaming"
        )

    def test_stream_options_preserved_when_streaming(self):
        """Test that stream_options is preserved when stream is in kwargs.

        Test scenario:
            When stream=True is included in kwargs, stream_options should remain.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            additional_kwargs={"stream_options": {"include_usage": True}},
        )
        kwargs = llm._get_model_kwargs(stream=True)
        assert "stream_options" in kwargs, (
            "stream_options should be preserved when streaming"
        )

    def test_o1_max_completion_tokens_rename(self, o1_llm: OpenAI):
        """Test that O1 models rename max_tokens to max_completion_tokens.

        Test scenario:
            Setting max_tokens on an O1 model should produce
            max_completion_tokens in the output kwargs.
        """
        o1_llm.max_tokens = 100
        kwargs = o1_llm._get_model_kwargs()
        assert "max_completion_tokens" in kwargs, (
            "O1 model should have max_completion_tokens"
        )
        assert kwargs["max_completion_tokens"] == 100, (
            f"Expected 100, got {kwargs['max_completion_tokens']}"
        )
        assert "max_tokens" not in kwargs, (
            "max_tokens should be removed for O1 models"
        )

    def test_o1_reasoning_effort_injected(self):
        """Test that reasoning_effort is injected for O1 models.

        Test scenario:
            Setting reasoning_effort on an O1 model should include it in kwargs.
        """
        llm = OpenAI(
            model="o1-mini", api_key="sk-test", reasoning_effort="high"
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["reasoning_effort"] == "high", (
            f"Expected reasoning_effort='high', got {kwargs.get('reasoning_effort')}"
        )

    def test_modalities_injection(self):
        """Test that modalities are injected into kwargs.

        Test scenario:
            Setting modalities=["text", "audio"] should include it in kwargs.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            modalities=["text", "audio"]
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["modalities"] == ["text", "audio"], (
            f"Expected modalities list, got {kwargs.get('modalities')}"
        )

    def test_audio_config_injection(self):
        """Test that audio_config is injected as 'audio' key.

        Test scenario:
            Setting audio_config should add an 'audio' key to kwargs.
        """
        audio_cfg = {"voice": "alloy", "format": "wav"}
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            audio_config=audio_cfg
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["audio"] == audio_cfg, (
            f"Expected audio config, got {kwargs.get('audio')}"
        )

    def test_additional_kwargs_merged(self):
        """Test that additional_kwargs are merged into the output.

        Test scenario:
            additional_kwargs like frequency_penalty should appear in the
            final kwargs dict.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            additional_kwargs={"frequency_penalty": 0.5}
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["frequency_penalty"] == 0.5, (
            f"Expected frequency_penalty=0.5, got {kwargs.get('frequency_penalty')}"
        )

    def test_runtime_kwargs_override(self, llm: OpenAI):
        """Test that per-call runtime kwargs override defaults.

        Test scenario:
            Passing temperature=0.9 at call time should override the
            instance-level temperature.
        """
        kwargs = llm._get_model_kwargs(temperature=0.9)
        assert kwargs["temperature"] == 0.9, (
            f"Expected runtime override temperature=0.9, got {kwargs['temperature']}"
        )


@pytest.mark.unit
class TestGetResponseTokenCounts:
    """Tests for OpenAI._get_response_token_counts."""

    def test_with_usage_attribute(self):
        """Test extraction from an object with a .usage attribute.

        Test scenario:
            SDK response objects have .usage with prompt_tokens, etc.
        """
        response = _make_chat_completion()
        counts = OpenAI._get_response_token_counts(response)
        assert counts["prompt_tokens"] == 10, (
            f"Expected prompt_tokens=10, got {counts.get('prompt_tokens')}"
        )
        assert counts["completion_tokens"] == 5, (
            f"Expected completion_tokens=5, got {counts.get('completion_tokens')}"
        )
        assert counts["total_tokens"] == 15, (
            f"Expected total_tokens=15, got {counts.get('total_tokens')}"
        )

    def test_with_dict_input(self):
        """Test extraction from a dict response.

        Test scenario:
            Legacy dict responses should have their usage field extracted.
        """
        response = {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            }
        }
        counts = OpenAI._get_response_token_counts(response)
        assert counts["total_tokens"] == 30, (
            f"Expected total_tokens=30, got {counts.get('total_tokens')}"
        )

    def test_with_none_usage_in_dict(self):
        """Test that None usage in dict returns empty dict.

        Test scenario:
            Some providers may return usage=None in the dict.
        """
        response = {"usage": None}
        counts = OpenAI._get_response_token_counts(response)
        assert counts == {}, f"Expected empty dict, got {counts}"

    def test_with_missing_usage_attribute(self):
        """Test that objects without usage return empty dict.

        Test scenario:
            Streaming chunks typically lack .usage, should return {}.
        """
        mock_response = MagicMock(spec=[])
        counts = OpenAI._get_response_token_counts(mock_response)
        assert counts == {}, f"Expected empty dict, got {counts}"

    def test_with_usage_attribute_error(self):
        """Test graceful handling when usage attribute access fails.

        Test scenario:
            If .usage exists but sub-attributes raise AttributeError, should
            return empty dict.
        """

        class BrokenUsage:
            @property
            def prompt_tokens(self):
                raise AttributeError("no prompt_tokens")

        class BrokenResponse:
            usage = BrokenUsage()

        counts = OpenAI._get_response_token_counts(BrokenResponse())
        assert counts == {}, f"Expected empty dict on AttributeError, got {counts}"

    def test_with_dict_missing_usage_key(self):
        """Test dict with no usage key returns zeroed counts.

        Test scenario:
            A dict without a 'usage' key defaults to empty usage dict,
            which returns zeroed token counts.
        """
        counts = OpenAI._get_response_token_counts({})
        assert counts == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }, f"Expected zeroed counts, got {counts}"

    def test_with_non_dict_non_object(self):
        """Test that a plain string returns empty dict.

        Test scenario:
            Completely unexpected input type should return {}.
        """
        counts = OpenAI._get_response_token_counts("unexpected")
        assert counts == {}, f"Expected empty dict, got {counts}"


@pytest.mark.unit
class TestUpdateMaxTokens:
    """Tests for OpenAI._update_max_tokens."""

    def test_noop_when_max_tokens_set(self, completion_llm: OpenAI):
        """Test that _update_max_tokens is a no-op when max_tokens is already set.

        Test scenario:
            When max_tokens is explicitly configured, the method should not
            modify the kwargs dict.
        """
        completion_llm.max_tokens = 100
        kwargs: dict[str, Any] = {}
        completion_llm._update_max_tokens(kwargs, "short prompt")
        assert "max_tokens" not in kwargs, (
            "Should not inject max_tokens when already set on instance"
        )

    def test_calculates_from_context_window(self, completion_llm: OpenAI):
        """Test max_tokens calculation from context window minus prompt tokens.

        Test scenario:
            When max_tokens is None and a tokenizer is available, the method
            should calculate remaining tokens from the context window.
        """
        kwargs: dict[str, Any] = {}
        completion_llm._update_max_tokens(kwargs, "hello world")
        assert "max_tokens" in kwargs, "Should inject max_tokens"
        assert kwargs["max_tokens"] > 0, (
            f"Calculated max_tokens should be positive, got {kwargs['max_tokens']}"
        )

    def test_prompt_exceeds_context_raises(self, completion_llm: OpenAI):
        """Test that a prompt exceeding the context window raises ValueError.

        Test scenario:
            A prompt with more tokens than the context window should raise
            ValueError.
        """
        huge_prompt = "word " * 100_000
        kwargs: dict[str, Any] = {}
        with pytest.raises(ValueError, match="too long"):
            completion_llm._update_max_tokens(kwargs, huge_prompt)


@pytest.mark.unit
class TestCompleteAudioRejection:
    """Tests for audio modality rejection in complete/acomplete."""

    def test_complete_rejects_audio(self):
        """Test that complete raises ValueError with audio modality.

        Test scenario:
            Calling complete() with modalities=["text","audio"] should
            raise ValueError.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            modalities=["text", "audio"]
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            llm.complete("test prompt")

    @pytest.mark.asyncio()
    async def test_acomplete_rejects_audio(self):
        """Test that acomplete raises ValueError with audio modality.

        Test scenario:
            Calling acomplete() with modalities=["text","audio"] should
            raise ValueError.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            modalities=["text", "audio"]
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            await llm.acomplete("test prompt")

    @pytest.mark.mock
    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_stream_chat_rejects_audio(self, MockSyncOpenAI: MagicMock):
        """Test that streaming chat raises ValueError with audio modality.

        Test scenario:
            Calling chat(stream=True) with audio modality should raise ValueError.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            modalities=["text", "audio"]
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            list(llm._stream_chat([
                Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
            ]))


@pytest.mark.unit
class TestChatRouting:
    """Tests for OpenAI.chat routing between chat completions and legacy."""

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_chat_routes_to_chat_completions(self, MockSyncOpenAI: MagicMock):
        """Test that chat() routes to _chat for chat models (non-streaming).

        Test scenario:
            gpt-4o-mini is a chat model, so chat() should call
            client.chat.completions.create.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_chat_completion()

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        response = llm.chat([Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])])

        mock_client.chat.completions.create.assert_called_once()
        assert response.message.content == "Hello!", (
            f"Expected 'Hello!', got '{response.message.content}'"
        )

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_chat_routes_to_completions(self, MockSyncOpenAI: MagicMock):
        """Test that chat() routes to legacy completions for non-chat models.

        Test scenario:
            text-davinci-003 is not a chat model, so chat() should go through
            the completion-to-chat adapter and call client.completions.create.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.completions.create.return_value = _make_completion("Hi!")

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        response = llm.chat([Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])])

        mock_client.completions.create.assert_called_once()
        assert response.message.content == "Hi!", (
            f"Expected 'Hi!', got '{response.message.content}'"
        )

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_chat_streaming_routes_to_stream_chat(
        self, MockSyncOpenAI: MagicMock
    ):
        """Test that chat(stream=True) routes to _stream_chat for chat models.

        Test scenario:
            gpt-4o-mini with stream=True should use streaming chat completions.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = _make_stream_chunks()

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        gen = llm.chat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])], stream=True
        )
        chunks = list(gen)

        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        final = chunks[-1]
        assert final.message.chunks[0].content == "Hello!", (
            f"Expected accumulated content 'Hello!', "
            f"got '{final.message.chunks[0].content}'"
        )


@pytest.mark.mock
class TestStreamChat:
    """Tests for OpenAI._stream_chat."""

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_stream_accumulates_tool_calls(self, MockSyncOpenAI: MagicMock):
        """Test that streaming chat accumulates tool call fragments.

        Test scenario:
            Tool call arguments split across chunks should be merged into
            a complete tool call in additional_kwargs.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = (
            _make_tool_call_stream_chunks()
        )

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        gen = llm._stream_chat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="search for test")])]
        )
        chunks = list(gen)

        # The first two chunks should have tool calls in additional_kwargs
        tool_chunks = [
            c for c in chunks
            if c.message.additional_kwargs.get("tool_calls")
        ]
        assert len(tool_chunks) >= 1, (
            "Should have at least one chunk with tool_calls"
        )

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_stream_empty_choices_handled(self, MockSyncOpenAI: MagicMock):
        """Test that streaming handles chunks with empty choices.

        Test scenario:
            A chunk with an empty choices list should use a default ChoiceDelta
            and not crash.
        """
        empty_chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[],
        )

        def gen_with_empty():
            yield empty_chunk
            yield from _make_stream_chunks()

        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = gen_with_empty()

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        chunks = list(
            llm._stream_chat(
                [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
            )
        )
        assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"


@pytest.mark.unit
class TestValidateChatWithToolsResponse:
    """Tests for OpenAI._validate_chat_with_tools_response."""

    def test_allows_parallel_calls(self, llm: OpenAI):
        """Test that parallel tool calls are preserved when allowed.

        Test scenario:
            With allow_parallel_tool_calls=True, a response with two tool
            calls should keep both.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="1", tool_name="a", tool_kwargs="{}"
                    ),
                    ToolCallBlock(
                        tool_call_id="2", tool_name="b", tool_kwargs="{}"
                    ),
                ],
            )
        )
        result = llm._validate_chat_with_tools_response(
            response, tools=[], allow_parallel_tool_calls=True
        )
        tool_blocks = [
            b for b in result.message.chunks if isinstance(b, ToolCallBlock)
        ]
        assert len(tool_blocks) == 2, (
            f"Expected 2 tool calls when parallel allowed, got {len(tool_blocks)}"
        )

    def test_forces_single_call_when_disallowed(self, llm: OpenAI):
        """Test that parallel tool calls are trimmed when disallowed.

        Test scenario:
            With allow_parallel_tool_calls=False, only the first tool call
            should remain.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="1", tool_name="a", tool_kwargs="{}"
                    ),
                    ToolCallBlock(
                        tool_call_id="2", tool_name="b", tool_kwargs="{}"
                    ),
                ],
            )
        )
        result = llm._validate_chat_with_tools_response(
            response, tools=[], allow_parallel_tool_calls=False
        )
        tool_blocks = [
            b for b in result.message.chunks if isinstance(b, ToolCallBlock)
        ]
        assert len(tool_blocks) == 1, (
            f"Expected 1 tool call when parallel disallowed, got {len(tool_blocks)}"
        )
        assert tool_blocks[0].tool_call_id == "1", (
            "Should keep the first tool call"
        )


@pytest.mark.unit
class TestGetToolCallsFromResponse:
    """Tests for OpenAI.get_tool_calls_from_response."""

    def test_extracts_from_tool_call_blocks(self, llm: OpenAI):
        """Test extraction from ToolCallBlock chunks (modern path).

        Test scenario:
            Response with ToolCallBlock chunks should extract tool calls
            correctly, parsing JSON arguments.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="call_1",
                        tool_name="search",
                        tool_kwargs='{"query": "hello"}',
                    )
                ],
            )
        )
        tool_calls = llm.get_tool_calls_from_response(response)
        assert len(tool_calls) == 1, (
            f"Expected 1 tool call, got {len(tool_calls)}"
        )
        assert tool_calls[0].tool_name == "search", (
            f"Expected tool name 'search', got '{tool_calls[0].tool_name}'"
        )
        assert tool_calls[0].tool_kwargs == {"query": "hello"}, (
            f"Expected parsed dict kwargs, got {tool_calls[0].tool_kwargs}"
        )

    def test_extracts_dict_kwargs_from_blocks(self, llm: OpenAI):
        """Test extraction when ToolCallBlock has dict kwargs (already parsed).

        Test scenario:
            If tool_kwargs is already a dict, it should be used directly.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="call_1",
                        tool_name="search",
                        tool_kwargs={"query": "hello"},
                    )
                ],
            )
        )
        tool_calls = llm.get_tool_calls_from_response(response)
        assert tool_calls[0].tool_kwargs == {"query": "hello"}, (
            f"Expected dict kwargs, got {tool_calls[0].tool_kwargs}"
        )

    def test_handles_invalid_json_in_blocks(self, llm: OpenAI):
        """Test that invalid JSON in ToolCallBlock falls back to empty dict.

        Test scenario:
            Malformed JSON in tool_kwargs should result in an empty dict.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="call_1",
                        tool_name="search",
                        tool_kwargs="INVALID JSON",
                    )
                ],
            )
        )
        tool_calls = llm.get_tool_calls_from_response(response)
        assert tool_calls[0].tool_kwargs == {}, (
            f"Expected empty dict for invalid JSON, got {tool_calls[0].tool_kwargs}"
        )

    def test_legacy_path_extracts_from_additional_kwargs(self, llm: OpenAI):
        """Test extraction via legacy additional_kwargs path.

        Test scenario:
            When response has no ToolCallBlock chunks but has tool_calls in
            additional_kwargs, the legacy extraction path should work.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content=None)],
                additional_kwargs={
                    "tool_calls": [
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="search",
                                arguments='{"query": "test"}',
                            ),
                        )
                    ]
                },
            )
        )
        tool_calls = llm.get_tool_calls_from_response(response)
        assert len(tool_calls) == 1, f"Expected 1 tool call, got {len(tool_calls)}"
        assert tool_calls[0].tool_name == "search", (
            f"Expected 'search', got '{tool_calls[0].tool_name}'"
        )
        assert tool_calls[0].tool_kwargs == {"query": "test"}, (
            f"Expected parsed kwargs, got {tool_calls[0].tool_kwargs}"
        )

    def test_legacy_path_invalid_tool_type_raises(self, llm: OpenAI):
        """Test that legacy path raises for non-function tool types.

        Test scenario:
            A tool call with type != "function" should raise ValueError.
        """
        mock_tool_call = MagicMock()
        mock_tool_call.type = "invalid_type"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = "{}"

        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content=None)],
                additional_kwargs={"tool_calls": [mock_tool_call]},
            )
        )
        with pytest.raises(ValueError, match="Invalid tool type"):
            llm.get_tool_calls_from_response(response)

    def test_no_tool_calls_raises_when_required(self, llm: OpenAI):
        """Test that error_on_no_tool_call=True raises when no calls found.

        Test scenario:
            A response with no tool calls should raise ValueError when
            error_on_no_tool_call=True.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT, chunks=[TextChunk(content="Just text.")])
        )
        with pytest.raises(ValueError, match="Expected at least one tool call"):
            llm.get_tool_calls_from_response(
                response, error_on_no_tool_call=True
            )

    def test_no_tool_calls_returns_empty_when_optional(self, llm: OpenAI):
        """Test that error_on_no_tool_call=False returns empty list.

        Test scenario:
            A response with no tool calls should return [] when
            error_on_no_tool_call=False.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT, chunks=[TextChunk(content="Just text.")])
        )
        result = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        assert result == [], f"Expected empty list, got {result}"


@pytest.mark.unit
class TestPrepareChatWithTools:
    """Tests for OpenAI._prepare_chat_with_tools."""

    def test_strict_mode_sets_strict_and_additional_properties(
        self, llm: OpenAI
    ):
        """Test that strict=True injects strict and additionalProperties.

        Test scenario:
            With strict=True, each tool spec should have strict=True and
            additionalProperties=False in the function parameters.
        """
        tool = _make_search_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], user_msg="test", strict=True
        )
        tool_spec = result["tools"][0]
        assert tool_spec["function"]["strict"] is True, (
            "Tool should have strict=True"
        )
        assert (
            tool_spec["function"]["parameters"]["additionalProperties"] is False
        ), "Tool parameters should have additionalProperties=False"

    def test_string_user_msg_wrapped_in_message(self, llm: OpenAI):
        """Test that a string user_msg is converted to a Message object.

        Test scenario:
            Passing a plain string as user_msg should result in a Message
            with role=USER in the messages list.
        """
        tool = _make_search_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], user_msg="hello"
        )
        messages = result["messages"]
        assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
        assert messages[0].role == MessageRole.USER, (
            f"Expected USER role, got {messages[0].role}"
        )
        assert messages[0].content == "hello", (
            f"Expected 'hello', got '{messages[0].content}'"
        )

    def test_message_user_msg_used_directly(self, llm: OpenAI):
        """Test that a Message user_msg is used directly.

        Test scenario:
            Passing a Message object as user_msg should append it as-is.
        """
        tool = _make_search_tool()
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi there")])
        result = llm._prepare_chat_with_tools(
            tools=[tool], user_msg=msg
        )
        assert result["messages"][-1] is msg, "Should use the exact Message object"

    def test_chat_history_preserved(self, llm: OpenAI):
        """Test that chat_history messages are included.

        Test scenario:
            Passing chat_history plus a user_msg should result in the
            history followed by the new message.
        """
        tool = _make_search_tool()
        history = [Message(role=MessageRole.USER, chunks=[TextChunk(content="previous")])]
        result = llm._prepare_chat_with_tools(
            tools=[tool],
            user_msg="current",
            chat_history=history,
        )
        assert len(result["messages"]) == 2, (
            f"Expected 2 messages, got {len(result['messages'])}"
        )

    def test_none_user_msg_does_not_append(self, llm: OpenAI):
        """Test that None user_msg does not add a message.

        Test scenario:
            Passing user_msg=None with empty chat_history should result
            in an empty messages list.
        """
        tool = _make_search_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], user_msg=None
        )
        assert len(result["messages"]) == 0, (
            f"Expected 0 messages, got {len(result['messages'])}"
        )


@pytest.mark.unit
class TestAsyncChat:
    """Tests for OpenAI.achat routing."""

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_achat_routes_to_chat_completions(
        self, MockAsyncOpenAI: MagicMock
    ):
        """Test that achat() routes to async chat completions.

        Test scenario:
            gpt-4o-mini should use the async chat completions endpoint.
        """
        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=_make_chat_completion())
        mock_client.chat.completions.create = create_fn

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        response = await llm.achat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
        )

        create_fn.assert_called_once()
        assert response.message.content == "Hello!", (
            f"Expected 'Hello!', got '{response.message.content}'"
        )

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_achat_routes_to_legacy_completions(
        self, MockAsyncOpenAI: MagicMock
    ):
        """Test that achat() routes to legacy completions for non-chat models.

        Test scenario:
            text-davinci-003 should route through the completion-to-chat adapter.
        """
        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=_make_completion("Hi!"))
        mock_client.completions.create = create_fn

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        response = await llm.achat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
        )

        create_fn.assert_called_once()
        assert response.message.content == "Hi!", (
            f"Expected 'Hi!', got '{response.message.content}'"
        )

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_achat_streaming(self, MockAsyncOpenAI: MagicMock):
        """Test that achat(stream=True) returns an async generator.

        Test scenario:
            Streaming achat should return an async generator that yields
            ChatResponse chunks.
        """
        async def mock_stream():
            for chunk in _make_stream_chunks():
                yield chunk

        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=mock_stream())
        mock_client.chat.completions.create = create_fn

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        gen = await llm.achat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])], stream=True
        )
        chunks = [chunk async for chunk in gen]

        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"


@pytest.mark.unit
class TestAsyncComplete:
    """Tests for OpenAI.acomplete."""

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_acomplete_chat_model(self, MockAsyncOpenAI: MagicMock):
        """Test that acomplete delegates to super().acomplete for chat models.

        Test scenario:
            gpt-4o-mini should use the ChatToCompletion mixin path.
        """
        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=_make_chat_completion("result text"))
        mock_client.chat.completions.create = create_fn

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        response = await llm.acomplete("test prompt")

        assert response.text is not None, "Response text should not be None"

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_acomplete_completion_model(
        self, MockAsyncOpenAI: MagicMock
    ):
        """Test that acomplete calls completions.create for legacy models.

        Test scenario:
            text-davinci-003 should call the completions endpoint directly.
        """
        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=_make_completion("result"))
        mock_client.completions.create = create_fn

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        response = await llm.acomplete("test prompt")

        create_fn.assert_called_once()
        assert response.text == "result", (
            f"Expected 'result', got '{response.text}'"
        )


@pytest.mark.unit
class TestAStreamChatAudioRejection:
    """Tests for audio modality rejection in async streaming."""

    @pytest.mark.asyncio()
    async def test_astream_chat_rejects_audio(self):
        """Test that achat(stream=True) raises ValueError with audio modality.

        Test scenario:
            Calling achat(stream=True) with audio modality should raise
            ValueError when the async generator is iterated.
        """
        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            modalities=["text", "audio"]
        )
        gen = await llm.achat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])], stream=True
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            async for _ in gen:
                pass


@pytest.mark.unit
class TestPydanticFieldValidation:
    """Tests for Pydantic field constraints on the OpenAI model."""

    def test_temperature_too_high(self):
        """Test that temperature > 2.0 is rejected.

        Test scenario:
            Pydantic should raise ValidationError for temperature=3.0.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OpenAI(model="gpt-4o-mini", api_key="sk-test", temperature=3.0)

    def test_temperature_too_low(self):
        """Test that temperature < 0.0 is rejected.

        Test scenario:
            Pydantic should raise ValidationError for temperature=-0.1.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OpenAI(model="gpt-4o-mini", api_key="sk-test", temperature=-0.1)

    def test_top_logprobs_too_high(self):
        """Test that top_logprobs > 20 is rejected.

        Test scenario:
            Pydantic should raise ValidationError for top_logprobs=21.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OpenAI(model="gpt-4o-mini", api_key="sk-test", top_logprobs=21)

    def test_max_tokens_must_be_positive(self):
        """Test that max_tokens=0 is rejected.

        Test scenario:
            Pydantic should raise ValidationError for max_tokens=0.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OpenAI(model="gpt-4o-mini", api_key="sk-test", max_tokens=0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected.

        Test scenario:
            model_config has extra='forbid', so unknown fields should raise.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            OpenAI(
                model="gpt-4o-mini", api_key="sk-test",
                nonexistent_field="value"
            )

    def test_valid_construction(self, llm: OpenAI):
        """Test that valid parameters produce a working instance.

        Test scenario:
            Standard parameters should construct without error.
        """
        assert llm.model == "gpt-4o-mini", (
            f"Expected model 'gpt-4o-mini', got '{llm.model}'"
        )


@pytest.mark.mock
class TestCompleteRouting:
    """Tests for OpenAI.complete routing."""

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_complete_chat_model_delegates_to_super(
        self, MockSyncOpenAI: MagicMock
    ):
        """Test that complete() delegates to super().complete for chat models.

        Test scenario:
            gpt-4o-mini should use the ChatToCompletion mixin, which calls
            chat.completions.create internally.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = (
            _make_chat_completion("Answer")
        )

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        response = llm.complete("What is 1+1?")

        mock_client.chat.completions.create.assert_called_once()
        assert response.text is not None, "Response text should not be None"

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_complete_legacy_model_calls_completions(
        self, MockSyncOpenAI: MagicMock
    ):
        """Test that complete() calls completions.create for legacy models.

        Test scenario:
            text-davinci-003 should call the completions endpoint.
        """
        mock_client = MockSyncOpenAI.return_value
        mock_client.completions.create.return_value = _make_completion("42")

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        response = llm.complete("What is the meaning of life?")

        mock_client.completions.create.assert_called_once()
        assert response.text == "42", (
            f"Expected '42', got '{response.text}'"
        )

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_complete_stream_legacy_model(self, MockSyncOpenAI: MagicMock):
        """Test streaming complete for legacy models.

        Test scenario:
            text-davinci-003 with stream=True should yield CompletionResponse
            chunks from the legacy endpoint.
        """
        def gen_completions():
            yield Completion(
                id="cmpl-test", object="text_completion", created=1700000000,
                model="text-davinci-003",
                choices=[
                    CompletionChoice(text="Hel", finish_reason="length", index=0)
                ],
            )
            yield Completion(
                id="cmpl-test", object="text_completion", created=1700000000,
                model="text-davinci-003",
                choices=[
                    CompletionChoice(text="lo", finish_reason="stop", index=0)
                ],
            )

        mock_client = MockSyncOpenAI.return_value
        mock_client.completions.create.return_value = gen_completions()

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        chunks = list(llm.complete("hi", stream=True))

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert chunks[-1].text == "Hello", (
            f"Expected accumulated text 'Hello', got '{chunks[-1].text}'"
        )


@pytest.mark.unit
class TestModelMetadataHelper:
    """Tests for ModelMetadata._get_model_name inherited by OpenAI."""

    def test_standard_model_name(self, llm: OpenAI):
        """Test that a standard model name is returned as-is.

        Test scenario:
            'gpt-4o-mini' should be returned unchanged.
        """
        assert llm._get_model_name() == "gpt-4o-mini", (
            f"Expected 'gpt-4o-mini', got '{llm._get_model_name()}'"
        )

    def test_legacy_fine_tuning_strips_suffix(self):
        """Test that legacy fine-tuning format strips the suffix.

        Test scenario:
            'curie:ft-acmeco-2021-03-03' should return 'curie'.
        """
        llm = OpenAI(model="curie:ft-acmeco-2021-03-03", api_key="sk-test")
        assert llm._get_model_name() == "curie", (
            f"Expected 'curie', got '{llm._get_model_name()}'"
        )

    def test_new_fine_tuning_extracts_base(self):
        """Test that new fine-tuning format extracts the base model.

        Test scenario:
            'ft:gpt-4o-mini:org:custom:id' should return 'gpt-4o-mini'.
        """
        llm = OpenAI(
            model="ft:gpt-4o-mini:org:custom:id", api_key="sk-test"
        )
        assert llm._get_model_name() == "gpt-4o-mini", (
            f"Expected 'gpt-4o-mini', got '{llm._get_model_name()}'"
        )


@pytest.mark.unit
class TestTokenizer:
    """Tests for ModelMetadata._tokenizer property."""

    def test_tokenizer_returns_encoder(self, llm: OpenAI):
        """Test that _tokenizer returns a tiktoken encoder.

        Test scenario:
            gpt-4o-mini should have a valid tiktoken tokenizer.
        """
        tokenizer = llm._tokenizer
        assert tokenizer is not None, "Tokenizer should not be None"
        tokens = tokenizer.encode("hello world")
        assert len(tokens) > 0, "Tokenizer should encode tokens"


@pytest.mark.mock
class TestChatWithLogprobs:
    """Tests for logprobs parsing in OpenAI._chat."""

    @patch("serapeum.openai.llm.chat_completions.LogProbParser")
    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_chat_with_logprobs(
        self, MockSyncOpenAI: MagicMock, MockLogProbParser: MagicMock
    ):
        """Test that logprobs in the response trigger LogProbParser.

        Test scenario:
            When the model returns logprobs, LogProbParser.from_tokens
            should be called to parse them.
        """
        from openai.types.chat.chat_completion import ChoiceLogprobs
        from openai.types.chat.chat_completion_token_logprob import (
            ChatCompletionTokenLogprob,
            TopLogprob,
        )

        logprob_data = [
            ChatCompletionTokenLogprob(
                token="Hello",
                logprob=-0.5,
                top_logprobs=[
                    TopLogprob(token="Hello", logprob=-0.5, bytes=None)
                ],
                bytes=None,
            ),
        ]
        from serapeum.core.base.llms.types import LogProb

        sentinel = [[LogProb(token="Hello", logprob=-0.5)]]
        MockLogProbParser.from_tokens.return_value = sentinel

        mock_response = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        role="assistant", content="Hello"
                    ),
                    finish_reason="stop",
                    index=0,
                    logprobs=ChoiceLogprobs(
                        content=logprob_data, refusal=None
                    ),
                )
            ],
        )

        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = mock_response

        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            logprobs=True, top_logprobs=1,
        )
        response = llm.chat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
        )

        MockLogProbParser.from_tokens.assert_called_once_with(logprob_data)
        assert response.logprob is not None, (
            "likelihood_score should be set from LogProbParser result"
        )

    @patch("serapeum.openai.llm.chat_completions.LogProbParser")
    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_complete_with_logprobs(
        self, MockSyncOpenAI: MagicMock, MockLogProbParser: MagicMock
    ):
        """Test that logprobs in completion response trigger LogProbParser.

        Test scenario:
            Legacy completion models with logprobs should call
            LogProbParser.from_completions.
        """
        from openai.types.completion_choice import Logprobs

        logprobs_obj = Logprobs(
            tokens=["Hello"],
            token_logprobs=[-0.5],
            top_logprobs=[{"Hello": -0.5}],
            text_offset=[0],
        )
        from serapeum.core.base.llms.types import LogProb

        sentinel = [[LogProb(token="Hello", logprob=-0.5)]]
        MockLogProbParser.from_completions.return_value = sentinel

        mock_response = Completion(
            id="cmpl-test",
            object="text_completion",
            created=1700000000,
            model="text-davinci-003",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            choices=[
                CompletionChoice(
                    text="Hello",
                    finish_reason="stop",
                    index=0,
                    logprobs=logprobs_obj,
                )
            ],
        )

        mock_client = MockSyncOpenAI.return_value
        mock_client.completions.create.return_value = mock_response

        llm = OpenAI(
            model="text-davinci-003", api_key="sk-test",
            logprobs=True, top_logprobs=1,
        )
        response = llm.complete("hi")

        MockLogProbParser.from_completions.assert_called_once_with(logprobs_obj)
        assert response.logprob is not None, (
            "likelihood_score should be set from LogProbParser result"
        )


@pytest.mark.mock
class TestStreamNullDelta:
    """Tests for edge cases in _stream_chat."""

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_stream_skips_null_delta(self, MockSyncOpenAI: MagicMock):
        """Test that streaming skips chunks where delta is None.

        Test scenario:
            A chunk with choices[0].delta = None should be skipped (continue).
        """
        null_delta_chunk = MagicMock()
        null_delta_chunk.choices = [MagicMock()]
        null_delta_chunk.choices[0].delta = None

        normal_chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=1700000000,
            model="gpt-4o-mini",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(role="assistant", content="Hi"),
                    finish_reason="stop",
                    index=0,
                )
            ],
        )

        def gen_chunks():
            yield null_delta_chunk
            yield normal_chunk

        mock_client = MockSyncOpenAI.return_value
        mock_client.chat.completions.create.return_value = gen_chunks()

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")
        chunks = list(
            llm._stream_chat(
                [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
            )
        )
        assert len(chunks) == 1, (
            f"Expected 1 non-null-delta chunk, got {len(chunks)}"
        )


@pytest.mark.mock
class TestStreamCompleteEmptyChoices:
    """Tests for _stream_complete edge cases."""

    @patch("serapeum.openai.llm.base.client.SyncOpenAI")
    def test_stream_complete_empty_choices(self, MockSyncOpenAI: MagicMock):
        """Test that streaming complete handles chunks with empty choices.

        Test scenario:
            A completion chunk with empty choices should yield an empty
            delta string.
        """
        empty_chunk = Completion(
            id="cmpl-test", object="text_completion", created=1700000000,
            model="text-davinci-003",
            choices=[],
        )
        normal_chunk = Completion(
            id="cmpl-test", object="text_completion", created=1700000000,
            model="text-davinci-003",
            choices=[
                CompletionChoice(text="Hi", finish_reason="stop", index=0)
            ],
        )

        def gen():
            yield empty_chunk
            yield normal_chunk

        mock_client = MockSyncOpenAI.return_value
        mock_client.completions.create.return_value = gen()

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        chunks = list(llm._stream_complete("test"))
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert chunks[0].delta == "", "Empty choices should yield empty delta"
        assert chunks[1].text == "Hi", (
            f"Expected accumulated text 'Hi', got '{chunks[1].text}'"
        )


@pytest.mark.mock
class TestAsyncChatWithLogprobs:
    """Tests for logprobs parsing in OpenAI._achat."""

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.chat_completions.LogProbParser")
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_achat_with_logprobs(
        self, MockAsyncOpenAI: MagicMock, MockLogProbParser: MagicMock
    ):
        """Test that logprobs in async response trigger LogProbParser.

        Test scenario:
            Async chat with logprobs enabled should call
            LogProbParser.from_tokens.
        """
        from openai.types.chat.chat_completion import ChoiceLogprobs
        from openai.types.chat.chat_completion_token_logprob import (
            ChatCompletionTokenLogprob,
            TopLogprob,
        )

        logprob_data = [
            ChatCompletionTokenLogprob(
                token="Hi",
                logprob=-0.3,
                top_logprobs=[
                    TopLogprob(token="Hi", logprob=-0.3, bytes=None)
                ],
                bytes=None,
            ),
        ]
        from serapeum.core.base.llms.types import LogProb

        sentinel = [[LogProb(token="Hi", logprob=-0.3)]]
        MockLogProbParser.from_tokens.return_value = sentinel

        mock_response = ChatCompletion(
            id="chatcmpl-test",
            object="chat.completion",
            created=1700000000,
            model="gpt-4o-mini",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            choices=[
                Choice(
                    message=ChatCompletionMessage(
                        role="assistant", content="Hi"
                    ),
                    finish_reason="stop",
                    index=0,
                    logprobs=ChoiceLogprobs(
                        content=logprob_data, refusal=None
                    ),
                )
            ],
        )

        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=mock_response)
        mock_client.chat.completions.create = create_fn

        llm = OpenAI(
            model="gpt-4o-mini", api_key="sk-test",
            logprobs=True, top_logprobs=1,
        )
        response = await llm.achat(
            [Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])]
        )

        MockLogProbParser.from_tokens.assert_called_once_with(logprob_data)
        assert response.logprob is not None, (
            "Async response should set likelihood_score from LogProbParser"
        )


@pytest.mark.mock
class TestAsyncCompleteWithLogprobs:
    """Tests for logprobs parsing in OpenAI._acomplete."""

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.chat_completions.LogProbParser")
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_acomplete_with_logprobs(
        self, MockAsyncOpenAI: MagicMock, MockLogProbParser: MagicMock
    ):
        """Test that logprobs in async completion trigger LogProbParser.

        Test scenario:
            Async complete with logprobs should call
            LogProbParser.from_completions.
        """
        from openai.types.completion_choice import Logprobs

        logprobs_obj = Logprobs(
            tokens=["Hello"],
            token_logprobs=[-0.5],
            top_logprobs=[{"Hello": -0.5}],
            text_offset=[0],
        )
        from serapeum.core.base.llms.types import LogProb

        sentinel = [[LogProb(token="Hello", logprob=-0.5)]]
        MockLogProbParser.from_completions.return_value = sentinel

        mock_response = Completion(
            id="cmpl-test",
            object="text_completion",
            created=1700000000,
            model="text-davinci-003",
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            choices=[
                CompletionChoice(
                    text="Hello",
                    finish_reason="stop",
                    index=0,
                    logprobs=logprobs_obj,
                )
            ],
        )

        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=mock_response)
        mock_client.completions.create = create_fn

        llm = OpenAI(
            model="text-davinci-003", api_key="sk-test",
            logprobs=True, top_logprobs=1,
        )
        response = await llm.acomplete("hi")

        MockLogProbParser.from_completions.assert_called_once_with(logprobs_obj)
        assert response.logprob is not None, (
            "Async completion should set likelihood_score from LogProbParser"
        )


@pytest.mark.mock
class TestAsyncStreamComplete:
    """Tests for OpenAI._astream_complete."""

    @pytest.mark.asyncio()
    @patch("serapeum.openai.llm.base.client.AsyncOpenAI")
    async def test_astream_complete_empty_choices(
        self, MockAsyncOpenAI: MagicMock
    ):
        """Test that async streaming complete handles empty choices.

        Test scenario:
            A completion chunk with empty choices should yield empty delta.
        """

        async def gen():
            yield Completion(
                id="cmpl-test", object="text_completion", created=1700000000,
                model="text-davinci-003",
                choices=[],
            )
            yield Completion(
                id="cmpl-test", object="text_completion", created=1700000000,
                model="text-davinci-003",
                choices=[
                    CompletionChoice(
                        text="world", finish_reason="stop", index=0
                    )
                ],
            )

        mock_client = MockAsyncOpenAI.return_value
        create_fn = AsyncMock(return_value=gen())
        mock_client.completions.create = create_fn

        llm = OpenAI(model="text-davinci-003", api_key="sk-test")
        result_gen = await llm.acomplete("hello", stream=True)
        chunks = [c async for c in result_gen]

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert chunks[0].delta == "", "Empty choices should yield empty delta"
        assert chunks[1].text == "world", (
            f"Expected accumulated 'world', got '{chunks[1].text}'"
        )


@pytest.mark.unit
class TestPrepareChatWithToolsNonFunctionCalling:
    """Tests for _prepare_chat_with_tools with non-function-calling models."""

    def test_strict_not_applied_for_non_fc_model(self):
        """Test that strict mode is not applied for non-function-calling models.

        Test scenario:
            For a non-function-calling model (text-davinci-003), the strict
            flag should not be injected into tool specs even when strict=True.
        """
        llm = OpenAI(model="text-davinci-003", api_key="sk-test", strict=True)
        tool = _make_search_tool()
        result = llm._prepare_chat_with_tools(tools=[tool], user_msg="test")
        tool_spec = result["tools"][0]
        assert "strict" not in tool_spec.get("function", {}), (
            "Non-FC model should not have strict in tool spec"
        )
