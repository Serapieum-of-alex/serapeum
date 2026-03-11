"""Unit and mock tests for the OpenAI Responses API provider class.

Targets ``serapeum.openai.llm.responses.Responses`` and covers:
- Constructor, model validators, field defaults
- class_name(), metadata, _should_use_structure_outputs(), _is_azure_client()
- _get_model_kwargs (all edge cases)
- chat / achat delegation (stream and non-stream)
- _chat / _achat (mocked client, response parsing, track_previous_responses)
- _stream_chat / _astream_chat (mocked streaming, accumulator)
- _prepare_chat_with_tools (tool specs, strict, tool_choice, message types)
- get_tool_calls_from_response (with and without tool calls)
- generate_tool_calls / agenerate_tool_calls (mocked flow)
- ResponsesOutputParser and ResponsesStreamAccumulator
- to_openai_message_dicts conversion

Run with: python -m pytest libs/providers/openai/tests/test_openai_responses.py -v -m "unit or mock"
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic import BaseModel, Field

from serapeum.core.base.llms.types import (
    ChatResponse,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
    DocumentBlock,
)
from serapeum.core.tools import CallableTool
from serapeum.core.prompts import PromptTemplate
from serapeum.openai.llm.responses import Responses
from serapeum.openai.data.models import O1_MODELS
from serapeum.openai.parsers import (
    ResponsesOutputParser,
    ResponsesStreamAccumulator,
    to_openai_message_dicts,
)

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseReasoningItem,
    ResponseOutputItem,
    ResponseOutputText,
    ResponseOutputItemDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseFunctionToolCall,
)
from openai.types.responses.response_reasoning_item import Content, Summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(**overrides: Any) -> Responses:
    """Create a Responses instance with a fake API key and optional overrides."""
    defaults: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "api_key": "fake-api-key",
    }
    defaults.update(overrides)
    return Responses(**defaults)


def _make_mock_response(
    *,
    response_id: str = "resp_abc123",
    text: str = "Hello world",
    reasoning_tokens: int | None = None,
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
    usage.output_tokens_details = MagicMock()
    if reasoning_tokens is not None:
        usage.output_tokens_details.reasoning_tokens = reasoning_tokens
    else:
        del usage.output_tokens_details.reasoning_tokens
    mock_response.usage = usage

    return mock_response


def _make_stream_events() -> list:
    """Build a sequence of mock streaming events."""
    return [
        ResponseTextDeltaEvent(
            content_index=0,
            item_id="item_001",
            output_index=0,
            delta="Hello",
            type="response.output_text.delta",
            sequence_number=1,
            logprobs=[],
        ),
        ResponseTextDeltaEvent(
            content_index=0,
            item_id="item_001",
            output_index=0,
            delta=" world",
            type="response.output_text.delta",
            sequence_number=2,
            logprobs=[],
        ),
    ]


def _add_tool() -> CallableTool:
    """Create a simple add tool for testing."""
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    return CallableTool.from_function(func=add)


def _search_tool() -> CallableTool:
    """Create a simple search tool for testing."""
    def search(query: str) -> str:
        """Search for information about a query."""
        return f"Results for {query}"
    return CallableTool.from_function(
        func=search, name="search_tool", description="A tool for searching information",
    )


def _user_messages(text: str = "Say hello") -> list[Message]:
    """Build a single-turn user message list."""
    return [Message(role=MessageRole.USER, chunks=[TextChunk(content=text)])]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm() -> Responses:
    """Create a default Responses instance with a fake API key."""
    return _make_llm()


@pytest.fixture
def llm_with_mocked_client(llm: Responses) -> Responses:
    """Responses instance with mocked sync and async clients."""
    llm._client = MagicMock()
    llm._async_client = AsyncMock()
    return llm


# ===========================================================================
# TestResponsesInit
# ===========================================================================


@pytest.mark.unit
class TestResponsesInit:
    """Tests for Responses constructor and model validators."""

    def test_default_fields(self, llm: Responses):
        """Test default field values after construction.

        Test scenario:
            Construct with model + api_key only; verify all defaults.
        """
        assert llm.model == "gpt-4o-mini", f"Expected model 'gpt-4o-mini', got '{llm.model}'"
        assert llm.temperature == 0.1, f"Expected temperature 0.1, got {llm.temperature}"
        assert llm.top_p == 1.0, f"Expected top_p 1.0, got {llm.top_p}"
        assert llm.max_output_tokens is None, f"Expected None, got {llm.max_output_tokens}"
        assert llm.reasoning_options is None, "reasoning_options should be None"
        assert llm.include is None, "include should be None"
        assert llm.instructions is None, "instructions should be None"
        assert llm.track_previous_responses is False, "track_previous_responses should be False"
        assert llm.store is False, "store should be False"
        assert llm.built_in_tools is None, "built_in_tools should be None"
        assert llm.truncation == "disabled", f"Expected 'disabled', got '{llm.truncation}'"
        assert llm.user is None, "user should be None"
        assert llm.call_metadata is None, "call_metadata should be None"
        assert llm.additional_kwargs == {}, "additional_kwargs should be empty dict"
        assert llm.strict is False, "strict should be False"
        assert llm.context_window is None, "context_window should be None"

    def test_o1_model_forces_temperature(self):
        """Test that O1 models force temperature to 1.0.

        Test scenario:
            Construct with an O1 model name; temperature should be overridden.
        """
        o1_model = next(iter(O1_MODELS))
        llm = _make_llm(model=o1_model)
        assert llm.temperature == 1.0, (
            f"O1 model should force temperature to 1.0, got {llm.temperature}"
        )

    def test_track_previous_responses_forces_store(self):
        """Test that track_previous_responses=True forces store=True.

        Test scenario:
            Construct with track_previous_responses=True and store=False.
        """
        llm = _make_llm(track_previous_responses=True, store=False)
        assert llm.store is True, "store should be True when tracking responses"

    def test_previous_response_id_injection(self):
        """Test that previous_response_id can be injected via constructor.

        Test scenario:
            Pass previous_response_id in constructor dict — it should be
            popped before validation and set on the private attribute.
        """
        llm = Responses(
            **{
                "model": "gpt-4o-mini",
                "api_key": "fake-key",
                "previous_response_id": "resp_prev123",
            }
        )
        assert llm._previous_response_id == "resp_prev123", (
            f"Expected 'resp_prev123', got '{llm._previous_response_id}'"
        )

    def test_previous_response_id_default_none(self, llm: Responses):
        """Test that _previous_response_id defaults to None.

        Test scenario:
            Normal construction without injecting previous_response_id.
        """
        assert llm._previous_response_id is None, (
            "_previous_response_id should default to None"
        )

    def test_custom_fields(self):
        """Test construction with custom field values.

        Test scenario:
            Provide non-default values for all configurable fields.
        """
        llm = _make_llm(
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=1024,
            instructions="Be concise.",
            truncation="auto",
            user="user_123",
            call_metadata={"tag": "test"},
            additional_kwargs={"seed": 42},
            strict=True,
            context_window=8192,
        )
        assert llm.temperature == 0.5, f"Expected 0.5, got {llm.temperature}"
        assert llm.top_p == 0.9, f"Expected 0.9, got {llm.top_p}"
        assert llm.max_output_tokens == 1024, f"Expected 1024, got {llm.max_output_tokens}"
        assert llm.instructions == "Be concise.", f"Got '{llm.instructions}'"
        assert llm.truncation == "auto", f"Expected 'auto', got '{llm.truncation}'"
        assert llm.user == "user_123", f"Got '{llm.user}'"
        assert llm.call_metadata == {"tag": "test"}, f"Got {llm.call_metadata}"
        assert llm.additional_kwargs == {"seed": 42}, f"Got {llm.additional_kwargs}"
        assert llm.strict is True, "strict should be True"
        assert llm.context_window == 8192, f"Expected 8192, got {llm.context_window}"

    def test_temperature_validation_range(self):
        """Test that temperature outside [0.0, 2.0] raises ValidationError.

        Test scenario:
            temperature=3.0 should fail Pydantic validation.
        """
        with pytest.raises(Exception, match="less than or equal to 2"):
            _make_llm(temperature=3.0)

    def test_max_output_tokens_must_be_positive(self):
        """Test that max_output_tokens <= 0 raises ValidationError.

        Test scenario:
            max_output_tokens=0 should fail Pydantic validation.
        """
        with pytest.raises(Exception, match="greater than 0"):
            _make_llm(max_output_tokens=0)


# ===========================================================================
# TestClassName
# ===========================================================================


@pytest.mark.unit
class TestClassName:
    """Tests for Responses.class_name()."""

    def test_class_name_returns_expected_string(self):
        """Test that class_name returns the canonical provider identifier.

        Test scenario:
            Static method should return 'openai_responses_llm'.
        """
        assert Responses.class_name() == "openai_responses_llm", (
            f"Expected 'openai_responses_llm', got '{Responses.class_name()}'"
        )


# ===========================================================================
# TestMetadata
# ===========================================================================


@pytest.mark.unit
class TestMetadata:
    """Tests for Responses.metadata property."""

    def test_metadata_default_context_window(self, llm: Responses):
        """Test metadata infers context window from model name.

        Test scenario:
            gpt-4o-mini should have a known context window size.
        """
        meta = llm.metadata
        assert meta.model_name == "gpt-4o-mini", f"Got '{meta.model_name}'"
        assert meta.is_chat_model is True, "Should be a chat model"
        assert meta.context_window > 0, (
            f"Expected positive context window, got {meta.context_window}"
        )

    def test_metadata_explicit_context_window(self):
        """Test metadata uses explicit context_window when set.

        Test scenario:
            context_window=4096 should override the inferred value.
        """
        llm = _make_llm(context_window=4096)
        meta = llm.metadata
        assert meta.context_window == 4096, f"Expected 4096, got {meta.context_window}"

    def test_metadata_num_output_with_max_tokens(self):
        """Test metadata.num_output reflects max_output_tokens.

        Test scenario:
            max_output_tokens=512 should be reported in metadata.
        """
        llm = _make_llm(max_output_tokens=512)
        meta = llm.metadata
        assert meta.num_output == 512, f"Expected 512, got {meta.num_output}"

    def test_metadata_num_output_default(self, llm: Responses):
        """Test metadata.num_output is -1 when max_output_tokens is None.

        Test scenario:
            Default construction with no max_output_tokens.
        """
        assert llm.metadata.num_output == -1, f"Expected -1, got {llm.metadata.num_output}"

    def test_metadata_is_function_calling(self, llm: Responses):
        """Test that gpt-4o-mini is reported as function-calling capable.

        Test scenario:
            is_function_calling_model should be True for gpt-4o-mini.
        """
        assert llm.metadata.is_function_calling_model is True, (
            "gpt-4o-mini should be a function calling model"
        )


# ===========================================================================
# TestShouldUseStructuredOutputs
# ===========================================================================


@pytest.mark.unit
class TestShouldUseStructuredOutputs:
    """Tests for Responses._should_use_structure_outputs()."""

    def test_always_returns_false(self, llm: Responses):
        """Test that Responses API always uses function-calling path.

        Test scenario:
            Regardless of model, _should_use_structure_outputs() is False.
        """
        assert llm._should_use_structure_outputs() is False, (
            "Responses API should always return False"
        )


# ===========================================================================
# TestIsAzureClient
# ===========================================================================


@pytest.mark.unit
class TestIsAzureClient:
    """Tests for Responses._is_azure_client()."""

    def test_non_azure_client(self, llm_with_mocked_client: Responses):
        """Test _is_azure_client returns False for standard OpenAI client.

        Test scenario:
            MagicMock client is not an AzureOpenAI instance.
        """
        assert llm_with_mocked_client._is_azure_client() is False, (
            "Mocked client should not be Azure"
        )

    def test_azure_client(self, llm_with_mocked_client: Responses):
        """Test _is_azure_client returns True when client is AzureOpenAI.

        Test scenario:
            Spec the mock as AzureOpenAI; should return True.
        """
        from openai import AzureOpenAI
        llm_with_mocked_client._client = MagicMock(spec=AzureOpenAI)
        assert llm_with_mocked_client._is_azure_client() is True, (
            "AzureOpenAI client should be detected"
        )


# ===========================================================================
# TestGetModelKwargs
# ===========================================================================


@pytest.mark.unit
class TestGetModelKwargs:
    """Tests for Responses._get_model_kwargs()."""

    def test_basic_kwargs(self, llm: Responses):
        """Test basic model kwargs contain expected keys.

        Test scenario:
            Default construction should include model, temperature, top_p, etc.
        """
        kwargs = llm._get_model_kwargs()
        assert kwargs["model"] == "gpt-4o-mini", f"Got '{kwargs['model']}'"
        assert kwargs["temperature"] == 0.1, f"Got {kwargs['temperature']}"
        assert kwargs["top_p"] == 1.0, f"Got {kwargs['top_p']}"
        assert kwargs["truncation"] == "disabled", f"Got '{kwargs['truncation']}'"
        assert kwargs["store"] is False, f"Got {kwargs['store']}"
        assert kwargs["tools"] == [], f"Got {kwargs['tools']}"

    def test_runtime_kwargs_override(self, llm: Responses):
        """Test that per-call kwargs override model defaults.

        Test scenario:
            Pass top_p=0.8 and max_output_tokens=100 at call time.
        """
        kwargs = llm._get_model_kwargs(top_p=0.8, max_output_tokens=100)
        assert kwargs["top_p"] == 0.8, f"Expected 0.8, got {kwargs['top_p']}"
        assert kwargs["max_output_tokens"] == 100, (
            f"Expected 100, got {kwargs['max_output_tokens']}"
        )

    def test_reasoning_options_excludes_sampling_params(self):
        """Test that reasoning_options causes sampling params to be excluded.

        Test scenario:
            Set reasoning_options; temperature, top_p, etc. should be absent.
        """
        llm = _make_llm(reasoning_options={"effort": "low"})
        kwargs = llm._get_model_kwargs()
        assert "top_p" not in kwargs, "top_p should be excluded with reasoning_options"
        assert "temperature" not in kwargs, "temperature should be excluded"
        assert "presence_penalty" not in kwargs, "presence_penalty should be excluded"
        assert "frequency_penalty" not in kwargs, "frequency_penalty should be excluded"
        assert "model" in kwargs, "model should still be present"

    def test_o1_model_with_reasoning_options(self):
        """Test O1 model with reasoning_options includes reasoning key.

        Test scenario:
            O1 model + reasoning_options should add 'reasoning' to kwargs.
        """
        o1_model = next(iter(O1_MODELS))
        llm = _make_llm(model=o1_model, reasoning_options={"effort": "high"})
        kwargs = llm._get_model_kwargs()
        assert kwargs["reasoning"] == {"effort": "high"}, f"Got {kwargs.get('reasoning')}"

    def test_non_o1_model_with_reasoning_options_no_reasoning_key(self):
        """Test non-O1 model with reasoning_options does NOT add reasoning key.

        Test scenario:
            gpt-4o-mini + reasoning_options should NOT have 'reasoning' key.
        """
        llm = _make_llm(model="gpt-4o-mini", reasoning_options={"effort": "low"})
        kwargs = llm._get_model_kwargs()
        assert "reasoning" not in kwargs, "Non-O1 model should not have reasoning key"

    def test_tools_merge_with_built_in_tools(self):
        """Test that per-call tools are appended to built_in_tools.

        Test scenario:
            built_in_tools has web_search; per-call tools has a function tool.
        """
        llm = _make_llm(built_in_tools=[{"type": "web_search"}])
        custom_tool = {"type": "function", "name": "my_func"}
        kwargs = llm._get_model_kwargs(tools=[custom_tool])
        assert {"type": "web_search"} in kwargs["tools"], (
            "built_in_tools should be included"
        )
        assert custom_tool in kwargs["tools"], "per-call tool should be appended"
        assert len(kwargs["tools"]) == 2, f"Expected 2 tools, got {len(kwargs['tools'])}"

    def test_tools_none_handled_gracefully(self, llm: Responses):
        """Test _get_model_kwargs handles tools=None without error.

        Test scenario:
            Pass tools=None (can happen from _prepare_chat_with_tools).
        """
        kwargs = llm._get_model_kwargs(tools=None)
        assert kwargs["tools"] == [], f"Expected empty list, got {kwargs['tools']}"

    def test_additional_kwargs_merged(self):
        """Test that additional_kwargs are merged into output.

        Test scenario:
            Set additional_kwargs={"seed": 42}; verify it appears in kwargs.
        """
        llm = _make_llm(additional_kwargs={"seed": 42})
        kwargs = llm._get_model_kwargs()
        assert kwargs["seed"] == 42, f"Expected seed=42, got {kwargs.get('seed')}"

    def test_previous_response_id_in_kwargs(self):
        """Test _previous_response_id is included in model kwargs.

        Test scenario:
            Inject previous_response_id; verify it's in the output kwargs.
        """
        llm = Responses(
            **{
                "model": "gpt-4o-mini",
                "api_key": "fake-key",
                "previous_response_id": "resp_prev_abc",
            }
        )
        kwargs = llm._get_model_kwargs()
        assert kwargs["previous_response_id"] == "resp_prev_abc", (
            f"Got '{kwargs['previous_response_id']}'"
        )

    def test_instructions_in_kwargs(self):
        """Test instructions field is included in kwargs.

        Test scenario:
            Set instructions="Be brief"; verify it appears.
        """
        llm = _make_llm(instructions="Be brief")
        kwargs = llm._get_model_kwargs()
        assert kwargs["instructions"] == "Be brief", f"Got '{kwargs['instructions']}'"


# ===========================================================================
# TestChat (delegation)
# ===========================================================================


@pytest.mark.mock
class TestChat:
    """Tests for Responses.chat() — delegation to _chat / _stream_chat."""

    def test_chat_delegates_to_chat_internal(
        self, llm_with_mocked_client: Responses
    ):
        """Test that chat(stream=False) calls _chat.

        Test scenario:
            Patch _chat; verify it's called with messages.
        """
        mock_resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT, chunks=[TextChunk(content="Hi")]
            ),
        )
        with patch.object(
            Responses, "_chat", return_value=mock_resp
        ) as mock_chat:
            result = llm_with_mocked_client.chat(_user_messages())
        mock_chat.assert_called_once()
        assert result.message.content == "Hi", f"Got '{result.message.content}'"

    def test_chat_stream_delegates_to_stream_chat(
        self, llm_with_mocked_client: Responses
    ):
        """Test that chat(stream=True) calls _stream_chat.

        Test scenario:
            Patch _stream_chat; verify it's called.
        """
        def fake_gen():
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    chunks=[TextChunk(content="Hi")],
                ),
                delta="Hi",
            )

        with patch.object(
            Responses, "_stream_chat", return_value=fake_gen()
        ) as mock_stream:
            chunks = list(
                llm_with_mocked_client.chat(_user_messages(), stream=True)
            )
        mock_stream.assert_called_once()
        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
        assert chunks[0].delta == "Hi", f"Got delta '{chunks[0].delta}'"


# ===========================================================================
# TestChatInternal (_chat)
# ===========================================================================


@pytest.mark.mock
class TestChatInternal:
    """Tests for Responses._chat() with mocked OpenAI SDK client."""

    def test_chat_returns_parsed_response(
        self, llm_with_mocked_client: Responses
    ):
        """Test _chat returns a properly parsed ChatResponse.

        Test scenario:
            Mock client to return a simple text response; verify parsing.
        """
        mock_resp = _make_mock_response(text="Hello from API")
        llm_with_mocked_client.client.responses.create.return_value = mock_resp

        result = llm_with_mocked_client._chat(_user_messages())

        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT, got {result.message.role}"
        )
        assert result.message.content == "Hello from API", (
            f"Got '{result.message.content}'"
        )
        assert result.raw is mock_resp, "raw should reference the SDK response"
        assert result.additional_kwargs["usage"] is mock_resp.usage, (
            "usage should be stored in additional_kwargs"
        )

    def test_chat_tracks_previous_response_id(self):
        """Test _chat stores response ID when track_previous_responses=True.

        Test scenario:
            Enable tracking; verify _previous_response_id is updated.
        """
        llm = _make_llm(track_previous_responses=True)
        llm._client = MagicMock()
        mock_resp = _make_mock_response(response_id="resp_track_123")
        llm.client.responses.create.return_value = mock_resp

        llm._chat(_user_messages())

        assert llm._previous_response_id == "resp_track_123", (
            f"Expected 'resp_track_123', got '{llm._previous_response_id}'"
        )

    def test_chat_does_not_track_when_disabled(
        self, llm_with_mocked_client: Responses
    ):
        """Test _chat does NOT update _previous_response_id when tracking off.

        Test scenario:
            Default (track_previous_responses=False); ID stays None.
        """
        mock_resp = _make_mock_response(response_id="resp_no_track")
        llm_with_mocked_client.client.responses.create.return_value = mock_resp

        llm_with_mocked_client._chat(_user_messages())

        assert llm_with_mocked_client._previous_response_id is None, (
            "Should not track when disabled"
        )

    def test_chat_copies_reasoning_tokens_to_thinking_blocks(self):
        """Test _chat writes reasoning_tokens onto ThinkingBlock chunks.

        Test scenario:
            Response includes reasoning_tokens in usage; thinking block
            should get num_tokens set.
        """
        llm = _make_llm()
        llm._client = MagicMock()
        mock_resp = _make_mock_response(reasoning_tokens=150)

        reasoning_item = ResponseReasoningItem(
            id="reason_1",
            summary=[],
            type="reasoning",
            content=[Content(text="Let me think...", type="reasoning_text")],
            encrypted_content=None,
            status=None,
        )
        output_msg = ResponseOutputMessage(
            type="message",
            content=[
                {"type": "output_text", "text": "Answer", "annotations": []}
            ],
            role="assistant",
            id="msg_002",
            status="completed",
        )
        mock_resp.output = [reasoning_item, output_msg]
        llm.client.responses.create.return_value = mock_resp

        result = llm._chat(_user_messages())

        thinking_blocks = [
            c for c in result.message.chunks if isinstance(c, ThinkingBlock)
        ]
        assert len(thinking_blocks) == 1, (
            f"Expected 1 thinking block, got {len(thinking_blocks)}"
        )
        assert thinking_blocks[0].num_tokens == 150, (
            f"Expected 150 reasoning tokens, got {thinking_blocks[0].num_tokens}"
        )

    def test_chat_passes_kwargs_to_client(
        self, llm_with_mocked_client: Responses
    ):
        """Test _chat forwards model kwargs to the SDK create call.

        Test scenario:
            Verify the client.responses.create was called with
            stream=False and model kwargs.
        """
        mock_resp = _make_mock_response()
        llm_with_mocked_client.client.responses.create.return_value = mock_resp

        llm_with_mocked_client._chat(_user_messages())

        call_kwargs = (
            llm_with_mocked_client.client.responses.create.call_args
        )
        assert call_kwargs.kwargs["stream"] is False, (
            "stream should be False for _chat"
        )
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini", (
            f"Got model '{call_kwargs.kwargs['model']}'"
        )


# ===========================================================================
# TestStreamChat (_stream_chat)
# ===========================================================================


@pytest.mark.mock
class TestStreamChat:
    """Tests for Responses._stream_chat() with mocked streaming events."""

    def test_stream_chat_yields_chunks(
        self, llm_with_mocked_client: Responses
    ):
        """Test _stream_chat yields ChatResponse for each event.

        Test scenario:
            Two text delta events should produce two yielded chunks.
        """
        events = _make_stream_events()
        llm_with_mocked_client.client.responses.create.return_value = events

        chunks = list(llm_with_mocked_client._stream_chat(_user_messages()))

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert chunks[0].delta == "Hello", (
            f"First delta should be 'Hello', got '{chunks[0].delta}'"
        )
        assert chunks[1].delta == " world", (
            f"Second delta should be ' world', got '{chunks[1].delta}'"
        )

    def test_stream_chat_all_chunks_are_assistant(
        self, llm_with_mocked_client: Responses
    ):
        """Test all streamed chunks have ASSISTANT role.

        Test scenario:
            Every yielded ChatResponse should have role=ASSISTANT.
        """
        llm_with_mocked_client.client.responses.create.return_value = (
            _make_stream_events()
        )

        for chunk in llm_with_mocked_client._stream_chat(_user_messages()):
            assert chunk.message.role == MessageRole.ASSISTANT, (
                f"Expected ASSISTANT, got {chunk.message.role}"
            )

    def test_stream_chat_tracks_previous_response_id(self):
        """Test _stream_chat updates _previous_response_id from accumulator.

        Test scenario:
            Enable tracking; the accumulator should propagate the response ID.
        """
        llm = _make_llm(track_previous_responses=True)
        llm._client = MagicMock()

        created_event = MagicMock()
        created_event.type = "response.created"
        response_obj = MagicMock()
        response_obj.id = "resp_stream_123"
        created_event.response = response_obj

        llm.client.responses.create.return_value = [created_event]

        with patch.object(
            ResponsesStreamAccumulator,
            "update",
            return_value=([], None),
        ), patch.object(
            ResponsesStreamAccumulator,
            "previous_response_id",
            new_callable=lambda: property(
                lambda self: "resp_stream_123"
            ),
        ):
            list(llm._stream_chat(_user_messages()))

        assert llm._previous_response_id == "resp_stream_123", (
            f"Expected 'resp_stream_123', got '{llm._previous_response_id}'"
        )

    def test_stream_chat_passes_stream_true(
        self, llm_with_mocked_client: Responses
    ):
        """Test _stream_chat calls create with stream=True.

        Test scenario:
            Verify the SDK create call uses stream=True.
        """
        llm_with_mocked_client.client.responses.create.return_value = []

        list(llm_with_mocked_client._stream_chat(_user_messages()))

        call_kwargs = (
            llm_with_mocked_client.client.responses.create.call_args
        )
        assert call_kwargs.kwargs["stream"] is True, (
            "stream should be True for _stream_chat"
        )


# ===========================================================================
# TestAChat (async delegation)
# ===========================================================================


@pytest.mark.mock
class TestAChat:
    """Tests for Responses.achat() — async delegation."""

    @pytest.mark.asyncio
    async def test_achat_delegates_to_achat_internal(
        self, llm_with_mocked_client: Responses
    ):
        """Test that achat(stream=False) calls _achat.

        Test scenario:
            Patch _achat; verify it's called.
        """
        mock_resp = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="Hi async")],
            ),
        )
        with patch.object(
            Responses, "_achat", return_value=mock_resp
        ) as mock_achat:
            result = await llm_with_mocked_client.achat(_user_messages())
        mock_achat.assert_called_once()
        assert result.message.content == "Hi async", (
            f"Got '{result.message.content}'"
        )

    @pytest.mark.asyncio
    async def test_achat_stream_delegates_to_astream_chat(
        self, llm_with_mocked_client: Responses
    ):
        """Test that achat(stream=True) calls _astream_chat.

        Test scenario:
            Patch _astream_chat; verify it's called.
        """
        async def fake_gen():
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    chunks=[TextChunk(content="Hi")],
                ),
                delta="Hi",
            )

        with patch.object(
            Responses, "_astream_chat", return_value=fake_gen()
        ) as mock_astream:
            gen = await llm_with_mocked_client.achat(
                _user_messages(), stream=True
            )
            chunks = [c async for c in gen]
        mock_astream.assert_called_once()
        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"


# ===========================================================================
# TestAChatInternal (_achat)
# ===========================================================================


@pytest.mark.mock
class TestAChatInternal:
    """Tests for Responses._achat() with mocked async client."""

    @pytest.mark.asyncio
    async def test_achat_returns_parsed_response(
        self, llm_with_mocked_client: Responses
    ):
        """Test _achat returns a properly parsed ChatResponse.

        Test scenario:
            Mock async client to return a simple text response.
        """
        mock_resp = _make_mock_response(text="Async hello")
        llm_with_mocked_client.async_client.responses.create = AsyncMock(
            return_value=mock_resp
        )

        result = await llm_with_mocked_client._achat(_user_messages())

        assert result.message.content == "Async hello", (
            f"Got '{result.message.content}'"
        )
        assert result.raw is mock_resp, "raw should reference the SDK response"
        assert result.additional_kwargs["usage"] is mock_resp.usage, (
            "usage should be in additional_kwargs"
        )

    @pytest.mark.asyncio
    async def test_achat_tracks_previous_response_id(self):
        """Test _achat stores response ID when track_previous_responses=True.

        Test scenario:
            Enable tracking; verify _previous_response_id is updated.
        """
        llm = _make_llm(track_previous_responses=True)
        llm._async_client = AsyncMock()
        mock_resp = _make_mock_response(response_id="resp_async_track")
        llm.async_client.responses.create = AsyncMock(return_value=mock_resp)

        await llm._achat(_user_messages())

        assert llm._previous_response_id == "resp_async_track", (
            f"Expected 'resp_async_track', got '{llm._previous_response_id}'"
        )


# ===========================================================================
# TestAStreamChat (_astream_chat)
# ===========================================================================


@pytest.mark.mock
class TestAStreamChat:
    """Tests for Responses._astream_chat() with mocked async streaming."""

    @pytest.mark.asyncio
    async def test_astream_chat_yields_chunks(
        self, llm_with_mocked_client: Responses
    ):
        """Test _astream_chat yields ChatResponse for each event.

        Test scenario:
            Two text delta events should produce two yielded chunks.
        """
        events = _make_stream_events()

        async def fake_stream():
            for event in events:
                yield event

        llm_with_mocked_client._async_client = AsyncMock()
        llm_with_mocked_client.async_client.responses.create = AsyncMock(
            return_value=fake_stream()
        )

        gen = await llm_with_mocked_client._astream_chat(_user_messages())
        chunks = [c async for c in gen]

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        assert chunks[0].delta == "Hello", (
            f"First delta should be 'Hello', got '{chunks[0].delta}'"
        )
        assert chunks[1].delta == " world", (
            f"Second delta should be ' world', got '{chunks[1].delta}'"
        )


# ===========================================================================
# TestPrepareChatWithTools
# ===========================================================================


@pytest.mark.unit
class TestPrepareChatWithTools:
    """Tests for Responses._prepare_chat_with_tools()."""

    def test_basic_tool_preparation(self, llm: Responses):
        """Test basic tool spec generation for Responses API format.

        Test scenario:
            One tool; verify flat format (type+name at top level).
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool],
            message="What is 2+2?",
        )

        assert len(result["tools"]) == 1, (
            f"Expected 1 tool, got {len(result['tools'])}"
        )
        assert result["tools"][0]["type"] == "function", (
            f"Expected 'function', got '{result['tools'][0]['type']}'"
        )
        assert result["tools"][0]["name"] == "add", (
            f"Expected 'add', got '{result['tools'][0]['name']}'"
        )
        assert "function" not in result["tools"][0], (
            "Responses API uses flat format, not nested 'function' key"
        )

    def test_string_message_converted_to_message_object(self, llm: Responses):
        """Test that a string message is converted to a Message.

        Test scenario:
            Pass message as string; verify it becomes a proper Message.
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool],
            message="Hello",
        )
        assert len(result["messages"]) == 1, (
            f"Expected 1 message, got {len(result['messages'])}"
        )
        msg = result["messages"][0]
        assert msg.role == MessageRole.USER, f"Expected USER, got {msg.role}"
        assert msg.content == "Hello", f"Got '{msg.content}'"

    def test_message_object_passed_through(self, llm: Responses):
        """Test that a Message object is passed through unchanged.

        Test scenario:
            Pass a Message directly; verify it appears in messages.
        """
        tool = _add_tool()
        msg = Message(
            role=MessageRole.USER, chunks=[TextChunk(content="Direct")]
        )
        result = llm._prepare_chat_with_tools(tools=[tool], message=msg)
        assert result["messages"][-1] is msg, "Message should be passed through"

    def test_chat_history_preserved(self, llm: Responses):
        """Test that chat_history is included in messages.

        Test scenario:
            Provide chat_history + new message; verify both are in result.
        """
        tool = _add_tool()
        history = [
            Message(
                role=MessageRole.USER, chunks=[TextChunk(content="prior")]
            ),
        ]
        result = llm._prepare_chat_with_tools(
            tools=[tool],
            message="new",
            chat_history=history,
        )
        assert len(result["messages"]) == 2, (
            f"Expected 2 messages, got {len(result['messages'])}"
        )

    def test_no_message_no_history(self, llm: Responses):
        """Test with no message and no chat_history.

        Test scenario:
            messages should be an empty list.
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(tools=[tool])
        assert result["messages"] == [], f"Expected empty, got {result['messages']}"

    def test_tool_required_sets_required(self, llm: Responses):
        """Test tool_required=True sets tool_choice='required'.

        Test scenario:
            Verify tool_choice value when tool_required=True.
        """
        tool = _search_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], tool_required=True
        )
        assert result["tool_choice"] == "required", (
            f"Expected 'required', got '{result['tool_choice']}'"
        )

    def test_tool_not_required_sets_auto(self, llm: Responses):
        """Test tool_required=False (default) sets tool_choice='auto'.

        Test scenario:
            Default behavior should use 'auto'.
        """
        tool = _search_tool()
        result = llm._prepare_chat_with_tools(tools=[tool])
        assert result["tool_choice"] == "auto", (
            f"Expected 'auto', got '{result['tool_choice']}'"
        )

    def test_explicit_tool_choice_overrides_tool_required(
        self, llm: Responses
    ):
        """Test that explicit tool_choice overrides tool_required.

        Test scenario:
            tool_required=True but tool_choice='none' -> should be 'none'.
        """
        tool = _search_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], tool_required=True, tool_choice="none",
        )
        assert result["tool_choice"] == "none", (
            f"Expected 'none', got '{result['tool_choice']}'"
        )

    def test_parallel_tool_calls_flag(self, llm: Responses):
        """Test allow_parallel_tool_calls is forwarded.

        Test scenario:
            allow_parallel_tool_calls=False should appear in result.
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], allow_parallel_tool_calls=False,
        )
        assert result["parallel_tool_calls"] is False, (
            f"Got {result['parallel_tool_calls']}"
        )

    def test_strict_mode_from_instance(self):
        """Test that instance-level strict=True adds strict to tool specs.

        Test scenario:
            strict=True on instance; tool specs should have strict=True.
        """
        llm = _make_llm(strict=True)
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(tools=[tool])
        assert result["tools"][0].get("strict") is True, (
            "Tool spec should have strict=True"
        )
        assert (
            result["tools"][0]["parameters"].get("additionalProperties")
            is False
        ), "additionalProperties should be False in strict mode"

    def test_strict_mode_override(self, llm: Responses):
        """Test per-call strict override.

        Test scenario:
            Instance strict=False, per-call strict=True.
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(tools=[tool], strict=True)
        assert result["tools"][0].get("strict") is True, (
            "Per-call strict=True should override instance"
        )

    def test_empty_tools_sets_none(self, llm: Responses):
        """Test that empty tools list results in tool_choice=None.

        Test scenario:
            Pass empty tools list; tool_choice should be None.
        """
        result = llm._prepare_chat_with_tools(tools=[])
        assert result["tools"] is None, f"Expected None, got {result['tools']}"
        assert result["tool_choice"] is None, (
            f"Expected None, got {result['tool_choice']}"
        )

    def test_extra_kwargs_forwarded(self, llm: Responses):
        """Test that extra kwargs are included in the result dict.

        Test scenario:
            Pass arbitrary extra_param=42.
        """
        tool = _add_tool()
        result = llm._prepare_chat_with_tools(
            tools=[tool], extra_param=42,
        )
        assert result["extra_param"] == 42, (
            f"Expected 42, got {result.get('extra_param')}"
        )


# ===========================================================================
# TestGetToolCallsFromResponse
# ===========================================================================


@pytest.mark.unit
class TestGetToolCallsFromResponse:
    """Tests for Responses.get_tool_calls_from_response() (inherited)."""

    def test_extracts_tool_calls(self, llm: Responses):
        """Test extracting tool calls from a response with ToolCallBlocks.

        Test scenario:
            Response has one ToolCallBlock; verify extraction.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="tc_001",
                        tool_name="search",
                        tool_kwargs='{"query": "test"}',
                    )
                ],
            ),
        )
        selections = llm.get_tool_calls_from_response(response)
        assert len(selections) == 1, f"Expected 1, got {len(selections)}"
        assert selections[0].tool_id == "tc_001", (
            f"Got '{selections[0].tool_id}'"
        )
        assert selections[0].tool_name == "search", (
            f"Got '{selections[0].tool_name}'"
        )
        assert selections[0].tool_kwargs == {"query": "test"}, (
            f"Got {selections[0].tool_kwargs}"
        )

    def test_multiple_tool_calls(self, llm: Responses):
        """Test extracting multiple tool calls.

        Test scenario:
            Response with two ToolCallBlocks.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="tc_001",
                        tool_name="search",
                        tool_kwargs='{"q": "a"}',
                    ),
                    ToolCallBlock(
                        tool_call_id="tc_002",
                        tool_name="add",
                        tool_kwargs='{"a": 1, "b": 2}',
                    ),
                ],
            ),
        )
        selections = llm.get_tool_calls_from_response(response)
        assert len(selections) == 2, f"Expected 2, got {len(selections)}"

    def test_no_tool_calls_raises_by_default(self, llm: Responses):
        """Test ValueError raised when no tool calls and error_on_no_tool_call=True.

        Test scenario:
            Response with only TextChunk; should raise ValueError.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="No tools here")],
            ),
        )
        with pytest.raises(ValueError, match="Expected at least one tool call"):
            llm.get_tool_calls_from_response(response)

    def test_no_tool_calls_returns_empty_when_allowed(self, llm: Responses):
        """Test empty list returned when no tool calls and error suppressed.

        Test scenario:
            Response with only text; error_on_no_tool_call=False.
        """
        response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="No tools")],
            ),
        )
        result = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False,
        )
        assert result == [], f"Expected empty list, got {result}"


# ===========================================================================
# TestGenerateToolCalls
# ===========================================================================


@pytest.mark.mock
class TestGenerateToolCalls:
    """Tests for generate_tool_calls() (inherited from FunctionCallingLLM)."""

    def test_generate_tool_calls_non_streaming(
        self, llm_with_mocked_client: Responses
    ):
        """Test generate_tool_calls calls chat and validates response.

        Test scenario:
            Mock _prepare_chat_with_tools and chat; verify flow.
        """
        tool = _add_tool()
        mock_response = ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="tc_1",
                        tool_name="add",
                        tool_kwargs='{"a": 2, "b": 3}',
                    )
                ],
            ),
        )
        with patch.object(
            Responses, "chat", return_value=mock_response,
        ):
            result = llm_with_mocked_client.generate_tool_calls(
                tools=[tool],
                message="What is 2+3?",
            )
        assert len(result.message.tool_calls) == 1, (
            f"Expected 1 tool call, got {len(result.message.tool_calls)}"
        )

    def test_generate_tool_calls_streaming(
        self, llm_with_mocked_client: Responses
    ):
        """Test generate_tool_calls with stream=True returns a generator.

        Test scenario:
            Mock chat(stream=True); verify generator is returned.
        """
        tool = _add_tool()

        def fake_gen():
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    chunks=[TextChunk(content="chunk")],
                ),
                delta="chunk",
            )

        with patch.object(
            Responses, "chat", return_value=fake_gen(),
        ):
            gen = llm_with_mocked_client.generate_tool_calls(
                tools=[tool],
                message="test",
                stream=True,
            )
            chunks = list(gen)
        assert len(chunks) == 1, f"Expected 1, got {len(chunks)}"


# ===========================================================================
# TestResponsesOutputParser
# ===========================================================================


@pytest.mark.unit
class TestResponsesOutputParser:
    """Tests for ResponsesOutputParser."""

    def test_parse_text_message(self):
        """Test parsing a simple text output message.

        Test scenario:
            Single ResponseOutputMessage with text -> TextChunk.
        """
        output = [
            ResponseOutputMessage(
                type="message",
                content=[
                    {
                        "type": "output_text",
                        "text": "Hello world",
                        "annotations": [],
                    }
                ],
                role="assistant",
                id="msg_1",
                status="completed",
            )
        ]
        result = ResponsesOutputParser(output).build()
        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT, got {result.message.role}"
        )
        assert len(result.message.chunks) == 1, (
            f"Expected 1 chunk, got {len(result.message.chunks)}"
        )
        assert isinstance(result.message.chunks[0], TextChunk), (
            f"Expected TextChunk, got {type(result.message.chunks[0])}"
        )
        assert result.message.chunks[0].content == "Hello world", (
            f"Got '{result.message.chunks[0].content}'"
        )

    def test_parse_reasoning_items(self):
        """Test parsing multiple reasoning items into ThinkingBlocks.

        Test scenario:
            Multiple reasoning items with various content/summary combos.
        """
        output: list[ResponseOutputItem] = [
            ResponseReasoningItem(
                id="r1",
                summary=[],
                type="reasoning",
                content=[
                    Content(text="thinking part 1", type="reasoning_text"),
                    Content(text="thinking part 2", type="reasoning_text"),
                ],
                encrypted_content=None,
                status=None,
            ),
            ResponseReasoningItem(
                id="r2",
                summary=[
                    Summary(text="summary", type="summary_text")
                ],
                type="reasoning",
                content=[
                    Content(text="more thinking", type="reasoning_text")
                ],
                encrypted_content=None,
                status=None,
            ),
        ]
        result = ResponsesOutputParser(output).build()
        thinking = [
            c for c in result.message.chunks if isinstance(c, ThinkingBlock)
        ]
        assert len(thinking) == 2, (
            f"Expected 2 thinking blocks, got {len(thinking)}"
        )
        assert thinking[0].content == "thinking part 1\nthinking part 2", (
            f"Got '{thinking[0].content}'"
        )
        assert thinking[1].content == "more thinking\nsummary", (
            f"Got '{thinking[1].content}'"
        )

    def test_parse_function_tool_call(self):
        """Test parsing a function tool call into ToolCallBlock.

        Test scenario:
            Single ResponseFunctionToolCall in output.
        """
        output: list[ResponseOutputItem] = [
            ResponseFunctionToolCall(
                arguments='{"key": "val"}',
                call_id="call_1",
                name="my_func",
                type="function_call",
                status="completed",
            ),
        ]
        result = ResponsesOutputParser(output).build()
        tool_calls = [
            c for c in result.message.chunks if isinstance(c, ToolCallBlock)
        ]
        assert len(tool_calls) == 1, f"Expected 1, got {len(tool_calls)}"
        assert tool_calls[0].tool_call_id == "call_1", (
            f"Got '{tool_calls[0].tool_call_id}'"
        )
        assert tool_calls[0].tool_name == "my_func", (
            f"Got '{tool_calls[0].tool_name}'"
        )
        assert tool_calls[0].tool_kwargs == '{"key": "val"}', (
            f"Got '{tool_calls[0].tool_kwargs}'"
        )

    def test_parse_mixed_output(self):
        """Test parsing a response with reasoning, tool calls, and text.

        Test scenario:
            Complex output with all item types.
        """
        output: list[ResponseOutputItem] = [
            ResponseReasoningItem(
                id="r1",
                summary=[],
                type="reasoning",
                content=[
                    Content(text="hello world", type="reasoning_text")
                ],
                encrypted_content=None,
                status=None,
            ),
            ResponseFunctionToolCall(
                arguments='{"hello": "world"}',
                call_id="1",
                name="test",
                type="function_call",
                status="completed",
            ),
            ResponseOutputMessage(
                id="m1",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="hey there",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        result = ResponsesOutputParser(output).build()
        thinking = [
            c for c in result.message.chunks if isinstance(c, ThinkingBlock)
        ]
        texts = [
            c for c in result.message.chunks if isinstance(c, TextChunk)
        ]
        tools = [
            c for c in result.message.chunks if isinstance(c, ToolCallBlock)
        ]
        assert len(thinking) == 1, (
            f"Expected 1 thinking, got {len(thinking)}"
        )
        assert len(texts) == 1, f"Expected 1 text, got {len(texts)}"
        assert len(tools) == 1, f"Expected 1 tool call, got {len(tools)}"

    def test_parse_reasoning_with_summary_only(self):
        """Test reasoning item with no content, only summaries.

        Test scenario:
            content=None, summary has two entries -> joined text.
        """
        output: list[ResponseOutputItem] = [
            ResponseReasoningItem(
                id="r1",
                summary=[
                    Summary(text="hello", type="summary_text"),
                    Summary(text="world", type="summary_text"),
                ],
                type="reasoning",
                content=None,
                encrypted_content=None,
                status=None,
            ),
        ]
        result = ResponsesOutputParser(output).build()
        thinking = [
            c for c in result.message.chunks if isinstance(c, ThinkingBlock)
        ]
        assert len(thinking) == 1, f"Expected 1, got {len(thinking)}"
        assert thinking[0].content == "hello\nworld", (
            f"Got '{thinking[0].content}'"
        )


# ===========================================================================
# TestResponsesStreamAccumulator
# ===========================================================================


@pytest.mark.unit
class TestResponsesStreamAccumulator:
    """Tests for ResponsesStreamAccumulator."""

    def test_text_delta_accumulation(self):
        """Test accumulating text delta events.

        Test scenario:
            Two text deltas should produce TextChunk with accumulated text.
        """
        acc = ResponsesStreamAccumulator()

        blocks, delta = acc.update(
            ResponseTextDeltaEvent(
                content_index=0,
                item_id="i1",
                output_index=0,
                delta="Hello",
                type="response.output_text.delta",
                sequence_number=1,
                logprobs=[],
            )
        )
        assert blocks == [TextChunk(content="Hello")], f"Got {blocks}"
        assert delta == "Hello", f"Got '{delta}'"

        blocks, delta = acc.update(
            ResponseTextDeltaEvent(
                content_index=0,
                item_id="i1",
                output_index=0,
                delta=" world",
                type="response.output_text.delta",
                sequence_number=2,
                logprobs=[],
            )
        )
        assert delta == " world", f"Got '{delta}'"

    def test_function_call_flow(self):
        """Test function call accumulation: added -> delta -> done.

        Test scenario:
            Full lifecycle of a function call through streaming events.
        """
        acc = ResponsesStreamAccumulator()

        tool_call_item = ResponseFunctionToolCall(
            id="fc_1",
            call_id="call_123",
            type="function_call",
            name="test_func",
            arguments="",
            status="in_progress",
        )

        acc.update(
            ResponseOutputItemAddedEvent(
                item=tool_call_item,
                output_index=0,
                sequence_number=1,
                type="response.output_item.added",
            )
        )

        blocks, _ = acc.update(
            ResponseFunctionCallArgumentsDeltaEvent(
                item_id="call_123",
                output_index=0,
                type="response.function_call_arguments.delta",
                delta='{"arg": "value"',
                sequence_number=2,
            )
        )
        assert blocks == [], "No blocks emitted during argument deltas"

        blocks, _ = acc.update(
            ResponseFunctionCallArgumentsDoneEvent(
                name="test_func",
                item_id="call_123",
                output_index=0,
                type="response.function_call_arguments.done",
                arguments='{"arg": "value"}',
                sequence_number=3,
            )
        )
        tool_blocks = [b for b in blocks if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 1, f"Expected 1, got {len(tool_blocks)}"
        assert tool_blocks[0].tool_name == "test_func", (
            f"Got '{tool_blocks[0].tool_name}'"
        )
        assert tool_blocks[0].tool_kwargs == '{"arg": "value"}', (
            f"Got '{tool_blocks[0].tool_kwargs}'"
        )
        assert tool_blocks[0].tool_call_id == "call_123", (
            f"Got '{tool_blocks[0].tool_call_id}'"
        )

    def test_annotation_event(self):
        """Test text annotation event is stored in additional_kwargs.

        Test scenario:
            Annotation added event should appear in annotations list.
        """
        acc = ResponsesStreamAccumulator()
        acc.update(
            ResponseOutputTextAnnotationAddedEvent(
                item_id="i1",
                output_index=0,
                content_index=0,
                annotation_index=0,
                type="response.output_text.annotation.added",
                annotation={
                    "type": "url",
                    "url": "https://example.com",
                },
                sequence_number=1,
            )
        )
        assert "annotations" in acc.additional_kwargs, (
            "Should have annotations key"
        )
        assert len(acc.additional_kwargs["annotations"]) == 1, (
            f"Expected 1, got {len(acc.additional_kwargs['annotations'])}"
        )
        assert acc.additional_kwargs["annotations"][0]["type"] == "url", (
            f"Got '{acc.additional_kwargs['annotations'][0]['type']}'"
        )

    def test_reasoning_item_done_event(self):
        """Test reasoning item done event creates ThinkingBlock.

        Test scenario:
            ResponseOutputItemDoneEvent with ReasoningItem.
        """
        acc = ResponsesStreamAccumulator()
        blocks, _ = acc.update(
            ResponseOutputItemDoneEvent(
                item=ResponseReasoningItem(
                    id="r1",
                    summary=[],
                    type="reasoning",
                    content=[
                        Content(text="first", type="reasoning_text"),
                        Content(text="second", type="reasoning_text"),
                    ],
                    encrypted_content=None,
                    status=None,
                ),
                output_index=0,
                sequence_number=1,
                type="response.output_item.done",
            )
        )
        thinking = [b for b in blocks if isinstance(b, ThinkingBlock)]
        assert len(thinking) == 1, f"Expected 1, got {len(thinking)}"
        assert thinking[0].content == "first\nsecond", (
            f"Got '{thinking[0].content}'"
        )


# ===========================================================================
# TestMessageConversion
# ===========================================================================


@pytest.mark.unit
class TestMessageConversion:
    """Tests for to_openai_message_dicts with is_responses_api=True."""

    def test_system_becomes_developer(self):
        """Test that SYSTEM role is converted to 'developer' for Responses API.

        Test scenario:
            System message -> developer role.
        """
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                chunks=[TextChunk(content="You are helpful.")],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert result[0]["role"] == "developer", f"Got '{result[0]['role']}'"

    def test_single_user_message_returns_string(self):
        """Test that a single USER message returns a plain string.

        Test scenario:
            Responses API optimises a lone user message to a raw string.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Hello")],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert result == "Hello", f"Got '{result}'"

    def test_user_message_in_multi_turn(self):
        """Test that USER role is preserved in multi-turn conversation.

        Test scenario:
            System + user messages produce a list with user dict.
        """
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                chunks=[TextChunk(content="Be helpful")],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Hello")],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert result[1]["role"] == "user", f"Got '{result[1]['role']}'"
        assert result[1]["content"] == "Hello", (
            f"Got '{result[1]['content']}'"
        )

    def test_tool_call_format(self):
        """Test tool call is formatted as flat function_call dict.

        Test scenario:
            ToolCallBlock should become a flat dict with type=function_call.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Use the tool")],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="tc_1",
                        tool_name="my_tool",
                        tool_kwargs='{"x": 1}',
                    )
                ],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        tc = result[1]
        assert tc["type"] == "function_call", f"Got '{tc['type']}'"
        assert tc["call_id"] == "tc_1", f"Got '{tc['call_id']}'"
        assert tc["name"] == "my_tool", f"Got '{tc['name']}'"
        assert tc["arguments"] == '{"x": 1}', (
            f"Got '{tc['arguments']}'"
        )

    def test_thinking_block_becomes_reasoning(self):
        """Test ThinkingBlock is converted to reasoning item.

        Test scenario:
            ThinkingBlock with additional_information should map to reasoning.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Think about it")],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ThinkingBlock(
                        content="I need to think",
                        additional_information={"id": "reason_abc"},
                    ),
                    TextChunk(content="Answer"),
                ],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        reasoning_items = [
            r for r in result
            if isinstance(r, dict) and r.get("type") == "reasoning"
        ]
        assert len(reasoning_items) == 1, (
            f"Expected 1, got {len(reasoning_items)}"
        )
        assert reasoning_items[0]["id"] == "reason_abc", (
            f"Got '{reasoning_items[0]['id']}'"
        )

    def test_complex_conversation(self):
        """Test conversion of a multi-turn conversation.

        Test scenario:
            System + user + assistant(tool_call) + assistant(text) messages.
        """
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                chunks=[TextChunk(content="You are a helpful assistant.")],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Capital of France?")],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ToolCallBlock(
                        tool_call_id="1",
                        tool_name="get_capital",
                        tool_kwargs="{'state': 'France'}",
                    )
                ],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="Paris")],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert len(result) == 4, f"Expected 4 messages, got {len(result)}"
        assert result[0]["role"] == "developer", f"Got '{result[0]['role']}'"
        assert result[1]["role"] == "user", f"Got '{result[1]['role']}'"
        assert result[2]["type"] == "function_call", (
            f"Got '{result[2].get('type')}'"
        )
        assert result[3]["role"] == "assistant", f"Got '{result[3]['role']}'"


# ===========================================================================
# TestGetModelName (from ModelMetadata mixin)
# ===========================================================================


@pytest.mark.unit
class TestGetModelName:
    """Tests for _get_model_name() inherited from ModelMetadata."""

    def test_standard_model(self):
        """Test standard model name is returned as-is.

        Test scenario:
            model='gpt-4o-mini' -> 'gpt-4o-mini'.
        """
        llm = _make_llm(model="gpt-4o-mini")
        assert llm._get_model_name() == "gpt-4o-mini", (
            f"Got '{llm._get_model_name()}'"
        )

    def test_legacy_finetune_format(self):
        """Test legacy fine-tuning format extracts base model.

        Test scenario:
            'ft-model:gpt-4' -> 'ft-model'.
        """
        llm = _make_llm(model="ft-model:gpt-4")
        assert llm._get_model_name() == "ft-model", (
            f"Got '{llm._get_model_name()}'"
        )

    def test_new_finetune_format(self):
        """Test new fine-tuning format extracts base model.

        Test scenario:
            'ft:gpt-4:org:custom:id' -> 'gpt-4'.
        """
        llm = _make_llm(model="ft:gpt-4:org:custom:id")
        assert llm._get_model_name() == "gpt-4", (
            f"Got '{llm._get_model_name()}'"
        )
