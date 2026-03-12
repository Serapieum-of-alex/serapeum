"""End-to-end tests for the OpenAI Responses API provider class.

These tests hit the real OpenAI API and require OPENAI_API_KEY to be set.
Run with: python -m pytest libs/providers/openai/tests/test_openai_responses_e2e.py -v -m e2e

Targets ``serapeum.openai.llm.responses.Responses`` and covers:
- Sync/async chat (non-streaming and streaming)
- Sync/async completion via ChatToCompletion mixin
- Structured outputs via parse/aparse (non-streaming and streaming)
- Function calling: single, parallel, tool_required, strict mode
- Tool execution round-trip (call -> result -> final answer)
- Stateful conversation continuation (track_previous_responses)
- Built-in tools (web_search)
- Token usage metadata
- Response structure invariants (role, chunks, deltas)
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from serapeum.core.llms import (
    ChatResponse,
    Message,
    MessageRole,
    TextChunk,
    ToolCallBlock,
)
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import CallableTool
from serapeum.openai.llm.responses import Responses

load_dotenv()

_has_key = os.getenv("OPENAI_API_KEY") is not None
_api_base = os.environ.get("OPENAI_API_BASE", "")
_is_azure = "azure" in _api_base.lower() or "cognitiveservices" in _api_base.lower()
_skip = not _has_key or _is_azure

skip_no_key = pytest.mark.skipif(
    _skip, reason="OPENAI_API_KEY not set or Azure endpoint"
)


# ---------------------------------------------------------------------------
# Pydantic models for structured output tests
# ---------------------------------------------------------------------------


class Person(BaseModel):
    """Simple person model for structured output tests."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


class MathResult(BaseModel):
    """Math operation result for structured output tests."""

    operation: str = Field(description="The math operation performed")
    result: float = Field(description="The numeric result")


class CityInfo(BaseModel):
    """City information for structured output tests."""

    name: str = Field(description="The city name")
    country: str = Field(description="The country the city is in")
    population: int | None = Field(default=None, description="Approximate population")


# ---------------------------------------------------------------------------
# Tool definitions for function calling tests
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum of a and b.
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The product of a and b.
    """
    return a * b


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.

    Returns:
        A weather description string.
    """
    return f"The weather in {city} is sunny and 22°C."


add_tool = CallableTool.from_function(func=add)
multiply_tool = CallableTool.from_function(func=multiply)
weather_tool = CallableTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model() -> str:
    """Return the OpenAI model name used for e2e tests.

    Returns:
        str: Model name, defaults to gpt-4o-mini.
    """
    return os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")


@pytest.fixture
def llm(model: str) -> Responses:
    """Create a Responses instance for e2e tests.

    Returns:
        Responses: Configured instance.
    """
    return Responses(model=model)


@pytest.fixture
def llm_strict(model: str) -> Responses:
    """Create a Responses instance with strict mode for e2e tests.

    Returns:
        Responses: Configured instance with strict=True.
    """
    return Responses(model=model, strict=True)


# ===========================================================================
# TestResponsesChat
# ===========================================================================


@pytest.mark.e2e
class TestResponsesChat:
    """E2e tests for Responses.chat() (sync, streaming and non-streaming)."""

    @skip_no_key
    def test_chat_non_streaming(self, llm: Responses):
        """Test basic non-streaming chat returns a valid response.

        Test scenario:
            Send a simple user message; verify assistant response structure.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello in exactly one word")],
            )
        ]
        response = llm.chat(messages)

        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT, got {response.message.role}"
        assert len(response.message.chunks) > 0, "Response should have chunks"
        assert response.message.content, "Response should have content"

    @skip_no_key
    def test_chat_streaming(self, llm: Responses):
        """Test streaming chat yields incremental chunks.

        Test scenario:
            Stream a response; verify chunks have deltas and correct role.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Count from 1 to 3")],
            )
        ]
        gen = llm.chat(messages, stream=True)
        chunks = list(gen)

        assert len(chunks) > 0, "Should yield at least one chunk"
        assert all(
            c.message.role == MessageRole.ASSISTANT for c in chunks
        ), "All chunks should have ASSISTANT role"
        accumulated = "".join(c.delta for c in chunks if c.delta is not None)
        assert len(accumulated) > 0, "Accumulated content should not be empty"

    @skip_no_key
    def test_chat_with_system_message(self, llm: Responses):
        """Test chat with a system/developer message.

        Test scenario:
            Send system + user messages; verify the system instruction
            affects the response (at least structurally valid).
        """
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                chunks=[
                    TextChunk(content="You must always respond in uppercase only.")
                ],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello")],
            ),
        ]
        response = llm.chat(messages)
        assert response.message.content, "Should have content"

    @skip_no_key
    def test_chat_response_has_usage(self, llm: Responses):
        """Test that chat response includes usage metadata.

        Test scenario:
            Verify additional_kwargs contains usage after a chat call.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Hi")],
            )
        ]
        response = llm.chat(messages)
        assert (
            "usage" in response.additional_kwargs
        ), "Response should include usage metadata"

    @skip_no_key
    def test_chat_response_has_raw(self, llm: Responses):
        """Test that chat response has raw SDK response attached.

        Test scenario:
            Verify raw field is not None after a chat call.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Hi")],
            )
        ]
        response = llm.chat(messages)
        assert response.raw is not None, "raw should be set"


# ===========================================================================
# TestResponsesAChat
# ===========================================================================


@pytest.mark.e2e
class TestResponsesAChat:
    """E2e tests for Responses.achat() (async, streaming and non-streaming)."""

    @skip_no_key
    @pytest.mark.asyncio
    async def test_achat_non_streaming(self, llm: Responses):
        """Test async non-streaming chat returns a valid response.

        Test scenario:
            Send a simple user message async; verify assistant response.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello")],
            )
        ]
        response = await llm.achat(messages)

        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT, got {response.message.role}"
        assert response.message.content, "Response should have content"

    @skip_no_key
    @pytest.mark.asyncio
    async def test_achat_streaming(self, llm: Responses):
        """Test async streaming chat yields incremental chunks.

        Test scenario:
            Stream async; verify chunks and accumulated content.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Count from 1 to 3")],
            )
        ]
        gen = await llm.achat(messages, stream=True)
        chunks = [c async for c in gen]

        assert len(chunks) > 0, "Should yield at least one chunk"
        assert all(
            c.message.role == MessageRole.ASSISTANT for c in chunks
        ), "All chunks should have ASSISTANT role"
        accumulated = "".join(c.delta for c in chunks if c.delta is not None)
        assert len(accumulated) > 0, "Accumulated content should not be empty"


# ===========================================================================
# TestResponsesComplete
# ===========================================================================


@pytest.mark.e2e
class TestResponsesComplete:
    """E2e tests for Responses.complete() (ChatToCompletion mixin)."""

    @skip_no_key
    def test_complete_non_streaming(self, llm: Responses):
        """Test non-streaming completion returns text.

        Test scenario:
            Simple completion; verify text is non-empty.
        """
        response = llm.complete("Write a one-sentence summary of Python.")
        assert response.text is not None, "text should not be None"
        assert len(response.text) > 0, "text should not be empty"

    @skip_no_key
    def test_complete_streaming(self, llm: Responses):
        """Test streaming completion yields chunks.

        Test scenario:
            Stream completion; verify at least one chunk with text.
        """
        gen = llm.complete(stream=True, prompt="Count to 3 briefly.")
        chunks = list(gen)
        assert len(chunks) > 0, "Should yield at least one chunk"
        assert chunks[-1].text is not None, "Last chunk should have text"


# ===========================================================================
# TestResponsesAComplete
# ===========================================================================


@pytest.mark.e2e
class TestResponsesAComplete:
    """E2e tests for Responses.acomplete() (async ChatToCompletion mixin)."""

    @skip_no_key
    @pytest.mark.asyncio
    async def test_acomplete_non_streaming(self, llm: Responses):
        """Test async non-streaming completion.

        Test scenario:
            Simple async completion; verify text.
        """
        response = await llm.acomplete("Write a one-sentence summary of Python.")
        assert response.text is not None, "text should not be None"
        assert len(response.text) > 0, "text should not be empty"

    @skip_no_key
    @pytest.mark.asyncio
    async def test_acomplete_streaming(self, llm: Responses):
        """Test async streaming completion.

        Test scenario:
            Stream async completion; verify chunks.
        """
        gen = await llm.acomplete(stream=True, prompt="Count to 3 briefly.")
        chunks = [c async for c in gen]
        assert len(chunks) > 0, "Should yield at least one chunk"
        assert chunks[-1].text is not None, "Last chunk should have text"


# ===========================================================================
# TestResponsesParse
# ===========================================================================


@pytest.mark.e2e
class TestResponsesParse:
    """E2e tests for Responses.parse() (structured output via tool forcing)."""

    @skip_no_key
    def test_parse_returns_model_instance(self, llm: Responses):
        """Test parse returns a valid Pydantic model instance.

        Test scenario:
            Ask for a Person named Alice age 25; verify field values.
        """
        result = llm.parse(
            Person,
            PromptTemplate(
                "Create a profile for a person named Alice who is 25 years old"
            ),
        )
        assert isinstance(result, Person), f"Expected Person, got {type(result)}"
        assert result.name == "Alice", f"Expected 'Alice', got '{result.name}'"
        assert result.age == 25, f"Expected 25, got {result.age}"

    @skip_no_key
    def test_parse_complex_model(self, llm: Responses):
        """Test parse with a more complex model.

        Test scenario:
            Ask for CityInfo about Paris; verify structure.
        """
        result = llm.parse(
            CityInfo,
            PromptTemplate(
                "Give me info about Paris, France. " "It has about 2 million people."
            ),
        )
        assert isinstance(result, CityInfo), f"Expected CityInfo, got {type(result)}"
        assert (
            "paris" in result.name.lower()
        ), f"Expected 'Paris' in name, got '{result.name}'"
        assert (
            "france" in result.country.lower()
        ), f"Expected 'France' in country, got '{result.country}'"

    @skip_no_key
    def test_parse_streaming(self, llm: Responses):
        """Test parse with stream=True yields partial models.

        Test scenario:
            Stream structured output; final yield should be a valid Person.
        """
        gen = llm.parse(
            Person,
            PromptTemplate("Create a profile for Bob who is 30 years old"),
            stream=True,
        )
        chunks = list(gen)
        assert len(chunks) > 0, "Should yield at least one chunk"
        final = chunks[-1]
        assert final is not None, "Final chunk should not be None"

    @skip_no_key
    def test_parse_math_result(self, llm: Responses):
        """Test parse with a math-oriented model.

        Test scenario:
            Ask for a MathResult of 7+3; verify operation and result.
        """
        result = llm.parse(
            MathResult,
            PromptTemplate("What is 7 + 3? Respond with the operation and result."),
        )
        assert isinstance(
            result, MathResult
        ), f"Expected MathResult, got {type(result)}"
        assert result.result == 10.0, f"Expected 10.0, got {result.result}"


# ===========================================================================
# TestResponsesAParse
# ===========================================================================


@pytest.mark.e2e
class TestResponsesAParse:
    """E2e tests for Responses.aparse() (async structured output)."""

    @skip_no_key
    @pytest.mark.asyncio
    async def test_aparse_returns_model(self, llm: Responses):
        """Test async parse returns a valid Pydantic model.

        Test scenario:
            Ask for a Person named Bob age 40; verify field values.
        """
        result = await llm.aparse(
            Person,
            PromptTemplate(
                "Create a profile for a person named Bob who is 40 years old"
            ),
        )
        assert isinstance(result, Person), f"Expected Person, got {type(result)}"
        assert result.name == "Bob", f"Expected 'Bob', got '{result.name}'"
        assert result.age == 40, f"Expected 40, got {result.age}"

    @skip_no_key
    @pytest.mark.asyncio
    async def test_aparse_streaming(self, llm: Responses):
        """Test async parse with stream=True.

        Test scenario:
            Stream async structured output; verify chunks yielded.
        """
        gen = await llm.aparse(
            Person,
            PromptTemplate("Create a profile for Carol who is 35 years old"),
            stream=True,
        )
        chunks = [c async for c in gen]
        assert len(chunks) > 0, "Should yield at least one chunk"


# ===========================================================================
# TestResponsesFunctionCalling
# ===========================================================================


@pytest.mark.e2e
class TestResponsesFunctionCalling:
    """E2e tests for function calling via Responses API."""

    @skip_no_key
    def test_generate_tool_calls_single(self, llm: Responses):
        """Test generate_tool_calls produces a single tool call.

        Test scenario:
            Ask to add 2+3 with add tool; verify tool call block.
        """
        response = llm.generate_tool_calls(
            tools=[add_tool],
            message="What is 2 + 3?",
        )
        tool_calls = response.message.tool_calls
        assert (
            len(tool_calls) >= 1
        ), f"Expected at least 1 tool call, got {len(tool_calls)}"
        assert (
            tool_calls[0].tool_name == "add"
        ), f"Expected 'add', got '{tool_calls[0].tool_name}'"

    @skip_no_key
    def test_generate_tool_calls_with_tool_required(self, llm: Responses):
        """Test tool_required=True forces a tool call.

        Test scenario:
            Even for a simple question, tool_required should force a call.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the capital of France?",
            tool_required=True,
        )
        tool_calls = response.message.tool_calls
        assert (
            len(tool_calls) >= 1
        ), f"Expected at least 1 tool call, got {len(tool_calls)}"

    @skip_no_key
    def test_generate_tool_calls_parallel(self, llm: Responses):
        """Test parallel tool calls when allow_parallel_tool_calls=True.

        Test scenario:
            Ask to add AND multiply; both tools should be called.
        """
        response = llm.generate_tool_calls(
            tools=[add_tool, multiply_tool],
            message=(
                "Calculate both 2+3 and 4*5. "
                "Use the add tool for addition and multiply tool for multiplication."
            ),
            allow_parallel_tool_calls=True,
            tool_required=True,
        )
        tool_calls = response.message.tool_calls
        assert (
            len(tool_calls) >= 2
        ), f"Expected at least 2 tool calls, got {len(tool_calls)}"

    @skip_no_key
    def test_generate_tool_calls_streaming(self, llm: Responses):
        """Test generate_tool_calls with stream=True.

        Test scenario:
            Stream tool call generation; verify chunks yielded.
        """
        gen = llm.generate_tool_calls(
            tools=[add_tool],
            message="What is 10 + 20?",
            stream=True,
            tool_required=True,
        )
        chunks = list(gen)
        assert len(chunks) > 0, "Should yield at least one chunk"

    @skip_no_key
    def test_get_tool_calls_from_response(self, llm: Responses):
        """Test extracting and parsing tool call arguments from a response.

        Test scenario:
            Generate tool calls then extract; verify parsed kwargs.
        """
        response = llm.generate_tool_calls(
            tools=[add_tool],
            message="What is 5 + 7?",
            tool_required=True,
        )
        selections = llm.get_tool_calls_from_response(response)
        assert len(selections) >= 1, f"Expected at least 1, got {len(selections)}"
        assert (
            selections[0].tool_name == "add"
        ), f"Expected 'add', got '{selections[0].tool_name}'"
        kwargs = selections[0].tool_kwargs
        assert isinstance(kwargs, dict), f"Expected dict, got {type(kwargs)}"
        assert (
            "a" in kwargs and "b" in kwargs
        ), f"Expected 'a' and 'b' keys, got {kwargs}"

    @skip_no_key
    def test_invoke_callable_round_trip(self, llm: Responses):
        """Test full tool execution round-trip: call -> execute -> return.

        Test scenario:
            Use invoke_callable with add tool; verify the tool output.
        """
        result = llm.invoke_callable(
            tools=[add_tool],
            message="What is 100 + 200?",
            tool_required=True,
        )
        assert (
            "300" in result.response
        ), f"Expected '300' in response, got '{result.response}'"

    @skip_no_key
    def test_strict_mode_tool_calling(self, llm_strict: Responses):
        """Test tool calling with strict mode enabled.

        Test scenario:
            strict=True should set strict on tool specs; call should still work.
        """
        response = llm_strict.generate_tool_calls(
            tools=[add_tool],
            message="What is 8 + 9?",
            tool_required=True,
        )
        tool_calls = response.message.tool_calls
        assert (
            len(tool_calls) >= 1
        ), f"Expected at least 1 tool call, got {len(tool_calls)}"


# ===========================================================================
# TestResponsesAsyncFunctionCalling
# ===========================================================================


@pytest.mark.e2e
class TestResponsesAsyncFunctionCalling:
    """E2e tests for async function calling via Responses API."""

    @skip_no_key
    @pytest.mark.asyncio
    async def test_agenerate_tool_calls(self, llm: Responses):
        """Test async generate_tool_calls.

        Test scenario:
            Async tool call generation; verify tool call block.
        """
        response = await llm.agenerate_tool_calls(
            tools=[add_tool],
            message="What is 3 + 4?",
            tool_required=True,
        )
        tool_calls = response.message.tool_calls
        assert len(tool_calls) >= 1, f"Expected at least 1, got {len(tool_calls)}"
        assert (
            tool_calls[0].tool_name == "add"
        ), f"Expected 'add', got '{tool_calls[0].tool_name}'"

    @skip_no_key
    @pytest.mark.asyncio
    async def test_agenerate_tool_calls_streaming(self, llm: Responses):
        """Test async streaming tool call generation.

        Test scenario:
            Stream async tool call generation.
        """
        gen = await llm.agenerate_tool_calls(
            tools=[add_tool],
            message="What is 6 + 7?",
            stream=True,
            tool_required=True,
        )
        chunks = [c async for c in gen]
        assert len(chunks) > 0, "Should yield at least one chunk"

    @skip_no_key
    @pytest.mark.asyncio
    async def test_ainvoke_callable_round_trip(self, llm: Responses):
        """Test async full tool execution round-trip.

        Test scenario:
            Use ainvoke_callable with add tool; verify the tool output.
        """
        result = await llm.ainvoke_callable(
            tools=[add_tool],
            message="What is 50 + 50?",
            tool_required=True,
        )
        assert (
            "100" in result.response
        ), f"Expected '100' in response, got '{result.response}'"


# ===========================================================================
# TestResponsesTrackPreviousResponses
# ===========================================================================


@pytest.mark.e2e
class TestResponsesTrackPreviousResponses:
    """E2e tests for stateful conversation via track_previous_responses."""

    @skip_no_key
    def test_stateful_conversation(self):
        """Test that track_previous_responses enables multi-turn context.

        Test scenario:
            Turn 1: Introduce a name. Turn 2: Ask what name was said.
            The model should remember from the previous response.
        """
        llm = Responses(
            model="gpt-4o-mini",
            track_previous_responses=True,
        )
        assert llm.store is True, "store should be True when tracking responses"

        r1 = llm.chat(
            [
                Message(
                    role=MessageRole.USER,
                    chunks=[
                        TextChunk(
                            content="My secret code word is 'pineapple'. "
                            "Remember it."
                        )
                    ],
                )
            ]
        )
        assert (
            llm._previous_response_id is not None
        ), "Should have a previous response ID after first call"
        first_id = llm._previous_response_id

        r2 = llm.chat(
            [
                Message(
                    role=MessageRole.USER,
                    chunks=[TextChunk(content="What was my secret code word?")],
                )
            ]
        )
        assert (
            "pineapple" in r2.message.content.lower()
        ), f"Model should remember 'pineapple', got: {r2.message.content}"
        assert (
            llm._previous_response_id != first_id
        ), "Previous response ID should be updated after second call"


# ===========================================================================
# TestResponsesBuiltInTools
# ===========================================================================


@pytest.mark.e2e
class TestResponsesBuiltInTools:
    """E2e tests for built-in tool usage (web_search, etc.)."""

    @skip_no_key
    def test_web_search_tool(self):
        """Test chat with web_search built-in tool.

        Test scenario:
            Ask a current-events question with web_search enabled.
        """
        llm = Responses(
            model="gpt-4o-mini",
            built_in_tools=[{"type": "web_search_preview"}],
        )
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="What is today's date?")],
            )
        ]
        response = llm.chat(messages)
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT, got {response.message.role}"
        assert len(response.message.chunks) > 0, "Should have chunks"


# ===========================================================================
# TestResponsesMetadata
# ===========================================================================


@pytest.mark.e2e
class TestResponsesMetadata:
    """E2e tests for Responses metadata and configuration."""

    @skip_no_key
    def test_class_name(self, llm: Responses):
        """Test class_name returns expected identifier.

        Test scenario:
            Verify the canonical class name string.
        """
        assert llm.class_name() == "openai_responses_llm", f"Got '{llm.class_name()}'"

    @skip_no_key
    def test_metadata_properties(self, llm: Responses):
        """Test metadata reflects the model configuration.

        Test scenario:
            Verify metadata fields for a real model.
        """
        meta = llm.metadata
        assert meta.is_chat_model is True, "Should be a chat model"
        assert (
            meta.context_window > 0
        ), f"Expected positive context window, got {meta.context_window}"
        assert (
            meta.model_name == llm.model
        ), f"Expected '{llm.model}', got '{meta.model_name}'"

    @skip_no_key
    def test_should_use_structure_outputs_false(self, llm: Responses):
        """Test _should_use_structure_outputs always returns False.

        Test scenario:
            Responses API never uses native JSON-schema path.
        """
        assert (
            llm._should_use_structure_outputs() is False
        ), "Should always be False for Responses API"
