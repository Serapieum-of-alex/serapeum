"""End-to-end tests for the OpenAI Chat Completions provider class.

These tests hit the real OpenAI API and require OPENAI_API_KEY to be set.
Run with: python -m pytest libs/providers/openai/tests/test_openai_e2e.py -v -m e2e

Targets ``serapeum.openai.llm.chat_completions.OpenAI`` and covers:
- Sync/async chat (non-streaming and streaming)
- Sync/async completion via ChatToCompletion mixin
- Structured outputs via parse/aparse (non-streaming and streaming)
- Function calling: single, parallel, specific choice, tool_required, strict mode
- Tool execution round-trip (call → result → final answer)
- Streaming with tool calls
- Token usage metadata
- Model validation and configuration
- Audio modality rejection
- Response structure invariants (role, chunks, deltas)
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from serapeum.core.base.llms.types import (
    ChatResponse,
    Message,
    MessageRole,
    TextChunk,
    ToolCallBlock,
)
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import CallableTool

load_dotenv()

_has_key = os.getenv("OPENAI_API_KEY") is not None
skip_no_key = pytest.mark.skipif(not _has_key, reason="OPENAI_API_KEY not set")


@pytest.fixture(scope="session")
def model():
    """Return the OpenAI model name used for e2e tests.

    Returns:
        str: Model name from OPENAI_MODEL env var or default.
    """
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@pytest.fixture()
def llm(model):
    """Create a default OpenAI instance for e2e tests.

    Returns:
        OpenAI: Instance configured from environment variables.
    """
    from serapeum.openai import OpenAI

    return OpenAI(
        model=model,
        api_base=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


@pytest.fixture()
def llm_with_logprobs(model):
    """Create an OpenAI instance with logprobs enabled.

    Returns:
        OpenAI: Instance with logprobs=True and top_logprobs=3.
    """
    from serapeum.openai import OpenAI

    return OpenAI(
        model=model,
        api_base=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        logprobs=True,
        top_logprobs=3,
    )


class Person(BaseModel):
    """A person with name and age."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


class Address(BaseModel):
    """A physical address."""
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class PersonWithAddress(BaseModel):
    """A person with a nested address."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    address: Address = Field(description="The person's home address")


def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny and 22C."


def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    return f"Search results for: {query}"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b


weather_tool = CallableTool.from_function(
    func=get_weather,
    name="get_weather",
    description="Get the current weather for a city",
)
search_tool = CallableTool.from_function(
    func=search_web,
    name="search_web",
    description="Search the web for information",
)
add_tool = CallableTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Add two numbers together",
)


@pytest.mark.e2e
@skip_no_key
class TestSyncChat:
    """Tests for OpenAI.chat (non-streaming)."""

    def test_single_turn(self, llm):
        """Test single-turn chat returns a non-empty assistant response.

        Test scenario:
            Send a simple user message and verify the response has content,
            the correct role, and contains at least one TextChunk.
        """
        response = llm.chat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello in one word.")],
            )]
        )
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )
        assert response.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {response.message.role}"
        )
        assert response.message.content is not None, "Content should not be None"
        assert len(response.message.content) > 0, "Content should not be empty"
        text_chunks = [c for c in response.message.chunks if isinstance(c, TextChunk)]
        assert len(text_chunks) >= 1, "Should contain at least one TextChunk"

    def test_with_system_prompt(self, llm):
        """Test that a system prompt influences the response.

        Test scenario:
            Instruct via system prompt to respond as a pirate.
            Verify the model produces a non-empty response.
        """
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                chunks=[TextChunk(
                    content="You are a pirate. Always respond like a pirate."
                )],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello.")],
            ),
        ]
        response = llm.chat(messages)
        assert response.message.content is not None, "Content should not be None"
        assert len(response.message.content) > 0, "Content should not be empty"

    def test_multi_turn_preserves_context(self, llm):
        """Test that multi-turn conversation preserves earlier context.

        Test scenario:
            Introduce a name in the first turn, then ask the model to recall it.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="My name is Alice.")],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="Nice to meet you, Alice!")],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="What is my name?")],
            ),
        ]
        response = llm.chat(messages)
        assert "Alice" in response.message.content, (
            f"Expected 'Alice' in response, got: {response.message.content}"
        )

    def test_raw_response_populated(self, llm):
        """Test that the raw API response object is attached.

        Test scenario:
            The raw field should contain the original SDK response object.
        """
        response = llm.chat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hi.")],
            )]
        )
        assert response.raw is not None, "raw should contain the SDK response"


@pytest.mark.e2e
@skip_no_key
class TestAsyncChat:
    """Tests for OpenAI.achat (non-streaming)."""

    @pytest.mark.asyncio()
    async def test_single_turn(self, llm):
        """Test async single-turn chat returns a non-empty response.

        Test scenario:
            Same as sync test but via achat.
        """
        response = await llm.achat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hello in one word.")],
            )]
        )
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )
        assert response.message.content is not None, "Content should not be None"
        assert len(response.message.content) > 0, "Content should not be empty"

    @pytest.mark.asyncio()
    async def test_multi_turn(self, llm):
        """Test async multi-turn conversation preserves context.

        Test scenario:
            Provide name context and verify model recalls it asynchronously.
        """
        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="My name is Bob.")],
            ),
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="Hi Bob!")],
            ),
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="What did I say my name was?")],
            ),
        ]
        response = await llm.achat(messages)
        assert "Bob" in response.message.content, (
            f"Expected 'Bob' in response, got: {response.message.content}"
        )


@pytest.mark.e2e
@skip_no_key
class TestSyncStreamChat:
    """Tests for OpenAI.chat(stream=True)."""

    def test_yields_multiple_chunks(self, llm):
        """Test that streaming chat yields multiple chunks.

        Test scenario:
            Ask the model to count — the response should arrive in multiple chunks
            with non-empty deltas.
        """
        gen = llm.chat(
            messages=[Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Count from 1 to 5.")],
            )],
            stream=True,
        )
        chunks = list(gen)
        assert len(chunks) > 1, (
            f"Expected multiple chunks, got {len(chunks)}"
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert len(deltas) > 0, "At least one chunk should have a non-empty delta"

    def test_accumulated_content(self, llm):
        """Test that each chunk's content is the cumulative text.

        Test scenario:
            Each successive chunk's message.content should grow or remain stable
            (never shrink), and the final chunk should contain all text.
        """
        gen = llm.chat(
            messages=[Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say the word 'hello'.")],
            )],
            stream=True,
        )
        chunks = list(gen)
        prev_len = 0
        for chunk in chunks:
            content = chunk.message.content or ""
            assert len(content) >= prev_len, (
                f"Content length should not decrease: {len(content)} < {prev_len}"
            )
            prev_len = len(content)
        final = chunks[-1]
        assert final.message.content is not None, "Final chunk should have content"

    def test_chunk_has_raw(self, llm):
        """Test that each streaming chunk carries a raw response.

        Test scenario:
            Every yielded chunk should have its raw field populated with the
            underlying SDK chunk object.
        """
        gen = llm.chat(
            messages=[Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Hi.")],
            )],
            stream=True,
        )
        for chunk in gen:
            assert chunk.raw is not None, "Each chunk should have a raw response"


@pytest.mark.e2e
@skip_no_key
class TestAsyncStreamChat:
    """Tests for OpenAI.achat(stream=True)."""

    @pytest.mark.asyncio()
    async def test_yields_multiple_chunks(self, llm):
        """Test that async streaming chat yields multiple chunks.

        Test scenario:
            Same as sync streaming test but via achat(stream=True).
        """
        gen = await llm.achat(
            messages=[Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Count from 1 to 5.")],
            )],
            stream=True,
        )
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert len(chunks) > 1, (
            f"Expected multiple chunks, got {len(chunks)}"
        )
        deltas = [c.delta for c in chunks if c.delta]
        assert len(deltas) > 0, "At least one chunk should have a non-empty delta"

    @pytest.mark.asyncio()
    async def test_accumulated_content(self, llm):
        """Test that async streaming content accumulates monotonically.

        Test scenario:
            Each chunk's content length should be >= the previous one.
        """
        gen = await llm.achat(
            messages=[Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say 'world'.")],
            )],
            stream=True,
        )
        prev_len = 0
        async for chunk in gen:
            content = chunk.message.content or ""
            assert len(content) >= prev_len, (
                f"Content length should not decrease: {len(content)} < {prev_len}"
            )
            prev_len = len(content)


@pytest.mark.e2e
@skip_no_key
class TestSyncCompletion:
    """Tests for OpenAI.complete (non-streaming, via ChatToCompletion mixin)."""

    def test_complete_basic(self, llm):
        """Test that complete returns a non-empty text response.

        Test scenario:
            Call complete with a prompt string and verify the response text.
        """
        response = llm.complete("The capital of France is")
        assert response.text is not None, "Text should not be None"
        assert len(response.text) > 0, "Text should not be empty"

    def test_complete_stream(self, llm):
        """Test streaming completion yields multiple chunks with deltas.

        Test scenario:
            Streaming completion should yield multiple incremental chunks
            that accumulate into a final text.
        """
        gen = llm.complete(prompt="Count from 1 to 5:", stream=True)
        chunks = list(gen)
        assert len(chunks) > 1, (
            f"Expected multiple chunks, got {len(chunks)}"
        )
        final_text = chunks[-1].text
        assert final_text is not None, "Final text should not be None"
        assert len(final_text) > 0, "Final text should not be empty"

    def test_complete_stream_deltas_are_nonempty(self, llm):
        """Test that at least some streaming completion chunks have deltas.

        Test scenario:
            The delta field should contain incremental text fragments for
            at least some chunks.
        """
        gen = llm.complete(prompt="Tell me a short fact.", stream=True)
        deltas = [c.delta for c in gen if c.delta]
        assert len(deltas) > 0, "At least one chunk should have a non-empty delta"


@pytest.mark.e2e
@skip_no_key
class TestAsyncCompletion:
    """Tests for OpenAI.acomplete."""

    @pytest.mark.asyncio()
    async def test_acomplete_basic(self, llm):
        """Test async completion returns a non-empty text response.

        Test scenario:
            Same as sync complete but via acomplete.
        """
        response = await llm.acomplete("The capital of Germany is")
        assert response.text is not None, "Text should not be None"
        assert len(response.text) > 0, "Text should not be empty"

    @pytest.mark.asyncio()
    async def test_acomplete_stream(self, llm):
        """Test async streaming completion yields multiple chunks.

        Test scenario:
            Async streaming completion should yield multiple chunks via
            an async generator.
        """
        gen = await llm.acomplete(prompt="Count from 1 to 3:", stream=True)
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert len(chunks) > 1, (
            f"Expected multiple chunks, got {len(chunks)}"
        )
        assert chunks[-1].text is not None, "Final text should not be None"


@pytest.mark.e2e
@skip_no_key
class TestParse:
    """Tests for OpenAI.parse (structured output prediction)."""

    def test_parse_flat_model(self, llm):
        """Test structured predict extracts a flat Pydantic model.

        Test scenario:
            Ask the model to create a person and verify the fields match.
        """
        prompt = PromptTemplate(
            "Create a person named Alice who is 30 years old."
        )
        result = llm.parse(schema=Person, prompt=prompt)
        assert isinstance(result, Person), (
            f"Expected Person, got {type(result)}"
        )
        assert result.name == "Alice", (
            f"Expected name 'Alice', got '{result.name}'"
        )
        assert result.age == 30, f"Expected age 30, got {result.age}"

    def test_parse_nested_model(self, llm):
        """Test structured predict handles nested Pydantic models.

        Test scenario:
            Create a person with a nested address and verify nested fields.
        """
        prompt = PromptTemplate(
            "Create a person named Bob, age 25, living at "
            "123 Main St, New York, USA."
        )
        result = llm.parse(schema=PersonWithAddress, prompt=prompt)
        assert isinstance(result, PersonWithAddress), (
            f"Expected PersonWithAddress, got {type(result)}"
        )
        assert result.name == "Bob", f"Expected name 'Bob', got '{result.name}'"
        assert result.age == 25, f"Expected age 25, got {result.age}"
        assert isinstance(result.address, Address), (
            f"Expected Address, got {type(result.address)}"
        )
        assert result.address.city == "New York", (
            f"Expected city 'New York', got '{result.address.city}'"
        )

    @pytest.mark.asyncio()
    async def test_aparse_flat_model(self, llm):
        """Test async structured predict extracts a Pydantic model.

        Test scenario:
            Same as sync parse but via aparse.
        """
        prompt = PromptTemplate(
            "Create a person named Carol who is 40 years old."
        )
        result = await llm.aparse(schema=Person, prompt=prompt)
        assert isinstance(result, Person), (
            f"Expected Person, got {type(result)}"
        )
        assert result.name == "Carol", (
            f"Expected name 'Carol', got '{result.name}'"
        )
        assert result.age == 40, f"Expected age 40, got {result.age}"

    def test_parse_streaming(self, llm):
        """Test streaming parse yields partial objects converging to final.

        Test scenario:
            Streaming parse should yield multiple partial objects; the last
            one should be a valid Person.
        """
        prompt = PromptTemplate(
            "Create a person named Dave who is 50 years old."
        )
        results = list(llm.parse(schema=Person, prompt=prompt, stream=True))
        assert len(results) > 0, "Streaming parse should yield at least one result"
        final = results[-1]
        assert isinstance(final, Person), (
            f"Expected Person, got {type(final)}"
        )
        assert final.name == "Dave", (
            f"Expected name 'Dave', got '{final.name}'"
        )

    @pytest.mark.asyncio()
    async def test_aparse_streaming(self, llm):
        """Test async streaming parse yields partial objects.

        Test scenario:
            Async streaming parse should yield multiple partial objects.
        """
        prompt = PromptTemplate(
            "Create a person named Eve who is 35 years old."
        )
        results = []
        async for partial in await llm.aparse(
            schema=Person, prompt=prompt, stream=True
        ):
            results.append(partial)
        assert len(results) > 0, (
            "Async streaming parse should yield at least one result"
        )
        final = results[-1]
        assert isinstance(final, Person), (
            f"Expected Person, got {type(final)}"
        )
        assert final.name == "Eve", (
            f"Expected name 'Eve', got '{final.name}'"
        )


@pytest.mark.e2e
@skip_no_key
class TestToolCallingSingleTool:
    """Tests for function calling with a single tool."""

    def test_tool_required_returns_tool_call(self, llm):
        """Test that tool_required=True forces a tool call.

        Test scenario:
            With tool_required=True the model must invoke the tool.
            Verify tool name and arguments are populated.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the weather in Paris?",
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 1, (
            f"Expected at least one tool call, got {len(tool_calls)}"
        )
        assert tool_calls[0].tool_name == "get_weather", (
            f"Expected tool 'get_weather', got '{tool_calls[0].tool_name}'"
        )
        assert "city" in tool_calls[0].tool_kwargs, (
            f"Expected 'city' in kwargs, got keys: {list(tool_calls[0].tool_kwargs)}"
        )

    def test_tool_call_id_populated(self, llm):
        """Test that each tool call has a non-empty tool_id.

        Test scenario:
            OpenAI always returns an id for each tool call.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the weather in London?",
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert tool_calls[0].tool_id is not None, "tool_id should not be None"
        assert len(tool_calls[0].tool_id) > 0, "tool_id should not be empty"

    def test_no_tool_needed_does_not_crash(self, llm):
        """Test that tool_required=False with irrelevant query works.

        Test scenario:
            The model may or may not call a tool — just verify no crash.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is 2 + 2?",
            tool_required=False,
        )
        assert response.message is not None, "Response message should exist"

    def test_error_on_no_tool_call_raises(self, llm):
        """Test that error_on_no_tool_call=True raises when no tool is invoked.

        Test scenario:
            Use tool_required=False with a query unlikely to trigger the tool,
            then request error_on_no_tool_call=True. If the model doesn't call
            a tool, ValueError should be raised; if it does, the test passes.
        """
        response = llm.generate_tool_calls(
            tools=[add_tool],
            message="Say hello.",
            tool_required=False,
        )
        tool_calls_present = bool(response.message.tool_calls)
        if not tool_calls_present:
            with pytest.raises(ValueError, match="Expected at least one tool call"):
                llm.get_tool_calls_from_response(
                    response, error_on_no_tool_call=True
                )


@pytest.mark.e2e
@skip_no_key
class TestToolCallingMultipleTools:
    """Tests for function calling with multiple tools."""

    def test_specific_tool_choice(self, llm):
        """Test that forcing a specific tool name selects that tool.

        Test scenario:
            Provide two tools but force tool_choice to search_web.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool, search_tool],
            message="Tell me about the Eiffel Tower",
            tool_choice="search_web",
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 1, (
            f"Expected at least one tool call, got {len(tool_calls)}"
        )
        assert tool_calls[0].tool_name == "search_web", (
            f"Expected 'search_web', got '{tool_calls[0].tool_name}'"
        )

    def test_parallel_tool_calls(self, llm):
        """Test that parallel tool calls returns multiple calls.

        Test scenario:
            Ask for weather in two cities with allow_parallel_tool_calls=True.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the weather in Paris and London?",
            allow_parallel_tool_calls=True,
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 2, (
            f"Expected at least 2 tool calls, got {len(tool_calls)}"
        )

    def test_strict_mode(self, llm):
        """Test that strict=True produces valid tool calls.

        Test scenario:
            Strict mode forces JSON schema adherence. Verify tool call succeeds.
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the weather in Tokyo?",
            tool_required=True,
            strict=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 1, (
            f"Expected at least one tool call, got {len(tool_calls)}"
        )
        assert tool_calls[0].tool_name == "get_weather", (
            f"Expected 'get_weather', got '{tool_calls[0].tool_name}'"
        )


@pytest.mark.e2e
@skip_no_key
class TestToolExecutionRoundTrip:
    """Tests for full tool call round-trip: call → execute → respond."""

    def test_tool_result_fed_back(self, llm):
        """Test complete tool round-trip: model calls tool, gets result, responds.

        Test scenario:
            1. Ask about weather — model calls get_weather tool
            2. Execute the tool locally with the model's arguments
            3. Send tool result back as a TOOL message
            4. Model produces a final natural-language response
        """
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            message="What is the weather in Paris?",
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        tc = tool_calls[0]
        tool_result = get_weather(**tc.tool_kwargs)

        messages = [
            Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="What is the weather in Paris?")],
            ),
            response.message,
            Message(
                role=MessageRole.TOOL,
                chunks=[TextChunk(content=tool_result)],
                additional_kwargs={"tool_call_id": tc.tool_id},
            ),
        ]
        final_response = llm.chat(messages)
        assert final_response.message.content is not None, (
            "Final response should have content"
        )
        assert len(final_response.message.content) > 0, (
            "Final response should not be empty"
        )


@pytest.mark.e2e
@skip_no_key
class TestStreamingWithToolCalls:
    """Tests for streaming chat with tool calls."""

    def test_sync_stream_with_tool_calls(self, llm):
        """Test sync streaming chat accumulates tool call fragments.

        Test scenario:
            Use stream=True with tool_required=True. The stream should
            produce chunks with tool call blocks that converge to a
            complete tool call.
        """
        prepared = llm._prepare_chat_with_tools(
            tools=[weather_tool],
            message="What is the weather in Berlin?",
            tool_required=True,
        )
        messages = prepared.pop("messages")
        gen = llm.chat(messages=messages, stream=True, **prepared)
        chunks = list(gen)
        assert len(chunks) > 0, "Should yield at least one chunk"

        final = chunks[-1]
        tool_call_chunks = [
            c for c in final.message.chunks if isinstance(c, ToolCallBlock)
        ]
        assert len(tool_call_chunks) >= 1, (
            f"Final chunk should have tool call blocks, got {len(tool_call_chunks)}"
        )
        assert tool_call_chunks[0].tool_name == "get_weather", (
            f"Expected 'get_weather', got '{tool_call_chunks[0].tool_name}'"
        )

    @pytest.mark.asyncio()
    async def test_async_stream_with_tool_calls(self, llm):
        """Test async streaming chat accumulates tool call fragments.

        Test scenario:
            Same as sync but via achat(stream=True).
        """
        prepared = llm._prepare_chat_with_tools(
            tools=[weather_tool],
            message="What is the weather in Berlin?",
            tool_required=True,
        )
        messages = prepared.pop("messages")
        gen = await llm.achat(messages=messages, stream=True, **prepared)
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert len(chunks) > 0, "Should yield at least one chunk"

        final = chunks[-1]
        tool_call_chunks = [
            c for c in final.message.chunks if isinstance(c, ToolCallBlock)
        ]
        assert len(tool_call_chunks) >= 1, (
            f"Final chunk should have tool call blocks, got {len(tool_call_chunks)}"
        )


@pytest.mark.e2e
@skip_no_key
class TestTokenUsage:
    """Tests for token usage metadata in responses."""

    def test_token_counts_in_chat_response(self, llm):
        """Test that chat response additional_kwargs contains token counts.

        Test scenario:
            Token usage (prompt, completion, total) should be present and
            total should equal prompt + completion.
        """
        response = llm.chat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hi.")],
            )]
        )
        kwargs = response.additional_kwargs
        assert "prompt_tokens" in kwargs, "prompt_tokens should be present"
        assert "completion_tokens" in kwargs, "completion_tokens should be present"
        assert "total_tokens" in kwargs, "total_tokens should be present"
        assert kwargs["prompt_tokens"] > 0, "prompt_tokens should be positive"
        assert kwargs["completion_tokens"] > 0, "completion_tokens should be positive"
        assert kwargs["total_tokens"] == (
            kwargs["prompt_tokens"] + kwargs["completion_tokens"]
        ), "total should equal prompt + completion"

    def test_token_counts_in_completion_response(self, llm):
        """Test that completion response contains token counts.

        Test scenario:
            Token usage via the completion interface should also be populated.
        """
        response = llm.complete("Hello")
        kwargs = response.additional_kwargs
        assert "prompt_tokens" in kwargs, "prompt_tokens should be present"
        assert "total_tokens" in kwargs, "total_tokens should be present"

    @pytest.mark.asyncio()
    async def test_token_counts_in_async_response(self, llm):
        """Test that async chat response contains token counts.

        Test scenario:
            Token usage should be present in async responses as well.
        """
        response = await llm.achat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hi.")],
            )]
        )
        kwargs = response.additional_kwargs
        assert "prompt_tokens" in kwargs, "prompt_tokens should be present"
        assert "total_tokens" in kwargs, "total_tokens should be present"


@pytest.mark.e2e
@skip_no_key
class TestLogprobs:
    """Tests for log-probability extraction."""

    def test_logprobs_in_chat_response(self, llm_with_logprobs):
        """Test that logprobs are returned when requested.

        Test scenario:
            With logprobs=True and top_logprobs=3, the response should
            contain likelihood_score data.
        """
        response = llm_with_logprobs.chat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content="Say hi.")],
            )]
        )
        assert response.logprob is not None, (
            "likelihood_score should be populated when logprobs=True"
        )
        assert len(response.logprob) > 0, (
            "likelihood_score should have at least one entry"
        )


@pytest.mark.e2e
@skip_no_key
class TestConfiguration:
    """Tests for OpenAI construction and configuration."""

    def test_metadata_properties(self, llm, model):
        """Test that metadata reflects the model's capabilities.

        Test scenario:
            Model name, chat/function-calling flags, and context window should
            match expectations for the configured model.
        """
        meta = llm.metadata
        assert meta.model_name == model, (
            f"Expected model_name '{model}', got '{meta.model_name}'"
        )
        assert meta.is_chat_model is True, "Should be a chat model"
        assert meta.is_function_calling_model is True, (
            "Should support function calling"
        )
        assert meta.context_window > 0, "Context window should be positive"

    def test_custom_temperature(self):
        """Test that custom temperature is stored on the instance.

        Test scenario:
            Non-O1 models should preserve the user-specified temperature.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(model="gpt-4o-mini", temperature=0.9)
        assert llm.temperature == 0.9, (
            f"Expected temperature 0.9, got {llm.temperature}"
        )

    def test_o1_model_forces_temperature(self):
        """Test that O1 models force temperature to 1.0.

        Test scenario:
            Creating an O1 model with temperature=0.5 should override to 1.0.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(model="o1-mini", api_key="sk-test", temperature=0.5)
        assert llm.temperature == 1.0, (
            f"O1 model should force temperature to 1.0, got {llm.temperature}"
        )

    def test_responses_only_model_rejected(self):
        """Test that Responses API-only models are rejected.

        Test scenario:
            Creating an OpenAI instance with a Responses-only model should
            raise ValueError directing the user to OpenAIResponses.
        """
        from serapeum.openai import OpenAI
        from serapeum.openai.data.models import RESPONSES_API_ONLY_MODELS

        if not RESPONSES_API_ONLY_MODELS:
            pytest.skip("No Responses API-only models defined")

        model_name = next(iter(RESPONSES_API_ONLY_MODELS))
        with pytest.raises(ValueError, match="Responses API"):
            OpenAI(model=model_name, api_key="sk-test")

    def test_max_tokens_limits_output(self, model):
        """Test that max_tokens limits response length.

        Test scenario:
            With max_tokens=5, the response should be very short.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(model=model, max_tokens=5)
        response = llm.chat(
            [Message(
                role=MessageRole.USER,
                chunks=[TextChunk(
                    content="Write a very long essay about the history of France."
                )],
            )]
        )
        assert response.message.content is not None, "Content should not be None"
        words = response.message.content.split()
        assert len(words) <= 20, (
            f"With max_tokens=5, expected <= 20 words, got {len(words)}"
        )

    def test_audio_modality_rejects_complete(self):
        """Test that complete raises ValueError when audio modality is set.

        Test scenario:
            Audio output is only supported via chat/achat, not complete.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(
            model="gpt-4o-mini",
            api_key="sk-test",
            modalities=["text", "audio"],
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            llm.complete("Hello")

    def test_audio_modality_rejects_stream_chat(self):
        """Test that streaming chat raises ValueError when audio modality is set.

        Test scenario:
            Audio output is not supported in streaming mode.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(
            model="gpt-4o-mini",
            api_key="sk-test",
            modalities=["text", "audio"],
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            list(llm.chat(
                messages=[Message(
                    role=MessageRole.USER,
                    chunks=[TextChunk(content="Hi")],
                )],
                stream=True,
            ))

    @pytest.mark.asyncio()
    async def test_audio_modality_rejects_acomplete(self):
        """Test that acomplete raises ValueError when audio modality is set.

        Test scenario:
            Async completion should also reject audio modality.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(
            model="gpt-4o-mini",
            api_key="sk-test",
            modalities=["text", "audio"],
        )
        with pytest.raises(ValueError, match="Audio is not supported"):
            await llm.acomplete("Hello")

    def test_class_name(self):
        """Test that class_name returns 'openai'.

        Test scenario:
            The canonical provider identifier should be 'openai'.
        """
        from serapeum.openai import OpenAI

        assert OpenAI.class_name() == "openai", (
            f"Expected 'openai', got '{OpenAI.class_name()}'"
        )

    def test_system_role_for_o1(self):
        """Test that O1 models report system_role as USER.

        Test scenario:
            O1 models don't accept system messages, so metadata.system_role
            should be MessageRole.USER.
        """
        from serapeum.openai import OpenAI

        llm = OpenAI(model="o1-mini", api_key="sk-test")
        assert llm.metadata.system_role == MessageRole.USER, (
            f"Expected USER system_role for O1, got {llm.metadata.system_role}"
        )

    def test_system_role_for_chat_model(self, llm):
        """Test that standard chat models report system_role as SYSTEM.

        Test scenario:
            Non-O1 models should use the standard SYSTEM role.
        """
        assert llm.metadata.system_role == MessageRole.SYSTEM, (
            f"Expected SYSTEM role, got {llm.metadata.system_role}"
        )
