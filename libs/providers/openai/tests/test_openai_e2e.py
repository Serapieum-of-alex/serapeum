"""End-to-end tests for the OpenAI class.

These tests hit the real OpenAI API and require OPENAI_API_KEY to be set.
Run with: python -m pytest libs/providers/openai/tests/test_openai_e2e.py -v -m e2e
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import CallableTool

load_dotenv()

_has_key = os.getenv("OPENAI_API_KEY") is not None
skip_no_key = pytest.mark.skipif(not _has_key, reason="OPENAI_API_KEY not set")


@pytest.fixture(scope="session")
def model():
    return os.environ.get("OPENAI_MODEL", "gpt-5-chat")


@pytest.fixture()
def llm():
    from serapeum.openai import OpenAI

    return OpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-chat"),
        api_base=os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Pydantic models for structured output tests
# ---------------------------------------------------------------------------
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class PersonWithAddress(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    address: Address = Field(description="The person's home address")


# ---------------------------------------------------------------------------
# Tool definitions for function-calling tests
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"The weather in {city} is sunny and 22°C."


def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    return f"Search results for: {query}"


weather_tool = CallableTool.from_function(
    func=get_weather, name="get_weather", description="Get the current weather for a city"
)
search_tool = CallableTool.from_function(
    func=search_web, name="search_web", description="Search the web for information"
)



@pytest.mark.e2e
@skip_no_key
class TestBasicChat:
    def test_chat_basic(self, llm):
        """Simple single-turn chat returns a non-empty response."""
        response = llm.chat([Message(role=MessageRole.USER, content="Say hello in one word.")])
        assert response.message is not None
        assert response.message.content is not None
        assert len(response.message.content) > 0

    def test_chat_with_system_prompt(self, llm):
        """System prompt influences the response."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a pirate. Always respond like a pirate."),
            Message(role=MessageRole.USER, content="Say hello."),
        ]
        response = llm.chat(messages)
        assert response.message.content is not None
        assert len(response.message.content) > 0

    def test_chat_multi_turn(self, llm):
        """Multi-turn conversation preserves context."""
        messages = [
            Message(role=MessageRole.USER, content="My name is Alice."),
            Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alice!"),
            Message(role=MessageRole.USER, content="What is my name?"),
        ]
        response = llm.chat(messages)
        assert "Alice" in response.message.content

    @pytest.mark.asyncio()
    async def test_achat_basic(self, llm):
        """Async chat returns a non-empty response."""
        response = await llm.achat([Message(role=MessageRole.USER, content="Say hello in one word.")])
        assert response.message is not None
        assert response.message.content is not None
        assert len(response.message.content) > 0


@pytest.mark.e2e
@skip_no_key
class TestStreaming:
    def test_stream_chat(self, llm):
        """Streaming chat yields multiple chunks with delta text."""
        gen = llm.chat(stream=True, messages=[Message(role=MessageRole.USER, content="Count from 1 to 5.")])
        chunks = list(gen)
        assert len(chunks) > 1
        # The final chunk should have accumulated content
        final = chunks[-1]
        assert final.message.content is not None
        assert len(final.message.content) > 0
        # At least one chunk should have a non-empty delta
        deltas = [c.delta for c in chunks if c.delta]
        assert len(deltas) > 0

    @pytest.mark.asyncio()
    async def test_astream_chat(self, llm):
        """Async streaming chat yields multiple chunks."""
        gen = await llm.achat(stream=True, messages=[Message(role=MessageRole.USER, content="Count from 1 to 5.")])
        chunks = []
        async for chunk in gen:
            chunks.append(chunk)
        assert len(chunks) > 1
        deltas = [c.delta for c in chunks if c.delta]
        assert len(deltas) > 0


@pytest.mark.e2e
@skip_no_key
class TestCompletion:
    def test_complete(self, llm):
        """Completion endpoint works for chat models via mixin."""
        response = llm.complete("The capital of France is")
        assert response.text is not None
        assert len(response.text) > 0

    def test_stream_complete(self, llm):
        """Streaming completion yields chunks with delta text."""
        gen = llm.complete(stream=True, prompt="Count from 1 to 5:")
        chunks = list(gen)
        assert len(chunks) > 1
        final_text = chunks[-1].text
        assert final_text is not None
        assert len(final_text) > 0


@pytest.mark.e2e
@skip_no_key
class TestParse:
    def test_parse(self, llm):
        """Structured predict extracts a Pydantic model."""
        prompt = PromptTemplate("Create a person named Alice who is 30 years old.")
        result = llm.parse(schema=Person, prompt=prompt)
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_nested(self, llm):
        """Structured predict handles nested Pydantic models."""
        prompt = PromptTemplate(
            "Create a person named Bob, age 25, living at 123 Main St, New York, USA."
        )
        result = llm.parse(schema=PersonWithAddress, prompt=prompt)
        assert isinstance(result, PersonWithAddress)
        assert result.name == "Bob"
        assert result.age == 25
        assert isinstance(result.address, Address)
        assert result.address.city == "New York"


@pytest.mark.e2e
@skip_no_key
class TestToolCalling:
    def test_chat_with_single_tool(self, llm):
        """Chat with a single tool returns a tool call."""
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            user_msg="What is the weather in Paris?",
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 1
        assert tool_calls[0].tool_name == "get_weather"
        assert "city" in tool_calls[0].tool_kwargs

    def test_chat_with_tool_choice_specific(self, llm):
        """Forcing a specific tool name selects that tool."""
        response = llm.generate_tool_calls(
            tools=[weather_tool, search_tool],
            user_msg="Tell me about the Eiffel Tower",
            tool_choice="search_web",
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 1
        assert tool_calls[0].tool_name == "search_web"

    def test_chat_with_tools_parallel(self, llm):
        """Parallel tool calls returns multiple tool calls in one response."""
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            user_msg="What is the weather in Paris and London?",
            allow_parallel_tool_calls=True,
            tool_required=True,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=True
        )
        assert len(tool_calls) >= 2

    def test_chat_with_tool_no_tool_needed(self, llm):
        """When no tool is required and the query doesn't need one, model may respond directly."""
        response = llm.generate_tool_calls(
            tools=[weather_tool],
            user_msg="What is 2 + 2?",
            tool_required=False,
        )
        tool_calls = llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )
        # Model may or may not call a tool — just verify no crash
        assert response.message is not None


@pytest.mark.e2e
@skip_no_key
class TestConfiguration:
    def test_metadata_properties(self, llm, model):
        """Metadata reflects the model's capabilities."""
        meta = llm.metadata
        assert meta.model_name == model
        assert meta.is_chat_model is True
        assert meta.is_function_calling_model is True
        assert meta.context_window == 128000

    def test_custom_temperature(self):
        """Custom temperature is stored on the instance."""
        from serapeum.openai import OpenAI

        # Use a non-O1 model — O1 models force temperature to 1.0
        llm = OpenAI(model="gpt-4o-mini", temperature=0.9)
        assert llm.temperature == 0.9

    def test_max_tokens_limits_output(self, model):
        """Setting max_tokens limits the response length."""
        from serapeum.openai import OpenAI

        llm = OpenAI(model=model, max_tokens=5)
        response = llm.chat(
            [Message(role=MessageRole.USER, content="Write a very long essay about the history of France.")]
        )
        # With max_tokens=5, the response should be very short
        assert response.message.content is not None
        # 5 tokens is roughly 3-8 words — be generous with the assertion
        words = response.message.content.split()
        assert len(words) <= 20


# ===========================================================================
# 7. Token usage / additional_kwargs
# ===========================================================================
@pytest.mark.e2e
@skip_no_key
class TestTokenUsage:
    def test_token_counts_in_response(self, llm):
        """Response additional_kwargs contains token usage counts."""
        response = llm.chat([Message(role=MessageRole.USER, content="Say hi.")])
        kwargs = response.additional_kwargs
        assert "prompt_tokens" in kwargs
        assert "completion_tokens" in kwargs
        assert "total_tokens" in kwargs
        assert kwargs["prompt_tokens"] > 0
        assert kwargs["completion_tokens"] > 0
        assert kwargs["total_tokens"] == kwargs["prompt_tokens"] + kwargs["completion_tokens"]
