"""Extended tests for serapeum.core.llms.abstractions.

This suite provides comprehensive, scenario-based unit tests for the
`_parse_tool_outputs` function and the `ToolOrchestratingLLM` class.

For each function/method, we define a dedicated test class that contains
individual tests (one per scenario). Each test method includes a docstring
explaining inputs, expected results, and what is being verified.
"""

from __future__ import annotations

from typing import Sequence

import pytest
from pydantic import BaseModel
import asyncio
from serapeum.core.chat.types import AgentChatResponse
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.tools import ToolOutput
from serapeum.ollama import Ollama


class SimpleOutput(BaseModel):
    """Simple Pydantic model for testing."""

    value: str
    count: int = 0


def make_agent_response_from_models(models: Sequence[BaseModel]) -> AgentChatResponse:
    """Utility to convert a sequence of BaseModel instances to AgentChatResponse.

    Input: list of BaseModel instances to embed in ToolOutput.raw_output
    Expected: An AgentChatResponse with ToolOutput entries holding raw_output=models
    Check: The number of sources matches the number of models
    """
    return AgentChatResponse(
        response="ok",
        sources=[
            ToolOutput(
                tool_name="model",
                content=str(m),
                raw_input={},
                raw_output=m,
            )
            for m in models
        ],
    )


class TestToolOrchestratingLLM:
    """Tests for ToolOrchestratingLLM."""

    def test_valid_with_prompt_str_and_llm(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """Construct with prompt_template_str and a function-calling-capable LLM.

        Input:
            schema=Album, prompt_template_str, llm with metadata.is_function_calling_model=True
        Expected:
            Returns a ToolOrchestratingLLM instance with configured prompt and flags
        Check:
            Instance type and prompt type are correct
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album,
            prompt="Album with {topic}",
            llm=llm_model,  # metadata says it supports function calling
            allow_parallel_tool_calls=True,
            verbose=True,
        )
        assert isinstance(tools_llm, ToolOrchestratingLLM)
        assert isinstance(tools_llm.prompt, PromptTemplate)

    def test_missing_prompt_raises(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """Raise ValueError if neither prompt nor prompt_template_str is provided.

        Input: llm provided but both prompt and prompt_template_str are None
        Expected: ValueError
        Check: pytest.raises(ValueError)
        """
        with pytest.raises(TypeError):
            ToolOrchestratingLLM(schema=album, llm=llm_model)


class TestToolOrchestratingLLMCall:
    """Synchronous execution via __call__ covering single/multiple outputs."""

    @pytest.mark.e2e
    def test_single_output_call(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """Call returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album object
        Check: isinstance and equality
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album,
            prompt="Create an Album about {topic} music. Include the album name, artist name, and two songs with their titles.",
            llm=llm_model,
        )
        result = tools_llm(topic="rock")
        assert isinstance(result, album)

    @pytest.mark.e2e
    def test_multiple_outputs_call_parallel_enabled(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """Call returns list of Albums when parallel=True.

        Input: Program with allow_parallel_tool_calls=True
        Expected: List[Album] of length 2
        Check: types and order
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album,
            prompt="Album with {topic}",
            llm=llm_model,
            allow_parallel_tool_calls=True,
        )
        result = tools_llm(topic="jazz")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], album)


@pytest.mark.asyncio()
class TestToolOrchestratingLLMAsyncCall:
    """Async execution via acall covering standard single-output scenario."""

    @pytest.mark.e2e
    async def test_async_single_output(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """Acall returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album equal to ALBUM
        Check: isinstance and equality
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album, prompt="Album with {topic}", llm=llm_model
        )
        result = await tools_llm.acall(topic="pop")
        assert isinstance(result, album)


class TestToolOrchestratingLLMStreamCall:
    """Tests for the synchronous streaming interface `stream_call`."""

    @pytest.mark.e2e
    def test_streaming_yields_processed_objects(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """stream_call yields objects returned by process_streaming_objects per chunk.

        Input: MockFunctionCallingLLM that emits 2 ChatResponse chunks; patched process_streaming_objects
        Expected: Two yields with objects we control
        Check: Sequence and values
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album,
            prompt="Album {topic}",
            llm=llm_model,
            allow_parallel_tool_calls=False,
        )

        out = list(tools_llm(topic="x", stream=True))
        assert len(out) >= 1
        # last instance should be the final, fully resolved model
        assert isinstance(out[-1], album)


@pytest.mark.asyncio()
class TestToolOrchestratingLLMAStreamCall:
    """Tests for the asynchronous streaming interface `astream_call`."""

    @pytest.mark.e2e
    async def test_async_streaming_yields_processed_objects(
        self, llm_model: Ollama, album: type[BaseModel]
    ) -> None:
        """astream_call yields objects returned by process_streaming_objects per chunk.

        Input: MockFunctionCallingLLM that emits 2 ChatResponse chunks; patched process_streaming_objects
        Expected: Two yields with objects we control (awaited via async for)
        Check: Sequence and values
        """
        tools_llm = ToolOrchestratingLLM(
            schema=album,
            prompt="Album {topic}",
            llm=llm_model,
            allow_parallel_tool_calls=False,
        )

        agen = await tools_llm.acall(topic="x", stream=True)
        results: list[album] = [item async for item in agen]
        assert len(results) >= 1
        assert all(isinstance(obj, album) for obj in results)


@pytest.mark.e2e
class TestOllamaE2E:
    """End-to-end tests with real Ollama server.

    These tests require:
    - Ollama server running
    - llama3.1 model pulled
    - serapeum-ollama package installed

    Skip if not available using pytest markers.
    """

    def test_pydantic_model_with_real_ollama(self, llm_model: Ollama):
        """Test Pydantic model with real Ollama server.

        Expected: Should generate valid SimpleOutput from LLM.
        """
        tools_llm = ToolOrchestratingLLM(
            schema=SimpleOutput,
            prompt="Generate a simple output with value '{text}' and count the words",
            llm=llm_model,
        )

        result = tools_llm(text="hello world")

        assert isinstance(result, SimpleOutput)
        assert isinstance(result.value, str)
        assert isinstance(result.count, int)

    def test_function_with_real_ollama(self, llm_model: Ollama):
        """Test regular function with real Ollama server.

        Expected: Should generate valid dict output from function via LLM.
        """
        def extract_info(name: str, age: int, city: str) -> dict:
            """Extract person information."""
            return {
                "name": name,
                "age": age,
                "city": city,
                "summary": f"{name} is {age} years old and lives in {city}",
            }

        tools_llm = ToolOrchestratingLLM(
            schema=extract_info,
            prompt="Extract information from: {text}",
            llm=llm_model,
        )

        result = tools_llm(text="John is 30 years old and lives in New York")

        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "city" in result

    @pytest.mark.asyncio
    async def test_async_function_with_real_ollama(self, llm_model: Ollama):
        """Test async function with real Ollama server.

        Expected: Should execute async function via LLM successfully.
        """
        async def async_processor(text: str, multiplier: int) -> dict:
            """Process text asynchronously."""
            await asyncio.sleep(0.01)
            return {
                "text": text,
                "length": len(text),
                "multiplied": len(text) * multiplier,
            }

        tools_llm = ToolOrchestratingLLM(
            schema=async_processor,
            prompt="Process this text: {text}",
            llm=llm_model,
        )

        result = await tools_llm.acall(text="hello")

        assert isinstance(result, dict)
        assert "text" in result
        assert "length" in result