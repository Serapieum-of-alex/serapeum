"""Extended tests for serapeum.core.structured_tools.tools_llm.

This suite provides comprehensive, scenario-based unit tests for the
`_parse_tool_outputs` function and the `ToolOrchestratingLLM` class.

For each function/method, we define a dedicated test class that contains
individual tests (one per scenario). Each test method includes a docstring
explaining inputs, expected results, and what is being verified.
"""
from __future__ import annotations

from typing import List, Sequence

import pytest

from pydantic import BaseModel

from serapeum.core.chat.models import AgentChatResponse
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.structured_tools.tools_llm import (
    ToolOrchestratingLLM,
)
from serapeum.core.tools import ToolOutput
from serapeum.llms.ollama import Ollama

from tests.llms.ollama.models import Album


LLM = Ollama(
    model="llama3.1",
    request_timeout=180,
)


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
    """Tests various construction scenarios for `from_defaults`."""

    def test_valid_with_prompt_str_and_llm(self) -> None:
        """Construct with prompt_template_str and a function-calling-capable LLM.

        Input:
            output_cls=Album, prompt_template_str, llm with metadata.is_function_calling_model=True
        Expected:
            Returns a ToolOrchestratingLLM instance with configured prompt and flags
        Check:
            Instance type and prompt type are correct
        """
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album with {topic}",
            llm=LLM,  # metadata says it supports function calling
            allow_parallel_tool_calls=True,
            verbose=True,
        )
        assert isinstance(tools_llm, ToolOrchestratingLLM)
        assert isinstance(tools_llm.prompt, PromptTemplate)

    def test_missing_prompt_raises(self) -> None:
        """Raise ValueError if neither prompt nor prompt_template_str is provided.

        Input: llm provided but both prompt and prompt_template_str are None
        Expected: ValueError
        Check: pytest.raises(ValueError)
        """
        with pytest.raises(TypeError):
            ToolOrchestratingLLM(output_cls=Album, llm=LLM)


class TestToolOrchestratingLLMCall:
    """Synchronous execution via __call__ covering single/multiple outputs."""

    def test_single_output_call(self) -> None:
        """Call returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album object
        Check: isinstance and equality
        """
        tools_llm = ToolOrchestratingLLM(Album, prompt="can you create Album with {topic}, and two random songs", llm=LLM)
        result = tools_llm(topic="rock")
        assert isinstance(result, Album)

    def test_multiple_outputs_call_parallel_enabled(self) -> None:
        """Call returns list of Albums when parallel=True.

        Input: Program with allow_parallel_tool_calls=True
        Expected: List[Album] of length 2
        Check: types and order
        """
        tools_llm = ToolOrchestratingLLM(
            Album,
            prompt="Album with {topic}",
            llm=LLM,
            allow_parallel_tool_calls=True,
        )
        result = tools_llm(topic="jazz")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Album)


@pytest.mark.asyncio()
class TestToolOrchestratingLLMAsyncCall:
    """Async execution via acall covering standard single-output scenario."""

    async def test_async_single_output(self) -> None:
        """acall returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album equal to ALBUM
        Check: isinstance and equality
        """
        tools_llm = ToolOrchestratingLLM(Album, prompt="Album with {topic}", llm=LLM)
        result = await tools_llm.acall(topic="pop")
        assert isinstance(result, Album)


class TestToolOrchestratingLLMStreamCall:
    """Tests for the synchronous streaming interface `stream_call`."""

    def test_streaming_yields_processed_objects(self) -> None:
        """stream_call yields objects returned by process_streaming_objects per chunk.

        Input: MockFunctionCallingLLM that emits 2 ChatResponse chunks; patched process_streaming_objects
        Expected: Two yields with objects we control
        Check: Sequence and values
        """
        llm = LLM
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album {topic}",
            llm=llm,
            allow_parallel_tool_calls=False,
        )

        out = tools_llm.stream_call(topic="x")
        out = list(out)
        assert len(out) == 2
        assert all(isinstance(obj, Album) for obj in out)


@pytest.mark.asyncio()
class TestToolOrchestratingLLMAStreamCall:
    """Tests for the asynchronous streaming interface `astream_call`."""

    async def test_async_streaming_yields_processed_objects(self) -> None:
        """astream_call yields objects returned by process_streaming_objects per chunk.

        Input: MockFunctionCallingLLM that emits 2 ChatResponse chunks; patched process_streaming_objects
        Expected: Two yields with objects we control (awaited via async for)
        Check: Sequence and values
        """
        llm = LLM
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album {topic}",
            llm=llm,
            allow_parallel_tool_calls=False,
        )

        agen = await tools_llm.astream_call(topic="x")
        results: list[Album] = []
        async for item in agen:  # type: ignore
            results.append(item)
        assert len(results) == 2
        assert all(isinstance(obj, Album) for obj in results)
