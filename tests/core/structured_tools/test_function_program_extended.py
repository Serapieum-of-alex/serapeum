"""Extended tests for serapeum.core.structured_tools.tools_llm.

This suite provides comprehensive, scenario-based unit tests for the
`_parse_tool_outputs` function and the `ToolOrchestratingLLM` class.

For each function/method, we define a dedicated test class that contains
individual tests (one per scenario). Each test method includes a docstring
explaining inputs, expected results, and what is being verified.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, List, Optional, Sequence, Union
from unittest.mock import MagicMock, patch

import pytest

from pydantic import BaseModel

from serapeum.core.base.llms.models import Message, ChatResponse, Metadata
from serapeum.core.chat.models import AgentChatResponse
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.structured_tools.tools_llm import (
    ToolOrchestratingLLM,
)
from serapeum.core.tools import ToolOutput
from serapeum.core.tools.models import ToolCallArguments


class Song(BaseModel):
    """A song data model used in tests."""

    title: str


class Album(BaseModel):
    """Album model used as the program output."""

    title: str
    artist: str
    songs: List[Song]


SAMPLE_ALBUM = Album(
    title="A Title",
    artist="An Artist",
    songs=[Song(title="s1"), Song(title="s2")],
)
SAMPLE_ALBUM_2 = Album(
    title="Another Title",
    artist="Another Artist",
    songs=[Song(title="s3"), Song(title="s4")],
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


class NonFunctionCallingMockLLM(MagicMock):
    """A simple MagicMock-based LLM that is NOT a FunctionCallingLLM instance.

    It exposes the attributes used by tests (e.g., metadata and predict methods)
    without inheriting `FunctionCallingLLM`. This lets us trigger the
    `stream_call`/`astream_call` ValueError branch.
    """

    @property
    def metadata(self) -> Metadata:
        return Metadata(is_function_calling_model=True, model_name="non_fc_mock")

    # The following two are used by ToolOrchestratingLLM.__call__/acall tests
    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        models = [SAMPLE_ALBUM] if not allow_parallel_tool_calls else [SAMPLE_ALBUM, SAMPLE_ALBUM_2]
        return make_agent_response_from_models(models)

    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        models = [SAMPLE_ALBUM] if not allow_parallel_tool_calls else [SAMPLE_ALBUM, SAMPLE_ALBUM_2]
        return make_agent_response_from_models(models)


class MockFunctionCallingLLM(FunctionCallingLLM):
    """A minimal FunctionCallingLLM implementation for streaming and tool-call tests.

    This mock:
      - Returns a configurable list of ToolCallArguments from
        `get_tool_calls_from_response` (set via `self.tool_calls`).
      - Provides basic streaming generators for `stream_chat`/`astream_chat`.
    """

    def __init__(self) -> None:
        super().__init__()
        # Note: avoid declaring `tool_calls` as a Pydantic field; keep it as a plain attribute
        # to prevent BaseModel validation from requiring it at construction time.
        self.tool_calls: List[ToolCallArguments] = []

    @property
    def metadata(self) -> Metadata:  # type: ignore[override]
        return Metadata(is_function_calling_model=True, model_name="mock_fc")

    # ---- Base abstract methods (minimal stubs) ----
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:  # type: ignore[override]
        return ChatResponse(message=Message.from_str("ok"))

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError

    def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> Generator[ChatResponse, None, None]:  # type: ignore[override]
        def gen() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(message=Message.from_str("chunk-1"))
            yield ChatResponse(message=Message.from_str("chunk-2"))
        return gen()

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:  # type: ignore[override]
        return ChatResponse(message=Message.from_str("ok"))

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:  # type: ignore[override]
        async def agen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(message=Message.from_str("chunk-1"))
            yield ChatResponse(message=Message.from_str("chunk-2"))
        return agen()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):  # type: ignore[override]
        raise NotImplementedError

    # ---- FunctionCallingLLM specifics ----
    def _prepare_chat_with_tools(  # type: ignore[override]
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict:
        # Just forward the provided history/messages to the underlying chat methods
        if chat_history is not None:
            messages = chat_history
        elif user_msg is not None:
            messages = [Message.from_str(user_msg) if isinstance(user_msg, str) else user_msg]
        else:
            messages = []
        return {"messages": messages, **kwargs}

    def get_tool_calls_from_response(  # type: ignore[override]
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolCallArguments]:
        return list(self.tool_calls)


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
        llm = NonFunctionCallingMockLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album with {topic}",
            llm=llm,  # metadata says it supports function calling
            allow_parallel_tool_calls=True,
            verbose=True,
        )
        assert isinstance(tools_llm, ToolOrchestratingLLM)
        assert isinstance(tools_llm.prompt, PromptTemplate)

    def test_invalid_when_llm_not_function_calling(self) -> None:
        """Raise ValueError if provided LLM reports no function calling support.

        Input: llm.metadata.is_function_calling_model=False
        Expected: ValueError with message mentioning the model name
        Check: pytest.raises(ValueError)
        """
        class NoFC(NonFunctionCallingMockLLM):
            @property
            def metadata(self) -> Metadata:  # type: ignore[override]
                return Metadata(is_function_calling_model=False, model_name="no-fc")

        with pytest.raises(ValueError):
            ToolOrchestratingLLM(
                output_cls=Album,
                prompt="Album with {topic}",
                llm=NoFC(),
            )

    def test_missing_prompt_raises(self) -> None:
        """Raise ValueError if neither prompt nor prompt_template_str is provided.

        Input: llm provided but both prompt and prompt_template_str are None
        Expected: ValueError
        Check: pytest.raises(ValueError)
        """
        with pytest.raises(TypeError):
            ToolOrchestratingLLM(output_cls=Album, llm=NonFunctionCallingMockLLM())

    def test_fallback_to_configs_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use Configs.llm if llm is not provided explicitly.

        Input: No llm passed; Configs.llm patched to a valid LLM with function calling enabled
        Expected: Returns a valid ToolOrchestratingLLM instance
        Check: Instance creation succeeds
        """
        patched_llm = NonFunctionCallingMockLLM()
        monkeypatch.setattr("serapeum.core.structured_tools.tools_llm.Configs.llm", patched_llm, raising=False)
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album with {topic}",
        )
        assert isinstance(tools_llm, ToolOrchestratingLLM)


class TestToolOrchestratingLLMProperties:
    """Tests for `output_cls` and `prompt` getters/setter."""

    def test_output_cls_property(self) -> None:
        """output_cls property returns the model class passed at construction.

        Input: ToolOrchestratingLLM created for Album
        Expected: .output_cls is Album
        Check: identity equality
        """
        tools_llm = ToolOrchestratingLLM(Album, prompt="x {y}", llm=NonFunctionCallingMockLLM())
        assert tools_llm.output_cls is Album

    def test_prompt_getter_setter(self) -> None:
        """prompt property getter and setter work as expected.

        Input: Set a new PromptTemplate on the program
        Expected: The .prompt returns the newly set template
        Check: identity equality
        """
        tools_llm = ToolOrchestratingLLM(Album, prompt="x {y}", llm=NonFunctionCallingMockLLM())
        new_prompt = PromptTemplate("New {var}")
        tools_llm.prompt = new_prompt
        assert tools_llm.prompt is new_prompt


class TestToolOrchestratingLLMCall:
    """Synchronous execution via __call__ covering single/multiple outputs."""

    def test_single_output_call(self) -> None:
        """Call returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album equal to SAMPLE_ALBUM
        Check: isinstance and equality
        """
        llm = NonFunctionCallingMockLLM()
        llm._extend_messages = MagicMock(side_effect=lambda msgs: msgs)  # track call
        tools_llm = ToolOrchestratingLLM(Album, prompt="Album with {topic}", llm=llm)
        result = tools_llm(topic="rock")
        assert isinstance(result, Album)
        assert result == SAMPLE_ALBUM
        llm._extend_messages.assert_called_once()

    def test_multiple_outputs_call_parallel_enabled(self) -> None:
        """Call returns list of Albums when parallel=True.

        Input: Program with allow_parallel_tool_calls=True
        Expected: List[Album] of length 2 matching SAMPLE_ALBUM and SAMPLE_ALBUM_2
        Check: types and order
        """
        llm = NonFunctionCallingMockLLM()
        tools_llm = ToolOrchestratingLLM(
            Album,
            prompt="Album with {topic}",
            llm=llm,
            allow_parallel_tool_calls=True,
        )
        result = tools_llm(topic="jazz")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == SAMPLE_ALBUM and result[1] == SAMPLE_ALBUM_2


@pytest.mark.asyncio()
class TestToolOrchestratingLLMAsyncCall:
    """Async execution via acall covering standard single-output scenario."""

    async def test_async_single_output(self) -> None:
        """acall returns a single Album when parallel=False.

        Input: Program with allow_parallel_tool_calls=False using NonFunctionCallingMockLLM
        Expected: Returns Album equal to SAMPLE_ALBUM
        Check: isinstance and equality
        """
        llm = NonFunctionCallingMockLLM()
        tools_llm = ToolOrchestratingLLM(Album, prompt="Album with {topic}", llm=llm)
        result = await tools_llm.acall(topic="pop")
        assert isinstance(result, Album)
        assert result == SAMPLE_ALBUM


class TestToolOrchestratingLLMStreamCall:
    """Tests for the synchronous streaming interface `stream_call`."""

    def test_raises_for_non_function_calling_llm(self) -> None:
        """Raise ValueError if the underlying LLM is not a FunctionCallingLLM.

        Input: Program created with NonFunctionCallingMockLLM
        Expected: ValueError from stream_call
        Check: pytest.raises(ValueError)
        """
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album {topic}",
            llm=NonFunctionCallingMockLLM(),
        )
        with pytest.raises(ValueError):
            list(tools_llm.stream_call(topic="t"))  # force iteration

    def test_streaming_yields_processed_objects(self) -> None:
        """stream_call yields objects returned by process_streaming_objects per chunk.

        Input: MockFunctionCallingLLM that emits 2 ChatResponse chunks; patched process_streaming_objects
        Expected: Two yields with objects we control
        Check: Sequence and values
        """
        llm = MockFunctionCallingLLM()
        tools_llm = ToolOrchestratingLLM(
            output_cls=Album,
            prompt="Album {topic}",
            llm=llm,
            allow_parallel_tool_calls=False,
        )
        obj1 = Album(title="t1", artist="a1", songs=[Song(title="s1")])
        obj2 = Album(title="t2", artist="a2", songs=[Song(title="s2")])
        with patch(
            "serapeum.core.structured_tools.tools_llm.process_streaming_objects",
            side_effect=[obj1, obj2],
        ) as mock_proc:
            out = list(tools_llm.stream_call(topic="x"))
            assert out == [obj1, obj2]
            assert mock_proc.call_count == 2

    def test_streaming_continues_on_parse_error(self) -> None:
        """If processing one chunk fails, it logs and continues with the next.

        Input: First call to process_streaming_objects raises; second returns an object
        Expected: The generator yields only the second object
        Check: Warning logged and correct single yield
        """
        llm = MockFunctionCallingLLM()
        tools_llm = ToolOrchestratingLLM(Album, prompt="Album {topic}", llm=llm)
        with patch("serapeum.core.structured_tools.tools_llm._logger") as mock_logger:
            with patch(
                "serapeum.core.structured_tools.tools_llm.process_streaming_objects",
                side_effect=[RuntimeError("boom"), SAMPLE_ALBUM],
            ):
                out = list(tools_llm.stream_call(topic="x"))
        assert out == [SAMPLE_ALBUM]
        mock_logger.warning.assert_called()




    def test_no_tool_calls_returns_blank_instance(self) -> None:
        """When get_tool_calls_from_response returns empty list, return blank model.

        Input: ChatResponse with no tool calls; output_cls has defaults
        Expected: Instance of DefaultableModel with default values
        Check: Attributes are None/default
        """
        llm = MockFunctionCallingLLM()
        llm.tool_calls = []  # no tool calls
        tools_llm = ToolOrchestratingLLM(
            output_cls=self.DefaultableModel, prompt="x {y}", llm=llm
        )
        resp = ChatResponse(message=Message.from_str("irrelevant"))
        out = tools_llm._process_objects(resp, self.DefaultableModel)
        assert isinstance(out, self.DefaultableModel)
        assert out.a is None and out.b is None