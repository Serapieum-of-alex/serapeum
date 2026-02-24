"""Unit/mock tests for Ollama.parse routing and private helpers.

All tests are @pytest.mark.mock — no running Ollama server required.
The existing e2e tests in test_ollama_structured_predict.py already cover full
integration across all Pydantic schema shapes.  These tests focus on:

  - Routing logic in parse(): which branch is taken based on
    structured_output_mode and stream flag.
  - _parse_default: format injection, kwargs handling, message formatting,
    return value, content coercion.
  - _stream_parse_default: generator laziness, format injection, processor
    construction args, exception swallowing, cur_objects update logic.
"""

from __future__ import annotations

from types import GeneratorType
from unittest.mock import MagicMock, call, patch

import pytest
from pydantic import BaseModel

from serapeum.core.llms.base import LLM
from serapeum.core.prompts import PromptTemplate
from serapeum.core.types import StructuredOutputMode
from serapeum.ollama import Ollama


class Simple(BaseModel):
    value: str


@pytest.fixture
def llm() -> Ollama:
    """Ollama instance that never touches the network.

    Client creation is lazy (on first property access), so constructing with
    just a model name is safe in unit tests.
    """
    return Ollama(model="test-model")


@pytest.fixture
def prompt() -> PromptTemplate:
    return PromptTemplate("Tell me about: {topic}")


def _chat_response(content: str | None) -> MagicMock:
    """Build a minimal ChatResponse-shaped mock with .message.content."""
    resp = MagicMock()
    resp.message.content = content
    return resp


class TestParseRouting:
    """parse() dispatches to the correct implementation branch."""

    @pytest.mark.mock
    def test_default_stream_false_calls_parse_default(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: DEFAULT mode (default), stream=False (default), topic="python".
        Expected: _parse_default called with (schema, prompt, None, {"topic": "python"}).
        Checks: return value is what _parse_default returned; super path not taken.
        """
        expected = Simple(value="x")
        with patch.object(llm, "_parse_default", return_value=expected) as mock_pd:
            result = llm.parse(Simple, prompt, topic="python")

        mock_pd.assert_called_once_with(Simple, prompt, None, {"topic": "python"})
        assert result is expected

    @pytest.mark.mock
    def test_default_stream_true_calls_stream_parse_default(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: DEFAULT mode, stream=True (keyword-only), topic="python".
        Expected: _stream_parse_default called with correct args; result is its return value.
        Checks: _stream_parse_default called once; prompt_args dict correctly collected.
        """
        mock_gen = iter([Simple(value="a")])
        with patch.object(
            llm, "_stream_parse_default", return_value=mock_gen
        ) as mock_spd:
            result = llm.parse(Simple, prompt, stream=True, topic="python")

        mock_spd.assert_called_once_with(Simple, prompt, None, {"topic": "python"})
        assert result is mock_gen

    @pytest.mark.mock
    def test_default_mode_explicit_llm_kwargs_forwarded(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: explicit llm_kwargs={"temperature": 0.0}, stream=False.
        Expected: _parse_default receives the same kwargs dict.
        Checks: llm_kwargs positional arg is passed through unchanged.
        """
        kwargs = {"temperature": 0.0}
        expected = Simple(value="y")
        with patch.object(llm, "_parse_default", return_value=expected) as mock_pd:
            llm.parse(Simple, prompt, kwargs, topic="x")

        mock_pd.assert_called_once_with(Simple, prompt, kwargs, {"topic": "x"})

    @pytest.mark.mock
    @pytest.mark.parametrize(
        "mode",
        [StructuredOutputMode.FUNCTION, StructuredOutputMode.LLM],
    )
    def test_non_default_stream_false_calls_super_parse(
        self, llm: Ollama, prompt: PromptTemplate, mode: StructuredOutputMode
    ) -> None:
        """
        Inputs: non-DEFAULT mode (FUNCTION or LLM), stream=False.
        Expected: LLM.parse (the base class implementation) is called.
        Checks: _parse_default is NOT called; super path IS taken.
        """
        llm.structured_output_mode = mode
        expected = Simple(value="super")
        with patch.object(LLM, "parse", return_value=expected) as mock_super, \
             patch.object(llm, "_parse_default") as mock_pd:
            result = llm.parse(Simple, prompt, topic="x")

        mock_super.assert_called_once()
        mock_pd.assert_not_called()
        assert result is expected

    @pytest.mark.mock
    @pytest.mark.parametrize(
        "mode",
        [StructuredOutputMode.FUNCTION, StructuredOutputMode.LLM],
    )
    def test_non_default_stream_true_calls_super_stream_parse(
        self, llm: Ollama, prompt: PromptTemplate, mode: StructuredOutputMode
    ) -> None:
        """
        Inputs: non-DEFAULT mode (FUNCTION or LLM), stream=True.
        Expected: LLM.stream_parse is called; _stream_parse_default is NOT called.
        Checks: super path taken for streaming in non-DEFAULT mode.
        """
        llm.structured_output_mode = mode
        mock_gen = iter([Simple(value="super-stream")])
        with patch.object(LLM, "stream_parse", return_value=mock_gen) as mock_super, \
             patch.object(llm, "_stream_parse_default") as mock_spd:
            result = llm.parse(Simple, prompt, stream=True, topic="x")

        mock_super.assert_called_once()
        mock_spd.assert_not_called()
        assert result is mock_gen

    @pytest.mark.mock
    def test_stream_is_keyword_only(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: True passed as the 4th positional argument (would be stream).
        Expected: TypeError — the * in the signature makes stream keyword-only.
        Checks: Python enforces keyword-only constraint at call time.
        """
        with pytest.raises(TypeError):
            llm.parse(Simple, prompt, None, True)  # type: ignore[call-arg]


class TestParseDefault:
    """_parse_default: format injection, kwargs handling, message formatting, return value."""

    @pytest.mark.mock
    def test_injects_format_into_chat_kwargs(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: schema=Simple, llm_kwargs=None.
        Expected: self.chat called with format=Simple.model_json_schema().
        Checks: 'format' key present with the correct schema value.
        """
        mock_resp = _chat_response('{"value": "hi"}')
        with patch.object(llm, "chat", return_value=mock_resp) as mock_chat:
            llm._parse_default(Simple, prompt, None, {"topic": "x"})

        _, chat_kwargs = mock_chat.call_args
        assert chat_kwargs.get("format") == Simple.model_json_schema()

    @pytest.mark.mock
    def test_none_llm_kwargs_uses_empty_dict(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: llm_kwargs=None.
        Expected: No error; chat is called successfully with a fresh dict.
        Checks: None is never accessed as a dict; call completes.
        """
        mock_resp = _chat_response('{"value": "ok"}')
        with patch.object(llm, "chat", return_value=mock_resp):
            result = llm._parse_default(Simple, prompt, None, {"topic": "x"})

        assert isinstance(result, Simple)

    @pytest.mark.mock
    def test_existing_kwargs_keys_preserved_alongside_format(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: llm_kwargs={"temperature": 0.1}.
        Expected: chat receives both 'temperature' and 'format' keys.
        Checks: format injection does not discard pre-existing kwargs.
        """
        mock_resp = _chat_response('{"value": "ok"}')
        with patch.object(llm, "chat", return_value=mock_resp) as mock_chat:
            llm._parse_default(Simple, prompt, {"temperature": 0.1}, {"topic": "x"})

        _, chat_kwargs = mock_chat.call_args
        assert "temperature" in chat_kwargs
        assert "format" in chat_kwargs

    @pytest.mark.mock
    def test_calls_format_messages_with_unpacked_prompt_args(
        self, llm: Ollama
    ) -> None:
        """
        Inputs: prompt_args={"topic": "AI"}.
        Expected: prompt.format_messages(topic="AI") is called (kwargs unpacked).
        Checks: prompt_args dict is correctly unpacked when calling format_messages.
        """
        mock_prompt = MagicMock(spec=PromptTemplate)
        mock_prompt.format_messages.return_value = []
        mock_resp = _chat_response('{"value": "ok"}')
        with patch.object(llm, "chat", return_value=mock_resp):
            llm._parse_default(Simple, mock_prompt, None, {"topic": "AI"})

        mock_prompt.format_messages.assert_called_once_with(topic="AI")

    @pytest.mark.mock
    def test_returns_validated_model_instance(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: response content '{"value": "hello"}'.
        Expected: Simple(value="hello") returned.
        Checks: model_validate_json result is what the method returns.
        """
        mock_resp = _chat_response('{"value": "hello"}')
        with patch.object(llm, "chat", return_value=mock_resp):
            result = llm._parse_default(Simple, prompt, None, {"topic": "x"})

        assert isinstance(result, Simple)
        assert result.value == "hello"

    @pytest.mark.mock
    def test_empty_content_string_passed_to_validate(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: response.message.content == "".
        Expected: model_validate_json("") called (empty string forwarded as-is).
        Checks: content or "" with empty string yields "".
        """
        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {}
        mock_resp = _chat_response("")
        with patch.object(llm, "chat", return_value=mock_resp):
            llm._parse_default(mock_schema, prompt, None, {"topic": "x"})

        mock_schema.model_validate_json.assert_called_once_with("")

    @pytest.mark.mock
    def test_none_content_coerced_to_empty_string(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: response.message.content == None.
        Expected: model_validate_json("") called — None coerced via `content or ""`.
        Checks: the `or ""` guard exercises the None branch.
        """
        mock_schema = MagicMock()
        mock_schema.model_json_schema.return_value = {}
        mock_resp = _chat_response(None)
        with patch.object(llm, "chat", return_value=mock_resp):
            llm._parse_default(mock_schema, prompt, None, {"topic": "x"})

        mock_schema.model_validate_json.assert_called_once_with("")


class TestStreamParseDefault:
    """_stream_parse_default: generator semantics, processor wiring, exception handling."""

    @pytest.mark.mock
    def test_returns_generator_before_any_io(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: schema=Simple, empty stream_chat iterator.
        Expected: Calling _stream_parse_default returns a GeneratorType immediately.
        Checks: stream_chat is NOT called until the generator is advanced.
        """
        with patch.object(llm, "stream_chat", return_value=iter([])) as mock_sc:
            result = llm._stream_parse_default(Simple, prompt, None, {"topic": "x"})

            assert isinstance(result, GeneratorType)
            mock_sc.assert_not_called()

    @pytest.mark.mock
    def test_injects_format_into_stream_chat_kwargs(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: schema=Simple, llm_kwargs=None.
        Expected: stream_chat called with format=Simple.model_json_schema().
        Checks: format key present with correct schema value.
        """
        obj = Simple(value="partial")
        with patch.object(
            llm, "stream_chat", return_value=iter([MagicMock()])
        ) as mock_sc, patch(
            "serapeum.ollama.llm.StreamingObjectProcessor"
        ) as MockProc:
            MockProc.return_value.process.return_value = obj
            list(llm._stream_parse_default(Simple, prompt, None, {"topic": "x"}))

        _, sc_kwargs = mock_sc.call_args
        assert sc_kwargs.get("format") == Simple.model_json_schema()

    @pytest.mark.mock
    def test_none_llm_kwargs_does_not_raise(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: llm_kwargs=None.
        Expected: generator completes without AttributeError.
        Checks: None is handled via `llm_kwargs or {}`.
        """
        obj = Simple(value="x")
        with patch.object(llm, "stream_chat", return_value=iter([MagicMock()])), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            MockProc.return_value.process.return_value = obj
            results = list(
                llm._stream_parse_default(Simple, prompt, None, {"topic": "x"})
            )

        assert results == [obj]

    @pytest.mark.mock
    def test_yields_objects_from_processor_in_order(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: stream_chat returns two chunks; processor returns one object per chunk.
        Expected: both objects yielded in chunk order.
        Checks: all successful processor results are yielded; count matches.
        """
        chunks = [MagicMock(), MagicMock()]
        objects = [Simple(value="a"), Simple(value="b")]
        with patch.object(llm, "stream_chat", return_value=iter(chunks)), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            MockProc.return_value.process.side_effect = objects
            results = list(
                llm._stream_parse_default(Simple, prompt, None, {"topic": "x"})
            )

        assert results == objects

    @pytest.mark.mock
    def test_processor_exception_swallowed_and_iteration_continues(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: 3 chunks; processor raises ValueError on chunk 2, succeeds on 1 and 3.
        Expected: only objects from chunks 1 and 3 yielded; no exception propagates.
        Checks: bare `except Exception: continue` branch is exercised.
        """
        chunks = [MagicMock(), MagicMock(), MagicMock()]
        obj_a = Simple(value="a")
        obj_c = Simple(value="c")
        with patch.object(llm, "stream_chat", return_value=iter(chunks)), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            MockProc.return_value.process.side_effect = [
                obj_a,
                ValueError("bad JSON fragment"),
                obj_c,
            ]
            results = list(
                llm._stream_parse_default(Simple, prompt, None, {"topic": "x"})
            )

        assert results == [obj_a, obj_c]

    @pytest.mark.mock
    def test_all_chunks_raise_yields_nothing(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: processor raises RuntimeError on every chunk.
        Expected: generator completes normally, yielding an empty sequence.
        Checks: exception swallowing works for every iteration step.
        """
        with patch.object(
            llm, "stream_chat", return_value=iter([MagicMock(), MagicMock()])
        ), patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            MockProc.return_value.process.side_effect = RuntimeError("boom")
            results = list(
                llm._stream_parse_default(Simple, prompt, None, {"topic": "x"})
            )

        assert results == []

    @pytest.mark.mock
    def test_cur_objects_wraps_single_result_for_next_call(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: processor returns a single BaseModel on both calls.
        Expected: second process() call receives cur_objects=[first_object].
        Checks: `objects if isinstance(objects, list) else [objects]` wrapping branch.
        """
        chunks = [MagicMock(), MagicMock()]
        obj1 = Simple(value="first")
        obj2 = Simple(value="second")
        with patch.object(llm, "stream_chat", return_value=iter(chunks)), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            mock_proc = MockProc.return_value
            mock_proc.process.side_effect = [obj1, obj2]
            list(llm._stream_parse_default(Simple, prompt, None, {"topic": "x"}))

        process_calls = mock_proc.process.call_args_list
        assert process_calls[0] == call(chunks[0], None)
        assert process_calls[1] == call(chunks[1], [obj1])

    @pytest.mark.mock
    def test_cur_objects_list_result_passed_as_is(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: processor returns a list[BaseModel] on the first call.
        Expected: second process() call receives that list directly (not re-wrapped).
        Checks: `isinstance(objects, list)` branch passes the list through unchanged.
        """
        chunks = [MagicMock(), MagicMock()]
        obj_list = [Simple(value="x"), Simple(value="y")]
        obj2 = Simple(value="second")
        with patch.object(llm, "stream_chat", return_value=iter(chunks)), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            mock_proc = MockProc.return_value
            mock_proc.process.side_effect = [obj_list, obj2]
            list(llm._stream_parse_default(Simple, prompt, None, {"topic": "x"}))

        process_calls = mock_proc.process.call_args_list
        assert process_calls[1] == call(chunks[1], obj_list)

    @pytest.mark.mock
    def test_processor_constructed_with_correct_kwargs(
        self, llm: Ollama, prompt: PromptTemplate
    ) -> None:
        """
        Inputs: schema=Simple.
        Expected: StreamingObjectProcessor(output_cls=Simple, flexible_mode=True,
                  allow_parallel_tool_calls=False).
        Checks: constructor receives exact keyword arguments — no accidental drift.
        """
        with patch.object(llm, "stream_chat", return_value=iter([])), \
             patch("serapeum.ollama.llm.StreamingObjectProcessor") as MockProc:
            list(llm._stream_parse_default(Simple, prompt, None, {"topic": "x"}))

        MockProc.assert_called_once_with(
            output_cls=Simple,
            flexible_mode=True,
            allow_parallel_tool_calls=False,
        )
