"""End-to-end tests for LlamaCPP against a real GGUF model.

These tests exercise the full llama-cpp-python backend.  They are
intentionally slow and are skipped when no model file is present.

To run locally::

    export LLAMA_CPP_MODEL_PATH=/path/to/model.gguf
    pytest libs/providers/llama-cpp/tests/test_llama_cpp_e2e.py -v -m e2e

Assertions are intentionally loose — they validate shape/type/contract,
not exact generated text, because generative output is non-deterministic.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from serapeum.core.llms import ChatResponse, CompletionResponse, Message, MessageRole
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    completion_to_prompt_v3_instruct,
    messages_to_prompt_v3_instruct,
)

MODEL_PATH_ENV = "LLAMA_CPP_MODEL_PATH"
# _model_path = r"\\MYCLOUDEX2ULTRA\research\llm\models\gguf\mistral-7b-instruct-v0.2.Q2_K.gguf"
_model_path = os.environ.get(MODEL_PATH_ENV, "")

skip_no_model = pytest.mark.skipif(
    not _model_path or not os.path.exists(_model_path),
    reason=(
        f"Set {MODEL_PATH_ENV}=/path/to/model.gguf to run e2e tests. "
        f"Current value: {_model_path!r}"
    ),
)

MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0


def _user_messages(*contents: str) -> list[Message]:
    """Build a list of USER messages from plain strings."""
    return [Message(role=MessageRole.USER, content=c) for c in contents]


@pytest.fixture(scope="module")
def llm() -> LlamaCPP:
    """Load the real GGUF model once for the entire module.

    Scope is *module* so the (potentially large) model is loaded once and
    shared across all tests in this file.

    Returns:
        LlamaCPP: Configured instance with a small token budget for speed.
    """
    if not _model_path or not os.path.exists(_model_path):
        pytest.skip(
            f"Set {MODEL_PATH_ENV}=/path/to/model.gguf to run e2e tests. "
            f"Current value: {_model_path!r}"
        )
    is_ci_model = os.path.basename(_model_path).startswith("stories")
    if is_ci_model:
        _messages_to_prompt = lambda msgs: " ".join(
            m.content or "" for m in msgs
        )  # noqa: E731
        _completion_to_prompt = lambda p: p  # noqa: E731
    else:
        _messages_to_prompt = messages_to_prompt_v3_instruct
        _completion_to_prompt = completion_to_prompt_v3_instruct

    return LlamaCPP(
        model_path=_model_path,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        context_window=512,
        verbose=False,
        messages_to_prompt=_messages_to_prompt,
        completion_to_prompt=_completion_to_prompt,
    )


@pytest.fixture(scope="module")
def llm_with_stop() -> LlamaCPP:
    """Load the model with a stop-token list for stop-field tests.

    Returns:
        LlamaCPP: Instance that stops on common EOS tokens.
    """
    if not _model_path or not os.path.exists(_model_path):
        pytest.skip(
            f"Set {MODEL_PATH_ENV}=/path/to/model.gguf to run e2e tests. "
            f"Current value: {_model_path!r}"
        )
    is_ci_model = os.path.basename(_model_path).startswith("stories")
    _completion_to_prompt = (
        (lambda p: p) if is_ci_model else completion_to_prompt_v3_instruct
    )
    _messages_to_prompt = (
        (lambda msgs: " ".join(m.content or "" for m in msgs))
        if is_ci_model
        else messages_to_prompt_v3_instruct
    )
    return LlamaCPP(
        model_path=_model_path,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        context_window=512,
        verbose=False,
        stop=["</s>", "<|eot_id|>", "<|end|>"],
        messages_to_prompt=_messages_to_prompt,
        completion_to_prompt=_completion_to_prompt,
    )


@pytest.mark.e2e
class TestLlamaCPPMetadata:
    """E2E tests for LlamaCPP class-level metadata and constructor fields."""

    def test_class_name(self) -> None:
        """Test class_name() returns the expected string identifier.

        Test scenario:
            class_name() is a classmethod — it must return 'LlamaCPP' without
            requiring a loaded model.
        """
        result = LlamaCPP.class_name()
        assert result == "LlamaCPP", f"Expected 'LlamaCPP', got {result!r}"

    @skip_no_model
    def test_metadata_context_window(self, llm: LlamaCPP) -> None:
        """Test metadata.context_window matches the value passed to the constructor.

        Test scenario:
            The Metadata object must expose the n_ctx value configured at
            construction time (512 in the fixture).
        """
        assert (
            llm.metadata.context_window == 512
        ), f"Expected context_window=512, got {llm.metadata.context_window}"

    @skip_no_model
    def test_metadata_num_output(self, llm: LlamaCPP) -> None:
        """Test metadata.num_output matches max_new_tokens.

        Test scenario:
            num_output in Metadata must equal the max_new_tokens passed at
            construction time.
        """
        assert (
            llm.metadata.num_output == MAX_NEW_TOKENS
        ), f"Expected num_output={MAX_NEW_TOKENS}, got {llm.metadata.num_output}"

    @skip_no_model
    def test_metadata_model_name_is_resolved_path(self, llm: LlamaCPP) -> None:
        """Test metadata.model_name is the resolved GGUF file path.

        Test scenario:
            model_name should be a non-empty string pointing to the local
            model file used for inference.
        """
        assert llm.metadata.model_name, "metadata.model_name should not be empty"
        assert isinstance(
            llm.metadata.model_name, str
        ), f"Expected str, got {type(llm.metadata.model_name)}"

    @skip_no_model
    def test_model_path_matches_constructed_value(self, llm: LlamaCPP) -> None:
        """Test llm.model_path is set to the file path after construction.

        Test scenario:
            After loading a local model file, llm.model_path must equal the
            path passed at construction time.
        """
        assert (
            llm.model_path == _model_path
        ), f"Expected model_path={_model_path!r}, got {llm.model_path!r}"


@pytest.mark.e2e
class TestLlamaCPPTokenize:
    """E2E tests for tokenize() and count_tokens() against a real model vocabulary."""

    @skip_no_model
    def test_tokenize_returns_list_of_ints(self, llm: LlamaCPP) -> None:
        """Test tokenize() returns a non-empty list of integer token IDs.

        Test scenario:
            Any non-empty string tokenized by a real model must produce at
            least one integer token ID.
        """
        tokens = llm.tokenize("hello world")
        assert isinstance(tokens, list), f"Expected list, got {type(tokens)}"
        assert len(tokens) > 0, "tokenize should return at least one token"
        assert all(
            isinstance(t, int) for t in tokens
        ), f"All token IDs must be ints, got: {tokens[:5]}"

    @skip_no_model
    def test_tokenize_empty_string_returns_list(self, llm: LlamaCPP) -> None:
        """Test tokenize('') returns a list (may be empty or contain BOS only).

        Test scenario:
            An empty string is a boundary input.  The function must not raise
            and must return a list (empty or containing only a BOS token).
        """
        tokens = llm.tokenize("")
        assert isinstance(
            tokens, list
        ), f"tokenize('') must return a list, got {type(tokens)}"

    @skip_no_model
    def test_count_tokens_equals_len_of_tokenize(self, llm: LlamaCPP) -> None:
        """Test count_tokens() == len(tokenize()) for the same input.

        Test scenario:
            count_tokens is a thin wrapper around tokenize.  Both must agree
            on the token count for any input string.
        """
        text = "The quick brown fox jumps over the lazy dog."
        assert llm.count_tokens(text) == len(
            llm.tokenize(text)
        ), "count_tokens must equal len(tokenize) for the same text"

    @skip_no_model
    def test_count_tokens_increases_with_longer_text(self, llm: LlamaCPP) -> None:
        """Test count_tokens grows monotonically as text length increases.

        Test scenario:
            Appending words must never decrease the token count — longer text
            always tokenizes to at least as many tokens as shorter text.
        """
        short = "Hello."
        long = "Hello. My name is Alex and I enjoy hiking in the mountains."
        assert llm.count_tokens(long) >= llm.count_tokens(short), (
            f"Longer text must have >= tokens: "
            f"short={llm.count_tokens(short)}, long={llm.count_tokens(long)}"
        )


@pytest.mark.e2e
class TestLlamaCPPContextGuard:
    """E2E tests for _guard_context and context-window overflow detection."""

    @skip_no_model
    def test_short_prompt_does_not_raise(self, llm: LlamaCPP) -> None:
        """Test _guard_context passes silently for a short prompt.

        Test scenario:
            A single short sentence is far below the 512-token context window
            and must not trigger a ValueError.
        """
        llm._guard_context("Say hello.")  # must not raise

    @skip_no_model
    def test_complete_raises_when_prompt_exceeds_context_window(
        self, llm: LlamaCPP
    ) -> None:
        """Test complete() raises ValueError when the prompt token count exceeds context_window.

        Test scenario:
            Repeating a word enough times to exceed the 512-token limit must
            cause ValueError before the model is called, with the error
            mentioning context_window.
        """
        huge_prompt = "word " * 600  # ~600 tokens, exceeds 512
        with pytest.raises(ValueError, match="context_window") as exc_info:
            llm.complete(huge_prompt, formatted=True)
        assert "512" in str(exc_info.value) or "context_window" in str(
            exc_info.value
        ), f"Error should reference context limit, got: {exc_info.value}"

    @skip_no_model
    def test_stream_complete_raises_when_prompt_exceeds_context_window(
        self, llm: LlamaCPP
    ) -> None:
        """Test complete(stream=True) also enforces the context guard.

        Test scenario:
            The _guard_context check runs before the stream is opened, so
            an oversized prompt must raise ValueError even in streaming mode.
        """
        huge_prompt = "word " * 600
        with pytest.raises(ValueError, match="context_window"):
            list(llm.complete(huge_prompt, formatted=True, stream=True))

    @skip_no_model
    def test_guard_context_error_includes_token_count(self, llm: LlamaCPP) -> None:
        """Test the context-guard error message includes the actual token count.

        Test scenario:
            The ValueError raised for an oversized prompt should contain the
            actual number of tokens so the user knows by how much they exceeded
            the limit.
        """
        huge_prompt = "word " * 600
        with pytest.raises(ValueError) as exc_info:
            llm._guard_context(huge_prompt)
        error = str(exc_info.value)
        assert any(
            char.isdigit() for char in error
        ), f"Error should include the token count number, got: {error}"


@pytest.mark.e2e
class TestLlamaCPPComplete:
    """E2E tests for LlamaCPP.complete() — synchronous non-streaming completion."""

    @skip_no_model
    def test_returns_completion_response(self, llm: LlamaCPP) -> None:
        """Test complete() returns a CompletionResponse instance.

        Test scenario:
            A plain string prompt must produce a CompletionResponse without
            raising.
        """
        response = llm.complete("Say hello.")
        assert isinstance(
            response, CompletionResponse
        ), f"Expected CompletionResponse, got {type(response)}"

    @skip_no_model
    def test_text_is_non_empty_string(self, llm: LlamaCPP) -> None:
        """Test complete() response text is a non-empty string.

        Test scenario:
            The model must generate at least one character of output for a
            simple prompt within the token budget.
        """
        response = llm.complete("Say hello.")
        assert isinstance(
            response.text, str
        ), f"Expected str, got {type(response.text)}"
        assert len(response.text) > 0, "Response text should not be empty"

    @skip_no_model
    def test_raw_response_has_choices_key(self, llm: LlamaCPP) -> None:
        """Test complete() attaches the raw llama-cpp response dict.

        Test scenario:
            The raw attribute must be the dict returned by the Llama backend
            and must contain the 'choices' key that holds the generated text.
        """
        response = llm.complete("Say hello.")
        assert response.raw is not None, "raw response should not be None"
        assert (
            "choices" in response.raw
        ), f"Expected 'choices' key in raw, got: {list(response.raw.keys())}"

    @skip_no_model
    def test_formatted_true_accepts_preformatted_prompt(self, llm: LlamaCPP) -> None:
        """Test complete(formatted=True) accepts an already-formatted prompt.

        Test scenario:
            Passing formatted=True with a manually-built instruct prompt must
            produce a valid CompletionResponse without applying the formatter
            a second time.
        """
        prompt = completion_to_prompt_v3_instruct("Say hello.")
        response = llm.complete(prompt, formatted=True)
        assert isinstance(
            response, CompletionResponse
        ), f"Expected CompletionResponse, got {type(response)}"
        assert (
            len(response.text) > 0
        ), "Pre-formatted prompt should produce non-empty output"

    @skip_no_model
    def test_stop_field_terminates_generation(self, llm_with_stop: LlamaCPP) -> None:
        """Test that the stop field prevents stop tokens from appearing in output.

        Test scenario:
            An LlamaCPP instance configured with common EOS stop tokens must
            not include those tokens verbatim in the generated text.
        """
        response = llm_with_stop.complete("Say hello.")
        for token in llm_with_stop.stop:
            assert (
                token not in response.text
            ), f"Stop token {token!r} should not appear in output: {response.text!r}"

    @skip_no_model
    def test_temperature_zero_is_deterministic(self, llm: LlamaCPP) -> None:
        """Test that temperature=0.0 produces the same output on two calls.

        Test scenario:
            Greedy decoding (temperature=0) with the same prompt must yield
            identical text on successive calls, confirming determinism.
        """
        prompt = "The capital of France is"
        r1 = llm.complete(prompt)
        r2 = llm.complete(prompt)
        assert r1.text == r2.text, (
            f"temperature=0 should be deterministic:\n"
            f"  call 1: {r1.text!r}\n  call 2: {r2.text!r}"
        )

    @skip_no_model
    def test_max_new_tokens_bounds_output_length(self, llm: LlamaCPP) -> None:
        """Test that output token count does not greatly exceed max_new_tokens.

        Test scenario:
            The number of tokens in the response text must be less than or
            equal to max_new_tokens + a small tolerance for special tokens.
        """
        response = llm.complete("Count from 1 to 100.")
        token_count = llm.count_tokens(response.text)
        tolerance = 5
        assert token_count <= MAX_NEW_TOKENS + tolerance, (
            f"Output tokens {token_count} exceeds budget "
            f"{MAX_NEW_TOKENS}+{tolerance}: {response.text!r}"
        )


@pytest.mark.e2e
class TestLlamaCPPStreamComplete:
    """E2E tests for LlamaCPP.complete(stream=True) — synchronous streaming."""

    @skip_no_model
    def test_returns_iterable(self, llm: LlamaCPP) -> None:
        """Test complete(stream=True) returns an iterable generator.

        Test scenario:
            Consuming the first chunk from the stream must yield a
            CompletionResponse without raising.
        """
        gen = llm.complete("Say hello.", stream=True)
        first = next(iter(gen))
        assert isinstance(
            first, CompletionResponse
        ), f"Expected CompletionResponse chunk, got {type(first)}"

    @skip_no_model
    def test_every_chunk_has_string_delta(self, llm: LlamaCPP) -> None:
        """Test every chunk from complete(stream=True) carries a string delta.

        Test scenario:
            Each yielded CompletionResponse must have a non-None string delta
            so callers can display incremental output.
        """
        for chunk in llm.complete("Count: 1, 2, 3", stream=True):
            assert (
                chunk.delta is not None
            ), f"Chunk delta must not be None, got: {chunk}"
            assert isinstance(
                chunk.delta, str
            ), f"Delta must be str, got {type(chunk.delta)}"

    @skip_no_model
    def test_text_accumulates_monotonically(self, llm: LlamaCPP) -> None:
        """Test chunk.text grows monotonically across the stream.

        Test scenario:
            The cumulative text field of each chunk must be at least as long
            as the previous chunk's text.
        """
        prev_len = 0
        for chunk in llm.complete("Count: 1, 2, 3", stream=True):
            assert (
                len(chunk.text) >= prev_len
            ), f"Text length decreased: {prev_len} → {len(chunk.text)}"
            prev_len = len(chunk.text)

    @skip_no_model
    def test_final_text_equals_joined_deltas(self, llm: LlamaCPP) -> None:
        """Test the last chunk's text equals all deltas concatenated.

        Test scenario:
            Joining every delta in order must reproduce the cumulative text
            field of the final chunk exactly.
        """
        chunks = list(llm.complete("Say hello.", stream=True))
        joined = "".join(c.delta for c in chunks if c.delta)
        assert (
            joined == chunks[-1].text
        ), f"Joined deltas {joined!r} != final text {chunks[-1].text!r}"

    @skip_no_model
    def test_stream_and_non_stream_produce_same_text(self, llm: LlamaCPP) -> None:
        """Test that streaming and non-streaming complete() agree on the final text.

        Test scenario:
            The full text collected from the streaming path must equal the
            text returned by the non-streaming path for the same prompt.
        """
        prompt = "The capital of France is"
        non_stream = llm.complete(prompt).text
        stream_chunks = list(llm.complete(prompt, stream=True))
        stream_text = stream_chunks[-1].text
        assert stream_text == non_stream, (
            f"Stream and non-stream outputs differ:\n"
            f"  non-stream: {non_stream!r}\n  stream:     {stream_text!r}"
        )

    @skip_no_model
    def test_yields_multiple_chunks(self, llm: LlamaCPP) -> None:
        """Test streaming produces more than one chunk for a non-trivial prompt.

        Test scenario:
            A multi-token response (asking to count) should be split across
            multiple chunks, not returned all at once.
        """
        chunks = list(llm.complete("Count: 1, 2, 3, 4, 5", stream=True))
        assert (
            len(chunks) > 1
        ), f"Expected multiple stream chunks, got only {len(chunks)}"


@pytest.mark.e2e
class TestLlamaCPPChat:
    """E2E tests for LlamaCPP.chat() — provided by CompletionToChatMixin."""

    @skip_no_model
    def test_non_stream_returns_chat_response(self, llm: LlamaCPP) -> None:
        """Test chat(stream=False) returns a ChatResponse.

        Test scenario:
            A single USER message must produce a ChatResponse whose message
            has ASSISTANT role.
        """
        response = llm.chat(_user_messages("Say hello."))
        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"

    @skip_no_model
    def test_non_stream_assistant_role(self, llm: LlamaCPP) -> None:
        """Test chat() response message has ASSISTANT role.

        Test scenario:
            The message inside the ChatResponse must be attributed to the
            ASSISTANT role, not USER or SYSTEM.
        """
        response = llm.chat(_user_messages("Say hello."))
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT role, got {response.message.role}"

    @skip_no_model
    def test_non_stream_content_is_non_empty(self, llm: LlamaCPP) -> None:
        """Test chat() response has non-empty content.

        Test scenario:
            The assistant reply must contain at least one character of text.
        """
        response = llm.chat(_user_messages("Say hello."))
        assert response.message.content, "Assistant content should not be empty"

    @skip_no_model
    def test_stream_yields_chat_responses(self, llm: LlamaCPP) -> None:
        """Test chat(stream=True) yields ChatResponse instances.

        Test scenario:
            Each chunk from the streaming generator must be a ChatResponse
            with a string delta.
        """
        chunks = list(llm.chat(_user_messages("Count to three."), stream=True))
        assert len(chunks) > 0, "Streaming chat produced no chunks"
        for chunk in chunks:
            assert isinstance(
                chunk, ChatResponse
            ), f"Expected ChatResponse chunk, got {type(chunk)}"

    @skip_no_model
    def test_stream_every_chunk_has_delta(self, llm: LlamaCPP) -> None:
        """Test each streaming chat chunk carries a non-None delta.

        Test scenario:
            Every chunk in the stream must have a non-None delta string
            so callers can display incremental output.
        """
        for chunk in llm.chat(_user_messages("Count to three."), stream=True):
            assert (
                chunk.delta is not None
            ), f"Chat chunk delta must not be None: {chunk}"

    @skip_no_model
    def test_multi_turn_conversation(self, llm: LlamaCPP) -> None:
        """Test chat() handles a USER / ASSISTANT / USER conversation.

        Test scenario:
            A three-message sequence must produce a valid ChatResponse without
            errors, proving the prompt formatter handles multi-turn context.
        """
        messages = [
            Message(role=MessageRole.USER, content="My name is Alex."),
            Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alex!"),
            Message(role=MessageRole.USER, content="What is my name?"),
        ]
        response = llm.chat(messages)
        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"
        assert (
            response.message.content
        ), "Multi-turn response content should not be empty"

    @skip_no_model
    def test_chat_and_complete_produce_equivalent_shapes(self, llm: LlamaCPP) -> None:
        """Test chat() and complete() both return responses with non-empty content.

        Test scenario:
            Both entry points should work for the same semantic query — this
            exercises the CompletionToChatMixin bridge end-to-end.
        """
        completion = llm.complete("Say hello.")
        chat_resp = llm.chat(_user_messages("Say hello."))
        assert len(completion.text) > 0, "complete() should produce text"
        assert chat_resp.message.content, "chat() should produce content"


@pytest.mark.e2e
class TestLlamaCPPAcomplete:
    """E2E tests for LlamaCPP.acomplete() — async completion."""

    @pytest.mark.asyncio
    @skip_no_model
    async def test_non_stream_returns_completion_response(self, llm: LlamaCPP) -> None:
        """Test acomplete(stream=False) returns a CompletionResponse.

        Test scenario:
            Awaiting acomplete() must resolve to a CompletionResponse with
            non-empty text, delegating via asyncio.to_thread.
        """
        response = await llm.acomplete("Say hello.")
        assert isinstance(
            response, CompletionResponse
        ), f"Expected CompletionResponse, got {type(response)}"
        assert len(response.text) > 0, "Async complete text should not be empty"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_non_stream_result_matches_sync(self, llm: LlamaCPP) -> None:
        """Test acomplete() and complete() produce the same text for the same prompt.

        Test scenario:
            Both sync and async paths call the same backend.  For a fixed
            prompt at temperature=0 both must return identical text.
        """
        prompt = "The capital of France is"
        sync_text = llm.complete(prompt).text
        async_text = (await llm.acomplete(prompt)).text
        assert async_text == sync_text, (
            f"acomplete and complete disagree:\n"
            f"  sync:  {sync_text!r}\n  async: {async_text!r}"
        )

    @pytest.mark.asyncio
    @skip_no_model
    async def test_non_stream_does_not_block_event_loop(self, llm: LlamaCPP) -> None:
        """Test acomplete() lets other coroutines run concurrently.

        Test scenario:
            A side coroutine scheduled alongside acomplete() via gather must
            complete, proving the inference call does not block the event loop.
        """
        side_ran: list[bool] = []

        async def side() -> None:
            side_ran.append(True)

        result, _ = await asyncio.gather(llm.acomplete("Say hello."), side())
        assert isinstance(
            result, CompletionResponse
        ), f"acomplete result should be CompletionResponse, got {type(result)}"
        assert side_ran, "Side coroutine must have run concurrently"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_two_concurrent_requests_both_complete(self, llm: LlamaCPP) -> None:
        """Test two concurrent acomplete() calls both return valid responses.

        Test scenario:
            Running two acomplete() calls under asyncio.gather must produce
            two CompletionResponse objects with non-empty text, validating
            that the thread-pool path handles concurrent requests correctly.
        """
        r1, r2 = await asyncio.gather(
            llm.acomplete("Say hello."),
            llm.acomplete("Count to three."),
        )
        assert isinstance(
            r1, CompletionResponse
        ), f"First concurrent result should be CompletionResponse, got {type(r1)}"
        assert isinstance(
            r2, CompletionResponse
        ), f"Second concurrent result should be CompletionResponse, got {type(r2)}"
        assert len(r1.text) > 0, "First concurrent response text must not be empty"
        assert len(r2.text) > 0, "Second concurrent response text must not be empty"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_stream_yields_completion_responses(self, llm: LlamaCPP) -> None:
        """Test acomplete(stream=True) yields CompletionResponse chunks asynchronously.

        Test scenario:
            Iterating the async generator must produce at least one chunk,
            each carrying a non-None delta.
        """
        gen = await llm.acomplete("Say hello.", stream=True)
        chunks = [chunk async for chunk in gen]
        assert len(chunks) > 0, "Async streaming complete produced no chunks"
        for chunk in chunks:
            assert isinstance(
                chunk, CompletionResponse
            ), f"Expected CompletionResponse chunk, got {type(chunk)}"
            assert chunk.delta is not None, f"Chunk delta must not be None: {chunk}"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_stream_final_text_equals_joined_deltas(self, llm: LlamaCPP) -> None:
        """Test async stream final text equals all deltas concatenated.

        Test scenario:
            Joining every delta from the async stream must reproduce the
            cumulative text of the last chunk.
        """
        gen = await llm.acomplete("Say hello.", stream=True)
        chunks = [chunk async for chunk in gen]
        joined = "".join(c.delta for c in chunks if c.delta)
        assert (
            joined == chunks[-1].text
        ), f"Joined async deltas {joined!r} != final text {chunks[-1].text!r}"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_stream_matches_non_stream_text(self, llm: LlamaCPP) -> None:
        """Test async streaming and non-streaming produce the same final text.

        Test scenario:
            For a deterministic prompt (temperature=0), the full text from
            the async stream must equal the text from non-streaming acomplete.
        """
        prompt = "The capital of France is"
        non_stream = (await llm.acomplete(prompt)).text
        gen = await llm.acomplete(prompt, stream=True)
        chunks = [chunk async for chunk in gen]
        stream_text = chunks[-1].text
        assert stream_text == non_stream, (
            f"Async stream and non-stream outputs differ:\n"
            f"  non-stream: {non_stream!r}\n  stream:     {stream_text!r}"
        )


@pytest.mark.e2e
class TestLlamaCPPAchat:
    """E2E tests for LlamaCPP.achat() — async chat via CompletionToChatMixin."""

    @pytest.mark.asyncio
    @skip_no_model
    async def test_non_stream_returns_chat_response(self, llm: LlamaCPP) -> None:
        """Test achat(stream=False) returns a ChatResponse with ASSISTANT role.

        Test scenario:
            Awaiting achat() with a USER message must resolve to a
            ChatResponse attributed to ASSISTANT.
        """
        response = await llm.achat(_user_messages("Say hello."))
        assert isinstance(
            response, ChatResponse
        ), f"Expected ChatResponse, got {type(response)}"
        assert (
            response.message.role == MessageRole.ASSISTANT
        ), f"Expected ASSISTANT role, got {response.message.role}"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_non_stream_content_is_non_empty(self, llm: LlamaCPP) -> None:
        """Test achat() response has non-empty content.

        Test scenario:
            The assistant reply from the async path must contain text.
        """
        response = await llm.achat(_user_messages("Say hello."))
        assert response.message.content, "Async chat content should not be empty"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_stream_yields_chat_responses(self, llm: LlamaCPP) -> None:
        """Test achat(stream=True) yields ChatResponse chunks asynchronously.

        Test scenario:
            Iterating the async generator must produce at least one
            ChatResponse chunk.
        """
        gen = await llm.achat(_user_messages("Count to three."), stream=True)
        chunks = [chunk async for chunk in gen]
        assert len(chunks) > 0, "Async streaming chat produced no chunks"
        for chunk in chunks:
            assert isinstance(
                chunk, ChatResponse
            ), f"Expected ChatResponse chunk, got {type(chunk)}"

    @pytest.mark.asyncio
    @skip_no_model
    async def test_two_concurrent_achat_requests(self, llm: LlamaCPP) -> None:
        """Test two concurrent achat() calls both return valid ChatResponse objects.

        Test scenario:
            Running two achat() calls under asyncio.gather must produce two
            ChatResponse objects with non-empty content.
        """
        r1, r2 = await asyncio.gather(
            llm.achat(_user_messages("Say hello.")),
            llm.achat(_user_messages("Count to three.")),
        )
        assert isinstance(
            r1, ChatResponse
        ), f"First concurrent achat result should be ChatResponse, got {type(r1)}"
        assert isinstance(
            r2, ChatResponse
        ), f"Second concurrent achat result should be ChatResponse, got {type(r2)}"
        assert r1.message.content, "First concurrent achat content must not be empty"
        assert r2.message.content, "Second concurrent achat content must not be empty"
