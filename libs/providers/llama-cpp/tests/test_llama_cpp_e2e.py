"""End-to-end tests for LlamaCPP.

These tests exercise the real llama-cpp-python backend against a local GGUF
model file.  They are intentionally slow and are skipped in CI unless an
actual model is present.

To run them locally::

    # Point at any small GGUF model you have on disk
    export LLAMA_CPP_MODEL_PATH=/path/to/model.gguf
    pytest libs/providers/llama-cpp/tests/test_llama_cpp_e2e.py -v -m e2e

The tests use a very small ``max_new_tokens`` budget (32 tokens) so they
complete quickly regardless of model size.  The assertions are intentionally
loose — they only check the shape/type of the response, not the exact text,
because generative output is non-deterministic.
"""

import os
import pytest

from serapeum.core.llms import Message, MessageRole, ChatResponse, CompletionResponse
from serapeum.llama_cpp import LlamaCPP

from serapeum.llama_cpp.utils import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

pytestmark = pytest.mark.e2e

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

MAX_NEW_TOKENS = 32  # keep tests fast regardless of model size
TEMPERATURE = 0.0   # deterministic greedy decoding where supported


def _user_messages(*contents: str) -> list[Message]:
    """Build a list of USER messages from plain strings."""
    return [Message(role=MessageRole.USER, content=c) for c in contents]



@pytest.fixture(scope="module")
def llm() -> LlamaCPP:
    """Create a LlamaCPP instance pointing at the local model.

    Scope is *module* so the (potentially large) model is loaded once and
    shared across all tests in this file.

    Returns:
        LlamaCPP: Configured instance with minimal token budget for speed.
    """
    if not _model_path or not os.path.exists(_model_path):
        pytest.skip(
            f"Set {MODEL_PATH_ENV}=/path/to/model.gguf to run e2e tests. "
            f"Current value: {_model_path!r}"
        )
    # Detect tiny CI model (stories260K) by filename and fall back to a plain
    # prompt formatter — the TinyStories base model has no instruct format.
    is_ci_model = os.path.basename(_model_path).startswith("stories")
    if is_ci_model:
        _messages_to_prompt = lambda msgs: " ".join(m.content or "" for m in msgs)  # noqa: E731
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


# ---------------------------------------------------------------------------
# TestLlamaCPPMetadata
# ---------------------------------------------------------------------------

class TestLlamaCPPMetadata:
    """Tests for LlamaCPP class-level and metadata API."""

    def test_class_name(self):
        """Test class_name() returns the expected string identifier.

        Test scenario:
            class_name() is a classmethod and should always return 'LlamaCPP'
            without needing a model instance.
        """
        result = LlamaCPP.class_name()
        assert result == "LlamaCPP", f"Expected 'LlamaCPP', got {result!r}"

    @skip_no_model
    def test_metadata_context_window(self, llm: LlamaCPP):
        """Test metadata.context_window reflects the configured value.

        Test scenario:
            The Metadata object should expose the n_ctx value passed to the
            constructor as context_window.
        """
        assert llm.metadata.context_window == 512, (
            f"Expected context_window=512, got {llm.metadata.context_window}"
        )

    @skip_no_model
    def test_metadata_num_output(self, llm: LlamaCPP):
        """Test metadata.num_output reflects max_new_tokens.

        Test scenario:
            num_output in metadata should match the max_new_tokens passed at
            construction time.
        """
        assert llm.metadata.num_output == MAX_NEW_TOKENS, (
            f"Expected num_output={MAX_NEW_TOKENS}, got {llm.metadata.num_output}"
        )

    @skip_no_model
    def test_metadata_model_name(self, llm: LlamaCPP):
        """Test metadata.model_name is a non-empty string.

        Test scenario:
            model_name should be the resolved path to the GGUF file.
        """
        assert llm.metadata.model_name, "metadata.model_name should not be empty"
        assert isinstance(llm.metadata.model_name, str), (
            f"Expected str, got {type(llm.metadata.model_name)}"
        )


# ---------------------------------------------------------------------------
# TestLlamaCPPComplete
# ---------------------------------------------------------------------------

class TestLlamaCPPComplete:
    """Tests for LlamaCPP.complete() — synchronous non-streaming completion."""

    @skip_no_model
    def test_returns_completion_response(self, llm: LlamaCPP):
        """Test complete() returns a CompletionResponse instance.

        Test scenario:
            A plain string prompt should produce a CompletionResponse with a
            non-empty text attribute.
        """
        response = llm.complete("Say hello.")
        assert isinstance(response, CompletionResponse), (
            f"Expected CompletionResponse, got {type(response)}"
        )

    @skip_no_model
    def test_text_is_non_empty_string(self, llm: LlamaCPP):
        """Test complete() response text is a non-empty string.

        Test scenario:
            The text attribute should contain at least one character of output.
        """
        response = llm.complete("Say hello.")
        assert isinstance(response.text, str), (
            f"Expected str, got {type(response.text)}"
        )
        assert len(response.text) > 0, "Response text should not be empty"

    @skip_no_model
    def test_raw_response_attached(self, llm: LlamaCPP):
        """Test complete() attaches the raw llama-cpp response dict.

        Test scenario:
            The raw attribute should be the dict returned by the Llama model
            and contain the 'choices' key.
        """
        response = llm.complete("Say hello.")
        assert response.raw is not None, "raw response should not be None"
        assert "choices" in response.raw, (
            f"Expected 'choices' key in raw response, got keys: {list(response.raw.keys())}"
        )

    @skip_no_model
    def test_formatted_true_skips_completion_to_prompt(self, llm: LlamaCPP):
        """Test complete(formatted=True) passes the prompt through unchanged.

        Test scenario:
            When formatted=True, completion_to_prompt is NOT called.  We pass
            an already-formatted Llama-3 prompt and verify we still get a valid
            CompletionResponse back (i.e., no crash and non-empty text).
        """
        prompt = completion_to_prompt_v3_instruct("Say hello.")
        response = llm.complete(prompt, formatted=True)
        assert isinstance(response, CompletionResponse), (
            f"Expected CompletionResponse, got {type(response)}"
        )
        assert len(response.text) > 0, "Response text should not be empty"


# ---------------------------------------------------------------------------
# TestLlamaCPPStreamComplete
# ---------------------------------------------------------------------------

class TestLlamaCPPStreamComplete:
    """Tests for LlamaCPP.complete(stream=True) — synchronous streaming completion."""

    @skip_no_model
    def test_returns_generator(self, llm: LlamaCPP):
        """Test complete(stream=True) returns an iterable generator.

        Test scenario:
            The return value should be iterable — consuming one chunk should
            not raise.
        """
        gen = llm.complete("Say hello.", stream=True)
        first_chunk = next(iter(gen))
        assert isinstance(first_chunk, CompletionResponse), (
            f"Expected CompletionResponse chunk, got {type(first_chunk)}"
        )

    @skip_no_model
    def test_chunks_have_delta(self, llm: LlamaCPP):
        """Test each chunk from complete(stream=True) carries a delta string.

        Test scenario:
            Every yielded CompletionResponse should have a non-None delta
            representing the incremental token text.
        """
        for chunk in llm.complete("Count: 1, 2, 3", stream=True):
            assert chunk.delta is not None, (
                f"Chunk delta should not be None, got chunk: {chunk}"
            )
            assert isinstance(chunk.delta, str), (
                f"Delta should be str, got {type(chunk.delta)}"
            )

    @skip_no_model
    def test_text_accumulates_across_chunks(self, llm: LlamaCPP):
        """Test that chunk.text grows monotonically across the stream.

        Test scenario:
            Each successive chunk's text should be at least as long as the
            previous one, since text is the cumulative concatenation of deltas.
        """
        prev_len = 0
        for chunk in llm.complete("Count: 1, 2, 3", stream=True):
            assert len(chunk.text) >= prev_len, (
                f"Text length decreased: {prev_len} -> {len(chunk.text)}"
            )
            prev_len = len(chunk.text)

    @skip_no_model
    def test_full_text_matches_joined_deltas(self, llm: LlamaCPP):
        """Test that the final chunk text equals all deltas concatenated.

        Test scenario:
            Streaming is consistent: joining every delta should reproduce the
            text field of the last chunk.
        """
        chunks = list(llm.complete("Say hello.", stream=True))
        joined = "".join(c.delta for c in chunks if c.delta)
        final_text = chunks[-1].text
        assert joined == final_text, (
            f"Joined deltas {joined!r} != final text {final_text!r}"
        )


# ---------------------------------------------------------------------------
# TestLlamaCPPChat
# ---------------------------------------------------------------------------

class TestLlamaCPPChat:
    """Tests for LlamaCPP.chat() — provided by CompletionToChatMixin."""

    @skip_no_model
    def test_non_stream_returns_chat_response(self, llm: LlamaCPP):
        """Test chat(stream=False) returns a ChatResponse.

        Test scenario:
            A single USER message should produce a ChatResponse whose message
            role is ASSISTANT.
        """
        messages = _user_messages("Say hello.")
        response = llm.chat(messages)
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )

    @skip_no_model
    def test_non_stream_assistant_role(self, llm: LlamaCPP):
        """Test chat() response has ASSISTANT role.

        Test scenario:
            The message inside the ChatResponse should be attributed to the
            ASSISTANT role.
        """
        messages = _user_messages("Say hello.")
        response = llm.chat(messages)
        assert response.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {response.message.role}"
        )

    @skip_no_model
    def test_non_stream_content_is_non_empty(self, llm: LlamaCPP):
        """Test chat() response has non-empty content.

        Test scenario:
            The assistant reply should contain at least one character.
        """
        messages = _user_messages("Say hello.")
        response = llm.chat(messages)
        assert response.message.content, "Assistant content should not be empty"

    @skip_no_model
    def test_stream_yields_chat_responses(self, llm: LlamaCPP):
        """Test chat(stream=True) yields ChatResponse instances.

        Test scenario:
            Each chunk from the streaming generator should be a ChatResponse
            with a string delta.
        """
        messages = _user_messages("Count to three.")
        chunks = list(llm.chat(messages, stream=True))
        assert len(chunks) > 0, "Streaming chat produced no chunks"
        for chunk in chunks:
            assert isinstance(chunk, ChatResponse), (
                f"Expected ChatResponse chunk, got {type(chunk)}"
            )

    @skip_no_model
    def test_stream_chunks_have_delta(self, llm: LlamaCPP):
        """Test each streaming chat chunk carries a delta.

        Test scenario:
            Every chunk in the stream should have a non-None delta string so
            callers can display incremental output.
        """
        messages = _user_messages("Count to three.")
        for chunk in llm.chat(messages, stream=True):
            assert chunk.delta is not None, (
                f"Chat chunk delta should not be None: {chunk}"
            )

    @skip_no_model
    def test_multi_turn_conversation(self, llm: LlamaCPP):
        """Test chat() handles a multi-turn conversation.

        Test scenario:
            A USER / ASSISTANT / USER message sequence should still produce a
            valid ChatResponse without errors.
        """
        messages = [
            Message(role=MessageRole.USER, content="My name is Alex."),
            Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alex!"),
            Message(role=MessageRole.USER, content="What is my name?"),
        ]
        response = llm.chat(messages)
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )
        assert response.message.content, "Multi-turn response content should not be empty"


# ---------------------------------------------------------------------------
# TestLlamaCPPAchat
# ---------------------------------------------------------------------------

class TestLlamaCPPAchat:
    """Tests for LlamaCPP.achat() — async chat via CompletionToChatMixin."""

    @skip_no_model
    @pytest.mark.asyncio
    async def test_non_stream_returns_chat_response(self, llm: LlamaCPP):
        """Test achat(stream=False) returns a ChatResponse.

        Test scenario:
            Awaiting achat() with a USER message should resolve to a
            ChatResponse with ASSISTANT role.
        """
        messages = _user_messages("Say hello.")
        response = await llm.achat(messages)
        assert isinstance(response, ChatResponse), (
            f"Expected ChatResponse, got {type(response)}"
        )
        assert response.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {response.message.role}"
        )

    @skip_no_model
    @pytest.mark.asyncio
    async def test_non_stream_content_is_non_empty(self, llm: LlamaCPP):
        """Test achat() response has non-empty content.

        Test scenario:
            The assistant reply from the async path should contain text.
        """
        messages = _user_messages("Say hello.")
        response = await llm.achat(messages)
        assert response.message.content, "Async chat content should not be empty"

    @skip_no_model
    @pytest.mark.asyncio
    async def test_stream_yields_chat_responses(self, llm: LlamaCPP):
        """Test achat(stream=True) yields ChatResponse instances asynchronously.

        Test scenario:
            Iterating the async generator should produce at least one
            ChatResponse chunk.
        """
        messages = _user_messages("Count to three.")
        gen = await llm.achat(messages, stream=True)
        chunks = [chunk async for chunk in gen]
        assert len(chunks) > 0, "Async streaming chat produced no chunks"
        for chunk in chunks:
            assert isinstance(chunk, ChatResponse), (
                f"Expected ChatResponse chunk, got {type(chunk)}"
            )


# ---------------------------------------------------------------------------
# TestLlamaCPPAcomplete
# ---------------------------------------------------------------------------

class TestLlamaCPPAcomplete:
    """Tests for LlamaCPP.acomplete() — async completion via CompletionToChatMixin."""

    @skip_no_model
    @pytest.mark.asyncio
    async def test_non_stream_returns_completion_response(self, llm: LlamaCPP):
        """Test acomplete(stream=False) returns a CompletionResponse.

        Test scenario:
            Awaiting acomplete() should resolve to a CompletionResponse with
            non-empty text, delegating to the synchronous complete().
        """
        response = await llm.acomplete("Say hello.")
        assert isinstance(response, CompletionResponse), (
            f"Expected CompletionResponse, got {type(response)}"
        )
        assert len(response.text) > 0, "Async complete text should not be empty"

    @skip_no_model
    @pytest.mark.asyncio
    async def test_stream_yields_completion_responses(self, llm: LlamaCPP):
        """Test acomplete(stream=True) yields CompletionResponse chunks asynchronously.

        Test scenario:
            Iterating the async generator from acomplete(stream=True) should
            produce at least one CompletionResponse with a delta.
        """
        gen = await llm.acomplete("Say hello.", stream=True)
        chunks = [chunk async for chunk in gen]
        assert len(chunks) > 0, "Async streaming complete produced no chunks"
        for chunk in chunks:
            assert isinstance(chunk, CompletionResponse), (
                f"Expected CompletionResponse chunk, got {type(chunk)}"
            )
            assert chunk.delta is not None, f"Chunk delta should not be None: {chunk}"
