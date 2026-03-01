"""Tests for CompletionToChatMixin."""

import pytest

from serapeum.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    Message,
    MessageRole,
)
from serapeum.core.llms.abstractions.mixins import CompletionToChatMixin


class ConcreteCompletionLLM(CompletionToChatMixin):
    """Minimal concrete class that satisfies CompletionToChatMixin's duck-typed contract.

    Implements the two methods the mixin requires:
    - ``messages_to_prompt`` (callable attribute)
    - ``complete(prompt, formatted=False, *, stream=False, **kwargs)``

    All calls are recorded so tests can assert on forwarded arguments.
    """

    def __init__(self, response_text: str = "OK"):
        """Initialize with a fixed response text.

        Args:
            response_text: The text each ``complete`` call will return.
        """
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.last_formatted: bool | None = None
        self.last_kwargs: dict = {}
        self.complete_call_count = 0   # tracks complete(stream=False) calls
        self.stream_call_count = 0     # tracks complete(stream=True) calls
        self.messages_to_prompt_call_count = 0

    def messages_to_prompt(self, messages) -> str:
        """Convert a sequence of messages to a single prompt string.

        Args:
            messages: Sequence of Message objects.

        Returns:
            A concatenated string of all message contents.
        """
        self.messages_to_prompt_call_count += 1
        return " ".join(m.content or "" for m in messages)

    def complete(self, prompt: str, formatted: bool = False, *, stream: bool = False, **kwargs):
        """Return a CompletionResponse or streaming generator, recording arguments.

        Args:
            prompt: The prompt string.
            formatted: Whether the prompt is already formatted.
            stream: If True, returns a generator yielding two CompletionResponse chunks.
            **kwargs: Extra keyword arguments forwarded by the mixin.

        Returns:
            A CompletionResponse when stream=False, or a generator of
            CompletionResponse objects when stream=True.
        """
        self.last_prompt = prompt
        self.last_formatted = formatted
        self.last_kwargs = kwargs
        if stream:
            self.stream_call_count += 1

            def gen():
                yield CompletionResponse(text="He", delta="He")
                yield CompletionResponse(text="Hello", delta="llo")

            return gen()
        else:
            self.complete_call_count += 1
            return CompletionResponse(text=self.response_text)


def _make_messages(*contents: str) -> list[Message]:
    """Build a list of user Messages from plain text strings.

    Args:
        *contents: One or more message content strings.

    Returns:
        A list of Message objects with USER role.
    """
    return [Message(role=MessageRole.USER, content=c) for c in contents]


@pytest.mark.unit
class TestCompletionToChatMixinChat:
    """Test CompletionToChatMixin.chat()."""

    def test_chat_non_stream_returns_chat_response(self):
        """chat(stream=False) must return a ChatResponse, not a generator.

        Inputs: A single-message list, stream=False (default).
        Expected: The return value is a ChatResponse instance.
        Checks: Type assertion on the return value.
        """
        llm = ConcreteCompletionLLM()
        messages = _make_messages("Hello")
        result = llm.chat(messages)

        assert isinstance(result, ChatResponse), (
            f"Expected ChatResponse, got {type(result)}"
        )

    def test_chat_non_stream_content_matches_completion_response(self):
        """chat(stream=False) content must match the underlying CompletionResponse text.

        Inputs: A single message, configured response_text="World".
        Expected: ChatResponse.message.content == "World".
        Checks: Content equality.
        """
        llm = ConcreteCompletionLLM(response_text="World")
        messages = _make_messages("Hi")
        result = llm.chat(messages)

        assert result.message.content == "World", (
            f"Expected 'World', got {result.message.content!r}"
        )

    def test_chat_non_stream_role_is_assistant(self):
        """chat(stream=False) must produce an ASSISTANT-role message.

        Inputs: Any message list.
        Expected: ChatResponse.message.role == MessageRole.ASSISTANT.
        Checks: Role equality.
        """
        llm = ConcreteCompletionLLM()
        result = llm.chat(_make_messages("ping"))

        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {result.message.role}"
        )

    def test_chat_non_stream_calls_messages_to_prompt(self):
        """chat() must call messages_to_prompt with the messages argument.

        Inputs: Two messages with known contents.
        Expected: messages_to_prompt is called exactly once and uses the messages.
        Checks: Call count and resulting prompt forwarded to complete().
        """
        llm = ConcreteCompletionLLM()
        messages = _make_messages("foo", "bar")
        llm.chat(messages)

        assert llm.messages_to_prompt_call_count == 1, (
            f"Expected messages_to_prompt to be called once, "
            f"was called {llm.messages_to_prompt_call_count} time(s)"
        )
        assert llm.last_prompt == "foo bar", (
            f"Prompt built from messages was {llm.last_prompt!r}, expected 'foo bar'"
        )

    def test_chat_non_stream_calls_complete_with_formatted_true(self):
        """chat() must pass formatted=True when calling complete().

        Inputs: Any message, stream=False.
        Expected: complete() is called with formatted=True because the prompt has
            already been formatted by messages_to_prompt.
        Checks: self.last_formatted is True after the call.
        """
        llm = ConcreteCompletionLLM()
        llm.chat(_make_messages("x"))

        assert llm.last_formatted is True, (
            f"Expected formatted=True to be forwarded to complete(), got {llm.last_formatted}"
        )

    def test_chat_non_stream_forwards_kwargs_to_complete(self):
        """chat() must forward extra kwargs to complete().

        Inputs: Extra kwargs temperature=0.5 and max_tokens=50.
        Expected: complete() receives these exact kwargs.
        Checks: self.last_kwargs matches the extra kwargs.
        """
        llm = ConcreteCompletionLLM()
        llm.chat(_make_messages("test"), temperature=0.5, max_tokens=50)

        assert llm.last_kwargs == {"temperature": 0.5, "max_tokens": 50}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )

    def test_chat_non_stream_result_is_to_chat_response_conversion(self):
        """chat(stream=False) result must equal CompletionResponse.to_chat_response().

        Inputs: Known response_text="check".
        Expected: The ChatResponse returned by chat() matches what
            CompletionResponse(text="check").to_chat_response() would produce.
        Checks: message.content and message.role equality.
        """
        llm = ConcreteCompletionLLM(response_text="check")
        result = llm.chat(_make_messages("q"))
        expected = CompletionResponse(text="check").to_chat_response()

        assert result.message.content == expected.message.content, (
            f"Content mismatch: {result.message.content!r} != {expected.message.content!r}"
        )
        assert result.message.role == expected.message.role, (
            f"Role mismatch: {result.message.role} != {expected.message.role}"
        )

    def test_chat_stream_returns_generator(self):
        """chat(stream=True) must return a generator, not a ChatResponse.

        Inputs: Any message list, stream=True.
        Expected: The return value is a generator object.
        Checks: import types.GeneratorType or check for __next__.
        """
        import types

        llm = ConcreteCompletionLLM()
        result = llm.chat(_make_messages("stream me"), stream=True)

        assert isinstance(result, types.GeneratorType), (
            f"Expected a generator, got {type(result)}"
        )

    def test_chat_stream_yields_chat_response_instances(self):
        """chat(stream=True) must yield ChatResponse objects.

        Inputs: Two-chunk stream from complete(stream=True).
        Expected: Each yielded item is a ChatResponse.
        Checks: Type of every chunk.
        """
        llm = ConcreteCompletionLLM()
        chunks = list(llm.chat(_make_messages("go"), stream=True))

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, ChatResponse), (
                f"Chunk {i} is {type(chunk)}, expected ChatResponse"
            )

    def test_chat_stream_chunk_contents(self):
        """chat(stream=True) chunks must carry the completion deltas.

        Inputs: complete(stream=True) yields ("He", delta="He") and ("Hello", delta="llo").
        Expected: Chunks have matching content and delta values.
        Checks: content and delta on each chunk.
        """
        llm = ConcreteCompletionLLM()
        chunks = list(llm.chat(_make_messages("stream"), stream=True))

        assert chunks[0].message.content == "He", (
            f"First chunk content: {chunks[0].message.content!r}"
        )
        assert chunks[0].delta == "He", (
            f"First chunk delta: {chunks[0].delta!r}"
        )
        assert chunks[1].message.content == "Hello", (
            f"Second chunk content: {chunks[1].message.content!r}"
        )
        assert chunks[1].delta == "llo", (
            f"Second chunk delta: {chunks[1].delta!r}"
        )

    def test_chat_stream_calls_complete_with_formatted_true(self):
        """chat(stream=True) must call complete(stream=True) with formatted=True.

        Inputs: Any message, stream=True.
        Expected: complete receives formatted=True.
        Checks: self.last_formatted after consuming the generator.
        """
        llm = ConcreteCompletionLLM()
        list(llm.chat(_make_messages("x"), stream=True))

        assert llm.last_formatted is True, (
            f"Expected formatted=True forwarded to complete(stream=True), got {llm.last_formatted}"
        )

    def test_chat_stream_forwards_kwargs_to_complete(self):
        """chat(stream=True) must forward extra kwargs to complete(stream=True).

        Inputs: Extra kwargs top_p=0.9 and stop=["<|end|>"].
        Expected: complete() receives these exact kwargs.
        Checks: self.last_kwargs matches after consuming the generator.
        """
        llm = ConcreteCompletionLLM()
        list(llm.chat(_make_messages("kw"), stream=True, top_p=0.9, stop=["<|end|>"]))

        assert llm.last_kwargs == {"top_p": 0.9, "stop": ["<|end|>"]}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["no-stream", "stream"])
    def test_chat_calls_messages_to_prompt_exactly_once(self, stream: bool):
        """chat() must call messages_to_prompt exactly once regardless of stream flag.

        Args:
            stream: Whether streaming is enabled.

        Inputs: A single message, with both stream=False and stream=True.
        Expected: messages_to_prompt is called exactly once in each case.
        Checks: Call count on messages_to_prompt_call_count.
        """
        llm = ConcreteCompletionLLM()
        result = llm.chat(_make_messages("once"), stream=stream)
        if stream:
            list(result)

        assert llm.messages_to_prompt_call_count == 1, (
            f"messages_to_prompt called {llm.messages_to_prompt_call_count} times "
            f"(stream={stream}), expected 1"
        )


@pytest.mark.unit
class TestCompletionToChatMixinAchat:
    """Test CompletionToChatMixin.achat()."""

    @pytest.mark.asyncio
    async def test_achat_non_stream_returns_chat_response(self):
        """achat(stream=False) must return a ChatResponse.

        Inputs: A single message, await achat() with default stream=False.
        Expected: The awaited result is a ChatResponse instance.
        Checks: isinstance check.
        """
        llm = ConcreteCompletionLLM()
        result = await llm.achat(_make_messages("hello async"))

        assert isinstance(result, ChatResponse), (
            f"Expected ChatResponse, got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_achat_non_stream_content_matches(self):
        """achat(stream=False) content must equal the configured response_text.

        Inputs: response_text="async world", single message.
        Expected: ChatResponse.message.content == "async world".
        Checks: Content equality.
        """
        llm = ConcreteCompletionLLM(response_text="async world")
        result = await llm.achat(_make_messages("q"))

        assert result.message.content == "async world", (
            f"Expected 'async world', got {result.message.content!r}"
        )

    @pytest.mark.asyncio
    async def test_achat_non_stream_role_is_assistant(self):
        """achat(stream=False) must produce an ASSISTANT-role message.

        Inputs: Any message.
        Expected: ChatResponse.message.role == MessageRole.ASSISTANT.
        Checks: Role equality.
        """
        llm = ConcreteCompletionLLM()
        result = await llm.achat(_make_messages("role check"))

        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT, got {result.message.role}"
        )

    @pytest.mark.asyncio
    async def test_achat_non_stream_delegates_to_acomplete(self):
        """achat(stream=False) must delegate to acomplete which calls complete.

        Inputs: Single message, stream=False.
        Expected: complete() is called exactly once (via acomplete delegate).
        Checks: complete_call_count == 1.
        """
        llm = ConcreteCompletionLLM()
        await llm.achat(_make_messages("delegate check"))

        assert llm.complete_call_count == 1, (
            f"Expected complete() to be called once via acomplete, "
            f"was called {llm.complete_call_count} time(s)"
        )

    @pytest.mark.asyncio
    async def test_achat_non_stream_forwards_kwargs(self):
        """achat(stream=False) must forward extra kwargs through acomplete to complete.

        Inputs: Extra kwargs seed=42.
        Expected: complete() receives seed=42 in its kwargs.
        Checks: self.last_kwargs == {"seed": 42}.
        """
        llm = ConcreteCompletionLLM()
        await llm.achat(_make_messages("kwarg"), seed=42)

        assert llm.last_kwargs == {"seed": 42}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_achat_non_stream_calls_messages_to_prompt(self):
        """achat() must call messages_to_prompt to build the prompt.

        Inputs: Two messages with known content.
        Expected: messages_to_prompt is called exactly once, the built prompt
            is forwarded to complete.
        Checks: Call count and last_prompt.
        """
        llm = ConcreteCompletionLLM()
        await llm.achat(_make_messages("alpha", "beta"))

        assert llm.messages_to_prompt_call_count == 1, (
            f"Expected messages_to_prompt called once, "
            f"got {llm.messages_to_prompt_call_count}"
        )
        assert llm.last_prompt == "alpha beta", (
            f"Expected prompt 'alpha beta', got {llm.last_prompt!r}"
        )

    @pytest.mark.asyncio
    async def test_achat_stream_returns_async_generator(self):
        """achat(stream=True) must return an async generator.

        Inputs: Any message, stream=True.
        Expected: The return value supports async iteration.
        Checks: Has __aiter__ and __anext__ attributes.
        """
        llm = ConcreteCompletionLLM()
        result = await llm.achat(_make_messages("async stream"), stream=True)

        assert hasattr(result, "__aiter__"), (
            f"Expected async generator (has __aiter__), got {type(result)}"
        )
        assert hasattr(result, "__anext__"), (
            f"Expected async generator (has __anext__), got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_achat_stream_yields_chat_response_instances(self):
        """achat(stream=True) must yield ChatResponse objects when iterated.

        Inputs: complete(stream=True) yields two chunks.
        Expected: Each async chunk is a ChatResponse.
        Checks: Type of every yielded item.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.achat(_make_messages("gen check"), stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, ChatResponse), (
                f"Chunk {i} is {type(chunk)}, expected ChatResponse"
            )

    @pytest.mark.asyncio
    async def test_achat_stream_chunk_contents_and_deltas(self):
        """achat(stream=True) chunks must carry the completion text and delta.

        Inputs: complete(stream=True) yields ("He", delta="He") and ("Hello", delta="llo").
        Expected: Yielded ChatResponses reflect those values.
        Checks: message.content and delta on each chunk.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.achat(_make_messages("deltas"), stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        assert chunks[0].message.content == "He", (
            f"First chunk content: {chunks[0].message.content!r}"
        )
        assert chunks[0].delta == "He", (
            f"First chunk delta: {chunks[0].delta!r}"
        )
        assert chunks[1].message.content == "Hello", (
            f"Second chunk content: {chunks[1].message.content!r}"
        )
        assert chunks[1].delta == "llo", (
            f"Second chunk delta: {chunks[1].delta!r}"
        )

    @pytest.mark.asyncio
    async def test_achat_stream_forwards_kwargs_to_complete(self):
        """achat(stream=True) must forward extra kwargs to complete(stream=True).

        Inputs: Extra kwargs repeat=3.
        Expected: complete receives repeat=3 in its kwargs.
        Checks: self.last_kwargs after consuming the async generator.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.achat(_make_messages("kw stream"), stream=True, repeat=3)
        async for _ in gen:
            pass

        assert llm.last_kwargs == {"repeat": 3}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )


@pytest.mark.unit
class TestCompletionToChatMixinAcomplete:
    """Test CompletionToChatMixin.acomplete()."""

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_returns_completion_response(self):
        """acomplete(stream=False) must return a CompletionResponse.

        Inputs: A plain prompt string, stream=False (default).
        Expected: The awaited result is a CompletionResponse instance.
        Checks: isinstance check.
        """
        llm = ConcreteCompletionLLM()
        result = await llm.acomplete("async prompt")

        assert isinstance(result, CompletionResponse), (
            f"Expected CompletionResponse, got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_text_matches(self):
        """acomplete(stream=False) text must equal the configured response_text.

        Inputs: response_text="done", prompt "irrelevant".
        Expected: CompletionResponse.text == "done".
        Checks: Text equality.
        """
        llm = ConcreteCompletionLLM(response_text="done")
        result = await llm.acomplete("any prompt")

        assert result.text == "done", (
            f"Expected text 'done', got {result.text!r}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_delegates_to_complete(self):
        """acomplete(stream=False) must delegate to self.complete synchronously.

        Inputs: Prompt string "test".
        Expected: complete() is called exactly once.
        Checks: complete_call_count == 1.
        """
        llm = ConcreteCompletionLLM()
        await llm.acomplete("test")

        assert llm.complete_call_count == 1, (
            f"Expected complete() called once, got {llm.complete_call_count}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_forwards_prompt(self):
        """acomplete(stream=False) must pass the prompt unchanged to complete().

        Inputs: Prompt "my exact prompt".
        Expected: complete() receives "my exact prompt" as its first argument.
        Checks: self.last_prompt equality.
        """
        llm = ConcreteCompletionLLM()
        await llm.acomplete("my exact prompt")

        assert llm.last_prompt == "my exact prompt", (
            f"Expected prompt 'my exact prompt', got {llm.last_prompt!r}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("formatted", [False, True], ids=["unformatted", "formatted"])
    async def test_acomplete_non_stream_forwards_formatted_flag(self, formatted: bool):
        """acomplete(stream=False) must pass the formatted flag to complete().

        Args:
            formatted: The formatted flag value to test.

        Inputs: formatted=False and formatted=True.
        Expected: complete() receives the same formatted value.
        Checks: self.last_formatted equality.
        """
        llm = ConcreteCompletionLLM()
        await llm.acomplete("p", formatted=formatted)

        assert llm.last_formatted == formatted, (
            f"Expected formatted={formatted}, got {llm.last_formatted}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_forwards_kwargs(self):
        """acomplete(stream=False) must forward extra kwargs to complete().

        Inputs: Extra kwargs n=1.
        Expected: complete() receives n=1 in its kwargs.
        Checks: self.last_kwargs equality.
        """
        llm = ConcreteCompletionLLM()
        await llm.acomplete("prompt", n=1)

        assert llm.last_kwargs == {"n": 1}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_stream_returns_async_generator(self):
        """acomplete(stream=True) must return an async generator.

        Inputs: Prompt string, stream=True.
        Expected: The return value has __aiter__ and __anext__.
        Checks: Attribute presence.
        """
        llm = ConcreteCompletionLLM()
        result = await llm.acomplete("stream prompt", stream=True)

        assert hasattr(result, "__aiter__"), (
            f"Expected async generator (__aiter__), got {type(result)}"
        )
        assert hasattr(result, "__anext__"), (
            f"Expected async generator (__anext__), got {type(result)}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_stream_yields_completion_response_instances(self):
        """acomplete(stream=True) must yield CompletionResponse objects.

        Inputs: complete(stream=True) produces two chunks.
        Expected: Each yielded item is a CompletionResponse.
        Checks: Type of every chunk.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.acomplete("chunks", stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, CompletionResponse), (
                f"Chunk {i} is {type(chunk)}, expected CompletionResponse"
            )

    @pytest.mark.asyncio
    async def test_acomplete_stream_chunk_text_and_delta(self):
        """acomplete(stream=True) chunks must carry the text and delta from complete(stream=True).

        Inputs: complete(stream=True) yields text="He",delta="He" then text="Hello",delta="llo".
        Expected: Async iteration yields matching CompletionResponse values.
        Checks: text and delta on each chunk.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.acomplete("delta check", stream=True)

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        assert chunks[0].text == "He", f"First chunk text: {chunks[0].text!r}"
        assert chunks[0].delta == "He", f"First chunk delta: {chunks[0].delta!r}"
        assert chunks[1].text == "Hello", f"Second chunk text: {chunks[1].text!r}"
        assert chunks[1].delta == "llo", f"Second chunk delta: {chunks[1].delta!r}"

    @pytest.mark.asyncio
    async def test_acomplete_stream_calls_complete_stream_not_non_stream(self):
        """acomplete(stream=True) must call complete(stream=True), not complete(stream=False).

        Inputs: stream=True.
        Expected: stream_call_count == 1, complete_call_count == 0.
        Checks: Both counters after consuming the generator.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.acomplete("dispatch check", stream=True)
        async for _ in gen:
            pass

        assert llm.stream_call_count == 1, (
            f"Expected complete(stream=True) called once, got {llm.stream_call_count}"
        )
        assert llm.complete_call_count == 0, (
            f"Expected complete(stream=False) NOT called, "
            f"but was called {llm.complete_call_count} time(s)"
        )

    @pytest.mark.asyncio
    async def test_acomplete_non_stream_calls_complete_not_stream(self):
        """acomplete(stream=False) must call complete(stream=False), not complete(stream=True).

        Inputs: stream=False (default).
        Expected: complete_call_count == 1, stream_call_count == 0.
        Checks: Both counters after the await.
        """
        llm = ConcreteCompletionLLM()
        await llm.acomplete("dispatch check non-stream")

        assert llm.complete_call_count == 1, (
            f"Expected complete(stream=False) called once, got {llm.complete_call_count}"
        )
        assert llm.stream_call_count == 0, (
            f"Expected complete(stream=True) NOT called, "
            f"but was called {llm.stream_call_count} time(s)"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("formatted", [False, True], ids=["unformatted", "formatted"])
    async def test_acomplete_stream_forwards_formatted_flag(self, formatted: bool):
        """acomplete(stream=True) must forward the formatted flag to complete(stream=True).

        Args:
            formatted: The formatted flag value to test.

        Inputs: formatted=False and formatted=True, stream=True.
        Expected: complete receives the same formatted value.
        Checks: self.last_formatted after consuming the generator.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.acomplete("formatted stream", formatted=formatted, stream=True)
        async for _ in gen:
            pass

        assert llm.last_formatted == formatted, (
            f"Expected formatted={formatted} forwarded to complete(stream=True), "
            f"got {llm.last_formatted}"
        )

    @pytest.mark.asyncio
    async def test_acomplete_stream_forwards_kwargs_to_complete(self):
        """acomplete(stream=True) must forward extra kwargs to complete(stream=True).

        Inputs: Extra kwargs verbose=True.
        Expected: complete receives verbose=True in its kwargs.
        Checks: self.last_kwargs after consuming the generator.
        """
        llm = ConcreteCompletionLLM()
        gen = await llm.acomplete("kw", stream=True, verbose=True)
        async for _ in gen:
            pass

        assert llm.last_kwargs == {"verbose": True}, (
            f"Forwarded kwargs mismatch: {llm.last_kwargs}"
        )
