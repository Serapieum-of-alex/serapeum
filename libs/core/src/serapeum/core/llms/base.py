"""Core LLM orchestration utilities and base high-level API.

This module defines the high-level LLM class with helpers for prompts,
streaming, and structured outputs.
"""

from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field, WithJsonSchema, field_validator, model_validator
from typing_extensions import Annotated

from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.base.llms.types import (
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Message,
    MessageList,
    MessageRole,
)
from serapeum.core.types import Model, StructuredLLMMode
from serapeum.core.output_parsers.types import BaseParser, TokenAsyncGen, TokenGen
from serapeum.core.prompts import BasePromptTemplate, PromptTemplate

if TYPE_CHECKING:
    from serapeum.core.llms.structured_output_llm import StructuredOutputLLM


@runtime_checkable
class MessagesToPromptType(Protocol):
    r"""Runtime protocol describing adapters that convert chat messages into prompts.

    Examples:
        - Join message contents into a newline-separated prompt
            ```python
            >>> from serapeum.core.llms.base import MessagesToPromptType
            >>> from serapeum.core.base.llms.types import Message, MessageRole, MessageList
            >>> def newline_join(message_list):
            ...     return '\n'.join(message.content or "" for message in message_list)
            ...
            >>> isinstance(newline_join, MessagesToPromptType)
            True

            ```
        - Validate message content before rendering the prompt
            ```python
            >>> from serapeum.core.base.llms.types import Message, MessageRole, MessageList
            >>> def validated_join(message_list):
            ...     contents = [message.content for message in message_list]
            ...     if any(content is None for content in contents):
            ...         raise ValueError("Missing content")
            ...     return " ".join(str(content) for content in contents)
            ...
            >>> validated_join(MessageList([Message(content="hi", role=MessageRole.USER)]))
            'hi'

            ```
    See Also:
        MessagesToPromptType.__call__: Details the expected callable signature.
    """

    def __call__(self, message_list: MessageList) -> str:
        """Render a MessageList into a single prompt string.

        Args:
            message_list (MessageList): MessageList containing chat messages to convert.

        Returns:
            str: Textual prompt synthesized from the provided messages.

        Raises:
            ValueError: Implementations may raise when a message payload is invalid.

        Examples:
            - Concatenate user and assistant messages into one prompt
                ```python
                >>> from serapeum.core.base.llms.types import Message, MessageRole, MessageList
                >>> def concatenate(message_list):
                ...     return " ".join((message.content or "").strip() for message in message_list)
                ...
                >>> concatenate(
                ...     MessageList([
                ...         Message(content="Hello", role=MessageRole.USER),
                ...         Message(content="world", role=MessageRole.ASSISTANT),
                ...     ])
                ... )
                'Hello world'

                ```
            - Reject empty message content before joining
                ```python
                >>> from serapeum.core.base.llms.types import Message, MessageRole, MessageList
                >>> def strict_concat(message_list):
                ...     for message in message_list:
                ...         if message.content is None:
                ...             raise ValueError("Missing content")
                ...     return " ".join(str(message.content) for message in message_list)
                ...
                >>> strict_concat(MessageList([Message(content="Ping", role=MessageRole.USER)]))
                'Ping'

                ```
        See Also:
            LLM._get_messages: Prepares message sequences prior to prompt rendering.
        """
        ...


@runtime_checkable
class CompletionToPromptType(Protocol):
    r"""Runtime protocol describing prompt adapters invoked before completions.

    Examples:
        - Check that an identity adapter satisfies the protocol
            ```python
            >>> from serapeum.core.llms.base import CompletionToPromptType
            >>> def identity(prompt: str) -> str:
            ...     return prompt
            ...
            >>> isinstance(identity, CompletionToPromptType)
            True

            ```
        - Compose multiple adapters to build reusable transformations
            ```python
            >>> def add_footer(prompt: str) -> str:
            ...     return prompt + "\\n--"
            ...
            >>> def upper_then_footer(prompt: str) -> str:
            ...     return add_footer(prompt.upper())
            ...
            >>> upper_then_footer("ok")
            'OK\n--'

            ```
    See Also:
        CompletionToPromptType.__call__: Signature and error handling requirements.
    """

    def __call__(self, prompt: str) -> str:
        """Transform a pre-formatted prompt prior to model execution.

        Args:
            prompt (str): The prompt string produced by an upstream formatter.

        Returns:
            str: A transformed prompt suitable for completion endpoints.

        Raises:
            ValueError: Implementations may raise when the prompt cannot be adapted.

        Examples:
            - Prefix a prompt with fixed metadata before submission
                ```python
                >>> def add_metadata(prompt: str) -> str:
                ...     return "SYSTEM: " + prompt
                ...
                >>> add_metadata("List three colors")
                'SYSTEM: List three colors'

                ```
            - Enforce non-empty prompts with validation
                ```python
                >>> def ensure_prompt(prompt: str) -> str:
                ...     if not prompt.strip():
                ...         raise ValueError("Prompt must not be empty")
                ...     return prompt.upper()
                ...
                >>> ensure_prompt("summarize the agenda")
                'SUMMARIZE THE AGENDA'

                ```
        See Also:
            LLM._extend_prompt: Applies system-level adornments to formatted prompts.
        """
        ...


def stream_response_to_tokens(
    completion_response_gen: CompletionResponseGen | ChatResponseGen,
) -> TokenGen:
    """Materialize a token generator from streaming completion responses.

    Args:
        completion_response_gen (CompletionResponseGen):
            Response iterator yielding completion deltas.

    Returns:
        TokenGen: Generator that yields delta strings ready for downstream consumption.

    Raises:
        AttributeError:
            If responses lack the ``delta`` attribute expected on completion payloads.

    Examples:
        - CompletionResponse:
            - Collect deltas produced by a completion stream
                ```python
                >>> from serapeum.core.base.llms.types import CompletionResponse
                >>> from serapeum.core.llms.base import stream_response_to_tokens
                >>> def responses():
                ...     yield CompletionResponse(text="Hello", delta="Hel")
                ...     yield CompletionResponse(text="Hello", delta="lo")
                ...
                >>> list(stream_response_to_tokens(responses()))
                ['Hel', 'lo']

                ```
            - Handle responses that omit delta text
                ```python
                >>> def responses():
                ...     yield CompletionResponse(text="partial", delta=None)
                ...     yield CompletionResponse(text="done", delta="")
                ...
                >>> list(stream_response_to_tokens(responses()))
                ['', '']

                ```
        - ChatResponse:
            - Collect assistant deltas from a chat stream
                ```python
                >>> from serapeum.core.base.llms.types import ChatResponse, Message, MessageRole
                >>> def responses():
                ...     yield ChatResponse(
                ...         message=Message(content="Hello", role=MessageRole.ASSISTANT),
                ...         delta="Hel",
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="Hello", role=MessageRole.ASSISTANT),
                ...         delta="lo",
                ...     )
                ...
                >>> list(stream_response_to_tokens(responses()))
                ['Hel', 'lo']

                ```
            - Support chat responses without deltas
                ```python
                >>> from serapeum.core.base.llms.types import ChatResponse, Message, MessageRole
                >>> def responses():
                ...     yield ChatResponse(
                ...         message=Message(content="Partial", role=MessageRole.ASSISTANT),
                ...         delta=None,
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="Partial", role=MessageRole.ASSISTANT),
                ...         delta="",
                ...     )
                ...
                >>> list(stream_response_to_tokens(responses()))
                ['', '']

                ```
    See Also:
        astream_response_to_tokens: Asynchronous variant returning an async generator of tokens.
    """

    def gen() -> TokenGen:
        for response in completion_response_gen:
            yield response.delta or ""

    return gen()


async def astream_response_to_tokens(
    completion_response_gen: CompletionResponseAsyncGen | ChatResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert async completion responses into an async token generator.

    Args:
        completion_response_gen (CompletionResponseAsyncGen): Async iterator yielding completion deltas.

    Returns:
        TokenAsyncGen: Async generator emitting delta strings from each response.

    Raises:
        AttributeError: If streamed responses do not expose a ``delta`` attribute.

    Examples:
        - CompletionResponse:
            - Gather tokens from an asynchronous completion stream
                ```python
                >>> import asyncio
                >>> from serapeum.core.base.llms.types import CompletionResponse
                >>> from serapeum.core.llms.base import astream_response_to_tokens
                >>> async def responses():
                ...     yield CompletionResponse(text="Hello", delta="Hel")
                ...     yield CompletionResponse(text="Hello", delta="lo")
                ...
                >>> async def collect():
                ...     generator = await astream_response_to_tokens(responses())
                ...     return [token async for token in generator]
                ...
                >>> asyncio.run(collect())
                ['Hel', 'lo']

                ```
            - Ensure empty delta values propagate as empty strings
                ```python
                >>> import asyncio
                >>> from serapeum.core.base.llms.types import CompletionResponse
                >>> async def responses():
                ...     yield CompletionResponse(text="partial", delta=None)
                ...     yield CompletionResponse(text="done", delta="")
                ...
                >>> async def collect():
                ...     generator = await astream_response_to_tokens(responses())
                ...     return [token async for token in generator]
                ...
                >>> asyncio.run(collect())
                ['', '']

                ```
        - ChatResponse:
            - Aggregate assistant deltas asynchronously
                ```python
                >>> import asyncio
                >>> from serapeum.core.base.llms.types import ChatResponse, Message, MessageRole
                >>> async def responses():
                ...     yield ChatResponse(
                ...         message=Message(content="Hi", role=MessageRole.ASSISTANT),
                ...         delta="H",
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="Hi", role=MessageRole.ASSISTANT),
                ...         delta="i",
                ...     )
                ...
                >>> async def collect():
                ...     generator = await astream_response_to_tokens(responses())
                ...     return [token async for token in generator]
                ...
                >>> asyncio.run(collect())
                ['H', 'i']

                ```
            - Surface empty delta entries when no new tokens are produced
                ```python
                >>> import asyncio
                >>> from serapeum.core.base.llms.types import ChatResponse, Message, MessageRole
                >>> async def responses():
                ...     yield ChatResponse(
                ...         message=Message(content="Partial", role=MessageRole.ASSISTANT),
                ...         delta=None,
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="Partial", role=MessageRole.ASSISTANT),
                ...         delta="",
                ...     )
                ...
                >>> async def collect():
                ...     generator = await astream_response_to_tokens(responses())
                ...     return [token async for token in generator]
                ...
                >>> asyncio.run(collect())
                ['', '']

                ```
    See Also:
        stream_response_to_tokens: Synchronous companion for blocking workloads.
    """

    async def gen() -> TokenAsyncGen:
        async for response in completion_response_gen:
            yield response.delta or ""

    return gen()


def default_completion_to_prompt(prompt: str) -> str:
    """Return the provided prompt unchanged.

    Args:
        prompt (str):
            Prompt string produced by a formatter or template.

    Returns:
        str:
            The input prompt without modification.

    Raises:
        Nothing:
            This helper performs no validation and never raises.

    Examples:
        - Use the identity adapter when no transformation is required
            ```python
            >>> default_completion_to_prompt("Draft a status report.")
            'Draft a status report.'

            ```
        - Combine with a fallback adapter to ensure a non-empty string
            ```python
            >>> def fallback_adapter(text: str) -> str:
            ...     transformed = default_completion_to_prompt(text)
            ...     return transformed or "Default prompt"
            ...
            >>> fallback_adapter("")
            'Default prompt'

            ```
    See Also:
        set_completion_to_prompt: Selects this identity helper when no adapter is provided.
    """
    return prompt


MessagesToPromptCallable = Annotated[
    MessagesToPromptType | None,
    WithJsonSchema({"type": "string"}),
]


CompletionToPromptCallable = Annotated[
    CompletionToPromptType | None,
    WithJsonSchema({"type": "string"}),
]


class LLM(BaseLLM, ABC):
    """Interactive abstraction around language model providers.

    Attributes:
        system_prompt (Optional[str]): Optional system-level preamble applied to every request.
        messages_to_prompt (MessagesToPromptCallable): Callable converting chat messages into prompts.
        completion_to_prompt (CompletionToPromptCallable): Callable adapting prepared prompts for completions.
        output_parser (Optional[BaseParser]): Parser used to coerce raw model text into structured values.
        pydantic_program_mode (StructuredLLMMode): Strategy for executing pydantic-based structured outputs.
        query_wrapper_prompt (Optional[BasePromptTemplate]): Legacy prompt wrapper retained for backwards compatibility.

    Examples:
        - Produce a simple completion by subclassing ``LLM``
            ```python
            >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
            >>> class EchoLLM(LLM):
            ...     metadata = Metadata.model_construct(is_chat_model=False)
            ...     def chat(self, messages, **kwargs):
            ...         raise NotImplementedError()
            ...     def stream_chat(self, messages, **kwargs):
            ...         raise NotImplementedError()
            ...     async def achat(self, messages, **kwargs):
            ...         raise NotImplementedError()
            ...     async def astream_chat(self, messages, **kwargs):
            ...         raise NotImplementedError()
            ...     def complete(self, prompt, formatted=False, **kwargs):
            ...         return CompletionResponse(text=prompt, delta=prompt)
            ...     def stream_complete(self, prompt, formatted=False, **kwargs):
            ...         raise NotImplementedError()
            ...     async def acomplete(self, prompt, formatted=False, **kwargs):
            ...         return CompletionResponse(text=prompt, delta=prompt)
            ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
            ...         raise NotImplementedError()
            ...
            >>> from serapeum.core.prompts import PromptTemplate
            >>> echo = EchoLLM()
            >>> echo.predict(PromptTemplate("{greeting}, world!"), greeting="Hello")
            'Hello, world!'

            ```
        - Parse structured output using ``structured_predict``
            ```python
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...
            >>> from serapeum.core.prompts import PromptTemplate
            >>> class StubLLM(EchoLLM):
            ...     def structured_predict(self, output_cls, prompt, **prompt_args):
            ...         return output_cls(name=prompt.format(**prompt_args))
            ...
            >>> stub = StubLLM()
            >>> stub.structured_predict(Person, PromptTemplate("{name}"), name="Ada").name
            'Ada'

            ```
    See Also:
        BaseLLM: Abstract interface specifying the contract implemented by ``LLM`` subclasses.
        StructuredOutputLLM: Wrapper that exposes structured interactions on top of an ``LLM`` instance.
    """

    system_prompt: str | None = Field(
        default=None, description="System prompt for LLM calls."
    )
    messages_to_prompt: MessagesToPromptCallable = Field(
        description="Function to convert a list of messages to an LLM prompt.",
        default=None,
        exclude=True,
    )
    completion_to_prompt: CompletionToPromptCallable = Field(
        description="Function to convert a completion to an LLM prompt.",
        default=None,
        exclude=True,
    )
    output_parser: BaseParser | None = Field(
        description="Output parser to parse, validate, and correct errors programmatically.",
        default=None,
        exclude=True,
    )
    pydantic_program_mode: StructuredLLMMode = StructuredLLMMode.DEFAULT

    # # deprecated
    query_wrapper_prompt: BasePromptTemplate | None = Field(
        description="Query wrapper prompt for LLM calls.",
        default=None,
        exclude=True,
    )

    @field_validator("messages_to_prompt")  # type: ignore[misc]
    @classmethod
    def set_messages_to_prompt(
        cls, messages_to_prompt: MessagesToPromptType | None
    ) -> MessagesToPromptType:
        r"""Select a message-to-prompt adapter, defaulting to MessageList.to_prompt().

        Args:
            messages_to_prompt (Optional[MessagesToPromptType]):
                Custom adapter supplied by the caller.

        Returns:
            MessagesToPromptType: Adapter that converts a MessageList into a prompt string.

        Examples:
            - Fall back to the default renderer when no adapter is provided
                ```python
                >>> LLM.set_messages_to_prompt(None)
                <function ...>

                ```
            - Preserve a custom adapter when one is supplied
                ```python
                >>> def reverse_messages(message_list):
                ...     return "\\n".join(message.content or "" for message in reversed(message_list))
                ...
                >>> LLM.set_messages_to_prompt(reverse_messages) is reverse_messages
                True

                ```
        See Also:
            MessageList.to_prompt: Provides the default message formatting behavior.
        """
        if messages_to_prompt is None:
            return lambda message_list: message_list.to_prompt()
        return messages_to_prompt

    @field_validator("completion_to_prompt")  # type: ignore[misc]
    @classmethod
    def set_completion_to_prompt(
        cls, completion_to_prompt: CompletionToPromptType | None
    ) -> CompletionToPromptType:
        """Ensure completion adapters always default to ``default_completion_to_prompt``.

        Args:
            completion_to_prompt (Optional[CompletionToPromptType]): Custom adapter transforming prompts.

        Returns:
            CompletionToPromptType: Adapter that prepares prompts for completion endpoints.

        Raises:
            Nothing: The validator guarantees a callable without raising.

        Examples:
            - Substitute the identity adapter when ``None`` is provided
                ```python
                >>> LLM.set_completion_to_prompt(None)
                <function default_completion_to_prompt at ...>

                ```
            - Preserve a custom adapter supplied by the caller
                ```python
                >>> def prefix(prompt: str) -> str:
                ...     return "PREFIX: " + prompt
                ...
                >>> LLM.set_completion_to_prompt(prefix) is prefix
                True

                ```
        See Also:
            default_completion_to_prompt: Built-in adapter used as the safe fallback.
        """
        return completion_to_prompt or default_completion_to_prompt

    @model_validator(mode="after")  # type: ignore[misc]
    def check_prompts(self) -> "LLM":
        """Populate prompt adapters after pydantic validation completes.

        Returns:
            LLM: The validated instance with guaranteed prompt adapters.

        Raises:
            Nothing: The validator ensures adapters exist without raising errors.

        Examples:
            - Automatically attach defaults when fields are unset
                ```python
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     Metadata,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> llm = DemoLLM()
                >>> callable(llm.messages_to_prompt)
                True

                ```
            - Respect explicitly provided adapters
                ```python
                >>> from serapeum.core.base.llms.types import Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> def identity_messages(messages):
                ...     return " ".join(message.content or "" for message in messages)
                ...
                >>> llm = DemoLLM(
                ...     messages_to_prompt=identity_messages,
                ...     completion_to_prompt=lambda prompt: prompt.upper(),
                ... )
                >>> llm.completion_to_prompt("hi")
                'HI'

                ```
        See Also:
            set_messages_to_prompt: Supplies the default message adapter when missing.
            set_completion_to_prompt: Supplies the default completion adapter when missing.
        """
        if self.completion_to_prompt is None:
            self.completion_to_prompt = default_completion_to_prompt
        if self.messages_to_prompt is None:
            self.messages_to_prompt = lambda message_list: message_list.to_prompt()
        return self

    def _get_prompt(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
        """Format a prompt template with LLM metadata and parser hooks.

        Args:
            prompt (BasePromptTemplate): Template describing the prompt structure.
            **prompt_args (Any): Named values injected into the template.

        Returns:
            str: Fully formatted prompt string ready for submission.

        Raises:
            Nothing: Validation occurs in the template and parser layers.

        Examples:
            - Expand a template without an output parser
                ```python
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> llm = DemoLLM()
                >>> llm._get_prompt(PromptTemplate("{subject} summary"), subject="Release")
                'Release summary'

                ```
            - Apply an output parser formatter before returning the result
                ```python
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> from serapeum.core.output_parsers import BaseParser
                >>> class UpperParser(BaseParser):
                ...     def parse(self, output: str) -> str:
                ...         return output.upper()
                ...     def format(self, query: str) -> str:
                ...         return query.upper()
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     output_parser = UpperParser()
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM()._get_prompt(PromptTemplate("summarize {item}"), item="notes")
                'SUMMARIZE NOTES'

                ```
        See Also:
            _extend_prompt: Adds system and wrapper prompts to the formatted string.
            _get_messages: Equivalent logic for chat-oriented prompt generation.
        """
        formatted_prompt = prompt.format(
            llm=self,
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            **prompt_args,
        )
        if self.output_parser is not None:
            formatted_prompt = self.output_parser.format(formatted_prompt)
        return self._extend_prompt(formatted_prompt)

    def _get_messages(
        self, prompt: BasePromptTemplate, **prompt_args: Any
    ) -> list[Message]:
        """Render chat messages from a prompt template.

        Args:
            prompt (BasePromptTemplate): Prompt template capable of producing chat messages.
            **prompt_args (Any): Named values inserted into the template.

        Returns:
            list[Message]: Sequence of messages ready for chat model consumption.

        Raises:
            ValueError: Propagated when the template cannot be formatted with ``prompt_args``.

        Examples:
            - Generate user-facing messages without an output parser
                ```python
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     Metadata,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> messages = DemoLLM()._get_messages(
                ...     ChatPromptTemplate.from_messages([("user", "Hello {name}!")]),
                ...     name="Ada",
                ... )
                >>> messages[0].content
                'Hello Ada!'

                ```
            - Apply an output parser hook to the formatted messages
                ```python
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     Metadata,
                ... )
                >>> from serapeum.core.output_parsers import BaseParser
                >>> class UpperParser(BaseParser):
                ...     def parse(self, output: str) -> str:
                ...         return output.upper()
                ...     def format_messages(self, messages):
                ...         for message in messages:
                ...             if message.content is not None:
                ...                 message.content = message.content.upper()
                ...         return messages
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     output_parser = UpperParser()
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> message = DemoLLM()._get_messages(
                ...     ChatPromptTemplate.from_messages([("user", "Hello {name}!")]),
                ...     name="Ada",
                ... )[0]
                >>> message.content
                'HELLO ADA!'

                ```
        See Also:
            _extend_messages: Adds system prompts to the generated message list.
            _get_prompt: Equivalent logic for completion-style prompts.
        """
        messages = prompt.format_messages(**prompt_args)
        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)
        return self._extend_messages(messages)

    def _parse_output(self, output: str) -> str:
        """Parse raw model output using the configured output parser.

        Args:
            output (str): Raw string returned by the model.

        Returns:
            str: Parsed representation of the model output.

        Raises:
            ValueError: Propagated when the configured parser rejects ``output``.

        Examples:
            - Return the text unchanged when no parser is configured
                ```python
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM()._parse_output("ready")
                'ready'

                ```
            - Coerce text via a custom parser before returning
                ```python
                >>> from serapeum.core.output_parsers import BaseParser
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class TrimParser(BaseParser):
                ...     def parse(self, output: str) -> str:
                ...         return output.strip()
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     output_parser = TrimParser()
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM()._parse_output("  ok  ")
                'ok'

                ```
        See Also:
            BaseParser.parse: Implements the parsing logic invoked here.
        """
        if self.output_parser is not None:
            return str(self.output_parser.parse(output))

        return output

    def _extend_prompt(
        self,
        formatted_prompt: str,
    ) -> str:
        r"""Add system and query wrapper prompts to a formatted prompt.

        Args:
            formatted_prompt (str): Fully formatted prompt string.

        Returns:
            str: Prompt extended with system and wrapper decorations.

        Raises:
            Nothing: Operations are concatenations that cannot fail.

        Examples:
            - Return the original text when no decorations are configured
                ```python
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM()._extend_prompt("Plan release notes")
                'Plan release notes'

                ```
            - Prepend a system prompt and wrap the query
                ```python
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     system_prompt = "You are an assistant."
                ...     query_wrapper_prompt = PromptTemplate("Question: {query_str}")
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM()._extend_prompt("List priorities")
                'You are an assistant.\\n\\nQuestion: List priorities'

                ```
        See Also:
            _extend_messages: Equivalent extension logic for chat message lists.
        """
        extended_prompt = formatted_prompt

        if self.system_prompt:
            extended_prompt = self.system_prompt + "\n\n" + extended_prompt

        if self.query_wrapper_prompt:
            extended_prompt = self.query_wrapper_prompt.format(
                query_str=extended_prompt
            )

        return extended_prompt

    def _extend_messages(self, messages: list[Message]) -> list[Message]:
        """Add optional system prompts to the chat message list.

        Args:
            messages (list[Message]): Sequence of user/assistant messages.

        Returns:
            list[Message]: Message list with system context prepended when configured.

        Raises:
            Nothing: Operates purely on in-memory message lists.

        Examples:
            - Leave messages unchanged when no system prompt is configured
                ```python
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     Message,
                ...     Metadata,
                ...     MessageRole,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> messages = [Message(content="Hi", role=MessageRole.USER)]
                >>> DemoLLM()._extend_messages(messages)[0].content
                'Hi'

                ```
            - Prepend a system prompt when configured
                ```python
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     Message,
                ...     Metadata,
                ...     MessageRole,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     system_prompt = "You are helpful."
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> extended = DemoLLM()._extend_messages([Message(content="Hi", role=MessageRole.USER)])
                >>> extended[0].role.value
                'system'

                ```
        See Also:
            _extend_prompt: Applies equivalent system context to string prompts.
        """
        if self.system_prompt:
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                *messages,
            ]
        return messages

    def structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Invoke the structured output program for synchronous predictions.

        Args:
            output_cls (type[BaseModel]): Pydantic model describing the expected output schema.
            prompt (PromptTemplate): Template used to gather inputs and instructions.
            llm_kwargs (dict[str, Any] | None): Provider-specific arguments forwarded to the underlying LLM.
            **prompt_args (Any): Additional template variables passed to ``prompt``.

        Returns:
            BaseModel: Instance of ``output_cls`` populated by the structured program.

        Raises:
            RuntimeError: Propagated when the structured program encounters execution failures.

        Examples:
            - Produce a structured response using a patched program
                ```python
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Person(BaseModel):
                ...     name: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> from serapeum.core.prompts import PromptTemplate
                >>> def fake_program(llm_kwargs=None, **kwargs):
                ...     return Person(name=kwargs["name"].title())
                ...
                >>> with patch(
                ...     "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...     return_value=fake_program,
                ... ):
                ...     DemoLLM().structured_predict(Person, PromptTemplate("{name}"), name="ada").name
                'Ada'

                ```
            - Forward ``llm_kwargs`` to the structured program
                ```python
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Stats(BaseModel):
                ...     parameter: str
                ...     config: dict
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> def fake_program(llm_kwargs=None, **kwargs):
                ...     return Stats(parameter=kwargs["name"], config=llm_kwargs or {})
                ...
                >>> from serapeum.core.prompts import PromptTemplate
                >>> with patch(
                ...     "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...     return_value=fake_program,
                ... ):
                ...     DemoLLM().structured_predict(
                ...         Stats,
                ...         PromptTemplate("{name}"),
                ...         llm_kwargs={"temperature": 0.3},
                ...         name="throughput",
                ...     ).config["temperature"]
                0.3

                ```
        See Also:
            astructured_predict: Async counterpart that awaits the structured program.
            stream_structured_predict: Streams partial structured outputs incrementally.
        """
        from serapeum.core.structured_tools.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = program(llm_kwargs=llm_kwargs, **prompt_args)

        return result

    async def astructured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Run the structured output program asynchronously.

        Args:
            output_cls (type[BaseModel]): Pydantic model describing the target schema.
            prompt (PromptTemplate): Template used to generate program inputs.
            llm_kwargs (dict[str, Any] | None): Optional provider arguments forwarded to the program.
            **prompt_args (Any): Additional inputs passed to the template.

        Returns:
            BaseModel: Awaited instance of ``output_cls`` produced by the structured program.

        Raises:
            RuntimeError: Propagated when the structured program fails during execution.

        Examples:
            - Await a patched structured program
                ```python
                >>> import asyncio
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Person(BaseModel):
                ...     name: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     async def acall(self, **kwargs):
                ...         return Person(name="Ada")
                ...
                >>> async def demo():
                ...     with patch(
                ...         "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...         return_value=FakeProgram(),
                ...     ):
                ...         from serapeum.core.prompts import PromptTemplate
                ...         result = await DemoLLM().astructured_predict(
                ...             Person,
                ...             PromptTemplate("{name}"),
                ...             name="ignored",
                ...         )
                ...     return result.name
                ...
                >>> asyncio.run(demo())
                'Ada'

                ```
            - Forward ``llm_kwargs`` through the async program interface
                ```python
                >>> import asyncio
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Report(BaseModel):
                ...     meta: dict
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     async def acall(self, **kwargs):
                ...         return Report(meta=kwargs["llm_kwargs"])
                ...
                >>> async def demo():
                ...     with patch(
                ...         "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...         return_value=FakeProgram(),
                ...     ):
                ...         from serapeum.core.prompts import PromptTemplate
                ...         result = await DemoLLM().astructured_predict(
                ...             Report,
                ...             PromptTemplate("{name}"),
                ...             llm_kwargs={"seed": 42},
                ...             name="ignored",
                ...         )
                ...     return result.meta["seed"]
                ...
                >>> asyncio.run(demo())
                42

                ```
        See Also:
            structured_predict: Blocking variant using the same structured program.
            astream_structured_predict: Emits partial values asynchronously during execution.
        """
        from serapeum.core.structured_tools.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = await program.acall(llm_kwargs=llm_kwargs, **prompt_args)

        return result

    def stream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Generator[Model | list[Model], None, None]:
        """Stream structured predictions as they become available.

        Args:
            output_cls (type[BaseModel]): Pydantic model describing the structured response.
            prompt (PromptTemplate): Template orchestrating the program execution.
            llm_kwargs (dict[str, Any] | None): Additional arguments forwarded to the underlying LLM.
            **prompt_args (Any): Keyword arguments interpolated into the template.

        Yields:
            Model | list[Model]: Incremental structured values emitted by the program.

        Raises:
            RuntimeError: Propagated from the structured program when streaming fails.

        Examples:
            - Iterate over partial models emitted by a patched program
                ```python
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Item(BaseModel):
                ...     value: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     def stream_call(self, llm_kwargs=None, **kwargs):
                ...         yield Item(value=kwargs["name"])
                ...         yield Item(value=kwargs["name"].upper())
                ...
                >>> from serapeum.core.prompts import PromptTemplate
                >>> with patch(
                ...     "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...     return_value=FakeProgram(),
                ... ):
                ...     tokens = [
                ...         part.value
                ...         for part in DemoLLM().stream_structured_predict(
                ...             Item,
                ...             PromptTemplate("{name}"),
                ...             name="signal",
                ...         )
                ...     ]
                >>> tokens
                ['signal', 'SIGNAL']

                ```
            - Stream lists when the program emits batch updates
                ```python
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Item(BaseModel):
                ...     value: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     def stream_call(self, llm_kwargs=None, **kwargs):
                ...         yield [Item(value="partial")]
                ...         yield [Item(value="final")]
                ...
                >>> from serapeum.core.prompts import PromptTemplate
                >>> with patch(
                ...     "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...     return_value=FakeProgram(),
                ... ):
                ...     batches = list(
                ...         DemoLLM().stream_structured_predict(
                ...             Item,
                ...             PromptTemplate("{name}"),
                ...             name="ignored",
                ...         )
                ...     )
                >>> [batch[0].value for batch in batches]
                ['partial', 'final']

                ```
        See Also:
            astream_structured_predict: Async variant yielding values via an async iterator.
            structured_predict: Non-streaming version that returns the final model directly.
        """
        from serapeum.core.structured_tools.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        result = program.stream_call(llm_kwargs=llm_kwargs, **prompt_args)
        for r in result:
            yield r

    async def _structured_astream_call(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Model | list[Model] | BaseModel | list[BaseModel], None]:
        """Obtain the async structured program stream without additional wrapping.

        Args:
            output_cls (type[Model]): Structured output model requested by the caller.
            prompt (PromptTemplate): Template defining the structured program execution.
            llm_kwargs (dict[str, Any] | None): Keyword arguments forwarded to the LLM.
            **prompt_args (Any): Arguments substituted into ``prompt``.

        Returns:
            AsyncGenerator[Model | list[Model]]: Async generator streaming structured values.

        Raises:
            RuntimeError: Propagated when the structured program fails to initialize.

        Examples:
            - Acquire the underlying async generator for custom post-processing
                ```python
                >>> import asyncio
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Item(BaseModel):
                ...     value: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     async def astream_call(self, llm_kwargs=None, **kwargs):
                ...         async def generator():
                ...             yield Item(value="partial")
                ...             yield Item(value="final")
                ...         return generator()
                ...
                >>> async def demo():
                ...     from serapeum.core.prompts import PromptTemplate
                ...     with patch(
                ...         "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...         return_value=FakeProgram(),
                ...     ):
                ...         stream = await DemoLLM()._structured_astream_call(
                ...             Item,
                ...             PromptTemplate("{name}"),
                ...             name="ignored",
                ...         )
                ...     collected = []
                ...     async for part in stream:
                ...         collected.append(part.value)
                ...     return collected
                ...
                >>> asyncio.run(demo())
                ['partial', 'final']

                ```
        See Also:
            astream_structured_predict: Public helper that wraps this coroutine for callers.
        """
        from serapeum.core.structured_tools.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        return await program.astream_call(llm_kwargs=llm_kwargs, **prompt_args)  # type: ignore[return-value]

    async def astream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Model | list[Model], None]:
        """Stream structured predictions asynchronously.

        Args:
            output_cls (type[BaseModel]): Structured response model expected from the program.
            prompt (PromptTemplate): Prompt orchestrating the structured interaction.
            llm_kwargs (dict[str, Any] | None): Provider arguments injected into the structured program.
            **prompt_args (Any): Additional inputs formatted into ``prompt``.

        Yields:
            Model | list[Model]: Structured values produced asynchronously.

        Raises:
            RuntimeError: Propagated when the underlying program encounters streaming issues.

        Examples:
            - Iterate over streamed values with patched program output
                ```python
                >>> import asyncio
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Item(BaseModel):
                ...     value: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     async def astream_call(self, llm_kwargs=None, **kwargs):
                ...         async def generator():
                ...             yield Item(value=kwargs["name"])
                ...             yield Item(value=kwargs["name"].upper())
                ...         return generator()
                ...
                >>> async def demo():
                ...     from serapeum.core.prompts import PromptTemplate
                ...     with patch(
                ...         "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...         return_value=FakeProgram(),
                ...     ):
                ...         stream = await DemoLLM().astream_structured_predict(
                ...             Item,
                ...             PromptTemplate("{name}"),
                ...             name="flow",
                ...         )
                ...     items = []
                ...     async for partial in stream:
                ...         items.append(partial.value)
                ...     return items
                ...
                >>> asyncio.run(demo())
                ['flow', 'FLOW']

                ```
            - Handle batch updates returned as lists
                ```python
                >>> import asyncio
                >>> from unittest.mock import patch
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class Item(BaseModel):
                ...     value: str
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class FakeProgram:
                ...     async def astream_call(self, llm_kwargs=None, **kwargs):
                ...         async def generator():
                ...             yield [Item(value="chunk")]
                ...             yield [Item(value="done")]
                ...         return generator()
                ...
                >>> async def demo():
                ...     from serapeum.core.prompts import PromptTemplate
                ...     with patch(
                ...         "serapeum.core.structured_tools.utils.get_program_for_llm",
                ...         return_value=FakeProgram(),
                ...     ):
                ...         stream = await DemoLLM().astream_structured_predict(
                ...             Item,
                ...             PromptTemplate("{name}"),
                ...             name="ignored",
                ...         )
                ...     values = []
                ...     async for batch in stream:
                ...         values.append(batch[0].value)
                ...     return values
                ...
                >>> asyncio.run(demo())
                ['chunk', 'done']

                ```
        See Also:
            stream_structured_predict: Synchronous counterpart yielding from a regular generator.
            _structured_astream_call: Internal helper that retrieves the structured async stream.
        """

        async def gen() -> AsyncGenerator[Model | list[Model], None]:
            from serapeum.core.structured_tools.utils import get_program_for_llm

            program = get_program_for_llm(
                output_cls,
                prompt,
                self,
                pydantic_program_mode=self.pydantic_program_mode,
            )

            result = await program.astream_call(llm_kwargs=llm_kwargs, **prompt_args)
            async for r in result:
                yield r

        return gen()

    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Generate a synchronous completion for the provided prompt.

        Args:
            prompt (BasePromptTemplate): Prompt template rendered prior to model invocation.
            **prompt_args (Any): Keyword arguments passed to ``prompt`` formatting helpers.

        Returns:
            str: Parsed text returned by the underlying model.

        Raises:
            ValueError: Propagated when prompt formatting or output parsing fails.

        Examples:
            - Produce a completion using a non-chat model
                ```python
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt.upper(), delta=prompt.upper())
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> DemoLLM().predict(PromptTemplate("{greet}"), greet="hi")
                'HI'

                ```
            - Return the assistant message when operating in chat mode
                ```python
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     ChatResponse,
                ...     Message,
                ...     MessageRole,
                ...     Metadata,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         return ChatResponse(
                ...             message=Message(content="pong", role=MessageRole.ASSISTANT)
                ...         )
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> prompt = ChatPromptTemplate.from_messages([("user", "{word}")])
                >>> DemoLLM().predict(prompt, word="ping")
                'pong'

                ```
        See Also:
            apredict: Asynchronous variant that awaits the model result.
            stream: Streams incremental tokens instead of returning a single string.
        """
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.chat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = self.complete(formatted_prompt, formatted=True)
            output = response.text
        parsed_output = self._parse_output(output)

        return parsed_output

    def stream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenGen:
        """Stream tokens produced by the model for the given prompt.

        Args:
            prompt (BasePromptTemplate): Prompt template rendered prior to streaming.
            **prompt_args (Any): Keyword arguments used to populate ``prompt``.

        Yields:
            str: Incremental token strings emitted by the model.

        Raises:
            NotImplementedError: If an output parser is configured for either the prompt or the LLM.

        Examples:
            - Stream completion tokens for a non-chat model
                ```python
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     CompletionResponseGen,
                ...     Metadata,
                ... )
                >>> def completion_stream():
                ...     yield CompletionResponse(text="run", delta="r")
                ...     yield CompletionResponse(text="run", delta="un")
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs) -> CompletionResponseGen:
                ...         return completion_stream()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> list(
                ...     DemoLLM().stream(PromptTemplate("{verb}!"), verb="run")
                ... )
                ['r', 'un']

                ```
            - Stream assistant deltas for a chat model
                ```python
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     ChatResponse,
                ...     ChatResponseGen,
                ...     Message,
                ...     MessageRole,
                ...     Metadata,
                ... )
                >>> def chat_stream():
                ...     yield ChatResponse(
                ...         message=Message(content="ok", role=MessageRole.ASSISTANT),
                ...         delta="o",
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="ok", role=MessageRole.ASSISTANT),
                ...         delta="k",
                ...     )
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs) -> ChatResponseGen:
                ...         return chat_stream()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> tokens = list(
                ...     DemoLLM().stream(
                ...         ChatPromptTemplate.from_messages([("user", "ping")])
                ...     )
                ... )
                >>> tokens
                ['o', 'k']

                ```
        See Also:
            astream: Asynchronous streaming counterpart returning an async generator.
            predict: Convenience wrapper that buffers the entire response.
        """
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.stream_chat(messages)
            stream_tokens = stream_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = self.stream_complete(formatted_prompt, formatted=True)
            stream_tokens = stream_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    async def apredict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Asynchronously generate a completion for the provided prompt.

        Args:
            prompt (BasePromptTemplate): Prompt template rendered prior to model invocation.
            **prompt_args (Any): Keyword arguments used during template formatting.

        Returns:
            str: Parsed model output produced asynchronously.

        Raises:
            ValueError: Propagated when prompt formatting or output parsing fails.

        Examples:
            - Await a completion for a non-chat model
                ```python
                >>> import asyncio
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt[::-1], delta=prompt[::-1])
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> async def demo():
                ...     return await DemoLLM().apredict(PromptTemplate("{word}"), word="abc")
                ...
                >>> asyncio.run(demo())
                'cba'

                ```
            - Await a chat response when the model operates in chat mode
                ```python
                >>> import asyncio
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     ChatResponse,
                ...     Message,
                ...     MessageRole,
                ...     Metadata,
                ... )
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         return ChatResponse(
                ...             message=Message(content="pong", role=MessageRole.ASSISTANT)
                ...         )
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> async def demo():
                ...     prompt = ChatPromptTemplate.from_messages([("user", "ping")])
                ...     return await DemoLLM().apredict(prompt)
                ...
                >>> asyncio.run(demo())
                'pong'

                ```
        See Also:
            predict: Blocking variant that returns immediately.
            astream: Streams asynchronous tokens without aggregating the response.
        """
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.achat(messages)
            output = chat_response.message.content or ""
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            response = await self.acomplete(formatted_prompt, formatted=True)
            output = response.text

        parsed_output = self._parse_output(output)
        return parsed_output

    async def astream(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> TokenAsyncGen:
        """Asynchronously stream tokens for the provided prompt.

        Args:
            prompt (BasePromptTemplate): Prompt template rendered before streaming.
            **prompt_args (Any): Keyword arguments supplied to ``prompt``.

        Returns:
            TokenAsyncGen: Async generator yielding incremental token strings.

        Raises:
            NotImplementedError: If an output parser is configured on the prompt or LLM.

        Examples:
            - Stream completion deltas asynchronously
                ```python
                >>> import asyncio
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     CompletionResponse,
                ...     CompletionResponseAsyncGen,
                ...     Metadata,
                ... )
                >>> async def completion_stream():
                ...     yield CompletionResponse(text="run", delta="r")
                ...     yield CompletionResponse(text="run", delta="un")
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         return completion_stream()
                ...
                >>> async def demo():
                ...     stream = await DemoLLM().astream(PromptTemplate("{verb}"), verb="run")
                ...     return [token async for token in stream]
                ...
                >>> asyncio.run(demo())
                ['r', 'un']

                ```
            - Stream chat deltas asynchronously for chat models
                ```python
                >>> import asyncio
                >>> from serapeum.core.prompts import ChatPromptTemplate
                >>> from serapeum.core.base.llms.types import (
                ...     ChatResponse,
                ...     ChatResponseAsyncGen,
                ...     Message,
                ...     MessageRole,
                ...     Metadata,
                ... )
                >>> async def chat_stream():
                ...     yield ChatResponse(
                ...         message=Message(content="ok", role=MessageRole.ASSISTANT),
                ...         delta="o",
                ...     )
                ...     yield ChatResponse(
                ...         message=Message(content="ok", role=MessageRole.ASSISTANT),
                ...         delta="k",
                ...     )
                ...
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=True)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs) -> ChatResponseAsyncGen:
                ...         return chat_stream()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> async def demo():
                ...     prompt = ChatPromptTemplate.from_messages([("user", "ping")])
                ...     stream = await DemoLLM().astream(prompt)
                ...     return [token async for token in stream]
                ...
                >>> asyncio.run(demo())
                ['o', 'k']

                ```
        See Also:
            stream: Blocking variant returning a synchronous generator of tokens.
            apredict: Aggregates the async stream into a final string.
        """
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.astream_chat(messages)
            stream_tokens = await astream_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = await self.astream_complete(
                formatted_prompt, formatted=True
            )
            stream_tokens = await astream_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    def as_structured_llm(
        self,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> "StructuredOutputLLM":
        """Wrap this LLM with structured output capabilities.

        Args:
            output_cls (type[BaseModel]): Pydantic model describing the structured response schema.
            **kwargs (Any): Additional keyword arguments forwarded to ``StructuredOutputLLM``.

        Returns:
            StructuredOutputLLM: Wrapper that exposes structured inference helpers.

        Raises:
            Nothing: Construction simply instantiates ``StructuredOutputLLM``.

        Examples:
            - Construct a structured wrapper with default options
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class Person(BaseModel):
                ...     name: str
                ...
                >>> llm = DemoLLM()
                >>> wrapper = llm.as_structured_llm(Person)
                >>> wrapper.llm is llm
                True

                ```
            - Pass configuration options through to ``StructuredOutputLLM``
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.base.llms.types import CompletionResponse, Metadata
                >>> class DemoLLM(LLM):
                ...     metadata = Metadata.model_construct(is_chat_model=False)
                ...     def chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def stream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def achat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     async def astream_chat(self, messages, **kwargs):
                ...         raise NotImplementedError()
                ...     def complete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     def stream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...     async def acomplete(self, prompt, formatted=False, **kwargs):
                ...         return CompletionResponse(text=prompt, delta=prompt)
                ...     async def astream_complete(self, prompt, formatted=False, **kwargs):
                ...         raise NotImplementedError()
                ...
                >>> class Configured(BaseModel):
                ...     value: str
                ...
                >>> wrapper = DemoLLM().as_structured_llm(Configured, retries=2)
                >>> wrapper.retries
                2

                ```
        See Also:
            StructuredOutputLLM: Provides structured prediction helpers built atop the base LLM.
        """
        from serapeum.core.llms.structured_output_llm import StructuredOutputLLM

        return StructuredOutputLLM(llm=self, output_cls=output_cls, **kwargs)
