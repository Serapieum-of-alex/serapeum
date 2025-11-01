from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Generator,
    AsyncGenerator,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
    Type,
)
from typing_extensions import Annotated

from serapeum.core.base.llms.models import (
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)
from pydantic import (
    BaseModel,
    WithJsonSchema,
    Field,
    field_validator,
    model_validator,
)
from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.base.llms.utils import (
    messages_to_prompt as generic_messages_to_prompt,
)

from serapeum.core.prompts import BasePromptTemplate, PromptTemplate
from serapeum.core.models import Model, StructuredLLMMode
from serapeum.core.output_parsers.models import (
    BaseOutputParser,
    TokenAsyncGen,
    TokenGen,
)

from serapeum.core.base.llms.models import (
    Message,
)


if TYPE_CHECKING:
    from serapeum.core.llm.structured_llm import StructuredLLM


@runtime_checkable
class MessagesToPromptType(Protocol):
    r"""Runtime protocol describing adapters that convert chat messages into prompts.

    Examples:
        - Join message contents into a newline-separated prompt
            ```python
            >>> from serapeum.core.llm.base import MessagesToPromptType
            >>> from serapeum.core.base.llms.models import Message, MessageRole
            >>> def newline_join(messages):
            ...     return '\n'.join(message.content or "" for message in messages)
            ...
            >>> isinstance(newline_join, MessagesToPromptType)
            True

            ```
        - Validate message content before rendering the prompt
            ```python
            >>> from serapeum.core.base.llms.models import Message, MessageRole
            >>> def validated_join(messages):
            ...     contents = [message.content for message in messages]
            ...     if any(content is None for content in contents):
            ...         raise ValueError("Missing content")
            ...     return " ".join(str(content) for content in contents)
            ...
            >>> validated_join([Message(content="hi", role=MessageRole.USER)])
            'hi'

            ```
    See Also:
        MessagesToPromptType.__call__: Details the expected callable signature.
    """

    def __call__(self, messages: Sequence[Message]) -> str:
        """Render a sequence of chat messages into a single prompt string.

        Args:
            messages (Sequence[Message]): Ordered chat messages to convert.

        Returns:
            str: Textual prompt synthesized from the provided messages.

        Raises:
            ValueError: Implementations may raise when a message payload is invalid.

        Examples:
            - Concatenate user and assistant messages into one prompt
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole
                >>> def concatenate(messages):
                ...     return " ".join((message.content or "").strip() for message in messages)
                ...
                >>> concatenate(
                ...     [
                ...         Message(content="Hello", role=MessageRole.USER),
                ...         Message(content="world", role=MessageRole.ASSISTANT),
                ...     ]
                ... )
                'Hello world'

                ```
            - Reject empty message content before joining
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole
                >>> def strict_concat(messages):
                ...     for message in messages:
                ...         if message.content is None:
                ...             raise ValueError("Missing content")
                ...     return " ".join(str(message.content) for message in messages)
                ...
                >>> strict_concat([Message(content="Ping", role=MessageRole.USER)])
                'Ping'

                ```
        See Also:
            LLM._get_messages: Prepares message sequences prior to prompt rendering.
        """
        ...


@runtime_checkable
class CompletionToPromptType(Protocol):
    """Runtime protocol describing prompt adapters invoked before completions.

    Examples:
        - Check that an identity adapter satisfies the protocol
            ```python
            >>> from serapeum.core.llm.base import CompletionToPromptType
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


def stream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseGen,
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
        - Collect deltas produced by a completion stream
            ```python
            >>> from serapeum.core.base.llms.models import CompletionResponse
            >>> from serapeum.core.llm.base import stream_completion_response_to_tokens
            >>> def responses():
            ...     yield CompletionResponse(text="Hello", delta="Hel")
            ...     yield CompletionResponse(text="Hello", delta="lo")
            ...
            >>> list(stream_completion_response_to_tokens(responses()))
            ['Hel', 'lo']

            ```
        - Handle responses that omit delta text
            ```python
            >>> def responses():
            ...     yield CompletionResponse(text="partial", delta=None)
            ...     yield CompletionResponse(text="done", delta="")
            ...
            >>> list(stream_completion_response_to_tokens(responses()))
            ['', '']

            ```
    See Also:
        astream_completion_response_to_tokens: Asynchronous variant returning an async generator of tokens.
    """

    def gen() -> TokenGen:
        for response in completion_response_gen:
            yield response.delta or ""

    return gen()


def stream_chat_response_to_tokens(
    chat_response_gen: ChatResponseGen,
) -> TokenGen:
    """Convert a stream completion response to a stream of tokens."""

    def gen() -> TokenGen:
        for response in chat_response_gen:
            yield response.delta or ""

    return gen()


async def astream_completion_response_to_tokens(
    completion_response_gen: CompletionResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in completion_response_gen:
            yield response.delta or ""

    return gen()


async def astream_chat_response_to_tokens(
    chat_response_gen: ChatResponseAsyncGen,
) -> TokenAsyncGen:
    """Convert a stream completion response to a stream of tokens."""

    async def gen() -> TokenAsyncGen:
        async for response in chat_response_gen:
            yield response.delta or ""

    return gen()


def default_completion_to_prompt(prompt: str) -> str:
    return prompt


MessagesToPromptCallable = Annotated[
    Optional[MessagesToPromptType],
    WithJsonSchema({"type": "string"}),
]


CompletionToPromptCallable = Annotated[
    Optional[CompletionToPromptType],
    WithJsonSchema({"type": "string"}),
]


class LLM(BaseLLM, ABC):
    """
    The LLM class is the main class for interacting with language models.

    Attributes:
        system_prompt (Optional[str]):
            System prompt for LLM calls.
        messages_to_prompt (Callable):
            Function to convert a list of messages to an LLM prompt.
        completion_to_prompt (Callable):
            Function to convert a completion to an LLM prompt.
        output_parser (Optional[BaseOutputParser]):
            Output parser to parse, validate, and correct errors programmatically.
        pydantic_program_mode (StructuredLLMMode):
            Pydantic program mode to use for structured prediction.
    """

    system_prompt: Optional[str] = Field(
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
    output_parser: Optional[BaseOutputParser] = Field(
        description="Output parser to parse, validate, and correct errors programmatically.",
        default=None,
        exclude=True,
    )
    pydantic_program_mode: StructuredLLMMode = StructuredLLMMode.DEFAULT

    # # deprecated
    query_wrapper_prompt: Optional[BasePromptTemplate] = Field(
        description="Query wrapper prompt for LLM calls.",
        default=None,
        exclude=True,
    )

    @field_validator("messages_to_prompt")
    @classmethod
    def set_messages_to_prompt(
        cls, messages_to_prompt: Optional[MessagesToPromptType]
    ) -> MessagesToPromptType:
        return messages_to_prompt or generic_messages_to_prompt

    @field_validator("completion_to_prompt")
    @classmethod
    def set_completion_to_prompt(
        cls, completion_to_prompt: Optional[CompletionToPromptType]
    ) -> CompletionToPromptType:
        return completion_to_prompt or default_completion_to_prompt

    @model_validator(mode="after")
    def check_prompts(self) -> "LLM":
        if self.completion_to_prompt is None:
            self.completion_to_prompt = default_completion_to_prompt
        if self.messages_to_prompt is None:
            self.messages_to_prompt = generic_messages_to_prompt
        return self


    def _get_prompt(self, prompt: BasePromptTemplate, **prompt_args: Any) -> str:
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
    ) -> List[Message]:
        messages = prompt.format_messages(llm=self, **prompt_args)
        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)
        return self._extend_messages(messages)

    def _parse_output(self, output: str) -> str:
        if self.output_parser is not None:
            return str(self.output_parser.parse(output))

        return output

    def _extend_prompt(
        self,
        formatted_prompt: str,
    ) -> str:
        """Add system and query wrapper prompts to base prompt."""
        extended_prompt = formatted_prompt

        if self.system_prompt:
            extended_prompt = self.system_prompt + "\n\n" + extended_prompt

        if self.query_wrapper_prompt:
            extended_prompt = self.query_wrapper_prompt.format(
                query_str=extended_prompt
            )

        return extended_prompt

    def _extend_messages(self, messages: List[Message]) -> List[Message]:
        """Add system prompt to chat message list."""
        if self.system_prompt:
            messages = [
                Message(role=MessageRole.SYSTEM, content=self.system_prompt),
                *messages,
            ]
        return messages

    def structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        r"""Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            llm_kwargs (Optional[Dict[str, Any]]):
                Arguments that are passed down to the LLM invoked by the program.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            BaseModel: The structured prediction output.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            output = llm.structured_predict(Test, prompt, topic="cats")
            print(output.name)
            ```
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
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        r"""Async Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            llm_kwargs (Optional[Dict[str, Any]]):
                Arguments that are passed down to the LLM invoked by the program.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            BaseModel: The structured prediction output.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            output = await llm.astructured_predict(Test, prompt, topic="cats")
            print(output.name)
            ```
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
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, List[Model]], None, None]:
        r"""Stream Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            llm_kwargs (Optional[Dict[str, Any]]):
                Arguments that are passed down to the LLM invoked by the program.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            Generator: A generator returning partial copies of the model or list of models.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            stream_output = llm.stream_structured_predict(Test, prompt, topic="cats")
            for partial_output in stream_output:
                # stream partial outputs until completion
                print(partial_output.name)
            ```
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
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[
        Union[Model, List[Model], "BaseModel", List["BaseModel"]], None
    ]:
        from serapeum.core.structured_tools.utils import get_program_for_llm

        program = get_program_for_llm(
            output_cls,
            prompt,
            self,
            pydantic_program_mode=self.pydantic_program_mode,
        )

        return await program.astream_call(llm_kwargs=llm_kwargs, **prompt_args)

    async def astream_structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        r"""Async Stream Structured predict.

        Args:
            output_cls (BaseModel):
                Output class to use for structured prediction.
            prompt (PromptTemplate):
                Prompt template to use for structured prediction.
            llm_kwargs (Optional[Dict[str, Any]]):
                Arguments that are passed down to the LLM invoked by the program.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            Generator: A generator returning partial copies of the model or list of models.

        Examples:
            ```python
            from pydantic import BaseModel

            class Test(BaseModel):
                \"\"\"My test class.\"\"\"
                name: str

            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please predict a Test with a random name related to {topic}.")
            stream_output = await llm.astream_structured_predict(Test, prompt, topic="cats")
            async for partial_output in stream_output:
                # stream partial outputs until completion
                print(partial_output.name)
            ```
        """

        async def gen() -> AsyncGenerator[Union[Model, List[Model]], None]:
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
        """Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            output = llm.predict(prompt, topic="cats")
            print(output)
            ```
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
        """Stream predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Yields:
            str: Each streamed token.

        Examples:
            ```python
            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            gen = llm.stream_predict(prompt, topic="cats")
            for token in gen:
                print(token, end="", flush=True)
            ```
        """
        self._log_template_data(prompt, **prompt_args)

        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = self.stream_chat(messages)
            stream_tokens = stream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = self.stream_complete(formatted_prompt, formatted=True)
            stream_tokens = stream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    async def apredict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any,
    ) -> str:
        """Async Predict for a given prompt.

        Args:
            prompt (BasePromptTemplate):
                The prompt to use for prediction.
            prompt_args (Any):
                Additional arguments to format the prompt with.

        Returns:
            str: The prediction output.

        Examples:
            ```python
            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            output = await llm.apredict(prompt, topic="cats")
            print(output)
            ```
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
        """Async stream predict for a given prompt.

        Args:
        prompt (BasePromptTemplate):
            The prompt to use for prediction.
        prompt_args (Any):
            Additional arguments to format the prompt with.

        Yields:
            str: An async generator that yields strings of tokens.

        Examples:
            ```python
            from serapeum.core.prompts import PromptTemplate

            prompt = PromptTemplate("Please write a random name related to {topic}.")
            gen = await llm.astream_predict(prompt, topic="cats")
            async for token in gen:
                print(token, end="", flush=True)
            ```
        """
        if self.metadata.is_chat_model:
            messages = self._get_messages(prompt, **prompt_args)
            chat_response = await self.astream_chat(messages)
            stream_tokens = await astream_chat_response_to_tokens(chat_response)
        else:
            formatted_prompt = self._get_prompt(prompt, **prompt_args)
            stream_response = await self.astream_complete(
                formatted_prompt, formatted=True
            )
            stream_tokens = await astream_completion_response_to_tokens(stream_response)

        if prompt.output_parser is not None or self.output_parser is not None:
            raise NotImplementedError("Output parser is not supported for streaming.")

        return stream_tokens

    def as_structured_llm(
        self,
        output_cls: Type[BaseModel],
        **kwargs: Any,
    ) -> "StructuredLLM":
        """Return a structured LLM around a given object."""
        from serapeum.core.llm.structured_llm import StructuredLLM

        return StructuredLLM(llm=self, output_cls=output_cls, **kwargs)
