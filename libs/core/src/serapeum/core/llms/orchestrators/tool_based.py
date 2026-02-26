"""Programs that orchestrate tools/function-calling to produce Pydantic outputs."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Literal,
    Type,
    overload,
)

from pydantic import BaseModel

from serapeum.core.configs.configs import Configs
from serapeum.core.llms.abstractions.function_calling import FunctionCallingLLM
from serapeum.core.llms.base import LLM
from serapeum.core.llms.orchestrators.types import BasePydanticLLM, Model
from serapeum.core.llms.orchestrators.utils import StreamingObjectProcessor
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.tools.callable_tool import CallableTool

if TYPE_CHECKING:
    from serapeum.core.chat import AgentChatResponse

_logger = logging.getLogger(__name__)


class ToolOrchestratingLLM(BasePydanticLLM[BaseModel]):
    """Orchestrate function-calling LLMs to produce structured outputs.

    This program converts either a Pydantic model or a regular Python function
    into a callable tool and asks a function-calling LLM to use that tool to
    produce structured data. It handles prompt formatting, invoking the LLM
    (sync/async), optional streaming, and parsing the tool outputs.

    The class automatically detects the type of ``schema`` and uses the
    appropriate factory method:
    - Pydantic models → ``CallableTool.from_model()``
    - Regular functions → ``CallableTool.from_function()``

    Attributes:
        _schema (Union[Type[Model], Callable]): Either a Pydantic model class
            or a callable function defining the expected output structure.
        _llm (FunctionCallingLLM): The language model with function-calling
            support. Must advertise support via ``llm.metadata.is_function_calling_model``.
        _prompt (BasePromptTemplate): The prompt template used to format chat
            messages for the LLM.
        _verbose (bool): Whether to enable verbose logging of interactions.
        _allow_parallel_tool_calls (bool): If True, the LLM is permitted to call
            the tool multiple times in one response and a list of outputs will be
            returned.
        _tool_choice (Optional[Union[str, Dict[str, Any]]]): Strategy for tool
            selection (implementation-dependent for the given LLM).

    See Also:
        - BasePydanticLLM: Base abstraction for Pydantic-powered programs
        - CallableTool.from_model: Helper to convert Pydantic models into tools
        - CallableTool.from_function: Helper to convert functions into tools
        - FunctionCallingLLM: LLM abstraction with function calling support

    Examples:
    - Basic construction with a real LLM (Ollama). No network calls occur during initialization.
        ```python
        >>> from pydantic import BaseModel
        >>> from serapeum.ollama import Ollama
        >>> from serapeum.core.llms import ToolOrchestratingLLM
        >>>
        >>> class Output(BaseModel):
        ...     value: int
        >>> llm = Ollama(model='llama3.1')
        >>> tools_llm = ToolOrchestratingLLM(
        ...     schema=Output,
        ...     prompt='You are a helpful assistant.',
        ...     llm=llm,
        ... )
        >>> isinstance(tools_llm, ToolOrchestratingLLM)
        True

        ```
    """

    def __init__(
        self,
        *,
        schema: Type[Model] | Callable[..., Any],
        prompt: BasePromptTemplate | str,
        llm: FunctionCallingLLM | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        allow_parallel_tool_calls: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the ToolOrchestratingLLM instance.

        Args:
            schema (Union[Type[Model], Callable[..., Any]]): Either a Pydantic
                model class or a callable function defining the expected output.
                Despite the name, this accepts plain callables (not only classes).
                If a Pydantic model is provided, it will be converted into a tool via
                ``CallableTool.from_model()``. If a callable function is provided,
                it will be converted via ``CallableTool.from_function()``.
            prompt (Union[BasePromptTemplate, str]): Template (or plain string)
                used to generate messages for the LLM. If a string is provided it
                is wrapped in a ``PromptTemplate``.
            llm (Optional[FunctionCallingLLM]): A language model instance that
                supports function calling. If ``None``, ``Configs.llm`` is used.
            tool_choice (Optional[Union[str, Dict[str, Any]]]): Strategy for tool
                selection. May be ``"auto"``, ``"none"``, a specific tool name, or
                a dict with model-specific options.
            allow_parallel_tool_calls (bool): Whether to allow the LLM to call
                the tool multiple times in a single response. When True, a list
                of outputs may be returned.
            verbose (bool): Enable verbose logging.

        Raises:
            AssertionError: If no LLM is provided and ``Configs.llm`` is not set.
            ValueError: If the provided LLM does not support function calling.
            TypeError: If schema is neither a Pydantic model nor a callable.

        See Also:
            - Configs: Global configuration for default LLM settings
            - CallableTool.from_model: Factory for Pydantic models
            - CallableTool.from_function: Factory for regular functions

        Examples:
        - Instantiate with a Pydantic model (recommended). No network calls occur during initialization.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.ollama import Ollama
            >>> from serapeum.core.llms import ToolOrchestratingLLM
            >>> class Output(BaseModel):
            ...     value: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     schema=Output,
            ...     prompt='Prompt here',
            ...     llm=Ollama(model='llama3.1'),
            ... )
            >>> tools_llm.schema is Output
            True

            ```

        - Instantiate with a regular function (alternative approach).
            ```python
            >>> from serapeum.ollama import Ollama
            >>> from serapeum.core.llms import ToolOrchestratingLLM
            >>> def calculate_sum(a: int, b: int) -> dict:
            ...     '''Calculate the sum of two numbers.'''
            ...     return {'result': a + b}
            >>> tools_llm = ToolOrchestratingLLM(
            ...     schema=calculate_sum,
            ...     prompt='Calculate the sum of {x} and {y}',
            ...     llm=Ollama(model='llama3.1'),
            ... )
            >>> callable(tools_llm.schema)
            True

            ```
        """
        self._schema = self._validate_schema(schema)
        self._llm = self._validate_llm(llm)
        self._prompt = self._validate_prompt(prompt)
        self._verbose = verbose
        self._allow_parallel_tool_calls = allow_parallel_tool_calls
        self._tool_choice = tool_choice

    @staticmethod
    def _validate_schema(
        schema: Type[Model] | Callable[..., Any],
    ) -> Type[Model] | Callable[..., Any]:
        """Validate that schema is a Pydantic model class or a callable.

        Args:
            schema (Union[Type[Model], Callable[..., Any]]): The value to validate.

        Returns:
            Union[Type[Model], Callable[..., Any]]: The validated schema unchanged.

        Raises:
            TypeError: If schema is neither a Pydantic BaseModel subclass nor a callable.

        Examples:
        - Accept a Pydantic model class.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.llms import ToolOrchestratingLLM
            >>> class Out(BaseModel):
            ...     x: int
            >>> ToolOrchestratingLLM._validate_schema(Out) is Out
            True

            ```
        - Accept a plain callable.
            ```python
            >>> from serapeum.core.llms import ToolOrchestratingLLM
            >>> def fn(x: int) -> dict:
            ...     return {"x": x}
            >>> ToolOrchestratingLLM._validate_schema(fn) is fn
            True

            ```
        - Reject non-callable, non-model values.
            ```python
            >>> from serapeum.core.llms import ToolOrchestratingLLM
            >>> ToolOrchestratingLLM._validate_schema(42)
            Traceback (most recent call last):
            ...
            TypeError: schema must be either a Pydantic BaseModel subclass or a callable function. Got <class 'int'>

            ```
        """
        if not (
            (isinstance(schema, type) and issubclass(schema, BaseModel))
            or callable(schema)
        ):
            raise TypeError(
                "schema must be either a Pydantic BaseModel subclass or a callable function. "
                f"Got {type(schema)}"
            )
        return schema

    def _create_tool(self) -> CallableTool:
        """Create a CallableTool from the schema.

        Automatically detects whether schema is a Pydantic model or a callable
        function and uses the appropriate factory method.

        Returns:
            CallableTool: Tool instance created from schema.

        Raises:
            TypeError: If schema is neither a Pydantic model nor a callable.
        """
        # Check if it's a Pydantic model (class that inherits from BaseModel)
        if isinstance(self._schema, type) and issubclass(
            self._schema, BaseModel
        ):
            return CallableTool.from_model(self._schema)
        # Check if it's a callable (function, method, or callable class)
        elif callable(self._schema):
            return CallableTool.from_function(self._schema)
        else:
            raise TypeError(
                f"schema must be either a Pydantic BaseModel subclass or a callable function. "
                f"Got {type(self._schema)}"
            )

    @staticmethod
    def _validate_prompt(prompt: BasePromptTemplate | str) -> BasePromptTemplate:
        """Validate and normalize a prompt input.

        If a plain string is provided, it is wrapped in a ``PromptTemplate``.
        Otherwise, a ``BasePromptTemplate`` instance is returned unchanged.

        Args:
            prompt (Union[BasePromptTemplate, str]): Prompt specification as a
                template object or raw template string.

        Returns:
            BasePromptTemplate: A prompt template instance suitable for use in
            the program.

        Raises:
            ValueError: If ``prompt`` is neither a ``BasePromptTemplate`` nor a
                string.

        Examples:
        - Convert a string to a PromptTemplate.
            ```python
            >>> prompt_template = ToolOrchestratingLLM._validate_prompt('Hello, {name}!')
            >>> print(prompt_template)  # doctest: +ELLIPSIS
            metadata={'prompt_type': <PromptType.CUSTOM: 'custom'>} ... template='Hello, {name}!'

            ```
        - Invalid type raises ValueError.
            ```python
            >>> ToolOrchestratingLLM._validate_prompt(123)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: prompt must be an instance of BasePromptTemplate or str.

            ```

        See Also:
            - PromptTemplate: Concrete prompt template implementation
        """
        if not isinstance(prompt, (BasePromptTemplate, str)):
            raise ValueError("prompt must be an instance of BasePromptTemplate or str.")
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)
        return prompt

    @staticmethod
    def _validate_llm(llm: LLM) -> LLM:
        """Validate that an LLM supports function calling and resolve defaults.

        If ``llm`` is falsy, attempts to use ``Configs.llm``. Ensures the chosen
        model advertises function-calling support via
        ``llm.metadata.is_function_calling_model``.

        Args:
            llm (LLM): The language model instance to validate. May be ``None`` to
                fallback to ``Configs.llm``.

        Returns:
            LLM: The validated language model instance.

        Raises:
            AssertionError: If neither ``llm`` nor ``Configs.llm`` is set.
            ValueError: If the chosen model does not support the function-calling API.

        Examples:
        - Accept a model that supports function calling (Ollama).
            ```python
            >>> from serapeum.ollama import Ollama
            >>> out = ToolOrchestratingLLM._validate_llm(Ollama(model='llama3.1'))
            >>> out.metadata.model_name
            'llama3.1'

            ```
        """
        llm = llm or Configs.llm
        if llm is None:
            raise AssertionError("llm must be provided or set in Configs.")

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.metadata.model_name} does not support "
                "function calling API. "
            )
        return llm

    @property
    def schema(self) -> Type[BaseModel] | Callable[..., Any]:
        """Get the output class or callable used to define the expected structure.

        Returns:
            Union[Type[BaseModel], Callable[..., Any]]: The Pydantic model class or
            callable function passed at construction time.

        Examples:
        - Inspect the configured output class (Pydantic model).
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.ollama import Ollama
            >>> class Out(BaseModel):
            ...     x: int
            >>> tools_llm = ToolOrchestratingLLM(schema=Out, prompt='prompt', llm=Ollama(model='llama3.1'))
            >>> tools_llm.schema is Out
            True

            ```
        - Inspect the configured output class (callable function).
            ```python
            >>> from serapeum.ollama import Ollama
            >>> def fn(x: int) -> dict:
            ...     return {"x": x}
            >>> tools_llm = ToolOrchestratingLLM(schema=fn, prompt='prompt', llm=Ollama(model='llama3.1'))
            >>> tools_llm.schema is fn
            True

            ```
        """
        return self._schema

    @property
    def prompt(self) -> BasePromptTemplate:
        """Get the current prompt template.

        Returns:
            BasePromptTemplate: The prompt template used for formatting LLM inputs.

        Examples:
        - Access the prompt after construction from a string.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts.base import PromptTemplate
            >>> from serapeum.ollama import Ollama
            >>> tools_llm = ToolOrchestratingLLM(
            ...     schema=type('M', (BaseModel,), {}),
            ...     prompt='Hi',
            ...     llm=Ollama(model='llama3.1'),
            ... )
            >>> isinstance(tools_llm.prompt, PromptTemplate)
            True

            ```
        """
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        """Set a new prompt template.

        Args:
            prompt (BasePromptTemplate): New prompt template to use.

        Examples:
        - Replace the prompt with a different template.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts.base import PromptTemplate
            >>> from serapeum.ollama import Ollama
            >>> tools_llm = ToolOrchestratingLLM(
            ...     schema=type('M', (BaseModel,), {}),
            ...     prompt='Hi',
            ...     llm=Ollama(model='llama3.1'),
            ... )
            >>> tools_llm.prompt = PromptTemplate('New prompt')
            >>> isinstance(tools_llm.prompt, PromptTemplate)
            True

            ```
        """
        self._prompt = prompt

    @overload
    def __call__(
        self,
        *args: Any,
        stream: Literal[False] = ...,
        llm_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> BaseModel | list[BaseModel]: ...

    @overload
    def __call__(
        self,
        *args: Any,
        stream: Literal[True],
        llm_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> Generator[Model | list[Model], None, None]: ...

    def __call__(
        self,
        *args: Any,
        stream: bool = False,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseModel | list[BaseModel] | Generator[Model | list[Model], None, None]:
        """Execute the program to generate structured output.

        Formats the prompt with provided kwargs, invokes the LLM with the function
        calling tool, and parses the response into structured Pydantic model(s).

        Args:
            *args (Any): Positional arguments (currently unused).
            stream (bool): If True, returns a generator yielding incremental results.
                Defaults to False.
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM (e.g., temperature, max_tokens). Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.

        Returns:
            Union[BaseModel, List[BaseModel]] when stream=False;
            Generator[Union[Model, List[Model]], None, None] when stream=True.

        Raises:
            ValueError: If stream=True and the LLM does not support function calling.

        See Also:
            - acall: Async version of this method
        """
        if stream:
            return self._stream_call(*args, llm_kwargs=llm_kwargs, **kwargs)
        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()
        # convert the prompt into messages
        messages = self.prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        if self._tool_choice is not None:
            llm_kwargs.setdefault("tool_choice", self._tool_choice)

        agent_response: AgentChatResponse = self._llm.invoke_callable(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return agent_response.parse_tool_outputs(
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    def _stream_call(
        self, *args: Any, llm_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Generator[Model | list[Model], None, None]:
        """Internal streaming implementation — use ``__call__(stream=True)``."""
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("Streaming is only supported for FunctionCallingLLM instances.")

        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()

        if self._tool_choice is not None:
            llm_kwargs.setdefault("tool_choice", self._tool_choice)

        messages = self._prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = self._llm.generate_tool_calls(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            stream=True,
            **llm_kwargs,
        )

        cur_objects = None
        processor = StreamingObjectProcessor(
            output_cls=self._schema,
            flexible_mode=True,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            llm=self._llm,
        )
        for partial_resp in chat_response_gen:
            try:
                objects = processor.process(partial_resp, cur_objects)
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception as e:
                _logger.warning(f"Failed to parse streaming response: {e}")
                continue

    @overload
    async def acall(
        self,
        *args: Any,
        stream: Literal[False] = ...,
        llm_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> BaseModel | list[BaseModel]: ...

    @overload
    async def acall(
        self,
        *args: Any,
        stream: Literal[True],
        llm_kwargs: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> AsyncGenerator[Model | list[Model], None]: ...

    async def acall(
        self,
        *args: Any,
        stream: bool = False,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseModel | list[BaseModel] | AsyncGenerator[Model | list[Model], None]:
        """Asynchronously execute the program to generate structured output.

        Args:
            *args (Any): Positional arguments (currently unused).
            stream (bool): If True, returns an async generator yielding incremental
                results. Defaults to False.
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM. Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.

        Returns:
            Union[BaseModel, List[BaseModel]] when stream=False;
            AsyncGenerator[Union[Model, List[Model]], None] when stream=True.

        Raises:
            ValueError: If stream=True and the LLM does not support function calling.

        See Also:
            - __call__: Synchronous version of this method
        """
        if stream:
            return await self._astream_call(*args, llm_kwargs=llm_kwargs, **kwargs)
        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()

        if self._tool_choice is not None:
            llm_kwargs.setdefault("tool_choice", self._tool_choice)

        agent_response = await self._llm.ainvoke_callable(
            [tool],
            chat_history=self._prompt.format_messages(**kwargs),
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return agent_response.parse_tool_outputs(
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    async def _astream_call(
        self, *args: Any, llm_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> AsyncGenerator[Model | list[Model], None]:
        """Internal async streaming implementation — use ``acall(stream=True)``."""
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("Streaming is only supported for FunctionCallingLLM instances.")

        tool = self._create_tool()
        llm_kwargs = llm_kwargs or {}

        if self._tool_choice is not None:
            llm_kwargs.setdefault("tool_choice", self._tool_choice)

        messages = self._prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = await self._llm.agenerate_tool_calls(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            stream=True,
            **llm_kwargs,
        )
        processor = StreamingObjectProcessor(
            output_cls=self._schema,
            flexible_mode=True,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            llm=self._llm,
        )

        async def gen() -> AsyncGenerator[Model | list[Model], None]:
            cur_objects = None
            async for partial_resp in chat_response_gen:
                try:
                    objects = processor.process(partial_resp, cur_objects)
                    cur_objects = objects if isinstance(objects, list) else [objects]
                    yield objects
                except Exception as e:
                    _logger.warning(f"Failed to parse streaming response: {e}")
                    continue

        return gen()
