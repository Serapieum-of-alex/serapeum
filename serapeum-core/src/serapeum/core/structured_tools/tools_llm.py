"""Programs that orchestrate tools/function-calling to produce Pydantic outputs."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel

from serapeum.core.configs.configs import Configs
from serapeum.core.llm.base import LLM
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.structured_tools.models import BasePydanticLLM, Model
from serapeum.core.structured_tools.utils import StreamingObjectProcessor
from serapeum.core.tools.callable_tool import CallableTool

if TYPE_CHECKING:
    from serapeum.core.chat.models import AgentChatResponse

_logger = logging.getLogger(__name__)


class ToolOrchestratingLLM(BasePydanticLLM[BaseModel]):
    """Orchestrate function-calling LLMs to produce structured outputs.

    This program converts either a Pydantic model or a regular Python function
    into a callable tool and asks a function-calling LLM to use that tool to
    produce structured data. It handles prompt formatting, invoking the LLM
    (sync/async), optional streaming, and parsing the tool outputs.

    The class automatically detects the type of ``output_cls`` and uses the
    appropriate factory method:
    - Pydantic models → ``CallableTool.from_model()``
    - Regular functions → ``CallableTool.from_function()``

    Attributes:
        _output_cls (Union[Type[Model], Callable]): Either a Pydantic model class
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
        >>> from serapeum.llms.ollama.base import Ollama
        >>> from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM
        >>>
        >>> class Output(BaseModel):
        ...     value: int
        >>> llm = Ollama(model='llama3.1')
        >>> tools_llm = ToolOrchestratingLLM(
        ...     output_cls=Output,
        ...     prompt='You are a helpful assistant.',
        ...     llm=llm,
        ... )
        >>> isinstance(tools_llm, ToolOrchestratingLLM)
        True

        ```
    """

    def __init__(
        self,
        output_cls: Union[Type[Model], Callable[..., Any]],
        prompt: Union[BasePromptTemplate, str],
        llm: Optional[FunctionCallingLLM] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        allow_parallel_tool_calls: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the ToolOrchestratingLLM instance.

        Args:
            output_cls (Union[Type[Model], Callable[..., Any]]): Either a Pydantic
                model class or a callable function defining the expected output.
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
            TypeError: If output_cls is neither a Pydantic model nor a callable.

        See Also:
            - Configs: Global configuration for default LLM settings
            - CallableTool.from_model: Factory for Pydantic models
            - CallableTool.from_function: Factory for regular functions

        Examples:
        - Instantiate with a Pydantic model (recommended). No network calls occur during initialization.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM
            >>> class Output(BaseModel):
            ...     value: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     Output,
            ...     'Prompt here',
            ...     Ollama(model='llama3.1'),
            ... )
            >>> tools_llm.output_cls is Output
            True

            ```

        - Instantiate with a regular function (alternative approach).
            ```python
            >>> from serapeum.llms.ollama.base import Ollama
            >>> from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM
            >>> def calculate_sum(a: int, b: int) -> dict:
            ...     '''Calculate the sum of two numbers.'''
            ...     return {'result': a + b}
            >>> tools_llm = ToolOrchestratingLLM(
            ...     calculate_sum,
            ...     'Calculate the sum of {x} and {y}',
            ...     Ollama(model='llama3.1'),
            ... )
            >>> callable(tools_llm.output_cls)
            True

            ```
        """
        self._output_cls = output_cls
        self._llm = self._validate_llm(llm)
        self._prompt = self._validate_prompt(prompt)
        self._verbose = verbose
        self._allow_parallel_tool_calls = allow_parallel_tool_calls
        self._tool_choice = tool_choice

    def _create_tool(self) -> CallableTool:
        """Create a CallableTool from the output_cls.

        Automatically detects whether output_cls is a Pydantic model or a callable
        function and uses the appropriate factory method.

        Returns:
            CallableTool: Tool instance created from output_cls.

        Raises:
            TypeError: If output_cls is neither a Pydantic model nor a callable.
        """
        # Check if it's a Pydantic model (class that inherits from BaseModel)
        if isinstance(self._output_cls, type) and issubclass(self._output_cls, BaseModel):
            return CallableTool.from_model(self._output_cls)
        # Check if it's a callable (function, method, or callable class)
        elif callable(self._output_cls):
            return CallableTool.from_function(self._output_cls)
        else:
            raise TypeError(
                f"output_cls must be either a Pydantic BaseModel subclass or a callable function. "
                f"Got {type(self._output_cls)}"
            )

    @staticmethod
    def _validate_prompt(prompt: Union[BasePromptTemplate, str]) -> BasePromptTemplate:
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
            >>> from serapeum.llms.ollama.base import Ollama
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
    def output_cls(self) -> Type[BaseModel]:
        """Get the output Pydantic model class.

        Returns:
            Type[BaseModel]: The Pydantic model class used for structured output.

        Examples:
        - Inspect the configured output class.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Out(BaseModel):
            ...     x: int
            >>> tools_llm = ToolOrchestratingLLM(Out, 'prompt', Ollama(model='llama3.1'))
            >>> tools_llm.output_cls is Out
            True

            ```
        """
        return self._output_cls

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
            >>> from serapeum.llms.ollama.base import Ollama
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=type('M', (BaseModel,), {}),
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
            >>> from serapeum.llms.ollama.base import Ollama
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=type('M', (BaseModel,), {}),
            ...     prompt='Hi',
            ...     llm=Ollama(model='llama3.1'),
            ... )
            >>> tools_llm.prompt = PromptTemplate('New prompt')
            >>> isinstance(tools_llm.prompt, PromptTemplate)
            True

            ```
        """
        self._prompt = prompt

    def __call__(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[BaseModel, List[BaseModel]]:
        """Execute the program to generate structured output.

        Formats the prompt with provided kwargs, invokes the LLM with the function
        calling tool, and parses the response into structured Pydantic model(s).

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM (e.g., temperature, max_tokens). Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.
                These should match variables in the prompt.

        Returns:
            BaseModel: A single Pydantic model instance when parallel calls are
            disabled, or a list of Pydantic models when enabled.

        Raises:
            ValueError: If the underlying LLM raises an error due to invalid
                arguments or internal failures.

        See Also:
            - acall: Async version of this method
            - stream_call: Streaming version for incremental results

        Examples:
        - Run a single structured prediction with a real LLM (Ollama).
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=Person,
            ...     prompt="Extract the person's name and age from the following text: {text}",
            ...     llm=Ollama(model='llama3.1', request_timeout=80),
            ... ) # doctest: +SKIP
            >>> result = tools_llm(text='My name is Alice and I am 30 years old.')  # doctest: +SKIP
            >>> print(result) # doctest: +SKIP
            name='Alice' age=30

            ```

        - Enable parallel tool-calls to receive multiple objects when the model chooses to call the tool more than once.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Item(BaseModel):
            ...     name: str
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=Item,
            ...     prompt='List three fruit names as separate tool calls.',
            ...     llm=Ollama(model='llama3.1', request_timeout=80),
            ...     allow_parallel_tool_calls=True,
            ... ) # doctest: +SKIP
            >>> results = tools_llm()  # doctest: +SKIP
            >>> print(results) # doctest: +SKIP)
            [Item(name='Apple'), Item(name='Banana'), Item(name='Orange')]

            ```
        """
        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()
        # convert the prompt into messages
        messages = self.prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        agent_response: AgentChatResponse = self._llm.predict_and_call(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return agent_response.parse_tool_outputs(
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    async def acall(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[BaseModel, List[BaseModel]]:
        """Asynchronously execute the program to generate structured output.

        Async version of ``__call__``. Formats the prompt with provided kwargs,
        asynchronously invokes the LLM with the function-calling tool, and parses
        the response into structured Pydantic model(s).

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM (e.g., temperature, max_tokens). Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.
                These should match variables in the prompt.

        Returns:
            BaseModel: A single Pydantic model instance when parallel calls are
            disabled, or a list of Pydantic models when enabled.

        See Also:
            - __call__: Synchronous version of this method
            - astream_call: Async streaming version for incremental results

        Examples:
        - Typical async usage with a real LLM (Ollama).
            ```python
            >>> import asyncio
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Out(BaseModel):
            ...     value: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     Out,
            ...     'Return an integer value for {thing}',
            ...     Ollama(model='llama3.1', request_timeout=80)
            ... )
            >>> result = asyncio.run(tools_llm.acall(thing='a number'))  # doctest: +SKIP
            >>> result  # doctest: +SKIP
            Out(value=123)

            ```
        """
        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()

        agent_response = await self._llm.apredict_and_call(
            [tool],
            chat_history=self._prompt.format_messages(**kwargs),
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return agent_response.parse_tool_outputs(
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )

    def stream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured output generation with incremental updates.

        Returns a generator that yields progressively refined structured objects as
        the LLM generates its response. Each item is a partial or complete instance
        of the output model, enabling progressive rendering in UIs.

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM (e.g., temperature, max_tokens). Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.

        Yields:
            Union[Model, List[Model]]: Progressive updates of the structured output.
            When ``allow_parallel_tool_calls`` is True, a list of models may be yielded.

        Raises:
            ValueError: If ``self._llm`` is not an instance of ``FunctionCallingLLM``.

        Warns:
            Logs a warning when parsing streaming responses fails and continues with
            the next chunk.

        See Also:
            - __call__: Non-streaming synchronous version
            - astream_call: Async streaming version

        Examples:
        - Iterate over streaming results with a real LLM (Ollama). Requires an Ollama server and a pulled model.
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Number(BaseModel):
            ...     n: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=Number,
            ...     prompt='Stream the numbers 1, 2, and 3 as separate tool calls.',
            ...     llm=Ollama(model='llama3.1', request_timeout=80),
            ... )
            >>> for obj in tools_llm.stream_call():  # doctest: +SKIP
            >>>      print(obj)
            n=1
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            n=1
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            n=1
            n=1

            ```
        """
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        llm_kwargs = llm_kwargs or {}
        tool = self._create_tool()

        messages = self._prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = self._llm.stream_chat_with_tools(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )

        cur_objects = None
        for partial_resp in chat_response_gen:
            try:
                processor = StreamingObjectProcessor(
                    output_cls=self._output_cls,
                    flexible_mode=True,
                    allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                    llm=self._llm,
                )
                objects = processor.process(partial_resp, cur_objects)

                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception as e:
                _logger.warning(f"Failed to parse streaming response: {e}")
                continue

    async def astream_call(
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        """Asynchronously stream structured output generation with incremental updates.

        Async counterpart to ``stream_call``. Yields progressively refined structured
        objects as the LLM generates its response.

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to
                pass to the LLM (e.g., temperature, max_tokens). Defaults to ``None``.
            **kwargs (Any): Keyword arguments used to format the prompt template.

        Returns:
            AsyncGenerator[Union[Model, List[Model]], None]:
                An async generator that yields progressive updates of the structured
                output. Must be awaited before iteration.
                When ``allow_parallel_tool_calls`` is True,
                a list of models may be yielded.

        Raises:
            ValueError: If ``self._llm`` is not a ``FunctionCallingLLM`` instance.

        Warns:
            Logs a warning when parsing streaming responses fails and continues with
            the next chunk.

        See Also:
            - acall: Non-streaming async version
            - stream_call: Synchronous streaming version

        Examples:
        - Consume async streaming results with a real LLM (Ollama). Requires an Ollama server and a pulled model.
            ```python
            >>> import asyncio
            >>> from pydantic import BaseModel
            >>> from serapeum.llms.ollama.base import Ollama
            >>> class Number(BaseModel):
            ...     n: int
            >>> tools_llm = ToolOrchestratingLLM(
            ...     output_cls=Number,
            ...     prompt='Stream the numbers 1, 2, and 3 as separate tool calls.',
            ...     llm=Ollama(model='llama3.1', request_timeout=80),
            ... )
            >>> async def consume():  # doctest: +SKIP
            >>>     async for obj in await tools_llm.astream_call():  # doctest: +SKIP
            >>>         print(obj)  # doctest: +SKIP
            >>> asyncio.run(consume())  # doctest: +SKIP
            n=1
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            n=1
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            Multiple outputs found, returning first one.
            If you want to return all outputs, set allow_parallel_tool_calls=True.
            n=1
            n=1

            ```
        """
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        tool = self._create_tool()

        messages = self._prompt.format_messages(**kwargs)
        messages = self._llm._extend_messages(messages)

        chat_response_gen = await self._llm.astream_chat_with_tools(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **(llm_kwargs or {}),
        )

        async def gen() -> AsyncGenerator[Union[Model, List[Model]], None]:
            cur_objects = None
            async for partial_resp in chat_response_gen:
                try:
                    processor = StreamingObjectProcessor(
                        output_cls=self._output_cls,
                        flexible_mode=True,
                        allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                        llm=self._llm,
                    )
                    objects = processor.process(partial_resp, cur_objects)

                    cur_objects = objects if isinstance(objects, list) else [objects]
                    yield objects
                except Exception as e:
                    _logger.warning(f"Failed to parse streaming response: {e}")
                    continue

        return gen()
