import logging
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    cast,
    Union,
    List,
    Generator,
    AsyncGenerator,
)

from pydantic import (
    BaseModel,
    ValidationError,
)
from serapeum.core.base.llms.models import ChatResponse
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.llm.base import LLM
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.configs.configs import Configs
from serapeum.core.structured_tools.models import BasePydanticProgram, Model
from serapeum.core.tools.callable_tool import CallableTool
from serapeum.core.structured_tools.utils import (
    process_streaming_objects,
    num_valid_fields,
)

_logger = logging.getLogger(__name__)


def _parse_tool_outputs(
    agent_response,
    allow_parallel_tool_calls: bool = False,
) -> Union[BaseModel, List[BaseModel]]:
    """Parse tool outputs from agent response.

    Extracts and returns structured output models from an agent's tool execution
    response. When parallel tool calls are disabled, only the first output is
    returned. When enabled, all outputs are returned as a list.

    Args:
        agent_response: The agent chat response containing tool outputs in its
            sources attribute. Each source should have a raw_output field
            containing a BaseModel instance.
        allow_parallel_tool_calls (bool, optional): If True, returns all tool
            outputs as a list. If False, returns only the first output and logs
            a warning if multiple outputs exist. Defaults to False.

    Returns:
        Union[BaseModel, List[BaseModel]]: A single BaseModel instance when
            allow_parallel_tool_calls is False, or a list of BaseModel instances
            when allow_parallel_tool_calls is True.

    Warns:
        Logs a warning message when multiple outputs are found but allow_parallel_tool_calls is False.

    Example:
        - Parse single tool output (parallel calls disabled):
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.chat.models import AgentChatResponse
            >>> from serapeum.core.tools import ToolOutput
            >>> from serapeum.core.structured_tools.tools_llm import _parse_tool_outputs
            >>>
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> person = Person(name="Alice", age=30)
            >>> tool_output = ToolOutput(
            ...     content="Person created",
            ...     tool_name="create_person",
            ...     raw_input={},
            ...     raw_output=person
            ... )
            >>> response = AgentChatResponse(
            ...     response="Created person",
            ...     sources=[tool_output]
            ... )
            >>> result = _parse_tool_outputs(response, allow_parallel_tool_calls=False)
            >>> print(result)
            name='Alice' age=30
            >>> result.name
            'Alice'
            >>> result.age
            30

            ```

        - Parse multiple tool outputs with parallel calls enabled:
            ```python
            >>> person1 = Person(name="Bob", age=25)
            >>> person2 = Person(name="Charlie", age=35)
            >>> tool_outputs = [
            ...     ToolOutput(
            ...         content="Person 1", tool_name="create_person",
            ...         raw_input={}, raw_output=person1
            ...     ),
            ...     ToolOutput(
            ...         content="Person 2", tool_name="create_person",
            ...         raw_input={}, raw_output=person2
            ...     )
            ... ]
            >>> response = AgentChatResponse(
            ...     response="Created persons", sources=tool_outputs
            ... )
            >>> results = _parse_tool_outputs(response, allow_parallel_tool_calls=True)
            >>> len(results)
            2
            >>> results[0].name
            'Bob'
            >>> results[1].name
            'Charlie'

            ```

        - Parse multiple outputs with parallel calls disabled (logs warning):
            ```python
            >>> response = AgentChatResponse(
            ...     response="Created persons", sources=tool_outputs
            ... )
            >>> result = _parse_tool_outputs(response, allow_parallel_tool_calls=False)
            >>> result.name
            'Bob'

            ```

    See Also:
        - get_function_tool: Creates callable tools from Pydantic models
        - ToolOrchestratingLLM.__call__: Main execution method using this parser
    """
    outputs = [cast(BaseModel, s.raw_output) for s in agent_response.sources]
    if allow_parallel_tool_calls:
        val = outputs
    else:
        if len(outputs) > 1:
            _logger.warning(
                "Multiple outputs found, returning first one. "
                "If you want to return all outputs, set allow_parallel_tool_calls=True."
            )

        val = outputs[0]

    return val


class ToolOrchestratingLLM(BasePydanticProgram[BaseModel]):
    """Function calling program that orchestrates LLM tool usage for structured outputs.

    This class enables LLMs with function calling capabilities to generate structured
    data by converting Pydantic models into callable tools. It manages the entire
    workflow of formatting prompts, invoking the LLM with tools, and parsing the
    structured outputs.

    The class supports both single and parallel tool calls, synchronous and
    asynchronous execution, and streaming responses.

    Attributes:
        _output_cls (Type[Model]):
            The Pydantic model class defining output structure.
        _llm (FunctionCallingLLM):
            The language model with function calling support.
        _prompt (BasePromptTemplate):
            Template for generating prompts.
        _verbose (bool):
            Whether to enable verbose logging.
        _allow_parallel_tool_calls (bool):
            Whether to allow multiple tool calls.
        _tool_choice (Optional[Union[str, Dict[str, Any]]]):
            Tool selection strategy.


    See Also:
        - get_function_tool: Creates tools from Pydantic models
        - BasePydanticProgram: Base class for Pydantic programs
        - FunctionCallingLLM: LLM interface with function calling support
    """

    def __init__(
        self,
        output_cls: Type[Model],
        prompt: Union[BasePromptTemplate, str],
        llm: Optional[FunctionCallingLLM] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        allow_parallel_tool_calls: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the ToolOrchestratingLLM instance.

        Args:
            output_cls (Type[Model]): Pydantic model class defining the structure
                of the expected output. This model will be converted to a callable
                tool for the LLM to use.
            prompt (BasePromptTemplate): Template for generating prompts that will be
                sent to the LLM. Can contain variables to be filled at call time.
            llm (FunctionCallingLLM, optional): A language model instance with function calling
                capabilities. Must have is_function_calling_model=True in metadata.
                If None, uses Configs.llm global configuration. Defaults to None.
            tool_choice (Optional[Union[str, Dict[str, Any]]], optional): Strategy
                for tool selection. Can be "auto", "none", a specific tool name, or
                a dictionary with detailed tool choice configuration. Defaults to None.
            allow_parallel_tool_calls (bool, optional): If True, allows the LLM to
                make multiple tool calls in a single response, returning a list of
                outputs. If False, only returns the first output. Defaults to False.
            verbose (bool, optional): If True, enables detailed logging of LLM
                interactions and tool calls. Defaults to False.

        Raises:
            ValueError: If the LLM doesn't support function calling.
            ValueError: If neither prompt nor prompt_template_str is provided.
            ValueError: If both prompt and prompt_template_str are provided.
            AssertionError: If llm is None after attempting to get from Configs.

        See Also:
            - __init__: Direct constructor if you already have all components
            - Configs: Global configuration for default LLM settings
        """
        self._output_cls = output_cls
        self._llm = self.validate_llm(llm)
        self._prompt = self.validate_prompt(prompt)
        self._verbose = verbose
        self._allow_parallel_tool_calls = allow_parallel_tool_calls
        self._tool_choice = tool_choice

    @staticmethod
    def validate_prompt(prompt: Union[BasePromptTemplate, str]) -> BasePromptTemplate:
        if not isinstance(prompt, (BasePromptTemplate, str)):
            raise ValueError(
                "prompt must be an instance of BasePromptTemplate or str."
            )
        if isinstance(prompt, str):
            prompt = PromptTemplate(prompt)
        return prompt

    @staticmethod
    def validate_llm(llm: LLM) -> LLM:
        llm = llm or Configs.llm  # type: ignore
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

        """
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        """Get the current prompt template.

        Returns:
            BasePromptTemplate: The prompt template used for formatting LLM inputs.

        """
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        """Set a new prompt template.

        Args:
            prompt (BasePromptTemplate): New prompt template to use.

        """
        self._prompt = prompt

    def __call__(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Execute the program to generate structured output.

        Formats the prompt with provided kwargs, invokes the LLM with the function
        calling tool, and parses the response into structured Pydantic model(s).

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]], optional): Additional keyword
                arguments to pass to the LLM (e.g., temperature, max_tokens).
                Defaults to None.
            **kwargs (Any): Keyword arguments used to format the prompt template.
                These should match the variables in the prompt template.

        Returns:
            BaseModel: A single Pydantic model instance if allow_parallel_tool_calls
                is False, or a list of Pydantic model instances if it's True.


        See Also:
            - acall: Async version of this method
            - stream_call: Streaming version for incremental results
        """
        llm_kwargs = llm_kwargs or {}
        tool = CallableTool.from_model(self._output_cls)
        # convert the prompt into messages
        messages = self.prompt.format_messages(llm=self._llm, **kwargs)
        messages = self._llm._extend_messages(messages)

        agent_response = self._llm.predict_and_call(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return _parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )  # type: ignore

    async def acall(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Asynchronously execute the program to generate structured output.

        Async version of __call__. Formats the prompt with provided kwargs,
        asynchronously invokes the LLM with the function calling tool, and parses
        the response into structured Pydantic model(s).

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]], optional): Additional keyword
                arguments to pass to the LLM (e.g., temperature, max_tokens).
                Defaults to None.
            **kwargs (Any): Keyword arguments used to format the prompt template.
                These should match the variables in the prompt template.

        Returns:
            BaseModel: A single Pydantic model instance if allow_parallel_tool_calls
                is False, or a list of Pydantic model instances if it's True.


        See Also:
            - __call__: Synchronous version of this method
            - astream_call: Async streaming version for incremental results
        """
        llm_kwargs = llm_kwargs or {}
        tool = CallableTool.from_model(self._output_cls)

        agent_response = await self._llm.apredict_and_call(
            [tool],
            chat_history=self._prompt.format_messages(llm=self._llm, **kwargs),
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return _parse_tool_outputs(
            agent_response,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
        )  # type: ignore

    def stream_call(  # type: ignore
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream structured output generation with incremental updates.

        Returns a generator that yields progressively refined structured objects as the
        LLM generates its response. Each yield provides a partial or complete instance
        of the output model, allowing for real-time updates and progressive rendering.

        Args:
            *args (Any):
                Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]], optional):
                Additional keyword arguments to pass to the LLM (e.g., temperature, max_tokens). Defaults to None.
            **kwargs (Any):
                Keyword arguments used to format the prompt template.

        Yields:
            Union[Model, List[Model]]:
                Progressive updates of the structured output. Each yielded value is a Pydantic model instance (or
                list of instances if allow_parallel_tool_calls is True) with incrementally more complete data.

        Raises:
            ValueError: If the LLM is not a FunctionCallingLLM instance.

        Warns:
            Logs warnings when parsing streaming responses fails, then continues with
            the next chunk.


        See Also:
            - __call__: Non-streaming synchronous version
            - astream_call: Async streaming version
            - _process_objects: Internal method for processing stream chunks
        """
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        llm_kwargs = llm_kwargs or {}
        tool = CallableTool.from_model(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
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
                objects = process_streaming_objects(
                    partial_resp,
                    self._output_cls,
                    cur_objects=cur_objects,
                    allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                    flexible_mode=True,
                    llm=self._llm,
                )
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects  # type: ignore
            except Exception as e:
                _logger.warning(f"Failed to parse streaming response: {e}")
                continue

    async def astream_call(  # type: ignore
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> AsyncGenerator[Union[Model, List[Model]], None]:
        """Asynchronously stream structured output generation with incremental updates.

        Async version of stream_call. Returns an async generator that yields progressively
        refined structured objects as the LLM generates its response. Enables concurrent
        streaming operations and integration with async frameworks.

        Args:
            *args (Any): Positional arguments (currently unused).
            llm_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments
                to pass to the LLM (e.g., temperature, max_tokens). Defaults to None.
            **kwargs (Any): Keyword arguments used to format the prompt template.

        Yields:
            Union[Model, List[Model]]: Progressive updates of the structured output.
                Each yielded value is a Pydantic model instance (or list of instances if
                allow_parallel_tool_calls is True) with incrementally more complete data.

        Raises:
            ValueError: If the LLM is not a FunctionCallingLLM instance.

        Warns:
            Logs warnings when parsing streaming responses fails, then continues with
            the next chunk.


        See Also:
            - acall: Non-streaming async version
            - stream_call: Synchronous streaming version
            - _process_objects: Internal method for processing stream chunks
        """
        if not isinstance(self._llm, FunctionCallingLLM):
            raise ValueError("stream_call is only supported for LLMs.")

        tool = CallableTool.from_model(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
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
                    objects = process_streaming_objects(
                        partial_resp,
                        self._output_cls,
                        cur_objects=cur_objects,
                        allow_parallel_tool_calls=self._allow_parallel_tool_calls,
                        flexible_mode=True,
                        llm=self._llm,
                    )
                    cur_objects = objects if isinstance(objects, list) else [objects]
                    yield objects  # type: ignore
                except Exception as e:
                    _logger.warning(f"Failed to parse streaming response: {e}")
                    continue

        return gen()
