import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Type,
    Union,
    List,
    Generator,
    AsyncGenerator,
)

from pydantic import (
    BaseModel,
)
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.llm.base import LLM
from serapeum.core.prompts.base import BasePromptTemplate, PromptTemplate
from serapeum.core.configs.configs import Configs
from serapeum.core.structured_tools.models import BasePydanticProgram, Model
from serapeum.core.tools.callable_tool import CallableTool
from serapeum.core.structured_tools.utils import process_streaming_objects

if TYPE_CHECKING:
    from serapeum.core.chat.models import AgentChatResponse

_logger = logging.getLogger(__name__)


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
            raise ValueError("prompt must be an instance of BasePromptTemplate or str.")
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

        agent_response: AgentChatResponse = self._llm.predict_and_call(
            [tool],
            chat_history=messages,
            verbose=self._verbose,
            allow_parallel_tool_calls=self._allow_parallel_tool_calls,
            **llm_kwargs,
        )
        return agent_response.parse_tool_outputs(
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
        return agent_response.parse_tool_outputs(
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
