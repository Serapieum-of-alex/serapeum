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
            >>> from serapeum.core.structured_tools.function_program import _parse_tool_outputs
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
    """Function Calling Program.

    Uses function calling LLMs to obtain a structured output.
    """

    def __init__(
        self,
        output_cls: Type[Model],
        llm: FunctionCallingLLM,
        prompt: BasePromptTemplate,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        allow_parallel_tool_calls: bool = False,
        verbose: bool = False,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._llm = llm
        self._prompt = prompt
        self._verbose = verbose
        self._allow_parallel_tool_calls = allow_parallel_tool_calls
        self._tool_choice = tool_choice

    @classmethod
    def from_defaults(
        cls,
        output_cls: Type[Model],
        prompt_template_str: Optional[str] = None,
        prompt: Optional[BasePromptTemplate] = None,
        llm: Optional[LLM] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> "ToolOrchestratingLLM":
        llm = llm or Configs.llm  # type: ignore
        assert llm is not None

        if not llm.metadata.is_function_calling_model:
            raise ValueError(
                f"Model name {llm.metadata.model_name} does not support "
                "function calling API. "
            )

        if prompt is None and prompt_template_str is None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt is not None and prompt_template_str is not None:
            raise ValueError("Must provide either prompt or prompt_template_str.")
        if prompt_template_str is not None:
            prompt = PromptTemplate(prompt_template_str)

        return cls(
            output_cls=output_cls,  # type: ignore
            llm=llm,  # type: ignore
            prompt=cast(PromptTemplate, prompt),
            tool_choice=tool_choice,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            verbose=verbose,
        )

    @property
    def output_cls(self) -> Type[BaseModel]:
        return self._output_cls

    @property
    def prompt(self) -> BasePromptTemplate:
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: BasePromptTemplate) -> None:
        self._prompt = prompt

    def __call__(
        self,
        *args: Any,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        tool = CallableTool.from_model(self._output_cls)

        messages = self._prompt.format_messages(llm=self._llm, **kwargs)
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

    def _process_objects(
        self,
        chat_response: ChatResponse,
        output_cls: Type[BaseModel],
        cur_objects: Optional[List[BaseModel]] = None,
    ) -> Union[Model, List[Model]]:
        """Process stream."""
        tool_calls = self._llm.get_tool_calls_from_response(
            chat_response,
            # error_on_no_tool_call=True
            error_on_no_tool_call=False,
        )
        # TODO: change
        if len(tool_calls) == 0:
            # if no tool calls, return single blank output_class
            return output_cls()  # type: ignore

        tool_fn_args = [call.tool_kwargs for call in tool_calls]
        objects = [
            output_cls.model_validate(tool_fn_arg) for tool_fn_arg in tool_fn_args
        ]

        if cur_objects is None or num_valid_fields(objects) > num_valid_fields(
            cur_objects
        ):
            cur_objects = objects

        # right now the objects are typed according to a flexible schema
        # try to do a pass to convert the objects to the output_cls
        new_cur_objects = []
        for obj in cur_objects:
            try:
                new_obj = self._output_cls.model_validate(obj.model_dump())
            except ValidationError as e:
                _logger.warning(f"Failed to parse object: {e}")
                new_obj = obj  # type: ignore
            new_cur_objects.append(new_obj)

        if self._allow_parallel_tool_calls:
            return new_cur_objects  # type: ignore
        else:
            if len(new_cur_objects) > 1:
                _logger.warning(
                    "Multiple outputs found, returning first one. "
                    "If you want to return all outputs, set output_multiple=True."
                )
            return new_cur_objects[0]  # type: ignore

    def stream_call(  # type: ignore
        self, *args: Any, llm_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Generator[Union[Model, List[Model]], None, None]:
        """Stream object.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
        """
        # TODO: we can extend this to non-function calling LLMs as well, coming soon
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
        """Stream objects.

        Returns a generator returning partials of the same object
        or a list of objects until it returns.
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
