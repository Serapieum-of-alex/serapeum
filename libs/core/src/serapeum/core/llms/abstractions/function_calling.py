"""LLM adapters and helpers for function/tool calling workflows."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Sequence, overload

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    Message,
    ToolCallArguments,
)
from serapeum.core.chat import AgentChatResponse
from serapeum.core.llms.base import LLM
from serapeum.core.tools.invoke import ExecutionConfig, ToolExecutor

if TYPE_CHECKING:
    from serapeum.core.tools.types import BaseTool


class FunctionCallingLLM(LLM, ABC):
    """LLM base with convenience helpers for tool/function calling flows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the function-calling LLM wrapper.

        Simply forwards all arguments to the base LLM constructor.
        """
        super().__init__(*args, **kwargs)

    @overload
    def generate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    def generate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseGen: ...

    def generate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseGen:
        """Chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            message=message,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        if stream:
            result = self.chat(stream=True, **chat_kwargs)
        else:
            response = self.chat(**chat_kwargs)
            result = self._validate_response(
                response,
                allow_parallel_tool_calls=allow_parallel_tool_calls,
            )
        return result

    @overload
    async def agenerate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def agenerate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen: ...

    async def agenerate_tool_calls(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseAsyncGen:
        """Async chat with function calling."""
        chat_kwargs = self._prepare_chat_with_tools(
            tools,
            message=message,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        if stream:
            result = await self.achat(stream=True, **chat_kwargs)
        else:
            response = await self.achat(**chat_kwargs)
            result = self._validate_response(
                response,
                allow_parallel_tool_calls=allow_parallel_tool_calls,
            )
        return result

    @abstractmethod
    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""

    @staticmethod
    def _validate_response(
        response: ChatResponse,
        allow_parallel_tool_calls: bool = False,
    ) -> ChatResponse:
        """Validate and normalize a chat-with-tools response.

        If ``allow_parallel_tool_calls`` is ``False``, the response is mutated
        to include at most a single tool call.

        Args:
            response: Response to validate.
            allow_parallel_tool_calls: Whether multiple tool calls are allowed.

        Returns:
            The validated response (possibly mutated in-place).
        """
        if not allow_parallel_tool_calls:
            response.force_single_tool_call()
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolCallArguments]:
        """Extract parsed tool-call arguments from a chat response.

        Reads :class:`~serapeum.core.base.llms.types.ToolCallBlock` entries
        from ``response.message.tool_calls``.  For each block, ``tool_kwargs``
        is parsed from JSON when stored as a string, or used directly when
        already a dict.

        Subclasses may override to add provider-specific fallback paths
        (e.g. reading from ``additional_kwargs``).

        Args:
            response: A :class:`~serapeum.core.base.llms.types.ChatResponse`
                that may contain tool calls.
            error_on_no_tool_call: When ``True`` (default), raises
                :exc:`ValueError` if no tool calls are found.
            **kwargs: Accepted for interface compatibility.

        Returns:
            One entry per tool call with ``tool_id``, ``tool_name``, and
            parsed ``tool_kwargs``.

        Raises:
            ValueError: If *error_on_no_tool_call* is ``True`` and no tool
                calls are present.
        """
        tool_call_blocks = response.message.tool_calls

        if tool_call_blocks:
            result = [tool_call.get_arguments() for tool_call in tool_call_blocks]
        elif error_on_no_tool_call:
            raise ValueError("Expected at least one tool call, but got 0 tool calls.")
        else:
            result = []

        return result

    def invoke_callable(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
        error_on_tool_error: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Predict and call the tool."""
        response = self.generate_tool_calls(
            tools,
            message=message,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        tool_calls = self.get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call
        )
        tool_executor = ToolExecutor(ExecutionConfig(verbose=verbose))
        tool_outputs = [
            tool_executor.execute_with_selection(tool_call, tools)
            for tool_call in tool_calls
        ]

        return self.parse_tool_outputs(
            tool_outputs, response, error_on_tool_error, allow_parallel_tool_calls
        )

    async def ainvoke_callable(
        self,
        tools: Sequence[BaseTool],
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
        error_on_tool_error: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Predict and call the tool."""
        response = await self.agenerate_tool_calls(
            tools,
            message=message,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

        tool_calls = self.get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call
        )
        tool_executor = ToolExecutor(ExecutionConfig(verbose=verbose))
        tool_tasks = [
            tool_executor.execute_async_with_selection(tool_call, tools)
            for tool_call in tool_calls
        ]

        tool_outputs = await asyncio.gather(*tool_tasks)
        result = self.parse_tool_outputs(
            tool_outputs, response, error_on_tool_error, allow_parallel_tool_calls
        )

        return result

    @staticmethod
    def parse_tool_outputs(
        tool_outputs: list[Any],
        response: ChatResponse,
        error_on_tool_error: bool,
        allow_parallel_tool_calls: bool,
    ) -> AgentChatResponse:
        tool_outputs_with_error = [
            tool_output for tool_output in tool_outputs if tool_output.is_error
        ]

        if error_on_tool_error and len(tool_outputs_with_error) > 0:
            error_text = "\n\n".join(
                [tool_output.content for tool_output in tool_outputs]
            )
            raise ValueError(error_text)

        if allow_parallel_tool_calls:
            output_text = "\n\n".join(
                [tool_output.content for tool_output in tool_outputs]
            )
            result = AgentChatResponse(response=output_text, sources=tool_outputs)
        elif len(tool_outputs) > 1:
            raise ValueError("Invalid")
        elif len(tool_outputs) == 0:
            result = AgentChatResponse(
                response=response.message.content or "", sources=tool_outputs
            )
        else:
            result = AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )
        return result
