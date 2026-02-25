"""LLM mixins and helpers for function/tool calling workflows."""

from __future__ import annotations
import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Sequence, overload

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    Message,
)
from serapeum.core.llms.base import LLM
from serapeum.core.tools.types import ToolCallArguments
from serapeum.core.tools.invoke import ExecutionConfig, ToolExecutor
from serapeum.core.chat import AgentChatResponse

if TYPE_CHECKING:
    from serapeum.core.tools.types import BaseTool


class FunctionCallingLLM(LLM):
    """LLM base with convenience helpers for tool/function calling flows."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the function-calling LLM wrapper.

        Simply forwards all arguments to the base LLM constructor.
        """
        super().__init__(*args, **kwargs)

    @overload
    def chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    def chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseGen: ...

    def chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = None,
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
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        if stream:
            return self.chat(stream=True, **chat_kwargs)
        response = self.chat(**chat_kwargs)
        return self._validate_chat_with_tools_response(
            response,
            tools,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    @overload
    async def achat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def achat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = ...,
        chat_history: list[Message] | None = ...,
        verbose: bool = ...,
        allow_parallel_tool_calls: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen: ...

    async def achat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = None,
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
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )
        if stream:
            return await self.achat(stream=True, **chat_kwargs)
        response = await self.achat(**chat_kwargs)
        return self._validate_chat_with_tools_response(
            response,
            tools,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            **kwargs,
        )

    @abstractmethod
    def _prepare_chat_with_tools(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare the arguments needed to let the LLM chat with tools."""

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence[BaseTool],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolCallArguments]:
        """Predict and call the tool."""
        raise NotImplementedError(
            "get_tool_calls_from_response is not supported by default."
        )

    def predict_and_call(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
        error_on_tool_error: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Predict and call the tool."""
        response = self.chat_with_tools(
            tools,
            user_msg=user_msg,
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

    async def apredict_and_call(
        self,
        tools: Sequence[BaseTool],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        error_on_no_tool_call: bool = True,
        error_on_tool_error: bool = False,
        **kwargs: Any,
    ) -> AgentChatResponse:
        """Predict and call the tool."""
        response = await self.achat_with_tools(
            tools,
            user_msg=user_msg,
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
        agent_response = self.parse_tool_outputs(
            tool_outputs, response, error_on_tool_error, allow_parallel_tool_calls
        )

        return agent_response

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
            agent_response = AgentChatResponse(
                response=output_text, sources=tool_outputs
            )
        elif len(tool_outputs) > 1:
            raise ValueError("Invalid")
        elif len(tool_outputs) == 0:
            agent_response = AgentChatResponse(
                response=response.message.content or "", sources=tool_outputs
            )
        else:
            agent_response = AgentChatResponse(
                response=tool_outputs[0].content, sources=tool_outputs
            )
        return agent_response
