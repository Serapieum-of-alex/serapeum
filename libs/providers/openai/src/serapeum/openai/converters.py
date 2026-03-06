"""Bidirectional converters between serapeum Message types and OpenAI API formats."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Sequence, Type

from deprecated import deprecated
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
from pydantic import BaseModel

from serapeum.core.llms import (
    Audio,
    ContentBlock,
    DocumentBlock,
    Image,
    LogProb,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
)
from serapeum.openai.models import O1_MODELS, O1_MODELS_WITHOUT_FUNCTION_CALLING

logger = logging.getLogger(__name__)

OpenAIToolCall = ChatCompletionMessageToolCall | ChoiceDeltaToolCall

# ---------------------------------------------------------------------------
# Shared block-level helpers
# ---------------------------------------------------------------------------



def _rewrite_system_to_developer(
    message_dict: dict[str, Any], model: str | None
) -> None:
    """Rewrite ``system`` role to ``developer`` for O1-family models."""
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
        and message_dict.get("role") == "system"
    ):
        message_dict["role"] = "developer"


def _strip_none_keys(message_dict: dict[str, Any], drop_none: bool) -> None:
    """Remove keys whose value is *None* when *drop_none* is set."""
    if drop_none:
        for key in [k for k, v in message_dict.items() if v is None]:
            message_dict.pop(key)


def _should_null_content(message: Message, has_tool_calls: bool) -> bool:
    """Return True when assistant content should be sent as ``None``.

    OpenAI requires ``content: null`` for assistant messages that only
    contain tool calls or function calls.
    """
    return (
        message.role == MessageRole.ASSISTANT
        and (
            "function_call" in message.additional_kwargs
            or "tool_calls" in message.additional_kwargs
            or has_tool_calls
        )
    )


class ChatFormat:
    """Convert serapeum content blocks to Chat Completions API dicts."""

    @staticmethod
    def text(block: TextChunk) -> dict[str, Any]:
        return {"type": "text", "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        b64_string, mimetype = block.as_base64()
        return {
            "type": "file",
            "filename": block.title,
            "file_data": f"data:{mimetype};base64,{b64_string}",
        }

    @staticmethod
    def image(block: Image) -> dict[str, Any]:
        if block.url:
            image_url: dict[str, Any] = {"url": str(block.url)}
        else:
            image_url = {"url": block.as_data_uri()}
        if block.detail:
            image_url["detail"] = block.detail
        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def audio(block: Audio) -> dict[str, Any]:
        audio_bytes = block.resolve_audio(as_base64=True).read()
        audio_str = audio_bytes.decode("utf-8")
        return {
            "type": "input_audio",
            "input_audio": {"data": audio_str, "format": block.format},
        }

    @staticmethod
    def tool_call(block: ToolCallBlock) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": block.tool_name,
                "arguments": block.tool_kwargs,
            },
            "id": block.tool_call_id,
        }

    # noinspection PyUnresolvedReferences
    content_converters: dict[type, Callable[..., dict[str, Any]]] = {
        TextChunk: text.__func__,
        DocumentBlock: document.__func__,
        Image: image.__func__,
        Audio: audio.__func__,
    }


class ResponsesFormat:
    """Convert serapeum content blocks to Responses API dicts."""

    @staticmethod
    def text(block: TextChunk, role: str) -> dict[str, Any]:
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        b64_string, mimetype = block.as_base64()
        return {
            "type": "input_file",
            "filename": block.title,
            "file_data": f"data:{mimetype};base64,{b64_string}",
        }

    @staticmethod
    def image(block: Image) -> dict[str, Any]:
        if block.url:
            url_str = str(block.url)
        else:
            url_str = block.as_data_uri()
        return {
            "type": "input_image",
            "image_url": url_str,
            "detail": block.detail or "auto",
        }

    @staticmethod
    def thinking(block: ThinkingBlock) -> dict[str, Any] | None:
        if block.content and "id" in block.additional_information:
            return {
                "type": "reasoning",
                "id": block.additional_information["id"],
                "summary": [
                    {"type": "summary_text", "text": block.content or ""}
                ],
            }
        return None

    @staticmethod
    def tool_call(block: ToolCallBlock) -> dict[str, Any]:
        return {
            "type": "function_call",
            "arguments": block.tool_kwargs,
            "call_id": block.tool_call_id,
            "name": block.tool_name,
        }

    # noinspection PyUnresolvedReferences
    content_converters: dict[type, Callable[..., dict[str, Any]]] = {
        DocumentBlock: document.__func__,
        Image: image.__func__,
    }

# ---------------------------------------------------------------------------
# Public converters (to OpenAI)
# ---------------------------------------------------------------------------


class ChatMessageConverter:
    """Converts a serapeum ``Message`` into a Chat Completions API dict.

    Usage::

        result = ChatMessageConverter(message, model="gpt-4o").build()
    """

    def __init__(
        self,
        message: Message,
        *,
        drop_none: bool = False,
        model: str | None = None,
    ) -> None:
        self._message = message
        self._model = model
        self._drop_none = drop_none
        self._content: list[dict[str, Any]] = []
        self._content_txt: str = ""
        self._tool_call_dicts: list[dict[str, Any]] = []

    def build(self) -> ChatCompletionMessageParam:
        """Convert the message and return the Chat Completions API dict."""
        audio_ref = self._try_audio_reference()
        if audio_ref is not None:
            result = audio_ref
        else:
            self._process_blocks()
            result = self._assemble()
            self._merge_legacy_kwargs(result)
        _rewrite_system_to_developer(result, self._model)
        _strip_none_keys(result, self._drop_none)
        return result  # type: ignore[return-value]

    def _try_audio_reference(self) -> dict[str, Any] | None:
        """Return a short-circuit dict for assistant messages with a reference audio id."""
        reference_audio_id = (
            self._message.additional_kwargs.get("reference_audio_id")
            if self._message.role == MessageRole.ASSISTANT
            else None
        )
        result = None
        if reference_audio_id:
            result = {
                "role": self._message.role.value,
                "audio": {"id": reference_audio_id},
            }
        return result

    def _process_blocks(self) -> None:
        """Iterate message chunks and dispatch to ``ChatFormat`` converters."""
        for block in self._message.chunks:
            if isinstance(block, TextChunk):
                self._content.append(ChatFormat.text(block))
                self._content_txt += block.content
            elif isinstance(block, ToolCallBlock):
                try:
                    self._tool_call_dicts.append(ChatFormat.tool_call(block))
                except Exception:
                    logger.warning(
                        f"It was not possible to convert ToolCallBlock with call id "
                        f"{block.tool_call_id or '`no call id`'} to a valid message, skipping..."
                    )
            else:
                converter = ChatFormat.content_converters.get(type(block))
                
                if converter:
                    self._content.append(converter(block))
                else:
                    raise ValueError(
                        f"Unsupported content block type: {type(block).__name__}"
                    )

    def _resolve_content(self) -> str | list[dict[str, Any]] | None:
        """Determine the final content value (string, list, or ``None``)."""
        has_tool_calls = len(self._tool_call_dicts) > 0
        content: str | list[dict[str, Any]] | None = self._content_txt

        if self._content_txt == "" and _should_null_content(self._message, has_tool_calls):
            content = None
        elif (
            self._message.role.value not in ("assistant", "tool", "system")
            and not all(isinstance(b, TextChunk) for b in self._message.chunks)
        ):
            content = self._content

        return content

    def _assemble(self) -> dict[str, Any]:
        """Construct the base message dict with role, content, and optional tool calls."""
        result: dict[str, Any] = {
            "role": self._message.role.value,
            "content": self._resolve_content(),
        }
        if self._tool_call_dicts:
            result["tool_calls"] = self._tool_call_dicts
        return result

    def _merge_legacy_kwargs(self, result: dict[str, Any]) -> None:
        """Merge ``tool_calls``/``function_call`` from ``additional_kwargs`` when no
        ``ToolCallBlock`` chunks exist, and pass through ``tool_call_id``."""
        has_tool_calls = len(self._tool_call_dicts) > 0
        if (
            "tool_calls" in self._message.additional_kwargs
            or "function_call" in self._message.additional_kwargs
        ) and not has_tool_calls:
            result.update(self._message.additional_kwargs)

        if "tool_call_id" in self._message.additional_kwargs:
            result["tool_call_id"] = self._message.additional_kwargs["tool_call_id"]


def to_openai_responses_message_dict(
    message: Message,
    drop_none: bool = False,
    model: str | None = None,
) -> str | dict[str, Any] | list[dict[str, Any]]:
    """Convert a single serapeum Message to an OpenAI Responses API dict."""
    content: list[dict[str, Any]] = []
    content_txt = ""
    tool_calls: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []

    for block in message.chunks:
        if isinstance(block, TextChunk):
            content.append(ResponsesFormat.text(block, message.role.value))
            content_txt += block.content
        elif isinstance(block, ThinkingBlock):
            item = ResponsesFormat.thinking(block)
            if item is not None:
                reasoning.append(item)
        elif isinstance(block, ToolCallBlock):
            tool_calls.append(ResponsesFormat.tool_call(block))
        else:
            converter = ResponsesFormat.content_converters.get(type(block))
            if converter:
                content.append(converter(block))
            else:
                raise ValueError(
                    f"Unsupported content block type: {type(block).__name__}"
                )

    # Legacy additional_kwargs tool calls take precedence over ToolCallBlock chunks
    if "tool_calls" in message.additional_kwargs:
        legacy_calls = [
            tc if isinstance(tc, dict) else tc.model_dump()
            for tc in message.additional_kwargs["tool_calls"]
        ]
        result: str | dict[str, Any] | list[dict[str, Any]] = [
            *reasoning, *legacy_calls
        ]
    elif tool_calls:
        result = [*reasoning, *tool_calls]
    elif message.role.value == "tool":
        # Tool output message
        call_id = message.additional_kwargs.get(
            "tool_call_id", message.additional_kwargs.get("call_id")
        )
        if call_id is None:
            raise ValueError(
                "tool_call_id or call_id is required in additional_kwargs for tool messages"
            )
        result = {
            "type": "function_call_output",
            "output": content_txt,
            "call_id": call_id,
        }
    elif (
        content_txt
        and all(item["type"] == "input_text" for item in content)
        and message.role.value == "user"
    ):
        # Plain text user message — some features (image generation, MCP)
        # require a bare string input.
        result = content_txt
    else:
        # Null-out content for assistant-only tool/function call messages
        final_content: str | list[dict[str, Any]] | None = content_txt
        if content_txt == "" and _should_null_content(message, False):
            final_content = None

        # Assistant, system, and developer roles require string content
        if (
            final_content is not None
            and message.role.value in ("system", "developer")
            or all(isinstance(block, TextChunk) for block in message.chunks)
        ):
            pass  # final_content is already the string form
        else:
            final_content = content

        message_dict: dict[str, Any] = {
            "role": message.role.value,
            "content": final_content,
        }
        _rewrite_system_to_developer(message_dict, model)
        _strip_none_keys(message_dict, drop_none)

        result = [*reasoning, message_dict] if reasoning else message_dict

    return result


def to_openai_message_dicts(
    messages: Sequence[Message],
    drop_none: bool = False,
    model: str | None = None,
    is_responses_api: bool = False,
) -> list[ChatCompletionMessageParam] | str:
    """Convert generic messages to OpenAI message dicts."""
    if is_responses_api:
        final_message_dicts = []
        for message in messages:
            message_dicts = to_openai_responses_message_dict(
                message,
                drop_none=drop_none,
                model="o3-mini",  # hardcode to ensure developer messages are used
            )
            if isinstance(message_dicts, list):
                final_message_dicts.extend(message_dicts)
            elif isinstance(message_dicts, str):
                final_message_dicts.append({"role": "user", "content": message_dicts})
            else:
                final_message_dicts.append(message_dicts)

        # If there is only one message, and it is a user message, return the content string directly
        if (
            len(final_message_dicts) == 1
            and final_message_dicts[0]["role"] == "user"
            and isinstance(final_message_dicts[0]["content"], str)
        ):
            return final_message_dicts[0]["content"]

        return final_message_dicts
    else:
        return [
            ChatMessageConverter(message, drop_none=drop_none, model=model).build()
            for message in messages
        ]


def from_openai_message(
    openai_message: ChatCompletionMessage, modalities: list[str]
) -> Message:
    """Convert openai message dict to generic message."""
    role = openai_message.role
    # NOTE: Azure OpenAI returns function calling messages without a content key
    if "text" in modalities and openai_message.content:
        blocks: list[ContentBlock] = [TextChunk(content=openai_message.content or "")]
    else:
        blocks: list[ContentBlock] = []

    additional_kwargs: dict[str, Any] = {}
    if openai_message.tool_calls:
        tool_calls: list[ChatCompletionMessageToolCall] = openai_message.tool_calls
        for tool_call in tool_calls:
            if tool_call.function:
                blocks.append(
                    ToolCallBlock(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name or "",
                        tool_kwargs=tool_call.function.arguments or {},
                    )
                )
        additional_kwargs.update(tool_calls=tool_calls)

    if openai_message.audio and "audio" in modalities:
        reference_audio_id = openai_message.audio.id
        audio_data = openai_message.audio.data
        additional_kwargs["reference_audio_id"] = reference_audio_id
        blocks.append(Audio(content=audio_data, format="mp3"))

    return Message(role=role, chunks=blocks, additional_kwargs=additional_kwargs)


def from_openai_token_logprob(
    openai_token_logprob: ChatCompletionTokenLogprob,
) -> list[LogProb]:
    """Convert a single openai token logprob to generic list of logprobs."""
    result = []
    if openai_token_logprob.top_logprobs:
        try:
            result = [
                LogProb(token=el.token, logprob=el.logprob, bytes=el.bytes or [])
                for el in openai_token_logprob.top_logprobs
            ]
        except Exception:
            print(openai_token_logprob)
            raise
    return result


def from_openai_token_logprobs(
    openai_token_logprobs: Sequence[ChatCompletionTokenLogprob],
) -> list[list[LogProb]]:
    """Convert openai token logprobs to generic list of LogProb."""
    result = []
    for token_logprob in openai_token_logprobs:
        if logprobs := from_openai_token_logprob(token_logprob):
            result.append(logprobs)
    return result


def from_openai_completion_logprob(
    openai_completion_logprob: dict[str, float],
) -> list[LogProb]:
    """Convert openai completion logprobs to generic list of LogProb."""
    return [
        LogProb(token=t, logprob=v, bytes=[])
        for t, v in openai_completion_logprob.items()
    ]


def from_openai_completion_logprobs(
    openai_completion_logprobs: Logprobs,
) -> list[list[LogProb]]:
    """Convert openai completion logprobs to generic list of LogProb."""
    result = []
    if openai_completion_logprobs.top_logprobs:
        result = [
            from_openai_completion_logprob(completion_logprob)
            for completion_logprob in openai_completion_logprobs.top_logprobs
        ]
    return result


def from_openai_messages(
    openai_messages: Sequence[ChatCompletionMessage], modalities: list[str]
) -> list[Message]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message(message, modalities) for message in openai_messages]


def from_openai_message_dict(message_dict: dict) -> Message:
    """Convert openai message dict to generic message."""
    role = message_dict["role"]
    # NOTE: Azure OpenAI returns function calling messages without a content key
    content = message_dict.get("content")
    blocks = []
    if isinstance(content, list):
        for elem in content:
            t = elem.get("type")
            if t == "text":
                blocks.append(TextChunk(content=elem.get("text")))
            elif t == "image_url":
                img = elem["image_url"]["url"]
                detail = elem["image_url"]["detail"]
                if img.startswith("data:"):
                    blocks.append(Image(content=img, detail=detail))
                else:
                    blocks.append(Image(url=img, detail=detail))
            elif t == "function_call":
                blocks.append(
                    ToolCallBlock(
                        tool_call_id=elem.get("call_id"),
                        tool_name=elem.get("name", ""),
                        tool_kwargs=elem.get("arguments", {}),
                    )
                )
            else:
                msg = f"Unsupported message type: {t}"
                raise ValueError(msg)
        content = None

    additional_kwargs = message_dict.copy()
    additional_kwargs.pop("role")
    additional_kwargs.pop("content", None)

    return Message(
        role=role, content=content, additional_kwargs=additional_kwargs, chunks=blocks
    )


def from_openai_message_dicts(message_dicts: Sequence[dict]) -> list[Message]:
    """Convert openai message dicts to generic messages."""
    return [from_openai_message_dict(message_dict) for message_dict in message_dicts]


@deprecated("Deprecated in favor of `to_openai_tool`, which should be used instead.")
def to_openai_function(pydantic_class: Type[BaseModel]) -> dict[str, Any]:
    """Deprecated in favor of `to_openai_tool`.

    Convert pydantic class to OpenAI function.
    """
    return to_openai_tool(pydantic_class, description=None)


def to_openai_tool(
    pydantic_class: Type[BaseModel], description: str | None = None
) -> dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.model_json_schema()
    schema_description = schema.get("description", None) or description
    function = {
        "name": schema["title"],
        "description": schema_description,
        "parameters": pydantic_class.model_json_schema(),
    }
    return {"type": "function", "function": function}


def update_tool_calls(
    tool_calls: list[ChoiceDeltaToolCall],
    tool_calls_delta: list[ChoiceDeltaToolCall] | None,
) -> list[ChoiceDeltaToolCall]:
    """Use the tool_calls_delta objects received from openai stream chunks to update the running tool_calls object.

    Args:
        tool_calls: the list of tool calls
        tool_calls_delta: the delta to update tool_calls

    Returns:
        The updated tool calls
    """
    # openai provides chunks consisting of tool_call deltas one tool at a time
    if tool_calls_delta is None or len(tool_calls_delta) == 0:
        return tool_calls

    tc_delta = tool_calls_delta[0]

    if len(tool_calls) == 0:
        tool_calls.append(tc_delta)
    else:
        # we need to either update latest tool_call or start a
        # new tool_call (i.e., multiple tools in this turn) and
        # accumulate that new tool_call with future delta chunks
        t = tool_calls[-1]
        if t.index != tc_delta.index:
            # the start of a new tool call, so append to our running tool_calls list
            tool_calls.append(tc_delta)
        else:
            # not the start of a new tool call, so update last item of tool_calls

            # validations to get passed by mypy
            assert t.function is not None
            assert tc_delta.function is not None

            # Initialize fields if they're None
            # OpenAI(or Compatible)'s streaming API can return partial tool call
            # information across multiple chunks where some fields may be None in
            # initial chunks and populated in subsequent ones
            if t.function.arguments is None:
                t.function.arguments = ""
            if t.function.name is None:
                t.function.name = ""
            if t.id is None:
                t.id = ""

            # Update with delta values
            t.function.arguments += tc_delta.function.arguments or ""
            t.function.name += tc_delta.function.name or ""
            t.id += tc_delta.id or ""
    return tool_calls
