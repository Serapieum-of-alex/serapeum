"""Bidirectional converters between serapeum Message types and OpenAI API formats."""

from __future__ import annotations

import logging
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


def to_openai_message_dict(
    message: Message,
    drop_none: bool = False,
    model: str | None = None,
) -> ChatCompletionMessageParam:
    """Convert a Message to an OpenAI message dict."""
    content = []
    content_txt = ""
    reference_audio_id = None
    for block in message.chunks:
        if message.role == MessageRole.ASSISTANT:
            reference_audio_id = message.additional_kwargs.get(
                "reference_audio_id", None
            )
            # if reference audio id is provided, we don't need to send the audio
            if reference_audio_id:
                continue

        if isinstance(block, TextChunk):
            content.append({"type": "text", "text": block.content})
            content_txt += block.content
        elif isinstance(block, DocumentBlock):
            if not block.data:
                file_buffer = block.resolve_document()
                b64_string = block._get_b64_string(file_buffer)
                mimetype = block._guess_mimetype()
            else:
                b64_string = block.data.decode("utf-8")
                mimetype = block._guess_mimetype()
            content.append(
                {
                    "type": "file",
                    "filename": block.title,
                    "file_data": f"data:{mimetype};base64,{b64_string}",
                }
            )
        elif isinstance(block, Image):
            if block.url:
                image_url = {"url": str(block.url)}
                if block.detail:
                    image_url["detail"] = block.detail

                content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    }
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                image_url = f"data:{block.image_mimetype};base64,{img_str}"

                image_url_dict = {"url": image_url}
                if block.detail:
                    image_url_dict["detail"] = block.detail

                content.append(
                    {
                        "type": "image_url",
                        "image_url": image_url_dict,
                    }
                )
        elif isinstance(block, Audio):
            audio_bytes = block.resolve_audio(as_base64=True).read()
            audio_str = audio_bytes.decode("utf-8")
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_str,
                        "format": block.format,
                    },
                }
            )
        elif isinstance(block, ToolCallBlock):
            try:
                function_dict = {
                    "type": "function",
                    "function": {
                        "name": block.tool_name,
                        "arguments": block.tool_kwargs,
                    },
                    "id": block.tool_call_id,
                }

                if len(content) == 0 or content[-1]["type"] != "text":
                    content.append(
                        {"type": "text", "text": "", "tool_calls": [function_dict]}
                    )
                elif content[-1]["type"] == "text" and "tool_calls" in content[-1]:
                    content[-1]["tool_calls"].append(function_dict)
                elif content[-1]["type"] == "text" and "tool_calls" not in content[-1]:
                    content[-1]["tool_calls"] = [function_dict]
            except Exception:
                logger.warning(
                    f"It was not possible to convert ToolCallBlock with call id {block.tool_call_id or '`no call id`'} to a valid message, skipping..."
                )
                continue
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    # NOTE: Sending a null value (None) for Tool Message to OpenAI will cause error
    # It's only Allowed to send None if it's an Assistant Message and either a function call or tool calls were performed
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    already_has_tool_calls = any(
        isinstance(block, ToolCallBlock) for block in message.chunks
    )
    content_txt = (
        None
        if content_txt == ""
        and message.role == MessageRole.ASSISTANT
        and (
            "function_call" in message.additional_kwargs
            or "tool_calls" in message.additional_kwargs
            or already_has_tool_calls
        )
        else content_txt
    )

    # If reference audio id is provided, we don't need to send the audio
    # NOTE: this is only a thing for assistant messages
    if reference_audio_id:
        message_dict = {
            "role": message.role.value,
            "audio": {"id": reference_audio_id},
        }
    else:
        # NOTE: Despite what the openai docs say, if the role is ASSISTANT, SYSTEM
        # or TOOL, 'content' cannot be a list and must be string instead.
        # Furthermore, if all blocks are text blocks, we can use the content_txt
        # as the content. This will avoid breaking openai-like APIs.
        message_dict = {
            "role": message.role.value,
            "content": (
                content_txt
                if message.role.value in ("assistant", "tool", "system")
                or all(isinstance(block, TextChunk) for block in message.chunks)
                else content
            ),
        }
        if already_has_tool_calls:
            existing_tool_calls = []
            for c in content:
                existing_tool_calls.extend(c.get("tool_calls", []))

            if existing_tool_calls:
                message_dict["tool_calls"] = existing_tool_calls

    # TODO: O1 models do not support system prompts
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
    ):
        if message_dict["role"] == "system":
            message_dict["role"] = "developer"

    if (
        "tool_calls" in message.additional_kwargs
        or "function_call" in message.additional_kwargs
    ) and not already_has_tool_calls:
        message_dict.update(message.additional_kwargs)

    if "tool_call_id" in message.additional_kwargs:
        message_dict["tool_call_id"] = message.additional_kwargs["tool_call_id"]

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    return message_dict  # type: ignore


def to_openai_responses_message_dict(
    message: Message,
    drop_none: bool = False,
    model: str | None = None,
) -> str | dict[str, Any] | list[dict[str, Any]]:
    """Convert a Message to an OpenAI message dict."""
    content = []
    content_txt = ""
    tool_calls = []
    reasoning = []

    for block in message.chunks:
        if isinstance(block, TextChunk):
            if message.role.value == "user":
                content.append({"type": "input_text", "text": block.content})
            else:
                content.append({"type": "output_text", "text": block.content})
            content_txt += block.content
        elif isinstance(block, DocumentBlock):
            if not block.data:
                file_buffer = block.resolve_document()
                b64_string = block._get_b64_string(file_buffer)
                mimetype = block._guess_mimetype()
            else:
                b64_string = block.data.decode("utf-8")
                mimetype = block._guess_mimetype()
            content.append(
                {
                    "type": "input_file",
                    "filename": block.title,
                    "file_data": f"data:{mimetype};base64,{b64_string}",
                }
            )
        elif isinstance(block, Image):
            if block.url:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": str(block.url),
                        "detail": block.detail or "auto",
                    }
                )
            else:
                img_bytes = block.resolve_image(as_base64=True).read()
                img_str = img_bytes.decode("utf-8")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{block.image_mimetype};base64,{img_str}",
                        "detail": block.detail or "auto",
                    }
                )
        elif isinstance(block, ThinkingBlock):
            if block.content and "id" in block.additional_information:
                reasoning.append(
                    {
                        "type": "reasoning",
                        "id": block.additional_information["id"],
                        "summary": [
                            {"type": "summary_text", "text": block.content or ""}
                        ],
                    }
                )
        elif isinstance(block, ToolCallBlock):
            tool_calls.extend(
                [
                    {
                        "type": "function_call",
                        "arguments": block.tool_kwargs,
                        "call_id": block.tool_call_id,
                        "name": block.tool_name,
                    }
                ]
            )
        else:
            msg = f"Unsupported content block type: {type(block).__name__}"
            raise ValueError(msg)

    if "tool_calls" in message.additional_kwargs:
        message_dicts = [
            tool_call if isinstance(tool_call, dict) else tool_call.model_dump()
            for tool_call in message.additional_kwargs["tool_calls"]
        ]

        return [*reasoning, *message_dicts]
    elif tool_calls:
        return [*reasoning, *tool_calls]

    # NOTE: Sending a null value (None) for Tool Message to OpenAI will cause error
    # It's only Allowed to send None if it's an Assistant Message and either a function call or tool calls were performed
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    content_txt = (
        None
        if content_txt == ""
        and message.role == MessageRole.ASSISTANT
        and (
            "function_call" in message.additional_kwargs
            or "tool_calls" in message.additional_kwargs
        )
        else content_txt
    )

    # NOTE: Despite what the openai docs say, if the role is ASSISTANT, SYSTEM
    # or TOOL, 'content' cannot be a list and must be string instead.
    # Furthermore, if all blocks are text blocks, we can use the content_txt
    # as the content. This will avoid breaking openai-like APIs.
    if message.role.value == "tool":
        call_id = message.additional_kwargs.get(
            "tool_call_id", message.additional_kwargs.get("call_id")
        )
        if call_id is None:
            raise ValueError(
                "tool_call_id or call_id is required in additional_kwargs for tool messages"
            )

        message_dict = {
            "type": "function_call_output",
            "output": content_txt,
            "call_id": call_id,
        }

        return message_dict

    # there are some cases (like image generation or MCP tool call) that only support the string input
    # this is why, if context_txt is a non-empty string, all the blocks are TextBlocks and the role is user, we return directly context_txt
    elif (
        isinstance(content_txt, str)
        and len(content_txt) != 0
        and all(item["type"] == "input_text" for item in content)
        and message.role.value == "user"
    ):
        return content_txt
    else:
        message_dict = {
            "role": message.role.value,
            "content": (
                content_txt
                if message.role.value in ("system", "developer")
                or all(isinstance(block, TextChunk) for block in message.chunks)
                else content
            ),
        }

    # TODO: O1 models do not support system prompts
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
    ):
        if message_dict["role"] == "system":
            message_dict["role"] = "developer"

    null_keys = [key for key, value in message_dict.items() if value is None]
    # if drop_none is True, remove keys with None values
    if drop_none:
        for key in null_keys:
            message_dict.pop(key)

    if reasoning:
        return [*reasoning, message_dict]

    return message_dict  # type: ignore


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
            to_openai_message_dict(
                message,
                drop_none=drop_none,
                model=model,
            )
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
        role=role, content=content, additional_kwargs=additional_kwargs, blocks=blocks
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
