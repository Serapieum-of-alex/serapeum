"""Bidirectional converters between serapeum Message types and OpenAI API formats."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Sequence, Type, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
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
        return cast(ChatCompletionMessageParam, result)

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
            elif isinstance(block, ThinkingBlock):
                logger.debug(
                    "ThinkingBlock skipped in Chat Completions path (not supported)"
                )
            elif isinstance(block, ToolCallBlock):
                self._tool_call_dicts.append(ChatFormat.tool_call(block))
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


class ResponsesMessageConverter:
    """Converts a serapeum ``Message`` into a Responses API dict or list of dicts.

    Usage::

        result = ResponsesMessageConverter(message, model="gpt-4o").build()
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
        self._reasoning: list[dict[str, Any]] = []

    def build(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert the message and return the Responses API representation."""
        self._process_blocks()
        result = self._assemble()
        return result

    def _process_blocks(self) -> None:
        """Iterate message chunks and dispatch to ``ResponsesFormat`` converters."""
        for block in self._message.chunks:
            if isinstance(block, TextChunk):
                self._content.append(
                    ResponsesFormat.text(block, self._message.role.value)
                )
                self._content_txt += block.content
            elif isinstance(block, ThinkingBlock):
                item = ResponsesFormat.thinking(block)
                if item is not None:
                    self._reasoning.append(item)
            elif isinstance(block, ToolCallBlock):
                self._tool_call_dicts.append(ResponsesFormat.tool_call(block))
            elif isinstance(block, Audio):
                raise ValueError(
                    "Audio blocks are not supported in the Responses API"
                )
            else:
                converter = ResponsesFormat.content_converters.get(type(block))
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
            (content is not None and self._message.role.value in ("system", "developer"))
            or all(isinstance(block, TextChunk) for block in self._message.chunks)
        ):
            pass  # content is already the string form
        else:
            content = self._content
        return content

    def _assemble_tool_output(self) -> dict[str, Any]:
        """Construct a ``function_call_output`` dict for tool-role messages."""
        call_id = self._message.additional_kwargs.get(
            "tool_call_id", self._message.additional_kwargs.get("call_id")
        )
        if call_id is None:
            raise ValueError(
                "tool_call_id or call_id is required in additional_kwargs for tool messages"
            )
        return {
            "type": "function_call_output",
            "output": self._content_txt,
            "call_id": call_id,
        }

    def _assemble_message_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Construct a standard message dict, optionally prefixed by reasoning items."""
        message_dict: dict[str, Any] = {
            "role": self._message.role.value,
            "content": self._resolve_content(),
        }
        # Responses API always uses "developer" instead of "system"
        if message_dict.get("role") == "system":
            message_dict["role"] = "developer"
        _strip_none_keys(message_dict, self._drop_none)
        result: dict[str, Any] | list[dict[str, Any]] = (
            [*self._reasoning, message_dict] if self._reasoning else message_dict
        )
        return result

    def _assemble(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Select the appropriate output shape based on message role and content."""
        if self._tool_call_dicts:
            result: dict[str, Any] | list[dict[str, Any]] = [
                *self._reasoning, *self._tool_call_dicts
            ]
        elif "tool_calls" in self._message.additional_kwargs:
            legacy_calls = [
                tc if isinstance(tc, dict) else tc.model_dump()
                for tc in self._message.additional_kwargs["tool_calls"]
            ]
            result = [*self._reasoning, *legacy_calls]
        elif self._message.role.value == "tool":
            result = self._assemble_tool_output()
        else:
            result = self._assemble_message_dict()
        return result


def to_openai_message_dicts(
    messages: Sequence[Message],
    drop_none: bool = False,
    model: str | None = None,
    is_responses_api: bool = False,
) -> list[ChatCompletionMessageParam] | str:
    """Convert generic messages to OpenAI message dicts.

    Returns ``str`` for the Responses API when the input is a single plain-text
    user message (the OpenAI Responses API ``input`` parameter accepts ``str``
    directly for features like image generation and MCP).
    """
    if is_responses_api:
        final_message_dicts: list[dict[str, Any]] = []
        for message in messages:
            message_dicts = ResponsesMessageConverter(
                message,
                drop_none=drop_none,
                model=model,
            ).build()
            if isinstance(message_dicts, list):
                final_message_dicts.extend(message_dicts)
            else:
                final_message_dicts.append(message_dicts)

        # Single user message → return the content string directly
        if (
            len(final_message_dicts) == 1
            and final_message_dicts[0]["role"] == "user"
            and isinstance(final_message_dicts[0]["content"], str)
        ):
            result: list[ChatCompletionMessageParam] | str = (
                final_message_dicts[0]["content"]
            )
        else:
            result = final_message_dicts
    else:
        result = [
            ChatMessageConverter(message, drop_none=drop_none, model=model).build()
            for message in messages
        ]
    return result


# ---------------------------------------------------------------------------
# "From OpenAI" parsers — typed SDK objects → serapeum types
# ---------------------------------------------------------------------------


class ChatMessageParser:
    """Parses a Chat Completions API ``ChatCompletionMessage`` into a serapeum ``Message``."""

    def __init__(self, openai_message: ChatCompletionMessage, modalities: list[str]) -> None:
        self._openai_message = openai_message
        self._modalities = modalities
        self._blocks: list[ContentBlock] = []
        self._additional_kwargs: dict[str, Any] = {}

    def build(self) -> Message:
        self._extract_text_content()
        self._extract_tool_calls()
        self._extract_audio()
        return Message(
            role=self._openai_message.role,
            chunks=self._blocks,
            additional_kwargs=self._additional_kwargs,
        )

    def _extract_text_content(self) -> None:
        # NOTE: Azure OpenAI returns function calling messages without a content key
        if "text" in self._modalities and self._openai_message.content:
            self._blocks.append(TextChunk(content=self._openai_message.content or ""))

    def _extract_tool_calls(self) -> None:
        if self._openai_message.tool_calls:
            tool_calls: list[ChatCompletionMessageToolCall] = self._openai_message.tool_calls
            for tool_call in tool_calls:
                if tool_call.function:
                    self._blocks.append(
                        ToolCallBlock(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name or "",
                            tool_kwargs=tool_call.function.arguments or {},
                        )
                    )
            self._additional_kwargs.update(tool_calls=tool_calls)

    def _extract_audio(self) -> None:
        if self._openai_message.audio and "audio" in self._modalities:
            reference_audio_id = self._openai_message.audio.id
            audio_data = self._openai_message.audio.data
            self._additional_kwargs["reference_audio_id"] = reference_audio_id
            self._blocks.append(Audio(content=audio_data, format="mp3"))

    @classmethod
    def batch(cls, messages: Sequence[ChatCompletionMessage], modalities: list[str]) -> list[Message]:
        return [cls(m, modalities).build() for m in messages]


# ---------------------------------------------------------------------------
# "From OpenAI" parsers — raw dicts → serapeum types
# ---------------------------------------------------------------------------


class DictMessageParser:
    """Parses a raw OpenAI message dict into a serapeum ``Message``."""

    _BLOCK_PARSERS: dict[str, Callable[..., ContentBlock]] = {}

    def __init__(self, message_dict: dict[str, Any]) -> None:
        self._message_dict = message_dict
        self._blocks: list[ContentBlock] = []

    def build(self) -> Message:
        content = self._message_dict.get("content")
        if isinstance(content, list):
            self._parse_content_blocks(content)
            content = None
        additional_kwargs = self._extract_additional_kwargs()
        return Message(
            role=self._message_dict["role"],
            content=content,
            additional_kwargs=additional_kwargs,
            chunks=self._blocks,
        )

    def _parse_content_blocks(self, content: list[dict[str, Any]]) -> None:
        for elem in content:
            t = elem.get("type")
            parser = self._BLOCK_PARSERS.get(t)  # type: ignore[arg-type]
            if parser:
                self._blocks.append(parser(elem))
            else:
                raise ValueError(f"Unsupported message type: {t}")

    def _extract_additional_kwargs(self) -> dict[str, Any]:
        additional_kwargs = self._message_dict.copy()
        additional_kwargs.pop("role")
        additional_kwargs.pop("content", None)
        return additional_kwargs

    @classmethod
    def batch(cls, dicts: Sequence[dict[str, Any]]) -> list[Message]:
        return [cls(d).build() for d in dicts]

    # -- block parsers (populated after class definition) --

    @staticmethod
    def _parse_text(elem: dict[str, Any]) -> TextChunk:
        return TextChunk(content=elem.get("text", ""))

    @staticmethod
    def _parse_image(elem: dict[str, Any]) -> Image:
        img = elem["image_url"]["url"]
        detail = elem["image_url"].get("detail")
        if img.startswith("data:"):
            result = Image(content=img, detail=detail)
        else:
            result = Image(url=img, detail=detail)
        return result

    @staticmethod
    def _parse_function_call(elem: dict[str, Any]) -> ToolCallBlock:
        return ToolCallBlock(
            tool_call_id=elem.get("call_id"),
            tool_name=elem.get("name", ""),
            tool_kwargs=elem.get("arguments", {}),
        )


DictMessageParser._BLOCK_PARSERS = {
    "text": DictMessageParser._parse_text,
    "image_url": DictMessageParser._parse_image,
    "function_call": DictMessageParser._parse_function_call,
    "output_text": DictMessageParser._parse_text,
    "input_text": DictMessageParser._parse_text,
}


# ---------------------------------------------------------------------------
# LogProb parsers
# ---------------------------------------------------------------------------


class LogProbParser:
    """Converts OpenAI logprob types to serapeum ``LogProb`` lists."""

    @staticmethod
    def from_token(openai_token_logprob: ChatCompletionTokenLogprob) -> list[LogProb]:
        result: list[LogProb] = []
        if openai_token_logprob.top_logprobs:
            result = [
                LogProb(token=el.token, logprob=el.logprob, bytes=el.bytes or [])
                for el in openai_token_logprob.top_logprobs
            ]
        return result

    @staticmethod
    def from_tokens(openai_token_logprobs: Sequence[ChatCompletionTokenLogprob]) -> list[list[LogProb]]:
        result: list[list[LogProb]] = []
        for token_logprob in openai_token_logprobs:
            if logprobs := LogProbParser.from_token(token_logprob):
                result.append(logprobs)
        return result

    @staticmethod
    def from_completion(openai_completion_logprob: dict[str, float]) -> list[LogProb]:
        return [
            LogProb(token=t, logprob=v, bytes=[])
            for t, v in openai_completion_logprob.items()
        ]

    @staticmethod
    def from_completions(openai_completion_logprobs: Logprobs) -> list[list[LogProb]]:
        result: list[list[LogProb]] = []
        if openai_completion_logprobs.top_logprobs:
            result = [
                LogProbParser.from_completion(completion_logprob)
                for completion_logprob in openai_completion_logprobs.top_logprobs
            ]
        return result


# ---------------------------------------------------------------------------
# Streaming tool call accumulator
# ---------------------------------------------------------------------------


class ToolCallAccumulator:
    """Accumulates streaming tool call deltas into complete tool calls."""

    def __init__(self) -> None:
        self._tool_calls: list[ChoiceDeltaToolCall] = []

    @property
    def tool_calls(self) -> list[ChoiceDeltaToolCall]:
        return self._tool_calls

    def update(self, tool_calls_delta: list[ChoiceDeltaToolCall] | None) -> None:
        if tool_calls_delta and len(tool_calls_delta) > 0:
            tc_delta = tool_calls_delta[0]
            if len(self._tool_calls) == 0:
                self._tool_calls.append(tc_delta)
            elif self._tool_calls[-1].index != tc_delta.index:
                self._tool_calls.append(tc_delta)
            else:
                self._merge_into_existing(self._tool_calls[-1], tc_delta)

    @staticmethod
    def _merge_into_existing(existing: ChoiceDeltaToolCall, delta: ChoiceDeltaToolCall) -> None:
        existing_fn = cast(ChoiceDeltaToolCallFunction, existing.function)
        delta_fn = cast(ChoiceDeltaToolCallFunction, delta.function)

        if existing_fn.arguments is None:
            existing_fn.arguments = ""
        if existing_fn.name is None:
            existing_fn.name = ""
        if existing.id is None:
            existing.id = ""

        existing_fn.arguments += delta_fn.arguments or ""
        existing_fn.name += delta_fn.name or ""
        existing.id += delta.id or ""


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def to_openai_tool(
    pydantic_class: Type[BaseModel], description: str | None = None
) -> dict[str, Any]:
    """Convert pydantic class to OpenAI tool."""
    schema = pydantic_class.model_json_schema()
    schema_description = schema.get("description", None) or description
    function = {
        "name": schema["title"],
        "description": schema_description,
        "parameters": schema,
    }
    return {"type": "function", "function": function}
