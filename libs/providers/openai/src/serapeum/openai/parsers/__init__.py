"""OpenAI parser and formatter subpackage.

Re-exports all public symbols so that ``from serapeum.openai.parsers import ...``
works as a drop-in replacement for the former ``serapeum.openai.converters`` module.
"""

from serapeum.openai.parsers.chat_parsers import (
    ChatMessageParser,
    DictMessageParser,
    LogProbParser,
    ToolCallAccumulator,
)
from serapeum.openai.parsers.formatters import (
    ChatFormat,
    ChatMessageConverter,
    OpenAIToolCall,
    ResponsesFormat,
    ResponsesMessageConverter,
    _rewrite_system_to_developer,
    _should_null_content,
    _strip_none_keys,
    to_openai_message_dicts,
    to_openai_tool,
)
from serapeum.openai.parsers.response_parsers import (
    ResponsesOutputParser,
    ResponsesStreamAccumulator,
    _build_reasoning_content,
)

__all__ = [
    "ChatFormat",
    "ChatMessageConverter",
    "ChatMessageParser",
    "DictMessageParser",
    "LogProbParser",
    "OpenAIToolCall",
    "ResponsesFormat",
    "ResponsesMessageConverter",
    "ResponsesOutputParser",
    "ResponsesStreamAccumulator",
    "ToolCallAccumulator",
    "_build_reasoning_content",
    "_rewrite_system_to_developer",
    "_should_null_content",
    "_strip_none_keys",
    "to_openai_message_dicts",
    "to_openai_tool",
]
