"""High-level LLM orchestration API and helpers.

This module provides a unified interface for working with LLMs in Serapeum.
It exports both core abstractions and high-level orchestrators.

Organization:
- Base LLM classes and types for core functionality
- Abstractions (llms.abstractions): Core LLM abstractions for providers
- Orchestrators (llms.orchestrators): High-level structured output generators
"""

from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.base.llms.types import (
    Audio,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Image,
    Message,
    MessageList,
    MessageRole,
    Metadata,
    TextChunk,
)

# Core LLM abstractions
from serapeum.core.llms.abstractions import (
    ChatToCompletionMixin,
    FunctionCallingLLM,
    StructuredOutputLLM,
)
from serapeum.core.llms.base import LLM

# High-level orchestrators
from serapeum.core.llms.orchestrators import (
    TextCompletionLLM,
    ToolOrchestratingLLM,
)

from serapeum.core.llms.custom import CustomLLM
__all__ = [
    # Base LLM and types
    "LLM",
    "BaseLLM",
    "Message",
    "ChatResponse",
    "ChatResponseAsyncGen",
    "ChatResponseGen",
    "CompletionResponse",
    "CompletionResponseAsyncGen",
    "CompletionResponseGen",
    "Metadata",
    "MessageRole",
    "Image",
    "TextChunk",
    "Audio",
    "MessageList",
    # Core abstractions
    "FunctionCallingLLM",
    "StructuredOutputLLM",
    "ChatToCompletionMixin",
    # Orchestrators
    "ToolOrchestratingLLM",
    "TextCompletionLLM",
    "CustomLLM"
]
