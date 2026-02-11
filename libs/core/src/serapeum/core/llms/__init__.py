"""High-level LLM orchestration API and helpers.

This module provides a unified interface for working with LLMs in Serapeum.
It exports both core abstractions and high-level orchestrators.

Organization:
- Base LLM classes and types for core functionality
- Abstractions (llms.abstractions): Core LLM abstractions for providers
- Orchestrators (llms.orchestrators): High-level structured output generators
"""

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
    MessageRole,
    Metadata,
    TextChunk,
    MessageList
)
from serapeum.core.base.llms.base import BaseLLM

from serapeum.core.llms.base import LLM

# Core LLM abstractions
from serapeum.core.llms.abstractions import (
    FunctionCallingLLM,
    StructuredOutputLLM,
)

# High-level orchestrators
from serapeum.core.llms.orchestrators import (
    ToolOrchestratingLLM,
    TextCompletionLLM,
)

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
    # Orchestrators
    "ToolOrchestratingLLM",
    "TextCompletionLLM",
]
