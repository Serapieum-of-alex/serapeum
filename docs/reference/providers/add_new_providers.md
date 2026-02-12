# Adding a New LLM Provider to Serapeum

This guide explains how to add a new LLM provider integration to the Serapeum framework.

## Overview

Serapeum uses a **provider-based architecture** where each LLM provider is implemented as a separate package under `libs/providers/`. Each provider package is self-contained and includes all features that provider offers (LLM, embeddings, etc.).

## Quick Start

To add a new provider (e.g., OpenAI):

1. Create package structure in `libs/providers/openai/`
2. Implement LLM class inheriting from `ChatToCompletionMixin` and `FunctionCallingLLM`
3. Add tests
4. Configure workspace
5. Document

---

## Step-by-Step Guide

### 1. Create Package Structure

Create a new provider package following this structure:

```
libs/providers/{provider-name}/
├── src/
│   └── serapeum/
│       └── providers/
│           └── {provider_name}/
│               ├── __init__.py
│               ├── llm.py              # Chat/completion LLM implementation
│               ├── embeddings.py       # Embeddings (if available)
│               └── shared/             # Shared utilities
│                   ├── __init__.py
│                   ├── client.py       # HTTP client, config
│                   └── errors.py       # Provider-specific errors
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_llm.py
│   ├── test_embeddings.py
│   └── fixtures/
├── pyproject.toml
└── README.md
```

### 2. Create `pyproject.toml`

**File**: `libs/providers/{provider}/pyproject.toml`

```toml
[project]
name = "serapeum-{provider}"
version = "0.1.0"
description = "{Provider} integration for Serapeum LLM framework"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "serapeum-core",
    "{provider-sdk}>=1.0.0",  # The official provider SDK
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/serapeum"]

# Workspace source - points to local serapeum-core during development
[tool.uv.sources]
serapeum-core = { workspace = true }

# Pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "strict"
asyncio_default_fixture_loop_scope = "function"

# Test markers
markers = [
    "e2e: End-to-end tests requiring external services",
    "integration: Integration tests",
    "unit: Unit tests",
]
```

### 3. Update Root Workspace

**File**: `pyproject.toml` (root)

Add your provider to the workspace members:

```toml
[tool.uv.workspace]
members = [
    "libs/core",
    "libs/providers/*",  # This includes your new provider
]

[tool.uv.sources]
serapeum-{provider} = { workspace = true }
```

### 4. Implement the LLM Class

**File**: `libs/providers/{provider}/src/serapeum/providers/{provider}/llm.py`

```python
"""LLM implementation for {Provider}."""

from typing import Any, Sequence

from pydantic import Field, PrivateAttr

from serapeum.core.llms import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    ChatToCompletionMixin,
    FunctionCallingLLM,
    Message,
    MessageList,
    Metadata,
)
from serapeum.core.tools import ToolCallArguments


class {Provider}LLM(ChatToCompletionMixin, FunctionCallingLLM):
    """LLM implementation for {Provider}.

    This class provides chat and completion interfaces for {Provider} models.
    Completion methods are automatically provided by ChatToCompletionMixin.

    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        api_key: API key for authentication (optional if set via env)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        **kwargs: Additional provider-specific options

    Examples:
        Basic chat:
        ```python
        from serapeum.providers.{provider} import {Provider}LLM

        llm = {Provider}LLM(model="model-name", api_key="your-key")
        response = llm.chat([Message(role=MessageRole.USER, content="Hello")])
        print(response.message.content)
        ```

        Streaming chat:
        ```python
        for chunk in llm.stream_chat([Message(role=MessageRole.USER, content="Hi")]):
            print(chunk.delta, end="", flush=True)
        ```

        Completion interface (via mixin):
        ```python
        response = llm.complete("What is 2+2?")
        print(response.text)
        ```
    """

    model: str = Field(description="Model identifier")
    api_key: str | None = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, gt=0)

    # Private attributes for client instances
    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ):
        """Initialize the {Provider} LLM."""
        super().__init__(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self._client = None
        self._async_client = None

    @classmethod
    def class_name(cls) -> str:
        """Return the class identifier."""
        return "{Provider}LLM"

    @property
    def metadata(self) -> Metadata:
        """Return LLM metadata."""
        return Metadata(
            context_window=self._get_context_window(),
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=True,
        )

    @property
    def client(self):
        """Lazy-loaded synchronous client."""
        if self._client is None:
            # Import and initialize the provider's SDK client
            from {provider}_sdk import Client
            self._client = Client(api_key=self.api_key or self._get_api_key_from_env())
        return self._client

    @property
    def async_client(self):
        """Lazy-loaded async client."""
        if self._async_client is None:
            from {provider}_sdk import AsyncClient
            self._async_client = AsyncClient(api_key=self.api_key or self._get_api_key_from_env())
        return self._async_client

    def _get_api_key_from_env(self) -> str:
        """Get API key from environment variable."""
        import os
        api_key = os.getenv("{PROVIDER}_API_KEY")
        if not api_key:
            raise ValueError(
                f"{PROVIDER}_API_KEY environment variable not set. "
                "Please set it or pass api_key parameter."
            )
        return api_key

    def _get_context_window(self) -> int:
        """Get context window size for the model."""
        # Map model names to context windows
        context_windows = {
            "model-small": 4096,
            "model-large": 128000,
            # Add more models
        }
        return context_windows.get(self.model, 4096)

    def _convert_messages(self, messages: MessageList) -> list[dict]:
        """Convert MessageList to provider's format."""
        provider_messages = []
        for msg in messages:
            provider_messages.append({
                "role": msg.role.value,
                "content": msg.content or "",
            })
        return provider_messages

    # ===== REQUIRED: Implement these 4 chat methods =====

    def chat(self, messages: MessageList, **kwargs: Any) -> ChatResponse:
        """Send a chat request.

        Args:
            messages: List of messages in the conversation
            **kwargs: Provider-specific options

        Returns:
            ChatResponse with the assistant's message
        """
        provider_messages = self._convert_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=provider_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
            ),
            raw=response,
        )

    def stream_chat(self, messages: MessageList, **kwargs: Any) -> ChatResponseGen:
        """Stream chat responses.

        Args:
            messages: List of messages in the conversation
            **kwargs: Provider-specific options

        Yields:
            ChatResponse chunks with delta content
        """
        provider_messages = self._convert_messages(messages)

        def gen():
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=provider_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )

            full_content = ""
            for chunk in response_stream:
                delta = chunk.choices[0].delta.content or ""
                full_content += delta

                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                    ),
                    delta=delta,
                    raw=chunk,
                )

        return gen()

    async def achat(self, messages: MessageList, **kwargs: Any) -> ChatResponse:
        """Async chat request.

        Args:
            messages: List of messages in the conversation
            **kwargs: Provider-specific options

        Returns:
            ChatResponse with the assistant's message
        """
        provider_messages = self._convert_messages(messages)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=provider_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=response.choices[0].message.content,
            ),
            raw=response,
        )

    async def astream_chat(
        self, messages: MessageList, **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Async streaming chat.

        Args:
            messages: List of messages in the conversation
            **kwargs: Provider-specific options

        Returns:
            Async generator yielding ChatResponse chunks
        """
        provider_messages = self._convert_messages(messages)

        async def gen():
            response_stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=provider_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs,
            )

            full_content = ""
            async for chunk in response_stream:
                delta = chunk.choices[0].delta.content or ""
                full_content += delta

                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                    ),
                    delta=delta,
                    raw=chunk,
                )

        return gen()

    # ===== COMPLETION METHODS =====
    # complete(), stream_complete(), acomplete(), astream_complete()
    # are automatically provided by ChatToCompletionMixin!

    # ===== FUNCTION CALLING (Optional) =====

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare chat request with tools."""
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = Message(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
        }

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
    ) -> list[ToolCallArguments]:
        """Extract tool calls from response."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])

        if not tool_calls:
            if error_on_no_tool_call:
                raise ValueError("No tool calls found in response")
            return []

        # Parse tool calls into ToolCallArguments
        selections = []
        for tool_call in tool_calls:
            selections.append(
                ToolCallArguments(
                    tool_id=tool_call.get("id", tool_call["function"]["name"]),
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=tool_call["function"]["arguments"],
                )
            )

        return selections
```

### 5. Implement Package `__init__.py`

**File**: `libs/providers/{provider}/src/serapeum/providers/{provider}/__init__.py`

```python
"""{Provider} integration for Serapeum LLM framework."""

from serapeum.providers.{provider}.llm import {Provider}LLM

# Add embeddings if available
# from serapeum.providers.{provider}.embeddings import {Provider}Embeddings

__all__ = [
    "{Provider}LLM",
    # "{Provider}Embeddings",
]
```

### 6. Create Tests

**File**: `libs/providers/{provider}/tests/test_llm.py`

```python
"""Tests for {Provider}LLM."""

import pytest
from unittest.mock import Mock, patch

from serapeum.core.base.llms.types import Message, MessageRole, MessageList
from serapeum.providers.{provider} import {Provider}LLM


@pytest.fixture
def mock_client():
    """Create a mock {Provider} client."""
    with patch("{provider}_sdk.Client") as mock:
        yield mock


@pytest.fixture
def llm(mock_client):
    """Create a {Provider}LLM instance with mocked client."""
    return {Provider}LLM(model="test-model", api_key="test-key")


class TestChat:
    """Test chat functionality."""

    def test_chat_basic(self, llm, mock_client):
        """Test basic chat."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello!"

        llm.client.chat.completions.create.return_value = mock_response

        # Test
        messages = MessageList.from_list([
            Message(role=MessageRole.USER, content="Hi")
        ])
        response = llm.chat(messages)

        assert response.message.content == "Hello!"
        assert response.message.role == MessageRole.ASSISTANT

    def test_stream_chat(self, llm, mock_client):
        """Test streaming chat."""
        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hel"))]),
            Mock(choices=[Mock(delta=Mock(content="lo!"))]),
        ]

        llm.client.chat.completions.create.return_value = iter(mock_chunks)

        # Test
        messages = MessageList.from_list([
            Message(role=MessageRole.USER, content="Hi")
        ])
        chunks = list(llm.stream_chat(messages))

        assert len(chunks) == 2
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo!"

    @pytest.mark.asyncio
    async def test_achat(self, llm, mock_client):
        """Test async chat."""
        # Mock async response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Async hello!"

        llm.async_client.chat.completions.create.return_value = mock_response

        # Test
        messages = MessageList.from_list([
            Message(role=MessageRole.USER, content="Hi")
        ])
        response = await llm.achat(messages)

        assert response.message.content == "Async hello!"


class TestCompletion:
    """Test completion functionality (from ChatToCompletionMixin)."""

    def test_complete(self, llm, mock_client):
        """Test that complete() works via mixin."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Answer: 42"

        llm.client.chat.completions.create.return_value = mock_response

        # Test completion interface
        response = llm.complete("What is the answer?")

        assert response.text == "Answer: 42"


@pytest.mark.e2e
class TestE2E:
    """End-to-end tests requiring actual API."""

    def test_real_chat(self):
        """Test with real API (requires API key)."""
        import os

        api_key = os.getenv("{PROVIDER}_API_KEY")
        if not api_key:
            pytest.skip("{PROVIDER}_API_KEY not set")

        llm = {Provider}LLM(model="model-name", api_key=api_key)
        response = llm.chat(
            MessageList.from_list([
                Message(role=MessageRole.USER, content="Say 'test'")
            ])
        )

        assert response.message.content
        assert isinstance(response.message.content, str)
```

### 7. Create README

**File**: `libs/providers/{provider}/README.md`

```markdown
# Serapeum {Provider} Integration

{Provider} integration for the Serapeum LLM framework.

## Installation

```bash
pip install serapeum-{provider}
```

## Usage

### Basic Chat

```python
from serapeum.providers.{provider} import {Provider}LLM
from serapeum.core.base.llms.types import Message, MessageRole

llm = {Provider}LLM(
    model="model-name",
    api_key="your-api-key",  # or set {PROVIDER}_API_KEY env var
)

response = llm.chat([
    Message(role=MessageRole.USER, content="Hello!")
])

print(response.message.content)
```

### Streaming

```python
for chunk in llm.stream_chat([
    Message(role=MessageRole.USER, content="Tell me a story")
]):
    print(chunk.delta, end="", flush=True)
```

### Completion Interface

```python
response = llm.complete("What is 2+2?")
print(response.text)
```

### Async

```python
import asyncio

async def main():
    response = await llm.achat([
        Message(role=MessageRole.USER, content="Hi")
    ])
    print(response.message.content)

asyncio.run(main())
```

## Configuration

- `model`: Model identifier
- `api_key`: API key (or set `{PROVIDER}_API_KEY` environment variable)
- `temperature`: Sampling temperature (0.0-1.0, default: 0.7)
- `max_tokens`: Maximum response tokens (default: 1000)

## Features

- ✅ Chat interface
- ✅ Streaming
- ✅ Async support
- ✅ Completion interface (via ChatToCompletionMixin)
- ✅ Function calling (if supported by provider)
- ✅ Embeddings (if available)

## Testing

```bash
# Unit tests (mocked)
uv run pytest libs/providers/{provider}/tests/ -m "not e2e"

# E2E tests (requires API key)
export {PROVIDER}_API_KEY="your-key"
uv run pytest libs/providers/{provider}/tests/ -m e2e
```

## License

Same as Serapeum core.
```

### 8. Build and Install

```bash
# Sync workspace dependencies
uv sync --dev

# Build the provider package
uv build --wheel -o dist libs/providers/{provider}

# Install locally for testing
uv pip install dist/serapeum_{provider}-0.1.0-py3-none-any.whl

# Or use workspace install
uv sync --package serapeum-{provider}
```

### 9. Test the Provider

```bash
# Run unit tests (mocked)
uv run pytest libs/providers/{provider}/tests/ -v -m "not e2e"

# Run E2E tests (requires API key)
export {PROVIDER}_API_KEY="your-api-key"
uv run pytest libs/providers/{provider}/tests/ -v -m e2e

# Run with coverage
uv run pytest libs/providers/{provider}/tests/ --cov=serapeum.providers.{provider}
```

---

## Key Points

### ✅ Do This

1. **Inherit from ChatToCompletionMixin FIRST**:
   ```python
   class YourLLM(ChatToCompletionMixin, FunctionCallingLLM):
   ```
   Order matters for Method Resolution Order (MRO)!

2. **Implement 4 chat methods only**:
   - `chat(messages, **kwargs)`
   - `stream_chat(messages, **kwargs)`
   - `achat(messages, **kwargs)`
   - `astream_chat(messages, **kwargs)`

3. **Get completion methods for free** from the mixin

4. **Use lazy-loaded clients** for better resource management

5. **Add proper error handling** for API errors

6. **Write comprehensive tests** with mocks and E2E tests

### ❌ Don't Do This

1. **Don't implement completion methods manually** - use the mixin
2. **Don't put FunctionCallingLLM before ChatToCompletionMixin** in inheritance
3. **Don't forget to handle async properly** (event loops, etc.)
4. **Don't hardcode API keys** - use environment variables
5. **Don't skip testing** - both mocked and E2E

---

## Examples

### Existing Providers

See these for reference:

- **Ollama**: `libs/providers/ollama/` - Complete implementation with streaming, async, and tool calling
- **Core abstractions**: `libs/core/src/serapeum/core/llms/abstractions/` - Base classes and mixins

### Minimal Example

For a minimal provider implementation, you only need ~100 lines of code:

```python
from serapeum.core.llms import ChatToCompletionMixin, FunctionCallingLLM

class MinimalLLM(ChatToCompletionMixin, FunctionCallingLLM):
    def chat(self, messages, **kwargs):
        # Call your provider's API
        return ChatResponse(...)

    def stream_chat(self, messages, **kwargs):
        # Stream from provider
        yield ChatResponse(...)

    async def achat(self, messages, **kwargs):
        # Async call
        return ChatResponse(...)

    async def astream_chat(self, messages, **kwargs):
        # Async stream
        async def gen():
            yield ChatResponse(...)
        return gen()

    # That's it! Completion methods come from ChatToCompletionMixin
```

---

## Checklist

Use this checklist when adding a new provider:

- [ ] Created package structure under `libs/providers/{provider}/`
- [ ] Created `pyproject.toml` with dependencies
- [ ] Updated root `pyproject.toml` workspace members
- [ ] Implemented LLM class with ChatToCompletionMixin
- [ ] Implemented 4 chat methods
- [ ] Verified completion methods work (from mixin)
- [ ] Added `_prepare_chat_with_tools()` for function calling
- [ ] Created comprehensive tests (unit + E2E)
- [ ] Created README.md with examples
- [ ] Added conftest.py with fixtures
- [ ] Tested with `uv sync --package serapeum-{provider}`
- [ ] Verified MRO is correct: `ChatToCompletionMixin` before `FunctionCallingLLM`
- [ ] All tests passing
- [ ] Added to documentation

---

## Next Steps

After implementing your provider:

1. **Add to documentation**: Update main docs with provider info
2. **Add examples**: Create example notebooks/scripts
3. **Publish**: Build and publish to PyPI (if public)
4. **Announce**: Add to changelog and release notes

---

## Need Help?
- **Example code**: See `examples/chat_to_completion_mixin_example.py`
- **Architecture**: See `docs/architecture/` for detailed docs

---

**Happy coding!** 🚀
