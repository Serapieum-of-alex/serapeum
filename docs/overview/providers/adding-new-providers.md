# Adding New Providers

This guide provides detailed instructions for implementing new provider integrations in the Serapeum framework.

## Overview

Serapeum uses a **provider-based organization** where each provider package contains all features that provider offers (LLM, embeddings, and any provider-specific capabilities). This keeps related code together and makes it easy to install only the providers you need.

## Prerequisites

Before implementing a new provider, you should:

- Understand the provider's API and SDK
- Have API credentials or access to the provider's service
- Be familiar with Serapeum's core abstractions ([read the codebase map](../core-package.md))
- Review the [Ollama provider](../../libs/providers/ollama/) as a reference implementation

---

## Directory Structure

Create a provider package following this structure:

```
libs/providers/{provider-name}/
├── src/
│   └── serapeum/
│       └── {provider_name}/
│           ├── __init__.py          # Public API exports
│           ├── llm.py               # Chat/completion implementation
│           ├── embeddings.py        # Embeddings (if available)
│           └── shared/              # Shared utilities
│               ├── __init__.py
│               ├── client.py        # HTTP client, config
│               └── errors.py        # Provider-specific errors
├── tests/
│   ├── __init__.py
│   ├── test_llm.py                  # LLM tests
│   ├── test_embeddings.py           # Embedding tests
│   └── conftest.py                  # Pytest fixtures
├── pyproject.toml                   # Package configuration
└── README.md                        # Provider documentation
```

---

## Implementation Steps

### 1. Create Package Structure

```bash
# Create directory structure
mkdir -p libs/providers/{provider}/src/serapeum/{provider_name}
mkdir -p libs/providers/{provider}/src/serapeum/{provider_name}/shared
mkdir -p libs/providers/{provider}/tests

# Create __init__.py files
touch libs/providers/{provider}/src/serapeum/{provider_name}/__init__.py
touch libs/providers/{provider}/src/serapeum/{provider_name}/shared/__init__.py
touch libs/providers/{provider}/tests/__init__.py
```

### 2. Implement LLM Class

Create `libs/providers/{provider}/src/serapeum/{provider_name}/llm.py`:

```python
"""LLM implementation for {Provider}."""

from typing import Any, Iterator, AsyncIterator
from serapeum.core.llms import FunctionCallingLLM
from serapeum.core.llms import (
    ChatResponse,
    CompletionResponse,
    Message,
    MessageList,
)

class ProviderLLM(FunctionCallingLLM):
    """LLM implementation for {Provider}."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ):
        """
        Initialize the {Provider} LLM.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3")
            api_key: API key for authentication
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__(**kwargs)

    @property
    def metadata(self) -> dict[str, Any]:
        """Get LLM metadata."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Implement completion endpoint.

        Args:
            prompt: Text prompt
            **kwargs: Additional arguments

        Returns:
            CompletionResponse with generated text
        """
        # TODO: Call provider API
        # response = provider_client.complete(prompt=prompt, **kwargs)
        # return CompletionResponse(text=response.text, raw=response)
        raise NotImplementedError("Implement completion logic")

    def _chat(
        self,
        messages: MessageList,
        **kwargs: Any
    ) -> ChatResponse:
        """
        Implement chat endpoint.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Returns:
            ChatResponse with assistant message
        """
        # TODO: Call provider API
        # response = provider_client.chat(messages=messages, **kwargs)
        # return ChatResponse(
        #     message=Message(role="assistant", content=response.content),
        #     raw=response
        # )
        raise NotImplementedError("Implement chat logic")

    def _stream_complete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> Iterator[CompletionResponse]:
        """
        Implement streaming completion.

        Args:
            prompt: Text prompt
            **kwargs: Additional arguments

        Yields:
            CompletionResponse chunks
        """
        # TODO: Stream from provider API
        # for chunk in provider_client.stream_complete(prompt=prompt, **kwargs):
        #     yield CompletionResponse(text=chunk.text, raw=chunk)
        raise NotImplementedError("Implement streaming completion")

    def _stream_chat(
        self,
        messages: MessageList,
        **kwargs: Any
    ) -> Iterator[ChatResponse]:
        """
        Implement streaming chat.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Yields:
            ChatResponse chunks
        """
        # TODO: Stream from provider API
        # for chunk in provider_client.stream_chat(messages=messages, **kwargs):
        #     yield ChatResponse(
        #         message=Message(role="assistant", content=chunk.content),
        #         raw=chunk
        #     )
        raise NotImplementedError("Implement streaming chat")

    async def _acomplete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> CompletionResponse:
        """
        Async completion implementation.

        Args:
            prompt: Text prompt
            **kwargs: Additional arguments

        Returns:
            CompletionResponse with generated text
        """
        # TODO: Implement async completion
        # response = await provider_async_client.complete(prompt=prompt, **kwargs)
        # return CompletionResponse(text=response.text, raw=response)
        raise NotImplementedError("Implement async completion")

    async def _achat(
        self,
        messages: MessageList,
        **kwargs: Any
    ) -> ChatResponse:
        """
        Async chat implementation.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Returns:
            ChatResponse with assistant message
        """
        # TODO: Implement async chat
        # response = await provider_async_client.chat(messages=messages, **kwargs)
        # return ChatResponse(
        #     message=Message(role="assistant", content=response.content),
        #     raw=response
        # )
        raise NotImplementedError("Implement async chat")

    async def _astream_complete(
        self,
        prompt: str,
        **kwargs: Any
    ) -> AsyncIterator[CompletionResponse]:
        """
        Async streaming completion.

        Args:
            prompt: Text prompt
            **kwargs: Additional arguments

        Yields:
            CompletionResponse chunks
        """
        # TODO: Implement async streaming completion
        # async for chunk in provider_async_client.stream_complete(prompt=prompt, **kwargs):
        #     yield CompletionResponse(text=chunk.text, raw=chunk)
        raise NotImplementedError("Implement async streaming completion")
        # Make this a proper async generator
        if False:
            yield

    async def _astream_chat(
        self,
        messages: MessageList,
        **kwargs: Any
    ) -> AsyncIterator[ChatResponse]:
        """
        Async streaming chat.

        Args:
            messages: Conversation messages
            **kwargs: Additional arguments

        Yields:
            ChatResponse chunks
        """
        # TODO: Implement async streaming chat
        # async for chunk in provider_async_client.stream_chat(messages=messages, **kwargs):
        #     yield ChatResponse(
        #         message=Message(role="assistant", content=chunk.content),
        #         raw=chunk
        #     )
        raise NotImplementedError("Implement async streaming chat")
        # Make this a proper async generator
        if False:
            yield
```

### 3. Implement Embeddings (if applicable)

Create `libs/providers/{provider}/src/serapeum/{provider_name}/embeddings.py`:

```python
"""Embedding implementation for {Provider}."""

from typing import Any
from serapeum.core.embeddings import BaseEmbedding

class ProviderEmbedding(BaseEmbedding):
    """Embedding implementation for {Provider}."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        batch_size: int = 10,
        **kwargs: Any
    ):
        """
        Initialize {Provider} embeddings.

        Args:
            model_name: Embedding model identifier
            api_key: API key for authentication
            batch_size: Batch size for embedding generation
            **kwargs: Additional provider-specific arguments
        """
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # TODO: Call provider embedding API
        # response = provider_client.embed(text=text, model=self.model_name)
        # return response.embedding
        raise NotImplementedError("Implement text embedding")

    def _get_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector optimized for retrieval
        """
        # Most providers use the same method for text and query
        # Override if provider has specific query optimization
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # TODO: Implement batch embedding
        # Can use simple loop or provider's batch API if available
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """
        Async text embedding.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # TODO: Implement async embedding
        # response = await provider_async_client.embed(text=text, model=self.model_name)
        # return response.embedding
        raise NotImplementedError("Implement async text embedding")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """
        Async query embedding.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return await self._aget_text_embedding(query)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Async batch embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # TODO: Implement async batch embedding
        # For simple implementation, can use asyncio.gather
        import asyncio
        return await asyncio.gather(*[
            self._aget_text_embedding(text) for text in texts
        ])
```

### 4. Create Shared Client (optional but recommended)

Create `libs/providers/{provider}/src/serapeum/{provider_name}/shared/client.py`:

```python
"""Shared HTTP client for {Provider}."""

import requests
from typing import Any

class ProviderClient:
    """HTTP client for {Provider} API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.provider.com/v1",
        timeout: float = 60.0,
        **kwargs: Any
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()

        # Set up authentication headers
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })

    def post(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Make POST request to provider API."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the session."""
        self.session.close()
```

### 5. Create Error Classes

Create `libs/providers/{provider}/src/serapeum/{provider_name}/shared/errors.py`:

```python
"""Provider-specific errors."""

class ProviderError(Exception):
    """Base error for {Provider} provider."""
    pass

class ProviderAPIError(ProviderError):
    """API request failed."""
    pass

class ProviderAuthError(ProviderError):
    """Authentication failed."""
    pass

class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""
    pass
```

### 6. Export Public API

Update `libs/providers/{provider}/src/serapeum/{provider_name}/__init__.py`:

```python
"""Serapeum {Provider} integration."""

from serapeum.{provider_name}.llm import ProviderLLM
from serapeum.{provider_name}.embeddings import ProviderEmbedding

__all__ = [
    "ProviderLLM",
    "ProviderEmbedding",
]
```

### 7. Add to Workspace

Update root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["libs/core", "libs/providers/*"]

[tool.uv.sources]
serapeum-core = { workspace = true }
serapeum-{provider} = { workspace = true }
```

### 8. Create Package Configuration

Create `libs/providers/{provider}/pyproject.toml`:

```toml
[project]
name = "serapeum-{provider}"
version = "0.1.0"
description = "{Provider} integration for Serapeum"
readme = "README.md"
license = {text = "GNU General Public License v3"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["llm", "ai", "{provider}", "serapeum"]
requires-python = ">=3.11,<4.0"
dependencies = [
    "serapeum-core",
    "{provider-sdk}>=1.0.0",  # e.g., "openai>=1.0.0"
    "requests>=2.32.0",
]

[project.urls]
homepage = "https://github.com/Serapieum-of-alex/serapeum"
repository = "https://github.com/Serapieum-of-alex/serapeum"
documentation = "https://serapieum-of-alex.github.io/serapeum/"

[tool.uv.sources]
serapeum-core = { workspace = true }

[dependency-groups]
dev = [
    "pytest>=8.4.2",
    "pytest-cov>=7.0.0",
    "pytest-asyncio>=1.2.0",
    "nest-asyncio>=1.6.0",
    "mypy>=1.13.0",
]

[tool.pytest.ini_options]
testpaths = "tests"
markers = [
    "e2e: end-to-end tests requiring provider service",
    "unit: unit tests",
    "integration: integration tests",
]

[tool.hatch.build.targets.wheel]
include = ["src/serapeum"]
packages = ["src/serapeum"]

[tool.hatch.build.targets.sdist]
include = ["src/serapeum"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 9. Write Tests

Create `libs/providers/{provider}/tests/test_llm.py`:

```python
"""Tests for {Provider} LLM."""

import pytest
from serapeum.{provider_name} import ProviderLLM
from serapeum.core.llms import Message, MessageRole

@pytest.mark.unit
def test_initialization():
    """Test LLM initialization."""
    llm = ProviderLLM(model="test-model", api_key="test-key")
    assert llm.model == "test-model"
    assert llm.api_key == "test-key"

@pytest.mark.unit
def test_metadata():
    """Test metadata property."""
    llm = ProviderLLM(model="test-model", temperature=0.5)
    metadata = llm.metadata
    assert metadata["model"] == "test-model"
    assert metadata["temperature"] == 0.5

@pytest.mark.e2e
def test_chat():
    """Test chat functionality (requires provider API)."""
    llm = ProviderLLM(model="test-model")
    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = llm.chat(messages)
    assert response.message.content
    assert response.message.role == MessageRole.ASSISTANT

@pytest.mark.e2e
def test_streaming():
    """Test streaming chat."""
    llm = ProviderLLM(model="test-model")
    messages = [Message(role=MessageRole.USER, content="Count to 3")]

    chunks = list(llm.chat(messages, stream=True))
    assert len(chunks) > 0
    assert chunks[-1].message.content

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_chat():
    """Test async chat."""
    llm = ProviderLLM(model="test-model")
    messages = [Message(role=MessageRole.USER, content="Hello")]
    response = await llm.achat(messages)
    assert response.message.content
```

Create `libs/providers/{provider}/tests/test_embeddings.py`:

```python
"""Tests for {Provider} embeddings."""

import pytest
from serapeum.{provider_name} import ProviderEmbedding

@pytest.mark.unit
def test_initialization():
    """Test embedding model initialization."""
    embed = ProviderEmbedding(model_name="test-embed", api_key="test-key")
    assert embed.model_name == "test-embed"
    assert embed.api_key == "test-key"

@pytest.mark.e2e
def test_text_embedding():
    """Test text embedding generation."""
    embed = ProviderEmbedding(model_name="test-embed")
    embedding = embed.get_text_embedding("Hello, world!")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.e2e
def test_batch_embeddings():
    """Test batch embedding generation."""
    embed = ProviderEmbedding(model_name="test-embed")
    texts = ["Hello", "World", "Test"]
    embeddings = embed.get_text_embeddings(texts)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_async_embedding():
    """Test async embedding generation."""
    embed = ProviderEmbedding(model_name="test-embed")
    embedding = await embed.aget_text_embedding("Hello, world!")
    assert isinstance(embedding, list)
    assert len(embedding) > 0
```

Create `libs/providers/{provider}/tests/conftest.py`:

```python
"""Pytest configuration for {Provider} tests."""

import pytest
import os

@pytest.fixture(scope="session")
def api_key():
    """Get API key from environment."""
    key = os.getenv("{PROVIDER}_API_KEY")
    if not key:
        pytest.skip("API key not found in environment")
    return key

@pytest.fixture
def llm(api_key):
    """Create LLM instance for testing."""
    from serapeum.{provider_name} import ProviderLLM
    return ProviderLLM(model="test-model", api_key=api_key)

@pytest.fixture
def embedding_model(api_key):
    """Create embedding model for testing."""
    from serapeum.{provider_name} import ProviderEmbedding
    return ProviderEmbedding(model_name="test-embed", api_key=api_key)
```

### 10. Create README

Create `libs/providers/{provider}/README.md`:

```markdown
# Serapeum {Provider} Provider

**{Provider} integration for the Serapeum LLM framework**

## Installation

```bash
pip install serapeum-{provider}
```

## Prerequisites

- {Provider} API key
- Python 3.11+

## Quick Start

```python
from serapeum.{provider_name} import ProviderLLM
from serapeum.core.llms import Message, MessageRole

# Initialize LLM
llm = ProviderLLM(
    model="model-name",
    api_key="your-api-key"
)

# Chat
messages = [Message(role=MessageRole.USER, content="Hello!")]
response = llm.chat(messages)
print(response.message.content)
```

## Features

- Chat and completion interfaces
- Streaming support
- Tool calling (if supported by provider)
- Structured outputs
- Embeddings (if available)
- Full async support

## Configuration

See the [documentation](https://serapieum-of-alex.github.io/serapeum/) for detailed configuration options.

## Testing

```bash
cd libs/providers/{provider}
uv run pytest
```

## Links

- [Documentation](https://serapieum-of-alex.github.io/serapeum/)
- [Serapeum Repository](https://github.com/Serapieum-of-alex/serapeum)
- [{Provider} Documentation]({provider_docs_url})
```

### 11. Add Documentation

Create `docs/overview/providers/{provider}.md` with:

- Installation instructions
- Configuration options
- Usage examples
- Available models
- Limitations and notes

---

## Why Provider-Based Organization?

**Benefits:**

- **Shared Infrastructure**: All provider features share client, auth, error handling
- **Single Installation**: Users install one package per provider (`pip install serapeum-{provider}`)
- **Co-located Code**: Related features are maintained together
- **Isolated Dependencies**: Provider SDKs don't conflict
- **Industry Standard**: Matches LangChain and other frameworks

**Example:**
```python
# Users install only what they need
pip install serapeum-ollama  # For local inference
pip install serapeum-openai  # For OpenAI API

# Clean imports
from serapeum.ollama import Ollama
from serapeum.openai import OpenAI
```

---

## Provider Development Checklist

When implementing a new provider, ensure:

- [ ] Inherits from `FunctionCallingLLM` for LLM
- [ ] Implements `BaseEmbedding` for embeddings (if applicable)
- [ ] Supports sync, async, and streaming operations
- [ ] All abstract methods are implemented
- [ ] Includes comprehensive unit tests (≥95% coverage)
- [ ] Includes e2e tests with appropriate markers
- [ ] Has README with examples and configuration
- [ ] Has documentation page in `docs/overview/providers/`
- [ ] Exports public API in `__init__.py`
- [ ] Added to workspace in root `pyproject.toml`
- [ ] Added to provider comparison table in `docs/overview/providers.md`
- [ ] Follows code style and type annotations
- [ ] Handles errors gracefully with provider-specific exceptions
- [ ] Includes proper logging
- [ ] Documentation includes troubleshooting section

---

## Reference Implementation

See the **[Ollama provider](https://github.com/Serapieum-of-alex/Serapeum/tree/main/libs/providers/ollama)** for a complete reference implementation showing:

- LLM implementation with streaming and async
- Embedding implementation with batching
- Shared client and error handling
- Comprehensive test suite with fixtures
- Complete documentation
- Proper error handling

Study the Ollama provider to understand:

- How to structure the package
- How to implement all required methods
- How to handle streaming responses
- How to write comprehensive tests
- How to document the provider

---

## Testing Your Provider

### Run Tests

```bash
# All tests
cd libs/providers/{provider}
uv run pytest

# Skip e2e tests
uv run pytest -m "not e2e"

# Only unit tests
uv run pytest -m unit

# With coverage
uv run pytest --cov=serapeum.{provider_name} --cov-report=html
```

### Test Coverage

Aim for ≥95% test coverage. Use coverage reports to identify untested code:

```bash
uv run pytest --cov=serapeum.{provider_name} --cov-report=term-missing
```

---

## Common Pitfalls

### 1. Not Implementing All Abstract Methods

Make sure to implement all required methods from `FunctionCallingLLM` and `BaseEmbedding`.

### 2. Inconsistent Message Formatting

Ensure messages are converted correctly to the provider's format.

### 3. Missing Error Handling

Always handle provider-specific errors and convert them to appropriate exceptions.

### 4. No Streaming Support

Streaming is a key feature. Implement both sync and async streaming.

### 5. Hardcoded Values

Use configuration parameters instead of hardcoding URLs, timeouts, etc.

---

## Support

For help implementing a new provider:

- Review the [Ollama provider](https://github.com/Serapieum-of-alex/Serapeum/tree/main/libs/providers/ollama) implementation
- Check the [API Reference](../../reference/core/llms/llm-classes-comparison.md)
- Open an issue on [GitHub](https://github.com/Serapieum-of-alex/Serapeum/issues)

---

## Next Steps

After implementing your provider:

1. Test thoroughly with both unit and e2e tests
2. Write comprehensive documentation
3. Add examples to the docs
4. Update the provider comparison table
5. Submit a pull request
6. Add your provider to the main documentation navigation

Good luck building your provider integration! 🚀
