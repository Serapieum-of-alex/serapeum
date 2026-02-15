# GitHub Copilot Instructions for Serapeum

## Project Overview

Serapeum is a modular Python LLM framework providing core abstractions for building applications with Large Language Models. The repository uses a **uv workspace** monorepo structure with multiple packages:

- **serapeum-core**: Provider-agnostic core (LLM interfaces, tools, prompts, parsers, structured outputs)
- **serapeum-ollama**: Ollama provider integration (LLM + embeddings)
- Additional provider packages follow the same pattern

**Key Repository Facts:**
- Uses **uv workspaces** for managing multiple packages in a monorepo
- Organizes integrations by **provider** (not by feature type like "llms" vs "embeddings")
- Each provider package contains all features that provider offers (LLM, embeddings, etc.)
- Follows **src layout** for proper package isolation
- Uses **PEP 420 namespace packages** (`serapeum.*`)

## Repository Structure

```
serapeum/
├── libs/
│   ├── core/                    # serapeum-core package
│   └── providers/
│       └── ollama/              # serapeum-ollama package
├── docs/                        # MkDocs documentation
├── examples/                    # Usage examples and notebooks
├── pyproject.toml              # Root workspace configuration
├── Taskfile.yml                # Task runner commands
└── uv.lock                     # Unified dependency lockfile
```

## Package Manager: uv

This project uses **uv** for dependency management and workspace orchestration.

### Common uv Commands

```bash
# Sync all workspace dependencies (from root)
uv sync --dev

# Sync specific package
uv sync --package serapeum-ollama

# Run tests
uv run pytest
uv run pytest libs/core/tests
uv run pytest -m "not e2e"              # Skip end-to-end tests

# Build packages
uv build --wheel -o dist libs/core
uv build --wheel -o dist libs/providers/ollama

# Run documentation server
uv run mkdocs serve
```

Use the **Taskfile** for common workflows:
```bash
task build:all           # Build all packages
task install:all         # Install all packages
task build:core-lib      # Build only core
task build:plugin:ollama # Build only ollama
```

## Architecture

### Layered Design

The framework follows a layered architecture from base abstractions to high-level orchestration:

**1. Base Layer** (`serapeum.core.base.llms`)
- `BaseLLM`: Core LLM protocol with sync/async support
- `Message`, `MessageList`, `ChatResponse`, `CompletionResponse`: Data models
- Abstract interfaces that all providers implement

**2. LLM Layer** (`serapeum.core.llms`)
- `LLM`: Prompt formatting and structured prediction
- `FunctionCallingLLM`: Tool-calling specialization (base for providers)
- `StructuredOutputLLM`: Wrapper that forces structured outputs

**3. Tools Layer** (`serapeum.core.tools`)
- `BaseTool`/`AsyncBaseTool`: Tool interfaces
- `CallableTool`: Create tools from functions or Pydantic models
- Automatic JSON schema generation and validation

**4. Orchestration Layer** (`serapeum.core.llms.orchestrators`)
- `ToolOrchestratingLLM`: Composes prompts, LLMs, and toolsets
- `TextCompletionLLM`: Text-completion style orchestration utilities
- Manages conversation flow with tool execution

**5. Provider Layer** (`serapeum.providers.*`)
- Concrete implementations (e.g., `serapeum.providers.ollama`)
- Each provider package is self-contained with all features
- Example: Ollama provider includes both LLM and embeddings

### Design Patterns

- **Async-First**: All LLM operations support both sync and async with streaming
- **Protocol-Based**: Uses Python protocols for flexible, duck-typed interfaces
- **Pydantic Integration**: Structured outputs and validation via Pydantic models
- **Tool Composition**: Tools can be created from functions or composed into workflows
- **Provider Organization**: Each provider package contains all features (LLM + embeddings + any provider-specific capabilities)

## Code Style Guidelines

### Python Standards
- Use Python 3.10+ features (type hints, pattern matching, etc.)
- Follow PEP 8 style guidelines
- Use `from __future__ import annotations` for forward references
- Prefer `list[T]` and `dict[K, V]` over `List[T]` and `Dict[K, V]`
- Use `Sequence`, `Mapping`, etc. from `typing` for abstract types
- Always include type hints for function parameters and return values

### Documentation
- Write comprehensive Google-style docstrings for all public classes, methods, and functions
- Include detailed descriptions, argument documentation, return values, and raised exceptions
- Add usage examples in docstrings with proper doctest formatting (add `# doctest: +SKIP` for examples requiring external services)
- Document both synchronous and asynchronous method variants
- Include "See Also" sections to link related functionality
- Keep docstrings up-to-date when modifying code

### Architecture Patterns
- Use abstract base classes from `serapeum.core` for common interfaces
- Implement both sync and async methods (`method()` and `amethod()`)
- Use Pydantic for configuration and validation with `BaseModel` or `ConfigDict`
- Private methods start with `_`, protected methods should be documented if part of extension API
- Use `@model_validator` for post-initialization setup in Pydantic models
- Prefer composition over inheritance where appropriate

### Async/Sync Support
- Always provide both synchronous and asynchronous versions of I/O operations
- Use `async`/`await` for async methods (prefix with `a`, e.g., `aget_embedding()`)
- Never block in async methods (use async clients, not sync clients in async context)
- Use proper async context managers (`async with`) for resource management

### Error Handling
- Use specific exception types, not generic `Exception`
- Validate inputs early and raise `ValueError` with clear messages
- Document all exceptions in docstrings under "Raises:" section
- Include helpful context in error messages (what failed, why, expected format)

### Testing Strategy

Tests use pytest with custom markers:

- `e2e`: End-to-end tests requiring external services
- `mock`: Tests using mocked dependencies
- `integration`: Integration tests
- `unit`: Unit tests

Run tests selectively:
```bash
# Skip tests requiring running services
uv run pytest -m "not e2e and not integration"

# Only unit tests
uv run pytest -m unit

# Test both sync and async variants
# Include edge cases (empty inputs, None values, invalid types)
```

## Module Organization

### Core Modules (`libs/core/src/serapeum/core/`)
- `base/`: Base abstractions (BaseLLM, BaseEmbedding, base protocols)
- `llms/`: LLM classes and orchestrators
- `embeddings/`: Embedding interfaces and utilities
- `tools/`: Tool abstractions and callable tools
- `schema/`: Common data structures and types
- `bridge/`: Integration utilities

### Provider Modules (`libs/providers/*/src/serapeum/providers/*/`)
- Each provider is a separate package (e.g., `serapeum.providers.ollama`)
- Structure:
  ```
  libs/providers/{provider-name}/
  ├── src/serapeum/providers/{provider_name}/
  │   ├── __init__.py
  │   ├── llm.py          # Chat/completion models
  │   ├── embeddings.py   # Embedding models (if available)
  │   └── shared/         # Shared utilities
  │       ├── client.py   # HTTP client, config
  │       └── errors.py   # Provider-specific errors
  ├── tests/
  ├── pyproject.toml
  └── README.md
  ```
- Implement provider-specific LLM and embedding classes
- Extend from core abstractions
- Include comprehensive README.md with setup and examples

### Storage and Stores (`000stores/`)
- Vector stores, document stores, and chat stores
- Key-value storage abstractions
- Storage context management

## Naming Conventions
- Classes: `PascalCase` (e.g., `OllamaEmbedding`, `TextCompletionLLM`)
- Functions/Methods: `snake_case` (e.g., `get_text_embedding`, `_format_query`)
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_private_attr`
- Pydantic private attributes: Use `PrivateAttr()` with underscore prefix
- Async methods: prefix with `a` (e.g., `aget_query_embedding`)

## Import Organization
1. Future imports (`from __future__ import annotations`)
2. Standard library imports
3. Third-party imports (grouped)
4. Local/internal imports (use relative imports within packages)

Example:
```python
from __future__ import annotations
from typing import Any, Sequence

from pydantic import Field, PrivateAttr

import ollama as ollama_sdk
from serapeum.core.base.embeddings import BaseEmbedding
```

## Key Design Principles
- **Modularity**: Keep components loosely coupled and highly cohesive
- **Documentation**: Over-document rather than under-document
- **Type Safety**: Use type hints everywhere, leverage Pydantic for validation
- **Dual Mode**: Support both sync and async operations consistently
- **Extensibility**: Design for easy extension by users (clear base classes, documented hooks)
- **Error Messages**: Make errors actionable with clear guidance
- **Provider Organization**: Group all provider features together (LLM + embeddings + utilities)

## Common Implementation Patterns

### Embedding Classes
- Extend `BaseEmbedding` from `serapeum.core.base.embeddings`
- Implement `_get_text_embedding()`, `_aget_text_embedding()`
- Implement `_get_query_embedding()`, `_aget_query_embedding()`
- Support batch operations with `_get_text_embeddings()`, `_aget_text_embeddings()`
- Use Pydantic for configuration
- Initialize clients in `@model_validator(mode="after")`

### LLM Classes
- Extend `FunctionCallingLLM` from `serapeum.core.llms.function_calling`
- Support streaming with generators/async generators
- Implement tool calling if supported by provider
- Handle both chat and completion modes where applicable
- Validate structured outputs against schemas
- Follow async/streaming patterns from Ollama implementation

### Tool Creation

Tools are created using `CallableTool`:

```python
from serapeum.core.tools import CallableTool

# From function
def my_tool(arg: str) -> str:
    """Tool description."""
    return f"Result: {arg}"

tool = CallableTool.from_function(my_tool)

# From Pydantic model
from pydantic import BaseModel

class MyToolInput(BaseModel):
    arg: str

tool = CallableTool.from_model(MyToolInput, my_tool)
```

The system handles:
- Automatic JSON schema generation
- Sync/async bridging
- Input validation and output parsing

### Structured Outputs

Use `LLM.structured_predict()` to force structured outputs:

```python
from pydantic import BaseModel

class Output(BaseModel):
    field: str

result = llm.structured_predict("prompt", output_schema=Output)
```

Supports:
- Direct model prediction
- Streaming structured outputs
- Error handling and retry mechanisms

## Adding New Provider Integrations

New provider packages should be organized by provider (following industry standard pattern):

### Implementation Steps

1. **Create package structure** in `libs/providers/{provider}/`
2. **Inherit from `FunctionCallingLLM`** in `serapeum.core.llms.function_calling`
3. **Implement required abstract methods** for chat/completion
4. **Follow async/streaming patterns** from Ollama implementation
5. **Add to workspace** in root `pyproject.toml`:
   ```toml
   [tool.uv.workspace]
   members = ["libs/core", "libs/providers/*"]

   [tool.uv.sources]
   serapeum-{provider} = { workspace = true }
   ```
6. **Create package pyproject.toml** with:
   ```toml
   [project]
   name = "serapeum-{provider}"
   dependencies = ["serapeum-core", "{provider-sdk}"]

   [tool.uv.sources]
   serapeum-core = { workspace = true }
   ```

### Why Provider-Based Organization?

- **Shared Infrastructure**: All provider features share client, auth, error handling
- **Single Installation**: Users install one package per provider
- **Co-located Code**: Related features are maintained together
- **Matches Industry Standard**: LangChain and other frameworks organize by provider
- Example: `serapeum-openai` would contain ChatOpenAI, OpenAIEmbeddings, etc.

## Important Architectural Decisions

Key decisions documented in `docs/architecture/`:
1. **Provider-based organization** (not feature-type based)
2. **libs/ directory** for all packages (industry standard for monorepos)
3. **uv workspaces** for unified dependency management
4. **Namespace packages** for `serapeum.*` hierarchy
5. **src layout** within each package for proper isolation

## Documentation Files
- Keep `README.md` files up-to-date in each package
- Update `docs/` when adding new features or modules
- Use Mermaid diagrams for architecture documentation in `docs/architecture/`
- Follow existing documentation structure and style
- Link documentation within mkdocs nav structure

## Response Guidelines
- **Do NOT create multiple markdown files** to explain analysis, decisions, or chat summaries unless explicitly requested by the user.
- Provide explanations and analysis inline in conversation, not as separate documentation files
- Only create documentation files when explicitly requested or when adding new features that require user-facing docs
- Keep responses concise and actionable
- Focus on code changes and direct answers rather than creating auxiliary markdown files


