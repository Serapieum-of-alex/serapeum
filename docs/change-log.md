## 0.1.0 (2026-02-16)

### Documentation Overhaul (#20)
- **Professional Home Page**: Added comprehensive landing page with feature cards, quick start examples, and architecture overview
- **Provider Documentation**:
    - Complete provider integrations overview with comparison table
    - Detailed Ollama provider guide covering installation, features, examples, and RAG integration
    - Step-by-step implementation guide for adding new providers with ready-to-use code templates
- **Enhanced Architecture Diagrams**: Added mermaid diagrams showing layered architecture, public API, component interactions, and data type hierarchy
- **Improved Navigation**: Reorganized documentation structure with provider subsections
- **Fixed Examples**: Corrected all code examples to use proper public API (chat_with_tools, parse)

### Embedding Support (#11)
- **BaseEmbedding Abstraction**: New core abstraction with sync/async interfaces
- **OllamaEmbedding Provider**: Full implementation supporting query, text, and batch operations
- **Embedding Data Models**: Added BaseNode, TextNode, ImageNode, LinkedNodes, and NodeInfo for document processing
- **Comprehensive Testing**: 3200+ lines of tests including unit, e2e, and validation tests
- **Performance Tools**: Added pytest-xdist for parallel testing and pytest-benchmark for performance testing

### Core Architecture (#3)
- **LLM Orchestration Framework**: Complete implementation of BaseLLM, FunctionCallingLLM, ToolOrchestratingLLM, and TextCompletionLLM
- **Tool System**: Robust tool execution with ToolExecutor, ToolCallArguments, and automatic schema validation
- **Sync/Async Support**: Full support for both synchronous and asynchronous operations
- **Streaming**: StreamingObjectProcessor for handling streaming structured outputs
- **Flexible Models**: MessageList, FlexibleModel, and comprehensive type system

### Dependency Management (#19)
- Removed httpx dependency in favor of stdlib urllib.request
- Added version constraints for pydantic and numpy for better compatibility
- Improved E2E test handling when optional dependencies unavailable
- Fixed sdist packaging configuration

### Schema & Tool Handling (#8)
- Moved schema helpers to core.utils.schemas for better code organization
- Generated concise required-field descriptions in tool metadata
- Improved tool_calls handling in Ollama provider with proper guards

### Documentation Quality (#7, #9)
- Standardized docstrings across core, test, and provider modules
- Added module-level, class, and method documentation for consistency
- Moved SECURITY.md into developer guide
- Expanded README content for root, core, and ollama packages

### Package Restructure (#17)
**BREAKING CHANGE**: Major restructure of packages and APIs

- **Package Organization**:
    - Core moved to `libs/core`
    - Providers moved to `libs/providers/ollama`
