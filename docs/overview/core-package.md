# Codebase Map

This page summarizes the main modules, key classes, and the public API surface of the `serapeum` package.

## Packages and Modules

### Core Framework (`serapeum.core`)

#### Base Abstractions

- **serapeum.core.base.llms**
  - `BaseLLM`: Foundation protocol for all LLM backends with sync/async chat and completion, streaming endpoints, and message conversion helpers
  - Core data models: `Message`, `MessageList`, `ChatResponse`, `CompletionResponse`, `Metadata`, `MessageRole`
  - Multimodal support: `TextChunk`, `Image`, `Audio`
  - Utilities for adapting chat endpoints to completion-style calls

- **serapeum.core.base.embeddings**
  - `BaseEmbedding`: Foundation protocol for embedding models
  - Core embedding types: `NodeType`, `BaseNode`, `LinkedNodes`, `NodeInfo`, `MetadataMode`
  - Utilities for working with document nodes and embeddings

#### LLM Layer

- **serapeum.core.llms**
  - `LLM`: High-level LLM orchestration built on `BaseLLM` with prompt/message formatting and structured prediction to Pydantic models
  - `FunctionCallingLLM`: Tool-calling specialization with chat with tools, tool call extraction/validation, and predict-and-call helpers
  - `StructuredOutputLLM`: Wrapper that forces structured outputs (`BaseModel`) from any LLM while keeping chat/completion interfaces
  - `ChatToCompletionMixin`: Adapter for using chat models in completion mode
  - Sync/async/streaming support across all abstractions

- **serapeum.core.llms.orchestrators**
  - `ToolOrchestratingLLM`: Composes prompts, an LLM, and a toolset to drive structured tool-calling conversations
  - `TextCompletionLLM`: Text-completion style orchestration utilities
  - `StreamingObjectProcessor`: Handles streaming structured outputs
  - Support for sync/async operations with streaming

#### Embeddings Layer

- **serapeum.core.embeddings**
  - `MockEmbedding`: Testing/development embedding implementation
  - Embedding utilities and helpers
  - Integration with node types for document processing

#### Tools System

- **serapeum.core.tools**
  - `BaseTool` / `AsyncBaseTool`: Core tool protocols
  - `CallableTool`: Create tools from Python functions or Pydantic models with automatic schema generation
  - `ToolMetadata`: Tool metadata and JSON schema utilities
  - `ToolOutput` / `ToolCallArguments`: Tool execution types
  - Automatic sync/async bridging and output parsing

#### Prompts

- **serapeum.core.prompts**
  - `PromptTemplate`: String-based prompts with variable/function mappings
  - `ChatPromptTemplate`: Message-based prompts for chat interfaces
  - Prompt-related utilities and type definitions

#### Output Parsing

- **serapeum.core.output_parsers**
  - `PydanticParser`: Parse LLM outputs into Pydantic models
  - Output parser protocols and base classes
  - Error handling and retry mechanisms for robust parsing

#### Chat Support

- **serapeum.core.chat**
  - `AgentChatResponse`: Aggregates model/tool outputs with sync/async streaming
  - Utilities for managing conversation state
  - Tool output parsing and aggregation

#### Configuration & Types

- **serapeum.core.configs**
  - `Configs`: Global configuration object
  - Default values and settings used across the framework

- **serapeum.core.types**
  - `SerializableModel`: Base model with JSON/pickle serialization helpers
  - `Model`: Pydantic model base
  - `StructuredLLMMode`: Enum for structured output modes

- **serapeum.core.utils**
  - Common utilities: sync/async helpers, base utilities
  - Shared functionality across modules

### Provider Integrations

Serapeum supports multiple LLM providers through dedicated integration packages. Each provider package contains all features that provider offers (LLM, embeddings, etc.).

**Available Providers:**

- **serapeum.ollama** - Complete Ollama integration with chat, completion, tool calling, structured outputs, and embeddings
- **serapeum.openai** - OpenAI API integration (in development)
- **serapeum.azure-openai** - Azure OpenAI Service integration (in development)

For detailed information about providers, installation, configuration, and usage examples, see the **[Provider Integrations Guide](providers.md)**.

## Key Public Classes

### LLM Abstractions
- `serapeum.core.base.llms.BaseLLM` - Base LLM protocol
- `serapeum.core.llms.LLM` - High-level LLM orchestration
- `serapeum.core.llms.FunctionCallingLLM` - Tool-calling LLM specialization
- `serapeum.core.llms.StructuredOutputLLM` - Structured output wrapper

### Orchestration
- `serapeum.core.llms.orchestrators.ToolOrchestratingLLM` - Tool-calling orchestrator
- `serapeum.core.llms.orchestrators.TextCompletionLLM` - Text completion orchestrator

### Tools
- `serapeum.core.tools.CallableTool` - Function/model-based tools
- `serapeum.core.tools.BaseTool` - Base tool protocol
- `serapeum.core.tools.AsyncBaseTool` - Async tool protocol

### Data Types
- `serapeum.core.base.llms.types.Message` - Individual messages
- `serapeum.core.base.llms.types.MessageList` - Message sequences
- `serapeum.core.base.llms.types.ChatResponse` - Chat responses
- `serapeum.core.base.llms.types.CompletionResponse` - Completion responses
- `serapeum.core.base.llms.types.Metadata` - LLM metadata

### Prompts
- `serapeum.core.prompts.PromptTemplate` - String templates
- `serapeum.core.prompts.ChatPromptTemplate` - Chat templates

### Base Models
- `serapeum.core.types.SerializableModel` - Serialization base
- `serapeum.core.types.Model` - Pydantic model base

### Embeddings
- `serapeum.core.base.embeddings.BaseEmbedding` - Base embedding protocol
- `serapeum.core.embeddings.MockEmbedding` - Mock implementation

### Provider Implementations
- `serapeum.ollama.Ollama` - Ollama LLM implementation
- `serapeum.ollama.OllamaEmbedding` - Ollama embeddings implementation

See the [Provider Integrations Guide](providers.md) for complete documentation on all providers.

## Representative Public Methods

### BaseLLM
- `chat(messages, **kwargs) → ChatResponse`
- `complete(prompt, formatted=False, **kwargs) → CompletionResponse`
- `stream_chat(...) → ChatResponseGen`
- `stream_complete(...) → CompletionResponseGen`
- `achat(...) → ChatResponse` (async)
- `acomplete(...) → CompletionResponse` (async)
- `astream_chat(...) → ChatResponseAsyncGen` (async)
- `astream_complete(...) → CompletionResponseAsyncGen` (async)

### LLM
- `predict(prompt: PromptTemplate, **kwargs) → str`
- `stream(prompt, **kwargs) → CompletionResponseGen`
- `apredict(...) → str` (async)
- `astream(...) → CompletionResponseAsyncGen` (async)
- `structured_predict(output_cls: type[BaseModel], prompt, **kwargs) → BaseModel`
- `stream_structured_predict(...) → Generator[BaseModel, None, None]`
- `astructured_predict(...) → BaseModel` (async)
- `astream_structured_predict(...) → AsyncGenerator[BaseModel, None]` (async)

### FunctionCallingLLM
- `chat_with_tools(tools, user_msg=None, chat_history=None, **kwargs) → ChatResponse`
- `predict_and_call(tools, user_msg=None, chat_history=None, **kwargs) → AgentChatResponse`
- `get_tool_calls_from_response(response, error_on_no_tool_call=True) → list[ToolCallArguments]`
- `stream_chat_with_tools(...) → ChatResponseGen`
- `astream_chat_with_tools(...) → ChatResponseAsyncGen` (async)

### CallableTool
- `from_function(func, name=None, description=None, **kwargs) → CallableTool` (class method)
- `from_model(model_cls, fn, name=None, description=None, **kwargs) → CallableTool` (class method)
- `call(input, **kwargs) → ToolOutput`
- `acall(input, **kwargs) → ToolOutput` (async)

### ToolOrchestratingLLM
- `__call__(**prompt_args, llm_kwargs=None) → BaseModel | Any`
- `acall(**prompt_args, llm_kwargs=None) → BaseModel | Any` (async)
- `stream_call(**prompt_args, llm_kwargs=None) → Generator`
- `astream_call(**prompt_args, llm_kwargs=None) → AsyncGenerator` (async)

### BaseEmbedding
- `get_text_embedding(text: str) → list[float]`
- `get_query_embedding(query: str) → list[float]`
- `get_text_embeddings(texts: list[str]) → list[list[float]]`
- `aget_text_embedding(text: str) → list[float]` (async)
- `aget_query_embedding(query: str) → list[float]` (async)
- `aget_text_embeddings(texts: list[str]) → list[list[float]]` (async)

## Data Flow (High Level)

### Basic LLM Flow
```
User input/messages
  → Prompt building (PromptTemplate / ChatPromptTemplate)
  → LLM (LLM or concrete provider like Ollama)
  → ChatResponse / CompletionResponse
```

### Structured Output Flow
```
User input
  → PromptTemplate
  → LLM.structured_predict(output_cls=MyModel, ...)
  → Pydantic BaseModel instance
```

### Tool-Calling Flow
```
User message
  → FunctionCallingLLM.predict_and_call(tools=[...])
  → LLM predicts tool calls
  → Tools executed (BaseTool/CallableTool)
  → ToolOutput aggregated
  → AgentChatResponse
```

### Orchestrated Tool Flow
```
User input
  → ToolOrchestratingLLM(llm=..., tools=[...], prompt=...)
  → Automatic tool selection and execution
  → Structured output (if output_cls specified)
```

### Embedding Flow
```
Documents/queries
  → BaseEmbedding.get_text_embeddings(texts)
  → Vector embeddings (list[list[float]])
  → Use for similarity search, RAG, etc.
```

See the [Architecture](../architecture/) section for diagrams and deeper internals, and the API Reference for exhaustive signatures.

## Architecture & API Overview

This section provides visual diagrams of the Serapeum architecture and public API.

### Layered Architecture

The framework follows a layered architecture from base abstractions to high-level orchestration:

```mermaid
graph TB
    subgraph "User Code Layer"
        U1[Your Application Code]
    end

    subgraph "Orchestration Layer"
        O1[ToolOrchestratingLLM<br/>Multi-step tool workflows]
        O2[TextCompletionLLM<br/>Text completion utilities]
    end

    subgraph "LLM Abstraction Layer"
        L1[LLM<br/>predict, structured_predict]
        L2[FunctionCallingLLM<br/>chat_with_tools, predict_and_call]
        L3[StructuredOutputLLM<br/>Force structured outputs]
    end

    subgraph "Base Protocol Layer"
        B1[BaseLLM<br/>chat, complete, stream_chat]
        B2[BaseEmbedding<br/>get_text_embedding]
    end

    subgraph "Provider Layer"
        P1[Ollama<br/>Local inference]
        P2[OpenAI<br/>coming soon]
        P3[Azure OpenAI<br/>coming soon]
        E1[OllamaEmbedding<br/>Local embeddings]
    end

    subgraph "Supporting Components"
        S1[PromptTemplate<br/>String templates]
        S2[CallableTool<br/>Function tools]
        S3[Message & MessageList<br/>Data models]
    end

    U1 --> O1
    U1 --> O2
    U1 --> L1
    U1 --> L2
    U1 --> S1
    U1 --> S2

    O1 --> L2
    O1 --> S1
    O1 --> S2

    L1 --> B1
    L2 --> B1
    L3 --> L1

    P1 -.implements.-> L2
    P2 -.implements.-> L2
    P3 -.implements.-> L2
    E1 -.implements.-> B2

    L1 --> S3
    L2 --> S3

    style U1 fill:#e1f5ff
    style O1 fill:#fff4e1
    style O2 fill:#fff4e1
    style L1 fill:#e8f5e9
    style L2 fill:#e8f5e9
    style L3 fill:#e8f5e9
    style B1 fill:#f3e5f5
    style B2 fill:#f3e5f5
    style P1 fill:#ffe5e5
    style P2 fill:#ffe5e5
    style P3 fill:#ffe5e5
    style E1 fill:#ffe5e5
```

### Core Public API

Key classes and their primary methods:

```mermaid
classDiagram
    class BaseLLM {
        <<abstract>>
        +chat(messages) ChatResponse
        +complete(prompt) CompletionResponse
        +stream_chat(messages) Generator
        +stream_complete(prompt) Generator
        +achat(messages) ChatResponse
        +acomplete(prompt) CompletionResponse
        +astream_chat(messages) AsyncGen
        +astream_complete(prompt) AsyncGen
    }

    class LLM {
        +predict(prompt, **kwargs) str
        +stream(prompt, **kwargs) Generator
        +apredict(prompt, **kwargs) str
        +astream(prompt, **kwargs) AsyncGen
        +structured_predict(output_cls, prompt, **kwargs) BaseModel
        +stream_structured_predict(...) Generator
        +astructured_predict(...) BaseModel
        +astream_structured_predict(...) AsyncGen
    }

    class FunctionCallingLLM {
        +chat_with_tools(tools, user_msg, chat_history) ChatResponse
        +predict_and_call(tools, user_msg, chat_history) AgentChatResponse
        +get_tool_calls_from_response(response) List~ToolCallArguments~
        +stream_chat_with_tools(tools, ...) Generator
        +astream_chat_with_tools(tools, ...) AsyncGen
    }

    class ToolOrchestratingLLM {
        -llm: FunctionCallingLLM
        -tools: List~BaseTool~
        -prompt: PromptTemplate
        +__call__(**prompt_args) Any
        +acall(**prompt_args) Any
        +stream_call(**prompt_args) Generator
        +astream_call(**prompt_args) AsyncGen
    }

    class CallableTool {
        <<interface>>
        +from_function(func) CallableTool$
        +from_model(model_cls, fn) CallableTool$
        +call(input, **kwargs) ToolOutput
        +acall(input, **kwargs) ToolOutput
    }

    class BaseEmbedding {
        <<abstract>>
        +get_text_embedding(text) List~float~
        +get_query_embedding(query) List~float~
        +get_text_embeddings(texts) List~List~float~~
        +aget_text_embedding(text) List~float~
        +aget_query_embedding(query) List~float~
        +aget_text_embeddings(texts) List~List~float~~
    }

    class PromptTemplate {
        +__init__(template, **kwargs)
        +format(**kwargs) str
        +format_messages(**kwargs) MessageList
    }

    class Message {
        +role: MessageRole
        +content: str | List
        +metadata: Dict
    }

    class MessageList {
        +messages: List~Message~
        +from_list(messages) MessageList$
        +to_list() List
    }

    BaseLLM <|-- LLM
    LLM <|-- FunctionCallingLLM
    ToolOrchestratingLLM --> FunctionCallingLLM : uses
    ToolOrchestratingLLM --> CallableTool : orchestrates
    ToolOrchestratingLLM --> PromptTemplate : composes
    LLM --> MessageList : uses
    LLM --> Message : uses
```

### Provider Implementation

How providers implement the core abstractions:

```mermaid
classDiagram
    class FunctionCallingLLM {
        <<abstract>>
        +_chat(messages) ChatResponse*
        +_complete(prompt) CompletionResponse*
        +_stream_chat(messages) Generator*
        +_stream_complete(prompt) Generator*
    }

    class BaseEmbedding {
        <<abstract>>
        +_get_text_embedding(text) List~float~*
        +_get_query_embedding(query) List~float~*
    }

    class Ollama {
        +model: str
        +base_url: str
        +temperature: float
        +_chat(messages) ChatResponse
        +_complete(prompt) CompletionResponse
        +_stream_chat(messages) Generator
        +_achat(messages) ChatResponse
    }

    class OllamaEmbedding {
        +model_name: str
        +batch_size: int
        +_get_text_embedding(text) List~float~
        +_get_text_embeddings(texts) List~List~float~~
        +_aget_text_embedding(text) List~float~
    }

    class OpenAI {
        <<coming soon>>
        +model: str
        +api_key: str
    }

    class AzureOpenAI {
        <<coming soon>>
        +deployment_name: str
        +endpoint: str
    }

    FunctionCallingLLM <|-- Ollama
    FunctionCallingLLM <|-- OpenAI
    FunctionCallingLLM <|-- AzureOpenAI
    BaseEmbedding <|-- OllamaEmbedding

    note for Ollama "Full local inference\nNo API costs\nPrivacy-focused"
    note for OpenAI "Cloud-based\nGPT-4, GPT-3.5\nPay-as-you-go"
```

### Component Interaction Flow

How components interact in a typical workflow:

```mermaid
sequenceDiagram
    participant User as User Code
    participant Orch as ToolOrchestratingLLM
    participant Prompt as PromptTemplate
    participant LLM as FunctionCallingLLM<br/>(e.g., Ollama)
    participant Tool as CallableTool
    participant Parser as OutputParser

    User->>Orch: orchestrator(query="What's 5+3?")
    Orch->>Prompt: format(query="What's 5+3?")
    Prompt-->>Orch: formatted messages
    Orch->>LLM: chat_with_tools(messages, tools)
    LLM-->>Orch: ChatResponse (tool call)
    Orch->>Orch: get_tool_calls_from_response()
    Orch->>Tool: call(operation="add", a=5, b=3)
    Tool-->>Orch: ToolOutput(result=8)
    Orch->>LLM: chat(messages + tool results)
    LLM-->>Orch: ChatResponse (final answer)
    alt structured output
        Orch->>Parser: parse(response, output_cls)
        Parser-->>Orch: Pydantic model
    end
    Orch-->>User: result

    Note over User,Parser: All steps support sync, async, and streaming
```

### Data Type Hierarchy

Core data models used across the API:

```mermaid
classDiagram
    class SerializableModel {
        <<base>>
        +to_dict() Dict
        +to_json() str
        +from_dict(data) Self$
    }

    class Message {
        +role: MessageRole
        +content: str | List~ContentBlock~
        +metadata: Dict
        +additional_kwargs: Dict
    }

    class MessageList {
        +messages: List~Message~
        +from_list(messages) MessageList$
        +to_list() List~Dict~
        +append(message) None
    }

    class ChatResponse {
        +message: Message
        +raw: Any
        +delta: str
        +metadata: Metadata
    }

    class CompletionResponse {
        +text: str
        +raw: Any
        +delta: str
        +metadata: Metadata
    }

    class AgentChatResponse {
        +response: str
        +sources: List~ToolOutput~
        +source_nodes: List
        +metadata: Dict
    }

    class ToolOutput {
        +content: str
        +tool_name: str
        +raw_input: Dict
        +raw_output: Any
        +is_error: bool
    }

    class ToolCallArguments {
        +tool_name: str
        +tool_kwargs: Dict
        +tool_id: str
    }

    SerializableModel <|-- Message
    SerializableModel <|-- MessageList
    SerializableModel <|-- ChatResponse
    SerializableModel <|-- CompletionResponse
    SerializableModel <|-- AgentChatResponse
    SerializableModel <|-- ToolOutput

    MessageList --> Message : contains
    ChatResponse --> Message : contains
    AgentChatResponse --> ToolOutput : contains
```

---

## Understanding the Diagrams

### How to Read These Diagrams

**Layered Architecture Diagram:**
- Shows the flow from your application code down to providers
- Each layer builds on the layer below it
- Colors indicate different architectural layers
- Dotted lines mean "implements" (providers implementing abstractions)

**Core Public API Diagram:**
- Shows the main classes you'll use in your code
- Methods are listed showing signatures and return types
- `$` symbol indicates static/class methods
- `~Type~` notation shows generic type parameters

**Provider Implementation Diagram:**
- Shows how to extend Serapeum with new providers
- Abstract methods marked with `*` must be implemented
- Notes provide quick provider characteristics

**Component Interaction Flow:**
- Shows the sequence of calls in a typical workflow
- Read top-to-bottom to follow the execution flow
- `alt` blocks show conditional paths

**Data Type Hierarchy:**
- Shows the relationships between data models
- All models inherit from `SerializableModel`
- Arrows show containment relationships

### Common Patterns

**Sync/Async/Streaming:**
- Every LLM method has 4 variants:
  - `method()` - Synchronous
  - `amethod()` - Asynchronous
  - `stream_method()` - Synchronous streaming
  - `astream_method()` - Asynchronous streaming

**Tool Integration:**
- Tools created via `CallableTool.from_function()` or `from_model()`
- Orchestrators automatically handle tool calling
- All tool operations return `ToolOutput`

**Structured Outputs:**
- Use `structured_predict()` with Pydantic models
- Works with both streaming and non-streaming
- Automatic validation and parsing

---

## Adding New Components

When extending Serapeum:

1. **New Provider**: Implement `FunctionCallingLLM` and optionally `BaseEmbedding`
2. **New Tool**: Create via `CallableTool.from_function()` or extend `BaseTool`
3. **New Orchestrator**: Compose existing `FunctionCallingLLM` + `PromptTemplate` + tools
4. **New Parser**: Implement `BaseOutputParser` protocol
