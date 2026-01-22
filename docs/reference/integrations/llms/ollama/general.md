# Ollama LLM Integration

This directory contains comprehensive documentation explaining the complete workflow of the `Ollama` class, from initialization to execution across various modes (chat, completion, streaming, tool calling, async).

## Overview

The `Ollama` class is a production-ready LLM integration that provides:
1. **Connection to Ollama server** (local or remote)
2. **Chat and completion APIs** with sync/async support
3. **Streaming responses** for real-time output
4. **Tool/function calling** for structured interactions
5. **Integration with orchestrators** (TextCompletionLLM, ToolOrchestratingLLM)

## Example Usage

### Basic Chat

```python
from serapeum.core.base.llms.models import Message, MessageRole
from serapeum.llms.ollama import Ollama

# Initialize Ollama
llm = Ollama(
    model="llama3.1",
    request_timeout=180,
)

# Send chat request
messages = [Message(role=MessageRole.USER, content="Say 'pong'.")]
response = llm.chat(messages)
print(response.message.content)  # "Pong!"
```

### With TextCompletionLLM

```python
from pydantic import BaseModel
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
from serapeum.llms.ollama import Ollama


class DummyModel(BaseModel):
    value: str


# Initialize Ollama
llm = Ollama(model="llama3.1", request_timeout=180)

# Create structured completion runner
text_llm = TextCompletionLLM(
    output_parser=PydanticParser(output_cls=DummyModel),
    prompt="Value: {value}",
    llm=llm,
)

# Execute and get structured output
result = text_llm(value="input")
# Returns: DummyModel(value="input")
```

### With ToolOrchestratingLLM

```python
from pydantic import BaseModel
from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


# Initialize Ollama
llm = Ollama(model="llama3.1", request_timeout=180)

# Create tool orchestrator
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create an album about {topic} with two random songs",
    llm=llm,
)

# Execute and get structured output via tool calling
result = tools_llm(topic="rock")
# Returns: Album(title="...", artist="...", songs=[...])
```

## Understanding the Workflow

### 1. [Execution Flow and Method Calls](./ollama_sequence.md)
Shows the chronological flow of method calls and interactions across all usage patterns.

**Best for**:
- Understanding the order of operations
- Seeing how Ollama communicates with the server
- Debugging execution flow
- Understanding integration patterns

**Key Flows**:
- Initialization phase (lazy client creation)
- Direct chat/completion calls
- Tool calling with schema conversion
- Streaming execution
- Integration with TextCompletionLLM and ToolOrchestratingLLM
- Async operations

### 2. [Architecture and Class Relationships](./ollama_class.md)
Illustrates the static structure, inheritance hierarchy, and relationships.

**Best for**:
- Understanding the architecture
- Seeing inheritance chain (BaseLLM → LLM → FunctionCallingLLM → Ollama)
- Identifying class responsibilities
- Understanding integration points

**Key Classes**:
- `Ollama`: Main LLM implementation
- `FunctionCallingLLM`: Tool calling abstraction
- `LLM`: High-level orchestration
- `BaseLLM`: Core interface
- `Client`/`AsyncClient`: HTTP communication
- Response models: `ChatResponse`, `CompletionResponse`, `Message`

### 3. [Data Transformations and Validation](./ollama_dataflow.md)
Tracks how data transforms through the system across different operation modes.

**Best for**:
- Understanding data transformations
- Identifying validation points
- Seeing error handling paths
- Understanding request/response formats

**Key Flows**:
- Initialization and configuration
- Chat request building and response parsing
- Completion via decorator pattern
- Tool schema conversion
- Streaming chunk processing
- Error handling pipelines

### 4. [Component Boundaries and Interactions](./ollama_components.md)
Shows component boundaries, responsibilities, and interaction patterns.

**Best for**:
- Understanding system architecture
- Seeing component responsibilities
- Identifying interaction patterns
- Understanding integration layers

**Key Components**:
- User space (application code)
- Ollama core (request building, response parsing, tool handling)
- Client layer (HTTP communication)
- Ollama server (model runtime, inference)
- Orchestrator layer (TextCompletionLLM, ToolOrchestratingLLM)

### 5. [Lifecycle and State Management](./ollama_state.md)
Depicts the lifecycle states, transitions, and state variables.

**Best for**:
- Understanding instance lifecycle
- Seeing state transitions
- Identifying error states
- Understanding concurrency considerations

**Key States**:
- Uninitialized → Configured (initialization)
- Configured → ClientInitialized (lazy client creation)
- Idle ↔ Processing* (request handling)
- Processing → Error → Idle (error handling)

### 6. [Usage Examples](./examples.md)
Comprehensive examples from real test cases.

**Best for**:
- Learning by example
- Understanding practical usage
- Seeing all API variants
- Integration patterns

**Key Examples**:
- Basic chat and completion
- Streaming operations
- Tool/function calling
- Integration with orchestrators
- Async operations
- Error handling

## Core Capabilities

### 1. Chat API
```
Direct conversation with the model:
- Single and multi-turn conversations
- System messages for context
- Image inputs (if model supports)
- Custom parameters (temperature, top_p, etc.)
```

### 2. Completion API
```
Text completion via decorator pattern:
- Converts prompt to chat message
- Delegates to chat API
- Extracts text from response
```

### 3. Streaming
```
Real-time response generation:
- Stream chat responses
- Stream completion responses
- Chunk-by-chunk processing
- Delta content access
```

### 4. Tool/Function Calling
```
Structured interactions with tools:
- Automatic schema conversion
- Single or parallel tool calls
- Tool call validation
- Streaming tool calls
```

### 5. Async Operations
```
Non-blocking execution:
- Async chat and completion
- Async streaming
- Concurrent request handling
- Separate async client per event loop
```

## Key Design Patterns

### 1. **Lazy Initialization**
Clients are created on first use, not during `__init__`:
```python
@property
def client(self) -> Client:
    if self._client is None:
        self._client = Client(host=self.base_url, timeout=self.request_timeout)
    return self._client
```

### 2. **Decorator Pattern**
Completion API wraps chat API for code reuse:
```python
@chat_to_completion_decorator
def complete(self, prompt: str, **kwargs) -> CompletionResponse:
    # Decorator handles conversion
    pass
```

### 3. **Template Method Pattern**
FunctionCallingLLM defines workflow, Ollama implements specifics:
```python
def chat_with_tools(self, messages, tools, **kwargs):
    prepared = self._prepare_chat_with_tools(messages, tools, **kwargs)  # Subclass
    response = self.chat(prepared)
    validated = self._validate_chat_with_tools_response(response, tools)  # Subclass
    return validated
```

### 4. **Adapter Pattern**
Ollama adapts between internal types and Ollama server format:
- `Message` → Ollama message dict
- `BaseTool` → Ollama tool schema
- Raw response dict → `ChatResponse`/`CompletionResponse`

## Integration Architecture

```
User Application
    ↓
ToolOrchestratingLLM / TextCompletionLLM
    ↓
Ollama
    ↓
Client / AsyncClient
    ↓
Ollama Server (HTTP)
    ↓
Model Runtime (llama3.1, etc.)
```

### TextCompletionLLM Integration
```
1. Formats prompt with variables
2. Checks is_chat_model → True
3. Calls Ollama.chat()
4. Parses response with PydanticParser
5. Returns validated model instance
```

### ToolOrchestratingLLM Integration
```
1. Converts output_cls to CallableTool
2. Formats prompt with variables
3. Calls Ollama.chat_with_tools()
4. Ollama converts tool to schema
5. Server returns tool_calls
6. Executes tool to create instance
7. Returns model instance(s)
```

## Performance Considerations

1. **Client Reuse**: Client created once and reused for all requests
2. **Async Support**: Separate async client for concurrent operations
3. **Streaming**: Reduces latency for long responses
4. **Connection Pooling**: HTTP client handles connection reuse
5. **Lazy Initialization**: Only create clients when needed

## Configuration Options

### Essential
- `model`: Model name (e.g., "llama3.1")
- `base_url`: Ollama server URL (default: "http://localhost:11434")
- `request_timeout`: Timeout in seconds (default: 60.0)

### Generation
- `temperature`: Sampling temperature (0.0-1.0, default: 0.75)
- `context_window`: Maximum context tokens (default: 3900)
- `json_mode`: Force JSON output (default: False)

### Advanced
- `keep_alive`: Model keep-alive duration (default: None)
- `additional_kwargs`: Additional Ollama options
- `client`: Pre-configured client (default: None, lazy-created)
- `async_client`: Pre-configured async client (default: None, lazy-created)

## Error Handling

### Network Errors
```
TimeoutError: Request timeout exceeded
ConnectionError: Cannot reach Ollama server
HTTPError: Server returned error status
```

### Parsing Errors
```
JSONDecodeError: Invalid JSON response
KeyError: Missing required field in response
ValueError: Invalid response format
```

### Configuration Errors
```
ValueError: Invalid model or URL
TypeError: Missing required field
AssertionError: Invalid parameter value
```

## Prerequisites

### Server Requirements
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1

# Start server (runs on port 11434 by default)
ollama serve
```

### Python Requirements
```bash
# Install serapeum-ollama
uv pip install serapeum-ollama

# Or install from source
uv pip install -e serapeum-integrations/llms/serapeum-ollama
```

## Common Patterns

### Pattern 1: Reusable Instance
```python
# Create once
llm = Ollama(model="llama3.1", request_timeout=180)

# Reuse many times
response1 = llm.chat(messages1)
response2 = llm.chat(messages2)
```

### Pattern 2: Streaming for Long Responses
```python
llm = Ollama(model="llama3.1", request_timeout=180)

for chunk in llm.stream_chat(messages):
    print(chunk.message.content, end="", flush=True)
```

### Pattern 3: Tool Calling for Structured Outputs
```python
llm = Ollama(model="llama3.1", request_timeout=180)

# Define tool from Pydantic model
tool = CallableTool.from_model(MyModel)

# Get structured output via tool calling
response = llm.chat_with_tools(messages, tools=[tool])
```

### Pattern 4: Async for Concurrency
```python
llm = Ollama(model="llama3.1", request_timeout=180)

# Process multiple requests concurrently
tasks = [llm.achat(messages) for messages in message_list]
responses = await asyncio.gather(*tasks)
```

## Troubleshooting

### Issue: Connection Refused
```
Solution: Ensure Ollama server is running
  $ ollama serve
```

### Issue: Model Not Found
```
Solution: Pull the model first
  $ ollama pull llama3.1
```

### Issue: Timeout
```
Solution: Increase request_timeout
  llm = Ollama(model="llama3.1", request_timeout=300)
```

### Issue: Invalid JSON Response
```
Solution: Enable json_mode
  llm = Ollama(model="llama3.1", json_mode=True)
```

## Next Steps

1. **Start with [Examples](./examples.md)** for practical usage patterns
2. **Review [Sequence Diagrams](./ollama_sequence.md)** to understand execution flow
3. **Study [Class Diagram](./ollama_class.md)** to understand architecture
4. **Explore [Data Flow](./ollama_dataflow.md)** to understand transformations
5. **Check [State Management](./ollama_state.md)** for lifecycle details

## See Also

- [Serapeum Core Documentation](../../core/) - Core abstractions and base classes
- [TextCompletionLLM](../../core/structured_tools/text_completion_llm/) - Structured completion orchestrator
- [ToolOrchestratingLLM](../../core/structured_tools/tool_orchestrating_llm/) - Tool-based orchestrator
- [Ollama Official Documentation](https://ollama.com/docs) - Ollama server documentation
