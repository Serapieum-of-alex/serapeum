# LLM Classes Comparison

This document provides a comprehensive comparison of the four main LLM classes in Serapeum's core library, explaining their purposes, relationships, and when to use each one.

## Overview

Serapeum provides four distinct LLM classes organized across two architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│  Orchestration Layer (High-level workflows)                 │
├─────────────────────────────────────────────────────────────┤
│  ToolOrchestratingLLM       │  TextCompletionLLM             │
│  (uses function calling)    │  (uses text parsing)          │
│  - Converts models to tools │  - Binds prompt+parser+LLM    │
│  - Executes tool calls      │  - Parses raw text output     │
│  - Returns Pydantic models  │  - Returns Pydantic models    │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
                            │
┌─────────────────────────────────────────────────────────────┐
│  LLM Layer (Core abstractions)                              │
├─────────────────────────────────────────────────────────────┤
│  FunctionCallingLLM         │  StructuredOutputLLM                 │
│  (base for providers)       │  (wrapper for structured IO)  │
│  - Tool calling interface   │  - Forces Pydantic outputs    │
│  - Provider implementations │  - Wraps any LLM              │
│  - Abstract methods         │  - Format conversion          │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Comparison

### 1. FunctionCallingLLM

**Location**: `libs/core/src/serapeum/core/llms/function_calling.py:21`
**Layer**: LLM Layer (core abstraction)
**Type**: Base class for provider implementations

#### Purpose
Provides the foundation for LLM providers that support function/tool calling. This is an abstract base class that concrete provider implementations (like Ollama, OpenAI) should inherit from.

#### Key Features
- Extends the base `LLM` class with tool-calling capabilities
- Provides convenience methods for tool workflows:
  - `chat_with_tools()` - Chat with function calling (sync)
  - `achat_with_tools()` - Chat with function calling (async)
  - `stream_chat_with_tools()` - Streaming chat with tools (sync)
  - `astream_chat_with_tools()` - Streaming chat with tools (async)
  - `predict_and_call()` - Predict and execute tool (sync)
  - `apredict_and_call()` - Predict and execute tool (async)
  - `get_tool_calls_from_response()` - Extract tool calls from response
- Abstract method `_prepare_chat_with_tools()` that providers must implement

#### When to Use
- **You're implementing a new provider** (e.g., OpenAI, Anthropic, Cohere)
- You need the base functionality for tool/function calling
- You're building low-level LLM integrations

#### Example
```python
from serapeum.core.llms import FunctionCallingLLM

class MyProviderLLM(FunctionCallingLLM):
    """Custom provider implementation."""

    def _prepare_chat_with_tools(self, tools, **kwargs):
        # Convert tools to provider-specific format
        tool_schemas = [tool.to_json_schema() for tool in tools]
        return {
            "messages": kwargs.get("chat_history", []),
            "tools": tool_schemas,
        }

    def get_tool_calls_from_response(self, response, **kwargs):
        # Extract tool calls from provider response
        return response.tool_calls
```

---

### 2. StructuredOutputLLM

**Location**: `libs/core/src/serapeum/core/llms/structured_output_llm.py:25`
**Layer**: LLM Layer (wrapper)
**Type**: Wrapper class for structured outputs

#### Purpose
Wraps an existing LLM to force all outputs into a specific Pydantic model format. Acts as an adapter that converts any LLM into a structured output generator.

#### Key Features
- Takes two inputs:
  - `llm`: Any LLM instance (base LLM, function-calling LLM, etc.)
  - `output_cls`: A Pydantic model class defining the output structure
- Delegates to the underlying LLM's `structured_predict()` method
- Converts all responses to JSON representations of the output model
- Maintains the same interface as the base LLM (chat, stream_chat, etc.)
- Supports streaming structured outputs

#### When to Use
- **You want to guarantee a specific output format** from any LLM
- You're wrapping an existing LLM to enforce schema compliance
- You need structured outputs without manually handling parsing

#### Example
```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
from serapeum.core.llms import StructuredOutputLLM
from serapeum.core.base.llms.types import Message

class PersonInfo(BaseModel):
    name: str
    age: int
    occupation: str

# Wrap an LLM to always return PersonInfo
base_llm = Ollama(model="llama3.1", request_timeout=90)
structured_llm = StructuredOutputLLM(
    llm=base_llm,
    output_cls=PersonInfo
)

# All responses will be PersonInfo instances
response = structured_llm.chat([
    Message(role="user", content="Tell me about Alice, a 30-year-old engineer")
])
print(response.raw)
PersonInfo(name='Alice', age=30, occupation='Engineer')
```

---

### 3. ToolOrchestratingLLM

**Location**: `libs/core/src/serapeum/core/llms/orchestrators/tool_based.py:33`
**Layer**: Orchestration Layer (high-level)
**Type**: Orchestrator for function-calling workflows

#### Purpose
High-level orchestrator that converts Pydantic models or Python functions into tools, executes them via function-calling, and returns structured outputs. This is the recommended way to get structured outputs from function-calling models.

#### Key Features
- **Automatic tool creation**: Converts Pydantic models OR Python functions to `CallableTool` instances
- **Full orchestration**: Handles prompt formatting, LLM invocation, tool execution, and output parsing
- **Flexible inputs**:
  - `output_cls`: Either a Pydantic model or a callable function
  - `prompt`: Template string or `BasePromptTemplate`
  - `llm`: A `FunctionCallingLLM` instance
- **Advanced capabilities**:
  - Streaming support via `stream_call()` and `astream_call()`
  - Parallel tool calls with `allow_parallel_tool_calls=True`
  - Custom tool selection with `tool_choice` parameter
- **Sync and async**: Both `__call__()` and `acall()` methods

#### When to Use
- **You want structured outputs from a function-calling model** (recommended approach)
- You're building applications that need reliable Pydantic outputs
- You want automatic tool creation from your data models
- You need streaming structured outputs
- You're using modern LLMs with function-calling support (GPT-4, Claude, Llama 3.1+)

#### Example with Pydantic Model
```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
from serapeum.core.llms import ToolOrchestratingLLM

class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: str
    temperature: float
    conditions: str

# Create orchestrator
weather_extractor = ToolOrchestratingLLM(
    output_cls=WeatherInfo,
    prompt="Extract weather information from: {text}",
    llm=Ollama(model="llama3.1"),
)

# Get structured output
result = weather_extractor(
    text="It's 72 degrees and sunny in San Francisco"
)
print(result)
WeatherInfo(location='San Francisco', temperature=72.0, conditions='sunny')
```

#### Example with Function
```python
from serapeum.llms.ollama import Ollama
from serapeum.core.llms import ToolOrchestratingLLM

def calculate_sum(a: int, b: int) -> dict:
    """Calculate the sum of two numbers."""
    return {"result": a + b}

# Create orchestrator with function
calculator = ToolOrchestratingLLM(
    output_cls=calculate_sum,
    prompt="Calculate the sum of {x} and {y}",
    llm=Ollama(model="llama3.1"),
)

result = calculator(x=5, y=3)
print(result)
{'result': 8}
```

#### Example with Streaming
```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
from serapeum.core.llms import ToolOrchestratingLLM

class Story(BaseModel):
    title: str
    content: str
    genre: str

story_generator = ToolOrchestratingLLM(
    output_cls=Story,
    prompt="Generate a short {genre} story",
    llm=Ollama(model="llama3.1", request_timeout=90),
)

# Stream partial results
for partial_story in story_generator.stream_call(genre="sci-fi"):
    print(partial_story)  # Progressively complete Story objects
```

---

### 4. TextCompletionLLM

**Location**: `libs/core/src/serapeum/core/llms/orchestrators/text_completion_llm.py:14`
**Layer**: Orchestration Layer (simpler alternative)
**Type**: Text-based structured output generator

#### Purpose
Provides structured outputs by parsing raw text completions (without using function calling). This is useful for models that don't support function calling or when you prefer text-based parsing.

#### Key Features
- **Simple pipeline**: Binds prompt + output parser + LLM together
- **Text-based parsing**: Uses `PydanticParser` to parse raw LLM output into Pydantic models
- **No function calling required**: Works with any LLM (chat or completion models)
- **Explicit parsing**: Uses output parsers to handle the conversion
- **Lightweight**: Less overhead than function-calling approaches

#### When to Use
- **Your LLM doesn't support function calling** (older models, smaller models)
- You prefer text-based parsing over function calling
- You want explicit control over the parsing logic
- You're working with completion-style models (non-chat)
- You need a simpler, more transparent approach

#### Example
```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM

class Task(BaseModel):
    title: str
    priority: int
    completed: bool

# Create text completion LLM
task_extractor = TextCompletionLLM(
    output_parser=PydanticParser(output_cls=Task),
    prompt="Extract task information from: {text}. Return as JSON.",
    llm=Ollama(model="llama3.1", request_timeout=90),
)

result = task_extractor(
    text="Finish the report - high priority, not done yet"
)
result
Task(title='Finish the report', priority=1, completed=False)
```

#### Example with Just output_cls
```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
from serapeum.core.llms import TextCompletionLLM

class Product(BaseModel):
    name: str
    price: float

# Parser is auto-created from output_cls
product_extractor = TextCompletionLLM(
    output_cls=Product,  # Parser created automatically
    prompt="Extract product: {description}",
    llm=Ollama(model="llama3.1", request_timeout=90),
)

result = product_extractor(description="iPhone 15 Pro - $999")
result
Product(name='iPhone 15 Pro', price=999.0)
```

---

## Comparison Matrix

| Feature | FunctionCallingLLM | StructuredOutputLLM | ToolOrchestratingLLM | TextCompletionLLM |
|---------|-------------------|---------------|---------------------|------------------|
| **Layer** | LLM | LLM | Orchestration | Orchestration |
| **Type** | Base class | Wrapper | Orchestrator | Pipeline |
| **Requires Function Calling** | N/A | No | Yes | No |
| **Primary Use Case** | Building providers | Enforcing output format | Structured outputs (recommended) | Text-based structured outputs |
| **Input** | N/A | LLM + output_cls | output_cls + prompt + LLM | prompt + parser + LLM |
| **Output** | ChatResponse | ChatResponse (with Pydantic in raw) | Pydantic model(s) | Pydantic model |
| **Streaming Support** | Yes | Yes | Yes | No |
| **Parallel Tool Calls** | N/A | No | Yes | No |
| **Complexity** | High (abstract) | Low | Medium | Low |
| **Flexibility** | High | Low | High | Medium |

## Decision Tree: Which Class Should I Use?

```
Are you implementing a new LLM provider?
├─ YES → Use FunctionCallingLLM (inherit from it)
└─ NO → Continue...

Do you need structured Pydantic outputs?
├─ NO → Use base LLM classes
└─ YES → Continue...

Does your LLM support function calling?
├─ NO → Use TextCompletionLLM
└─ YES → Continue...

Do you just want to wrap an existing LLM to enforce a format?
├─ YES → Use StructuredOutputLLM
└─ NO → Use ToolOrchestratingLLM (recommended for most use cases)
```

## Best Practices

### For Application Developers

1. **Default to `ToolOrchestratingLLM`** for structured outputs with modern LLMs
   - Most flexible and powerful
   - Handles tool creation automatically
   - Supports streaming and parallel calls

2. **Use `TextCompletionLLM`** when:
   - Your model doesn't support function calling
   - You prefer explicit text parsing
   - You need simpler, more predictable behavior

3. **Use `StructuredOutputLLM`** when:
   - You have an existing LLM instance you want to wrap
   - You just need to enforce an output format
   - You don't need tool orchestration features

### For Framework Developers

1. **Inherit from `FunctionCallingLLM`** when building provider integrations
   - Implement `_prepare_chat_with_tools()` for your provider's format
   - Implement `get_tool_calls_from_response()` to extract tool calls
   - Follow the async/streaming patterns from existing providers (e.g., Ollama)

2. **Compose higher-level abstractions** using the orchestration layer
   - Build on `ToolOrchestratingLLM` for complex workflows
   - Create domain-specific wrappers around `TextCompletionLLM`

## Code References

- **FunctionCallingLLM**: `libs/core/src/serapeum/core/llms/function_calling.py`
- **StructuredOutputLLM**: `libs/core/src/serapeum/core/llms/structured_output_llm.py`
- **ToolOrchestratingLLM**: `libs/core/src/serapeum/core/llms/orchestrators/tool_based.py`
- **TextCompletionLLM**: `libs/core/src/serapeum/core/llms/orchestrators/text_completion_llm.py`

## Related Documentation

- [Callable Tools Guide](../tools/callable_tools.md)
- [Provider Integration Guide](../../../architecture/integration-guide.md)
- [Architecture Overview](../../../overview/codebase-map.md)
