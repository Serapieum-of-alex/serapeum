# Serapeum OpenAI Provider

**OpenAI integration for the Serapeum LLM framework**

The `serapeum-openai` package provides complete OpenAI backend support for
Serapeum, with two main classes covering both OpenAI APIs:

- **`OpenAI`**: Chat Completions API for GPT-4o, GPT-4, GPT-3.5, and
  instruct models
- **`OpenAIResponses`**: Responses API for reasoning models (o3, o4-mini)
  with built-in tools and stateful conversations

Both classes implement the `serapeum.core.llms.FunctionCallingLLM` interface,
making them compatible with all Serapeum orchestrators and tools.

## Features

- **Chat & Completion**: Sync/async support with streaming
- **Tool Calling**: Function calling with automatic schema generation
- **Structured Outputs**: Native JSON-schema support on compatible models,
  with automatic fallback to function-calling approach
- **Responses API**: Built-in tools (file search, web search, code
  interpreter), reasoning-effort control, stateful conversations
- **Model Registry**: YAML-based model metadata with automatic capability
  detection (context windows, JSON-schema support, function calling)
- **Retry Logic**: OpenAI-specific exception classification with automatic
  retries for transient errors
- **Token Counting**: Tiktoken integration for accurate token counting

## Installation

### From Source

```bash
cd libs/providers/openai
uv sync --active
```

### From PyPI (when published)

```bash
pip install serapeum-openai
```

## Prerequisites

You need an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass it directly when creating the LLM instance.

## Quick Start

### Chat Completions API

```python
from serapeum.openai import OpenAI
from serapeum.core.llms import Message, MessageRole

llm = OpenAI(model="gpt-4o", temperature=0.7)

messages = [
    Message(role=MessageRole.USER, content="Explain quantum computing in one sentence.")
]
response = llm.chat(messages)
print(response.message.content)
```

### Responses API

```python
from serapeum.openai import OpenAIResponses
from serapeum.core.llms import Message, MessageRole

llm = OpenAIResponses(model="o3", reasoning_options={"effort": "medium"})

messages = [
    Message(role=MessageRole.USER, content="Solve this step by step: what is 2^10?")
]
response = llm.chat(messages)
print(response.message.content)
```

## LLM Features

### Streaming

```python
from serapeum.openai import OpenAI
from serapeum.core.llms import Message, MessageRole

llm = OpenAI(model="gpt-4o")

messages = [Message(role=MessageRole.USER, content="Write a haiku about coding.")]

for chunk in llm.chat(messages, stream=True):
    print(chunk.delta, end="", flush=True)
print()
```

### Async Operations

```python
import asyncio
from serapeum.openai import OpenAI
from serapeum.core.llms import Message, MessageRole

llm = OpenAI(model="gpt-4o")

async def main():
    response = await llm.achat([
        Message(role=MessageRole.USER, content="Hello!")
    ])
    print(response.message.content)

    # Async streaming
    async for chunk in await llm.achat(
        [Message(role=MessageRole.USER, content="Count to 5.")],
        stream=True,
    ):
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(main())
```

### Structured Outputs

Extract structured data using Pydantic models. On compatible models (GPT-4o
and later), native JSON-schema mode is used automatically:

```python
from pydantic import BaseModel, Field
from serapeum.openai import OpenAI
from serapeum.core.prompts import PromptTemplate


class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job title")


llm = OpenAI(model="gpt-4o")

person = llm.parse(
    schema=Person,
    prompt=PromptTemplate("Extract person info from: {text}"),
    text="John Doe is a 32-year-old software engineer at Tech Corp.",
)

print(f"{person.name}, {person.age}, works as {person.occupation}")
```

### Tool Calling

```python
from serapeum.openai import OpenAI
from serapeum.core.tools import CallableTool
from serapeum.core.llms import Message, MessageRole


def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location."""
    return f"The weather in {location} is 22 {unit} and sunny."


llm = OpenAI(model="gpt-4o")

weather_tool = CallableTool.from_function(get_weather)
messages = [Message(role=MessageRole.USER, content="What's the weather in Paris?")]

response = llm.generate_tool_calls(
    tools=[weather_tool],
    chat_history=messages,
)

tool_calls = llm.get_tool_calls_from_response(
    response, error_on_no_tool_call=False
)
if tool_calls:
    for call in tool_calls:
        print(f"Tool: {call.tool_name}, Args: {call.tool_kwargs}")
```

### Responses API with Built-in Tools

```python
from serapeum.openai import OpenAIResponses
from serapeum.core.llms import Message, MessageRole

llm = OpenAIResponses(
    model="o3",
    reasoning_options={"effort": "medium"},
    built_in_tools=[{"type": "web_search_preview"}],
)

messages = [
    Message(role=MessageRole.USER, content="What happened in tech news today?")
]
response = llm.chat(messages)
print(response.message.content)
```

## Configuration

### OpenAI (Chat Completions)

```python
from serapeum.openai import OpenAI

llm = OpenAI(
    model="gpt-4o",                       # Required: model name
    temperature=0.7,                       # Sampling temperature (0.0-2.0)
    max_tokens=1024,                       # Max tokens to generate
    logprobs=False,                        # Return log probabilities
    top_logprobs=None,                     # Number of top logprobs per token
    strict=False,                          # Strict JSON-schema mode
    reasoning_effort=None,                 # For reasoning models
    api_key="sk-...",                      # API key (or OPENAI_API_KEY env)
    api_base=None,                         # Custom API base URL
    timeout=60.0,                          # Request timeout in seconds
    default_headers=None,                  # Additional HTTP headers
    additional_kwargs={},                  # Extra API parameters
)
```

### OpenAIResponses (Responses API)

```python
from serapeum.openai import OpenAIResponses

llm = OpenAIResponses(
    model="o3",                            # Required: model name
    temperature=1.0,                       # Sampling temperature
    top_p=1.0,                             # Nucleus sampling
    max_output_tokens=None,                # Max output tokens
    reasoning_options={"effort": "medium"},# Reasoning effort control
    instructions=None,                     # System-level instructions
    built_in_tools=None,                   # Built-in tools list
    store=True,                            # Store responses server-side
    track_previous_responses=False,        # Stateful conversation tracking
    strict=False,                          # Strict JSON-schema mode
    api_key="sk-...",                      # API key (or OPENAI_API_KEY env)
    api_base=None,                         # Custom API base URL
    timeout=60.0,                          # Request timeout
)
```

## Package Layout

```
src/serapeum/openai/
├── __init__.py              # Lazy imports (OpenAI, OpenAIResponses)
├── llm/
│   ├── chat_completions.py  # OpenAI class (Chat Completions API)
│   ├── responses.py         # OpenAIResponses class (Responses API)
│   └── base/
│       ├── client.py        # SDK client lifecycle and credentials
│       ├── model.py         # Model metadata and tokenizer
│       └── structured.py    # Native JSON-schema structured output
├── data/
│   ├── models.py            # Model query functions and constants
│   └── models.yaml          # Model registry (context windows, capabilities)
├── parsers/
│   ├── chat_parsers.py      # Chat completion response parsers
│   ├── formatters.py        # Message format converters
│   └── response_parsers.py  # Responses API output parsers
├── retry.py                 # OpenAI-specific retry logic
└── utils.py                 # Credential resolution, constants
```

## Testing

```bash
# All tests (from repo root)
python -m pytest libs/providers/openai/tests

# Skip end-to-end tests (don't require API key)
python -m pytest libs/providers/openai/tests -m "not e2e"

# Only unit tests
python -m pytest libs/providers/openai/tests -m unit
```

**Note:** End-to-end tests require a valid `OPENAI_API_KEY` environment
variable.

## Links

- **Homepage**:
  [github.com/Serapieum-of-alex/serapeum](https://github.com/Serapieum-of-alex/serapeum)
- **Documentation**:
  [serapieum-of-alex.github.io/serapeum](https://serapieum-of-alex.github.io/serapeum)
- **OpenAI API**:
  [platform.openai.com/docs](https://platform.openai.com/docs)
