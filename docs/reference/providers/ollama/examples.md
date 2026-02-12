# Ollama Usage Examples

This guide provides comprehensive examples covering all possible ways to use the `Ollama` LLM class based on real test cases from the codebase.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Initialization Patterns](#initialization-patterns)
3. [Chat Operations](#chat-operations)
4. [Completion Operations](#completion-operations)
5. [Streaming Operations](#streaming-operations)
6. [Tool/Function Calling](#toolfunction-calling)
7. [Integration with Orchestrators](#integration-with-orchestrators)
8. [Async Operations](#async-operations)

---

## Basic Usage

### Simple Chat

The most straightforward way to use `Ollama`:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

# Initialize Ollama LLM
llm = Ollama(
    model="llama3.1",
    request_timeout=180,
)

# Create a message
messages = [Message(role=MessageRole.USER, content="Say 'pong'.")]

# Send chat request
response = llm.chat(messages)
print(response.message.content)  # "Pong!"
```

### Simple Completion

Using the completion API:

```python
from serapeum.ollama import Ollama

# Initialize Ollama LLM
llm = Ollama(
    model="llama3.1",
    request_timeout=180,
)

# Send completion request
response = llm.complete("Say 'pong'.")
print(response.text)  # "Pong!"
```

---

## Initialization Patterns

### 1. Basic Initialization

Minimal configuration:

```python
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1")
```

### 2. Full Configuration

With all common parameters:

```python
from serapeum.ollama import Ollama

llm = Ollama(
    model="llama3.1",
    base_url="http://localhost:11434",
    temperature=0.8,
    context_window=4096,
    request_timeout=180.0,
    json_mode=True,
    keep_alive="5m",
    additional_kwargs={"top_p": 0.9, "top_k": 40}
)
```

### 3. With Custom Client

Pre-configured Ollama client:

```python
from ollama import Client
from serapeum.ollama import Ollama

# Create custom client
client = Client(host="http://localhost:11434", timeout=300)

# Pass to Ollama
llm = Ollama(
    model="llama3.1",
    client=client,
)
```

### 4. JSON Mode for Structured Outputs

Enable JSON formatting:

```python
from serapeum.ollama import Ollama

llm = Ollama(
    model="llama3.1",
    json_mode=True,  # Forces JSON output
    request_timeout=180,
)
```

---

## Chat Operations

### 1. Single Turn Chat

Basic conversation:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

messages = [
    Message(role=MessageRole.USER, content="What is 2+2?")
]

response = llm.chat(messages)
print(response.message.content)  # "4"
```

### 2. Multi-turn Conversation

With conversation history:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful math tutor."),
    Message(role=MessageRole.USER, content="What is 2+2?"),
    Message(role=MessageRole.ASSISTANT, content="2+2 equals 4."),
    Message(role=MessageRole.USER, content="What about 3+3?"),
]

response = llm.chat(messages)
print(response.message.content)  # "3+3 equals 6."
```

### 3. Chat with Parameters

Passing custom parameters:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

messages = [Message(role=MessageRole.USER, content="Write a creative story.")]

# Override default settings
response = llm.chat(
    messages,
    temperature=0.9,      # Higher for creativity
    top_p=0.95,
    max_tokens=500,
)
```

### 4. Chat with Images

Multi-modal input (if supported by model):

```python
from serapeum.core.base.llms.types import Message, MessageRole, Image
from serapeum.ollama import Ollama

llm = Ollama(model="llama4", request_timeout=180)  # Vision model

# Create message with image
image = Image(path="docs/reference/providers/ollama/images/baharia-oasis.jpg")
messages = [
    Message(
        role=MessageRole.USER,
        content="What's in this image?",
        images=[image]
    )
]

response = llm.chat(messages)
print(response.message.content)
```

---

## Completion Operations

### 1. Basic Completion

Simple text completion:

```python
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

prompt = "The capital of France is"
response = llm.complete(prompt)
print(response.text)  # "Paris"
```

### 2. Completion with Parameters

Custom generation settings:

```python
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

response = llm.complete(
    "Once upon a time",
    temperature=0.8,
    max_tokens=200,
)
print(response.text)
```

### 3. JSON Completion

Force JSON output:

```python
from serapeum.ollama import Ollama

llm = Ollama(
    model="llama3.1",
    json_mode=True,
    request_timeout=180,
)

prompt = 'Return {"name": "John", "age": 30} as JSON'
response = llm.complete(prompt)
print(response.text)  # {"name": "John", "age": 30}
```

---

## Streaming Operations

### 1. Stream Chat

Real-time streaming chat:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

messages = [Message(role=MessageRole.USER, content="Count from 1 to 5.")]

# Stream responses
for chunk in llm.stream_chat(messages):
    print(chunk.message.content, end="", flush=True)
    # Outputs: "1" " 2" " 3" " 4" " 5"
```

### 2. Stream Completion

Real-time streaming completion:

```python
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

prompt = "Write a haiku about coding:"

# Stream completion
for chunk in llm.stream_complete(prompt):
    print(chunk.text, end="", flush=True)
```

### 3. Processing Stream with Delta

Access incremental content:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

messages = [Message(role=MessageRole.USER, content="Tell me a joke.")]

full_response = ""
for chunk in llm.stream_chat(messages):
    delta = chunk.delta  # Incremental content
    if delta:
        full_response += delta
        print(delta, end="", flush=True)

print(f"\n\nFull response: {full_response}")
```

---

## Tool/Function Calling

### 1. Basic Tool Calling

Using tools with Ollama:

```python
from pydantic import BaseModel
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


def create_album(title: str, artist: str, songs: list[str]) -> Album:
    """Create an album with the given information."""
    return Album(title=title, artist=artist, songs=songs)


llm = Ollama(model="llama3.1", request_timeout=180)

# Create tool from function
tool = CallableTool.from_function(create_album)

message =  Message(
    role=MessageRole.USER,
    content="Create a rock album with two songs"
)

# Call with tools
response = llm.chat_with_tools(tools=[tool], user_msg=message)

# Extract tool calls
tool_calls = llm.get_tool_calls_from_response(response)
print(tool_calls)
```

### 2. Tool Calling from Pydantic Model

Create tools from Pydantic models:

```python
from pydantic import BaseModel, Field
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class Album(BaseModel):
    """An music album."""
    title: str = Field(description="Album title")
    artist: str = Field(description="Artist name")
    songs: list[str] = Field(description="List of song titles")


llm = Ollama(model="llama3.1", request_timeout=180)

# Create tool from Pydantic model
tool = CallableTool.from_model(Album)

message = Message(
    role=MessageRole.USER,
    content="Create a jazz album with title 'Blue Notes' by Miles Davis with 3 songs"
)

response = llm.chat_with_tools(tools=[tool], user_msg=message)

# Extract and execute tool call
tool_calls = llm.get_tool_calls_from_response(response)
for tool_call in tool_calls:
    # Execute tool
    result = tool.call(**tool_call.tool_kwargs)
    print(result)  # Album instance
```

### 3. Single Tool Call Mode

Force single tool call:

```python
from pydantic import BaseModel
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


llm = Ollama(model="llama3.1", request_timeout=180)

tool = CallableTool.from_model(Album)

message = Message(
    role=MessageRole.USER,
    content="Create two albums"
)

# Force single tool call
response = llm.chat_with_tools(
    tools=[tool],
    user_msg=message,
    allow_parallel_tool_calls=False,  # Only one tool call allowed
)

tool_calls = llm.get_tool_calls_from_response(response)
print(len(tool_calls))  # 1 (even if model tried to return multiple)
```

### 4. Parallel Tool Calls

Allow multiple tool calls:

```python
from pydantic import BaseModel
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


llm = Ollama(model="llama3.1", request_timeout=180)

tool = CallableTool.from_model(Album)

message = Message(
    role=MessageRole.USER,
    content="Create two albums: one rock album and one jazz album"
)

# Allow parallel tool calls
response = llm.chat_with_tools(
    tools=[tool],
    user_msg=message,
    allow_parallel_tool_calls=True,
)

tool_calls = llm.get_tool_calls_from_response(response)
print(len(tool_calls))  # 2 (if model returns multiple)

for tool_call in tool_calls:
    result = tool.call(**tool_call.tool_kwargs)
    print(result)
```

### 5. Streaming with Tools

Stream tool calls:

```python
from pydantic import BaseModel
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


llm = Ollama(model="llama3.1", request_timeout=180)

tool = CallableTool.from_model(Album)

message = Message(
    role=MessageRole.USER,
    content="Create a pop album"
)

# Stream with tools
for chunk in llm.stream_chat_with_tools(tools=[tool], user_msg=message):
    # Process streaming tool calls
    if chunk.message.additional_kwargs.get("tool_calls"):
        print(f"Tool call chunk: {chunk.message.additional_kwargs['tool_calls']}")
```

---

## Integration with Orchestrators

### 1. With TextCompletionLLM

Use Ollama with `TextCompletionLLM` for structured outputs:

```python
from pydantic import BaseModel
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama


class DummyModel(BaseModel):
    value: str


# Initialize Ollama
llm = Ollama(model="llama3.1", request_timeout=180)

# Create parser
parser = PydanticParser(output_cls=DummyModel)

# Create TextCompletionLLM
text_llm = TextCompletionLLM(
    output_parser=parser,
    prompt="Value: {value}",
    llm=llm,
)

# Execute
result = text_llm(value="input")
print(result.value)  # "input"
```

### 2. With ToolOrchestratingLLM

Use Ollama with `ToolOrchestratingLLM` for tool-based workflows:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


# Initialize Ollama
llm = Ollama(model="llama3.1", request_timeout=180)

# Create ToolOrchestratingLLM
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create an album about {topic} with two random songs",
    llm=llm,
)

# Execute - returns Album instance
result = tools_llm(topic="rock")
print(result.title)
print(result.artist)
print(result.songs)
```

### 3. Parallel Tool Execution

Using `ToolOrchestratingLLM` with parallel tools:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


llm = Ollama(model="llama3.1", request_timeout=180)

# Enable parallel tool calls
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create albums about {topic}",
    llm=llm,
    allow_parallel_tool_calls=True,
)

# Returns list of Album instances
results = tools_llm(topic="jazz")
print(len(results))  # Potentially multiple albums
for album in results:
    print(f"{album.title} by {album.artist}")
```

### 4. Streaming with ToolOrchestratingLLM

Stream tool execution results:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


llm = Ollama(model="llama3.1", request_timeout=180)

tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create albums about {topic}",
    llm=llm,
    allow_parallel_tool_calls=False,
)

# Stream results
for album in tools_llm.stream_call(topic="rock"):
    print(f"Received: {album.title}")
```

---

## Async Operations

### 1. Async Chat

Non-blocking chat:

```python
import asyncio
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.llms.ollama import Ollama


async def async_chat_example():
    llm = Ollama(model="llama3.1", request_timeout=180)

    messages = [Message(role=MessageRole.USER, content="Hello!")]

    response = await llm.achat(messages)
    print(response.message.content)


asyncio.run(async_chat_example())
```

### 2. Async Completion

Non-blocking completion:

```python
import asyncio
from serapeum.llms.ollama import Ollama


async def async_complete_example():
    llm = Ollama(model="llama3.1", request_timeout=180)

    response = await llm.acomplete("Say hello")
    print(response.text)


asyncio.run(async_complete_example())
```

### 3. Async Streaming Chat

Non-blocking streaming:

```python
import asyncio
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.llms.ollama import Ollama


async def async_stream_example():
    llm = Ollama(model="llama3.1", request_timeout=180)

    messages = [Message(role=MessageRole.USER, content="Count to 5")]

    async for chunk in await llm.astream_chat(messages):
        print(chunk.message.content, end="", flush=True)


asyncio.run(async_stream_example())
```

### 4. Concurrent Async Requests

Process multiple requests concurrently:

```python
import asyncio
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.llms.ollama import Ollama


async def process_multiple():
    llm = Ollama(model="llama3.1", request_timeout=180)

    prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]

    # Create tasks
    tasks = [
        llm.achat([Message(role=MessageRole.USER, content=prompt)])
        for prompt in prompts
    ]

    # Execute concurrently
    responses = await asyncio.gather(*tasks)

    for prompt, response in zip(prompts, responses):
        print(f"{prompt} -> {response.message.content}")


asyncio.run(process_multiple())
```

### 5. Async with ToolOrchestratingLLM

Async tool orchestration:

```python
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


async def async_tool_example():
    llm = Ollama(model="llama3.1", request_timeout=180)

    tools_llm = ToolOrchestratingLLM(
        output_cls=Album,
        prompt="Create an album about {topic}",
        llm=llm,
    )

    result = await tools_llm.acall(topic="pop")
    print(result.title)


asyncio.run(async_tool_example())
```

### 6. Async Streaming with Tools

Async streaming tool execution:

```python
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama


class Album(BaseModel):
    title: str
    artist: str
    songs: list[str]


async def async_stream_tool_example():
    llm = Ollama(model="llama3.1", request_timeout=180)

    tools_llm = ToolOrchestratingLLM(
        output_cls=Album,
        prompt="Create albums about {topic}",
        llm=llm,
        allow_parallel_tool_calls=False,
    )

    stream = await tools_llm.astream_call(topic="rock")
    async for album in stream:
        print(f"Received: {album.title}")


asyncio.run(async_stream_tool_example())
```

---

## Best Practices

### 1. Reuse LLM Instances

Create once, use many times:

```python
from serapeum.llms.ollama import Ollama

# ✓ Good: Create once
llm = Ollama(model="llama3.1", request_timeout=180)

# Reuse for multiple calls
response1 = llm.chat(messages1)
response2 = llm.chat(messages2)

# ✗ Bad: Don't recreate for each call
def process(messages):
    llm = Ollama(model="llama3.1")  # Inefficient
    return llm.chat(messages)
```

### 2. Use Appropriate Timeout

Set timeout based on expected response time:

```python
from serapeum.llms.ollama import Ollama

# Short timeout for simple queries
quick_llm = Ollama(model="llama3.1", request_timeout=30)

# Longer timeout for complex queries
complex_llm = Ollama(model="llama3.1", request_timeout=300)
```

### 3. Handle Errors Gracefully

Always handle potential errors:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.llms.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

try:
    response = llm.chat([Message(role=MessageRole.USER, content="Hello")])
except TimeoutError:
    print("Request timed out")
except ConnectionError:
    print("Could not connect to Ollama server")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 4. Use JSON Mode for Structured Outputs

Enable when expecting JSON:

```python
from serapeum.llms.ollama import Ollama

# Enable JSON mode
llm = Ollama(
    model="llama3.1",
    json_mode=True,
    request_timeout=180,
)

# LLM will always return valid JSON
```

### 5. Monitor Response Metadata

Use metadata for monitoring:

```python
from serapeum.core.base.llms.types import Message, MessageRole
from serapeum.llms.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=180)

response = llm.chat([Message(role=MessageRole.USER, content="Hello")])

# Access metadata
print(f"Model: {response.additional_kwargs.get('model')}")
print(f"Tokens: {response.additional_kwargs.get('eval_count')}")
print(f"Duration: {response.additional_kwargs.get('total_duration')}")
```

---

## See Also

- [Execution Flow and Method Calls](./ollama_sequence.md) - Detailed sequence diagrams
- [Architecture and Class Relationships](./ollama_class.md) - Class structure
- [Data Transformations and Validation](./ollama_dataflow.md) - Data flow details
- [Component Boundaries and Interactions](./ollama_components.md) - System components
- [Lifecycle and State Management](./ollama_state.md) - State management
