# Serapeum Ollama Provider

**Ollama integration for the Serapeum LLM framework**

The `serapeum-ollama` package provides complete Ollama backend support for Serapeum, including:

- **Chat & Completion**: Full-featured LLM interface with sync/async support
- **Streaming**: Real-time token streaming for both chat and structured outputs
- **Tool Calling**: Function calling with automatic schema generation
- **Structured Outputs**: Type-safe extraction using Pydantic models
- **Embeddings**: Local embedding generation for RAG and semantic search

This adapter implements the `serapeum.core.llms.FunctionCallingLLM` interface, making it compatible with all Serapeum orchestrators and tools.

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [LLM Features](#llm-features)
  - [Basic Chat](#basic-chat)
  - [Streaming](#streaming)
  - [Async Operations](#async-operations)
  - [Structured Outputs](#structured-outputs)
  - [Tool Calling](#tool-calling)
  - [Completion Style Usage](#completion-style-usage)
- [Embeddings](#embeddings)
  - [Basic Embedding Generation](#basic-embedding-generation)
  - [Batch Embeddings](#batch-embeddings)
  - [Async Embeddings](#async-embeddings)
- [Configuration](#configuration)
- [Examples](#examples)
- [Testing](#testing)
- [Links](#links)

## Installation

### From Source

```bash
# Install from the repository
cd libs/providers/ollama
uv sync
```

### From PyPI (when published)

```bash
pip install serapeum-ollama
```

## Prerequisites

You need a running Ollama server with at least one model available:

1. **Install Ollama**: Visit [ollama.com](https://ollama.com/) and follow installation instructions
2. **Start the server**: Run `ollama serve` (usually runs on `http://localhost:11434`)
3. **Pull a model**:
   ```bash
   # For chat/completion
   ollama pull llama3.1

   # For embeddings
   ollama pull nomic-embed-text
   ```

Verify installation:
```bash
ollama list
```

## Quick Start

### Basic Usage

```python
from serapeum.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole

# Initialize the model
llm = Ollama(model="llama3.1", request_timeout=120)

# Simple chat
messages = [Message(role=MessageRole.USER, content="Explain quantum computing in one sentence.")]
response = llm.chat(messages)
print(response.message.content)
```

## LLM Features

### Basic Chat

The `Ollama` class provides a complete chat interface:

```python
from serapeum.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole, MessageList

llm = Ollama(
    model="llama3.1",
    temperature=0.7,
    request_timeout=120
)

# Single message
response = llm.chat([
    Message(role=MessageRole.USER, content="What is the capital of France?")
])
print(response.message.content)  # "The capital of France is Paris."

# Multi-turn conversation
conversation = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    Message(role=MessageRole.USER, content="What's 2+2?"),
    Message(role=MessageRole.ASSISTANT, content="4"),
    Message(role=MessageRole.USER, content="And if I add 3?"),
]

response = llm.chat(MessageList.from_list(conversation))
print(response.message.content)  # "7"

# Access token usage
if hasattr(response.raw, 'usage'):
    print(f"Tokens used: {response.raw['usage']['total_tokens']}")
```

### Streaming

Stream responses token-by-token for real-time feedback:

```python
from serapeum.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole

llm = Ollama(model="llama3.1")

messages = [Message(role=MessageRole.USER, content="Write a haiku about coding.")]

# Synchronous streaming
print("Streaming response: ", end="")
for chunk in llm.stream_chat(messages):
    print(chunk.delta, end="", flush=True)
print()  # newline

# Get the complete message from the last chunk
full_response = chunk.message.content
```

### Async Operations

Full async support for concurrent operations:

```python
import asyncio
from serapeum.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole

async def main():
    llm = Ollama(model="llama3.1")

    # Async chat
    response = await llm.achat([
        Message(role=MessageRole.USER, content="Hello!")
    ])
    print(response.message.content)

    # Async streaming
    messages = [Message(role=MessageRole.USER, content="Count to 5.")]
    stream = await llm.astream_chat(messages)

    async for chunk in stream:
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(main())
```

### Structured Outputs

Extract structured data using Pydantic models:

```python
from pydantic import BaseModel, Field
from serapeum.ollama import Ollama
from serapeum.core.prompts import PromptTemplate

class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job title")

llm = Ollama(model="llama3.1", json_mode=True)

prompt = PromptTemplate(
    "Extract person information from: {text}\n"
    "Return a JSON object with name, age, and occupation."
)

# Synchronous structured prediction
person = llm.structured_predict(
    output_cls=Person,
    prompt=prompt,
    text="John Doe is a 32-year-old software engineer at Tech Corp."
)

print(f"{person.name}, {person.age}, works as {person.occupation}")
# Output: John Doe, 32, works as software engineer

# Streaming structured outputs
for partial in llm.stream_structured_predict(
    output_cls=Person,
    prompt=prompt,
    text="Jane Smith, age 28, data scientist"
):
    if isinstance(partial, list):
        partial = partial[0]
    print(f"Partial: {partial}")

# Async structured prediction
async def get_structured():
    person = await llm.astructured_predict(
        output_cls=Person,
        prompt=prompt,
        text="Alice Johnson is 45 and works as a CEO."
    )
    return person

import asyncio
result = asyncio.run(get_structured())
print(result)
```

### Tool Calling

Create tools from functions or Pydantic models and let the LLM use them:

```python
from pydantic import BaseModel, Field
from serapeum.ollama import Ollama
from serapeum.core.tools import CallableTool
from serapeum.core.llms.orchestrators import ToolOrchestratingLLM
from serapeum.core.prompts import PromptTemplate

# Define tools using Pydantic models
class WeatherInput(BaseModel):
    location: str = Field(description="City name, e.g., 'San Francisco'")
    unit: str = Field(description="Temperature unit: 'celsius' or 'fahrenheit'")

def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location."""
    # Simulated weather data
    return f"The weather in {location} is 72°{unit[0].upper()} and sunny."

class CalculatorInput(BaseModel):
    operation: str = Field(description="Math operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

def calculate(operation: str, a: float, b: float) -> float:
    """Perform basic math operations."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float('inf')
    }
    return ops.get(operation, 0)

# Create tools
weather_tool = CallableTool.from_model(
    WeatherInput,
    get_weather,
    name="get_weather",
    description="Get current weather for a location"
)

calculator_tool = CallableTool.from_model(
    CalculatorInput,
    calculate,
    name="calculate",
    description="Perform basic arithmetic operations"
)

# Create orchestrator with tools
llm = Ollama(model="llama3.1", request_timeout=120, json_mode=True)

orchestrator = ToolOrchestratingLLM(
    llm=llm,
    prompt=PromptTemplate("Answer the user's question: {query}"),
    tools=[weather_tool, calculator_tool],
)

# Use tools via natural language
result = orchestrator(query="What's 15 multiplied by 8?")
print(result)  # Uses calculator_tool automatically

result = orchestrator(query="What's the weather in Paris?")
print(result)  # Uses weather_tool automatically

# You can also use tools directly with the base LLM
from serapeum.core.base.llms.types import Message, MessageRole

messages = [Message(role=MessageRole.USER, content="What's 25 + 17?")]
response = llm.chat_with_tools(
    tools=[calculator_tool],
    chat_history=messages,
)

# Check if model wants to call a tool
tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
if tool_calls:
    for call in tool_calls:
        print(f"Tool: {call.tool_name}")
        print(f"Arguments: {call.tool_kwargs}")

        # Execute the tool
        if call.tool_name == "calculate":
            result = calculate(**call.tool_kwargs)
            print(f"Result: {result}")
```

### Completion Style Usage

Use prompt templates for completion-style interactions:

```python
from pydantic import BaseModel
from serapeum.ollama import Ollama
from serapeum.core.prompts import PromptTemplate

llm = Ollama(model="llama3.1", temperature=0.8)

# Simple template
prompt = PromptTemplate("Write a tagline for a company that makes {product}.")
response = llm.predict(prompt, product="eco-friendly water bottles")
print(response)

# Multi-variable template
prompt = PromptTemplate(
    "Write a {style} poem about {topic} in {lines} lines."
)
response = llm.predict(
    prompt,
    style="haiku",
    topic="artificial intelligence",
    lines="3"
)
print(response)

# With output parsing
from serapeum.core.output_parsers import PydanticParser

class Summary(BaseModel):
    title: str
    main_points: list[str]
    conclusion: str

parser = PydanticParser(output_cls=Summary)
prompt = PromptTemplate(
    "Summarize this text as JSON: {text}\n"
    "Include title, main_points (array), and conclusion.",
    output_parser=parser
)

llm_json = Ollama(model="llama3.1", json_mode=True)
summary = llm_json.predict(
    prompt,
    text="Artificial intelligence is transforming industries. It automates tasks, "
         "provides insights, and enables new capabilities. However, it also raises "
         "ethical concerns about privacy and job displacement."
)
print(summary.title)
print(summary.main_points)
print(summary.conclusion)
```

## Embeddings

The `OllamaEmbedding` class provides local embedding generation:

### Basic Embedding Generation

```python
from serapeum.ollama import OllamaEmbedding

# Initialize embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Generate single embedding
text_embedding = embed_model.get_text_embedding("Machine learning is fascinating.")
print(f"Embedding dimension: {len(text_embedding)}")
print(f"First 5 values: {text_embedding[:5]}")

# Query embedding (optimized for retrieval)
query_embedding = embed_model.get_query_embedding("What is machine learning?")
```

### Batch Embeddings

Generate embeddings for multiple texts efficiently:

```python
from serapeum.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    embed_batch_size=32,  # Process 32 texts at a time
)

documents = [
    "Python is a high-level programming language.",
    "Machine learning enables computers to learn from data.",
    "Neural networks are inspired by biological neurons.",
    "Deep learning uses multi-layer neural networks.",
    "Natural language processing deals with text and speech.",
]

# Batch embedding generation
embeddings = embed_model.get_text_embeddings(documents)
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Use with similarity search
import numpy as np

query = "What is deep learning?"
query_emb = embed_model.get_query_embedding(query)

# Calculate cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [
    (doc, cosine_similarity(query_emb, emb))
    for doc, emb in zip(documents, embeddings)
]

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)
print("\nMost similar documents:")
for doc, score in similarities[:3]:
    print(f"  {score:.3f}: {doc}")
```

### Async Embeddings

Use async operations for better performance:

```python
import asyncio
from serapeum.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text")

async def embed_documents():
    
    # Async single embedding
    embedding = await embed_model.aget_text_embedding("Hello, world!")
    print(f"Embedding generated: {len(embedding)} dimensions")

    # Async batch embeddings
    documents = [
        "Document 1 about AI",
        "Document 2 about ML",
        "Document 3 about NLP",
    ]
    text_embed = await embed_model.aget_text_embedding_batch(documents)
    print(f"Generated {len(text_embed)} embeddings asynchronously")

    # Async query embedding
    query_embed = await embed_model.aget_query_embedding("What is AI?")

    return text_embed, query_embed

asyncio.run(embed_documents())
```

### Advanced Embedding Configuration

```python
from serapeum.ollama import OllamaEmbedding

# Configure with instructions for better retrieval
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    embed_batch_size=16,
    keep_alive="10m",  # Keep model loaded for 10 minutes
    query_instruction="Represent this query for retrieving relevant documents: ",
    text_instruction="Represent this document for retrieval: ",
    ollama_additional_kwargs={
        # Add any Ollama-specific parameters
    },
)

# Instructions are automatically prepended
documents = "AI is transforming healthcare."
doc_embeddings = embed_model.get_text_embedding(documents)

query = "How is AI used in medicine?"
query_embedding = embed_model.get_query_embedding(query)

# The model internally processes:
# - Document: "Represent this document for retrieval: AI is transforming healthcare."
# - Query: "Represent this query for retrieving relevant documents: How is AI used in medicine?"
```

### Integration with Serapeum

Combine embeddings with LLMs for RAG (Retrieval-Augmented Generation):

```python
from serapeum.ollama import Ollama, OllamaEmbedding
from serapeum.core.base.llms.types import Message, MessageRole

# Initialize both LLM and embeddings
llm = Ollama(model="llama3.1")
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Document store (simplified)
knowledge_base = [
    "The Eiffel Tower is in Paris, France.",
    "The Great Wall of China is in China.",
    "The Statue of Liberty is in New York, USA.",
    "The Colosseum is in Rome, Italy.",
]

# Generate embeddings for knowledge base
kb_embeddings = embed_model.get_text_embeddings(knowledge_base)

# User query
query = "Where is the Eiffel Tower?"
query_emb = embed_model.get_query_embedding(query)

# Simple similarity search
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [
    (doc, cosine_similarity(query_emb, emb))
    for doc, emb in zip(knowledge_base, kb_embeddings)
]
similarities.sort(key=lambda x: x[1], reverse=True)
context = similarities[0][0]

# Use LLM with retrieved context
messages = [
    Message(
        role=MessageRole.SYSTEM,
        content=f"Answer based on this context: {context}"
    ),
    Message(role=MessageRole.USER, content=query)
]

response = llm.chat(messages)
print(response.message.content)
# Output: "The Eiffel Tower is in Paris, France."
```

## Configuration

### LLM Configuration

The `Ollama` class accepts these parameters:

```python
from serapeum.ollama import Ollama

llm = Ollama(
    model="llama3.1",                    # Required: Ollama model name
    base_url="http://localhost:11434",   # Ollama server URL
    temperature=0.75,                    # Sampling temperature (0.0-1.0)
    context_window=3900,                 # Max context tokens
    request_timeout=60.0,                # Request timeout in seconds
    json_mode=False,                     # Enable JSON formatting
    is_function_calling_model=True,      # Whether model supports tools
    keep_alive="5m",                     # How long to keep model loaded
    additional_kwargs={                  # Provider-specific options
        "num_predict": 100,              # Max tokens to generate
        "top_k": 40,                     # Top-k sampling
        "top_p": 0.9,                    # Top-p (nucleus) sampling
        "repeat_penalty": 1.1,           # Repetition penalty
    }
)
```

**Key Parameters:**

- `model`: Model identifier (e.g., `"llama3.1"`, `"mistral:latest"`)
- `base_url`: Ollama server endpoint (default: `http://localhost:11434`)
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = very random)
- `json_mode`: Request JSON-formatted responses when `True`
- `request_timeout`: Timeout for API calls (increase for slower models)
- `keep_alive`: Duration to keep model in memory (e.g., `"5m"`, `"1h"`)
- `additional_kwargs`: Pass any Ollama-specific options

### Embedding Configuration

The `OllamaEmbedding` class parameters:

```python
from serapeum.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",         # Required: embedding model name
    base_url="http://localhost:11434",     # Ollama server URL
    embed_batch_size=10,                   # Batch size (1-2048)
    keep_alive="5m",                       # Model keep-alive duration
    query_instruction=None,                # Prefix for queries
    text_instruction=None,                 # Prefix for documents
    ollama_additional_kwargs={},           # Ollama API options
    client_kwargs={},                      # Client configuration
)
```

### Available Models

**Chat/Completion Models:**
- `llama3.1` - Meta's Llama 3.1 (8B, 70B, 405B)
- `llama3.2` - Latest Llama model
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B MoE
- `codellama` - Code-specialized Llama
- `gemma2` - Google's Gemma 2

**Embedding Models:**
- `nomic-embed-text` - General-purpose embeddings (768d)
- `mxbai-embed-large` - High-quality embeddings (1024d)
- `snowflake-arctic-embed` - Snowflake's embedding model

Pull models with:
```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Examples

Complete examples are available in the `examples/` directory:

- `basic_chat.py` - Simple chat interactions
- `streaming_example.py` - Streaming responses
- `tool_calling_example.py` - Using tools with LLMs
- `structured_outputs.py` - Extracting structured data
- `embeddings_rag.py` - RAG with embeddings
- `async_operations.py` - Async patterns

## Testing

Run tests for the Ollama provider:

```bash
# All tests
cd libs/providers/ollama
uv run pytest

# Skip end-to-end tests (don't require Ollama server)
uv run pytest -m "not e2e"

# Only unit tests
uv run pytest -m unit

# With coverage
uv run pytest --cov=serapeum.ollama
```

**Note:** End-to-end tests require a running Ollama server with models available.

## Notes

- **Server Required**: Ollama must be running (`ollama serve`) before using this adapter
- **Tool Calling**: Depends on model capabilities and Ollama version (some models don't support tools)
- **JSON Mode**: Improves structured output reliability when the model supports it
- **Timeouts**: Increase `request_timeout` for larger models or complex tasks
- **Async**: All async methods use a per-event-loop client for thread safety

## Links

- **Homepage**: [https://github.com/Serapieum-of-alex/serapeum](https://github.com/Serapieum-of-alex/serapeum)
- **Documentation**: [https://serapeum.readthedocs.io/](https://serapeum.readthedocs.io/)
- **Ollama**: [https://ollama.com/](https://ollama.com/)
- **Changelog**: [HISTORY.rst](https://github.com/Serapieum-of-alex/serapeum/blob/main/HISTORY.rst)

---

**Questions or issues?** Open an issue on [GitHub](https://github.com/Serapieum-of-alex/serapeum/issues).
