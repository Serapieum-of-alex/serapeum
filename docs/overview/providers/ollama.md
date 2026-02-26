# Ollama Provider

**Local LLM inference for the Serapeum framework**

The `serapeum-ollama` package provides complete Ollama backend support for Serapeum, enabling you to run powerful language models locally on your machine.

<div class="grid cards" markdown>

-   :material-server: **Local Inference**

    ---

    Run models locally on your machine without external API dependencies

-   :material-lock: **Privacy First**

    ---

    All data stays on your machine. No internet connection required after model download

-   :material-scale-balance: **Free & Open Source**

    ---

    No API costs. Use any Ollama-compatible model

-   :material-flash: **Complete Features**

    ---

    Chat, completion, streaming, tool calling, structured outputs, and embeddings

</div>

---

## Features

- **Chat & Completion**: Full-featured LLM interface with multi-turn conversations
- **Streaming**: Real-time token streaming for both chat and structured outputs
- **Tool Calling**: Function calling with automatic schema generation
- **Structured Outputs**: Type-safe extraction using Pydantic models
- **Embeddings**: Local embedding generation for RAG and semantic search
- **Async Support**: Full async/await support for all operations

---

## Installation

### Install Serapeum-Ollama

```bash
pip install serapeum-ollama
```

### Install Ollama Server

**1. Download and Install Ollama**

Visit [ollama.com](https://ollama.com/) and follow the installation instructions for your platform:

- **macOS**: Download the .app or use `brew install ollama`
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Windows**: Download from ollama.com

**2. Start the Ollama Server**

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

**3. Pull Models**

Download the models you want to use:

```bash
# Chat/Completion models
ollama pull llama3.1          # Meta Llama 3.1 (recommended)
ollama pull llama3.2          # Latest Llama
ollama pull mistral           # Mistral 7B
ollama pull mixtral           # Mixtral 8x7B
ollama pull codellama         # Code-specialized
ollama pull gemma2            # Google Gemma 2

# Embedding models
ollama pull nomic-embed-text  # General embeddings (768d)
ollama pull mxbai-embed-large # High-quality (1024d)
```

**4. Verify Installation**

```bash
ollama list
```

You should see the models you've downloaded.

---

## Quick Start

```python
import os
from serapeum.ollama import Ollama
from serapeum.core.llms import Message, MessageRole

# Initialize the LLM
llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    temperature=0.7
)

# Simple chat
messages = [
    Message(role=MessageRole.USER, content="Explain quantum computing in one sentence.")
]
response = llm.chat(messages)
print(response.message.content)
```

---

## Chat & Completion

### Basic Chat

```python
import os
from serapeum.ollama import Ollama
from serapeum.core.llms import Message, MessageRole, MessageList

llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
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

### Completion Style

Use prompt templates for completion-style interactions:

```python
import os
from serapeum.ollama import Ollama
from serapeum.core.prompts import PromptTemplate

llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    temperature=0.8
)

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
```

---

## Streaming

Stream responses token-by-token for real-time feedback:

### Sync Streaming

```python
import os
from serapeum.ollama import Ollama
from serapeum.core.llms import Message, MessageRole

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"))
messages = [Message(role=MessageRole.USER, content="Write a haiku about coding.")]

# Synchronous streaming
print("Streaming response: ", end="")
for chunk in llm.chat(messages, stream=True):
    print(chunk.delta, end="", flush=True)
print()

# Get the complete message from the last chunk
full_response = chunk.message.content
```

### Async Streaming

```python
import asyncio
import os
from serapeum.ollama import Ollama
from serapeum.core.llms import Message, MessageRole

async def stream_example():
    llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"))
    messages = [Message(role=MessageRole.USER, content="Count to 5.")]

    stream = await llm.achat(messages, stream=True)
    async for chunk in stream:
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(stream_example())
```

---

## Structured Outputs

Extract structured data using Pydantic models:

```python
import os
from pydantic import BaseModel, Field
from serapeum.ollama import Ollama
from serapeum.core.prompts import PromptTemplate


class Person(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job title")

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), json_mode=True)

prompt = PromptTemplate(
    "Extract person information from: {text}\n"
    "Return a JSON object with name, age, and occupation."
)

# Synchronous structured prediction
person = llm.parse(
    schema=Person,
    prompt=prompt,
    text="John Doe is a 32-year-old software engineer at Tech Corp."
)

print(f"{person.name}, {person.age}, works as {person.occupation}")
# Output: John Doe, 32, works as software engineer

# Streaming structured outputs
for partial in llm.stream_parse(
        schema=Person,
        prompt=prompt,
        text="Jane Smith, age 28, data scientist"
):
    if isinstance(partial, list):
        partial = partial[0]
    print(f"Partial: {partial}")

# Async structured prediction
async def get_structured():
    person = await llm.aparse(
        schema=Person,
        prompt=prompt,
        text="Alice Johnson is 45 and works as a CEO."
    )
    return person

import asyncio


result = asyncio.run(get_structured())
print(result)
```

---

## Tool Calling

Create tools from functions or Pydantic models and let the LLM use them:

```python
import os
from serapeum.ollama import Ollama
from serapeum.core.tools import CallableTool


def search_flights(origin: str, destination: str) -> dict:
    """Return estimated round-trip flight cost between two cities."""
    # Mock data
    prices = {
        ("london", "tokyo"): 850,
        ("new york", "paris"): 620,
        ("sydney", "dubai"): 540,
    }
    key = (origin.lower(), destination.lower())
    cost = prices.get(key, 700)
    return {"origin": origin, "destination": destination, "round_trip_cost_usd": cost}


def search_hotels(city: str, nights: int) -> dict:
    """Return estimated hotel cost for a stay in a city."""
    # Mock data — price per night
    per_night = {
        "tokyo": 180,
        "paris": 210,
        "dubai": 160,
    }
    rate = per_night.get(city.lower(), 150)
    return {"city": city, "nights": nights, "rate_per_night_usd": rate, "total_usd": rate * nights}


search_flight_tool = CallableTool.from_function(search_flights)
search_hotels_tool = CallableTool.from_function(search_hotels)

tools = [
    search_flight_tool,
    search_hotels_tool,
]

llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    request_timeout=120,
)

response = llm.invoke_callable(
    tools=tools,
    user_msg="I'm planning a 7-night trip from London to Tokyo. What are the flight and hotel costs?",
    allow_parallel_tool_calls=True,
)
print(response)
```

### Direct Tool Calling

You can also use tools directly with the base LLM:

```python
import os
from pydantic import BaseModel, Field
from serapeum.core.llms import Message, MessageRole
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama


class CalculatorInput(BaseModel):
    """CalculatorInput data(operation, a, b)"""
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


calculator_tool = CallableTool.from_model(CalculatorInput)

messages = [Message(role=MessageRole.USER, content="What's 25 + 17?")]

llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    request_timeout=120,
)

response = llm.generate_tool_calls(
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

---

## Embeddings

Generate embeddings for RAG and semantic search:

### Basic Embedding Generation
- The embedding API is only available in the local ollama server.
- The embedding API is not available in the public cloud.
- you need to install the ollama server locally. [Run Ollama Server](https://docs.ollama.com/)

```python
from serapeum.ollama import OllamaEmbedding

# Initialize embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
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
import numpy as np

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    batch_size=32,  # Process 32 texts at a time
)

documents = [
    "Python is a high-level programming language.",
    "Machine learning enables computers to learn from data.",
    "Neural networks are inspired by biological neurons.",
    "Deep learning uses multi-layer neural networks.",
    "Natural language processing deals with text and speech.",
]

# Batch embedding generation
embeddings = embed_model.get_text_embedding_batch(documents)
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Use with similarity search
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

    return text_embed

asyncio.run(embed_documents())
```

### Advanced Configuration

```python
from serapeum.ollama import OllamaEmbedding

# Configure with instructions for better retrieval
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    batch_size=16,
    keep_alive="10m",  # Keep model loaded for 10 minutes
    query_instruction="Represent this query for retrieving relevant documents: ",
    text_instruction="Represent this document for retrieval: ",
)

# Instructions are automatically prepended
documents = "AI is transforming healthcare."
doc_embeddings = embed_model.get_text_embedding(documents)

query = "How is AI used in medicine?"
query_embedding = embed_model.get_query_embedding(query)
```

---

## RAG Integration

Combine LLM and embeddings for Retrieval-Augmented Generation:

```python
import os
from serapeum.ollama import Ollama, OllamaEmbedding
from serapeum.core.llms import Message, MessageRole
import numpy as np

# Initialize both components
llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"))
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Knowledge base
knowledge_base = [
    "The Eiffel Tower is in Paris, France.",
    "The Great Wall of China is in China.",
    "The Statue of Liberty is in New York, USA.",
]

# Generate embeddings for knowledge base
kb_embeddings = embed_model.get_text_embedding_batch(knowledge_base)

# User query
query = "Where is the Eiffel Tower?"
query_emb = embed_model.get_query_embedding(query)

# Similarity search
similarities = [
    (doc, embed_model.similarity(query_emb, emb))
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

---

## Configuration

### LLM Configuration

```python
import os
from serapeum.ollama import Ollama

llm = Ollama(
    model="qwen3.5:397b",                    # Required: Ollama model name
    api_key=os.environ.get("OLLAMA_API_KEY"),
    base_url="https://api.ollama.com",   # Ollama server URL
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

- **model**: Model identifier (e.g., `"qwen3.5:397b"`, `"mistral:latest"`)
- **base_url**: Ollama cloud server endpoint (default: `https://api.ollama.com`)
- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **json_mode**: Request JSON-formatted responses when `True`
- **request_timeout**: Timeout for API calls (increase for slower models)
- **keep_alive**: Duration to keep model in memory (e.g., `"5m"`, `"1h"`)
- **additional_kwargs**: Pass any Ollama-specific options

### Embedding Configuration

```python
from serapeum.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",         # Required: embedding model name
    base_url="http://localhost:11434",     # Ollama server URL
    batch_size=10,                         # Batch size (1-2048)
    keep_alive="5m",                       # Model keep-alive duration
    query_instruction=None,                # Prefix for queries
    text_instruction=None,                 # Prefix for documents
    ollama_additional_kwargs={},           # Ollama API options
)
```

---

## Available Models

### Chat/Completion Models

| Model | Size | Description | Best For |
|-------|------|-------------|----------|
| `llama3.1` | 8B-405B | Meta's Llama 3.1 | General purpose (recommended) |
| `llama3.2` | 3B-90B | Latest Llama model | Latest features |
| `mistral` | 7B | Mistral 7B | Fast inference |
| `mixtral` | 8x7B | Mixtral MoE | High quality, efficient |
| `codellama` | 7B-70B | Code-specialized Llama | Code generation |
| `gemma2` | 9B-27B | Google Gemma 2 | Google's latest |

### Embedding Models

| Model | Dimensions | Description |
|-------|-----------|-------------|
| `nomic-embed-text` | 768 | General-purpose embeddings |
| `mxbai-embed-large` | 1024 | High-quality embeddings |
| `snowflake-arctic-embed` | 1024 | Snowflake's model |

**Download models:**

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

**List available models:**

```bash
ollama list
```

---

## Async Operations

Full async support for concurrent operations:

```python
import asyncio
import os
from serapeum.ollama import Ollama
from serapeum.core.llms import Message, MessageRole

async def main():
    llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"))

    # Async chat
    response = await llm.achat([
        Message(role=MessageRole.USER, content="Hello!")
    ])
    print(response.message.content)

    # Async streaming
    messages = [Message(role=MessageRole.USER, content="Count to 5.")]
    stream = await llm.achat(messages, stream=True)

    async for chunk in stream:
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(main())
```

---

## Testing

Run tests for the Ollama provider:

```bash
# All tests
cd libs/providers/ollama
uv run pytest

# Skip tests requiring Ollama server
uv run pytest -m "not e2e"

# Only unit tests
uv run pytest -m unit

# With coverage
uv run pytest --cov=serapeum.ollama
```

**Note:** End-to-end tests require a running Ollama server with models available.

---

## Notes & Limitations

- **Server Required**: Ollama must be running (`ollama serve`) before using this provider
- **Tool Calling**: Depends on model capabilities (llama3.1+ recommended for best results)
- **JSON Mode**: Improves structured output quality when enabled
- **Timeouts**: Increase `request_timeout` for larger models or complex tasks
- **Local Only**: All inference happens on your machine
- **Model Availability**: Only models you've downloaded with `ollama pull` are available

---

## Troubleshooting

### Connection Issues

```python
# Check if Ollama is running
import requests
try:
    response = requests.get("http://localhost:11434/api/tags")
    print("Ollama is running")
    print(f"Available models: {response.json()}")
except Exception as e:
    print(f"Ollama is not running: {e}")
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.1
```

### Timeout Issues

```python
import os
from serapeum.ollama import Ollama
# Increase timeout for slower models
llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    request_timeout=300  # 5 minutes
)
```

---

## Additional Resources

- [Ollama Official Documentation](https://github.com/ollama/ollama)
- [Ollama Model Library](https://ollama.com/library)
- [Serapeum Core Documentation](../core-package.md)
- [API Reference](../../reference/providers/ollama/api_reference.md)

---

## Support

For Ollama-specific issues:
- [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)

For Serapeum integration issues:
- [Serapeum GitHub Issues](https://github.com/Serapieum-of-alex/Serapeum/issues)
