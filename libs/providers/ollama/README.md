# Serapeum Ollama

Serapeum Ollama is the Ollama backend adapter for Serapeum Core. It implements
the `serapeum.core.llms.LLM` interface on top of the Ollama Python client,
supporting chat, streaming, async calls, and tool/function calling.

Use this package when you want Serapeum to talk to a local or remote Ollama
server.

## Requirements

- Python 3.11+
- An Ollama server running locally or remotely
  - Install: https://ollama.com/
  - Run: `ollama serve`
  - Pull a model, e.g.: `ollama pull llama3.1`

## Installation

From the repo:

```bash
cd libs/providers/ollama
python -m pip install -e .
```

## Quick start

### Basic chat

```python
from serapeum.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole

llm = Ollama(model="llama3.1", request_timeout=120)
messages = [Message(role=MessageRole.USER, content="Say pong.")]
response = llm.chat(messages)
print(response)
```

### Completion style usage

```python
from serapeum.ollama import Ollama
from serapeum.core.prompts import PromptTemplate

llm = Ollama(model="llama3.1")
prompt = PromptTemplate("Hello, {name}!")
print(llm.predict(prompt, name="Serapeum"))
```

### JSON mode + structured outputs

```python
from pydantic import BaseModel
from serapeum.ollama import Ollama
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.prompts import PromptTemplate


class Greeting(BaseModel):
    message: str


llm = Ollama(model="llama3.1", json_mode=True)
parser = PydanticParser(output_cls=Greeting)
prompt = PromptTemplate(
    'Return JSON like {"message": "<text>"}. Text: {text}',
    output_parser=parser,
)

result = llm.predict(prompt, text="Hello")  # requires running Ollama server
```

### Tool/function calling

```python
from pydantic import BaseModel, Field
from serapeum.ollama import Ollama
from serapeum.core.llms import ToolOrchestratingLLM


class Album(BaseModel):
    name: str = Field(description="Album name")
    artist: str = Field(description="Artist name")


llm = Ollama(model="llama3.1", request_timeout=120, json_mode=True)
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create an album about {topic}.",
    llm=llm,
)

result = tools_llm(topic="jazz")  # requires running Ollama server
```

## Configuration

`Ollama` accepts the following key settings:

- `model`: Ollama model name (required), e.g. `llama3.1` or `llama3.1:latest`
- `base_url`: Ollama server URL (default `http://localhost:11434`)
- `temperature`: sampling temperature (default `0.75`)
- `context_window`: maximum tokens for prompt + response
- `request_timeout`: timeout (seconds) for requests
- `json_mode`: request JSON format when supported
- `additional_kwargs`: provider-specific options passed to Ollama
- `is_function_calling_model`: set False if your model does not support tools
- `keep_alive`: keep model loaded for a period (e.g. `"5m"`)

# Embeddings

The `serapeum-ollama` package contains Serapeum integrations for generating embeddings using [Ollama](https://ollama.ai/), a tool for running large language models locally.

Ollama allows you to run embedding models on your local machine, providing privacy, cost savings, and the ability to work offline. This integration enables you to use Ollama's embedding models seamlessly with Serapeum's vector store and retrieval systems.

## Installation

To install the `serapeum-ollama` package, run the following command:

```bash
pip install serapeum-ollama
```

You'll also need to have Ollama installed and running on your machine. Visit [ollama.ai](https://ollama.ai/) to download and install Ollama.

## Prerequisites

Before using this integration, ensure you have:

1. **Ollama installed**: Download from [ollama.ai](https://ollama.ai/)
2. **Ollama running**: Start the Ollama service (usually runs on `http://localhost:11434` by default)
3. **An embedding model pulled**: Pull an embedding model using Ollama CLI:
   ```bash
   ollama pull nomic-embed-text
   # or
   ollama pull embeddinggemma
   ```

## Basic Usage

### Simple Embedding Generation

```python
from serapeum.ollama import OllamaEmbedding

# Initialize the embedding model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # or "embeddinggemma"
    base_url="http://localhost:11434",  # default Ollama URL
)

# Generate an embedding for a single text
text_embedding = embed_model.get_text_embedding("Hello, world!")
print(f"Embedding dimension: {len(text_embedding)}")

# Generate an embedding for a query
query_embedding = embed_model.get_query_embedding("What is AI?")
```

### Batch Embedding Generation

```python
# Generate embeddings for multiple texts at once
texts = [
    "The capital of France is Paris.",
    "Python is a programming language.",
    "Machine learning is a subset of AI.",
]

embeddings = embed_model.get_text_embeddings(texts)
print(f"Generated {len(embeddings)} embeddings")
```

## Integration with Serapeum

### Using with Custom LLM

You can combine Ollama embeddings with other LLMs (including Ollama LLMs):

```python
from serapeum.core.configs import Configs
from serapeum.ollama import OllamaEmbedding
from serapeum.ollama import Ollama

# Set both LLM and embedding model
Configs.llm = Ollama(model="llama3.1", base_url="http://localhost:11434")
Configs.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Your documents and indexing code here...
```

## Configuration Options

The `OllamaEmbedding` class supports several configuration options:

```python
from serapeum.ollama import OllamaEmbedding
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",  # Required: Ollama model name
    base_url="http://localhost:11434",  # Optional: Ollama server URL (default: http://localhost:11434)
    embed_batch_size=10,  # Optional: Batch size for embeddings (default: 10)
    keep_alive="5m",  # Optional: How long to keep model in memory (default: "5m")
    query_instruction=None,  # Optional: Instruction to prepend to queries
    text_instruction=None,  # Optional: Instruction to prepend to text
    ollama_additional_kwargs={},  # Optional: Additional kwargs for Ollama API
    client_kwargs={},  # Optional: Additional kwargs for Ollama client
)
```

### Parameter Details

- **`model_name`** (required): The name of the Ollama embedding model to use (e.g., `"nomic-embed-text"`, `"embeddinggemma"`)
- **`base_url`** (optional): The base URL of your Ollama server. Defaults to `"http://localhost:11434"`
- **`embed_batch_size`** (optional): Number of texts to process in each batch. Must be between 1 and 2048. Defaults to 10
- **`keep_alive`** (optional): Controls how long the model stays loaded in memory after a request. Can be a duration string (e.g., `"5m"`, `"10s"`) or a number of seconds. Defaults to `"5m"`
- **`query_instruction`** (optional): Instruction text to prepend to query strings before embedding
- **`text_instruction`** (optional): Instruction text to prepend to document text before embedding
- **`ollama_additional_kwargs`** (optional): Additional keyword arguments to pass to the Ollama API
- **`client_kwargs`** (optional): Additional keyword arguments for the Ollama client (e.g., authentication headers)

## Using Instructions for Better Retrieval

Some embedding models benefit from prepending instructions to queries and documents. This can improve retrieval quality:

```python
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    query_instruction="Represent the question for retrieving supporting documents:",
    text_instruction="Represent the document for retrieval:",
)

# The instructions will be automatically prepended
query_embedding = embed_model.get_query_embedding("What is machine learning?")
# Internally processes: "Represent the question for retrieving supporting documents: What is machine learning?"

text_embedding = embed_model.get_text_embedding(
    "Machine learning is a method of data analysis."
)
# Internally processes: "Represent the document for retrieval: Machine learning is a method of data analysis."
```

## Async Usage

The integration supports asynchronous operations for better performance:

```python
import asyncio
from serapeum.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(model_name="nomic-embed-text")


async def main():
    # Async single embedding
    embedding = await embed_model.aget_text_embedding("Hello, world!")

    # Async batch embeddings
    embeddings = await embed_model.aget_text_embeddings(
        [
            "Text 1",
            "Text 2",
            "Text 3",
        ]
    )

    # Async query embedding
    query_embedding = await embed_model.aget_query_embedding("What is AI?")


asyncio.run(main())
```

## Remote Ollama Server

If you're running Ollama on a remote server, specify the `base_url`:

```python
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://your-remote-server:11434",
)
```

## Available Models

Popular embedding models available in Ollama include:

- **`nomic-embed-text`**: General-purpose embedding model
- **`embeddinggemma`**: Google's Gemma-based embedding model
- **`mxbai-embed-large`**: Large embedding model for better quality

Pull a model using:

```bash
ollama pull nomic-embed-text
```

## Notes

- Ollama must be running (`ollama serve`) before using this adapter.
- Tool calling behavior depends on the underlying model and Ollama version.
- For structured outputs, JSON mode improves reliability when the model supports it.

## Testing

From `libs/providers/ollama/`:

```bash
python -m pytest
```

## Links

- Homepage: https://github.com/Serapieum-of-alex/serapeum
- Docs: https://serapeum.readthedocs.io/
- Changelog: https://github.com/Serapieum-of-alex/serapeum/HISTORY.rst
