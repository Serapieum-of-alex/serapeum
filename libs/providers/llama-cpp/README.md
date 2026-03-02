# Serapeum llama.cpp Provider

**Local GGUF model inference for the Serapeum LLM framework**

The `serapeum-llama-cpp` package runs quantised GGUF models locally using the
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) backend. It provides:

- **Completion & Chat**: Sync and async text generation with streaming
- **Multiple Model Sources**: Load from a local path, direct URL, or HuggingFace Hub
- **Prompt Formatters**: Ready-made formatters for Llama 2/Mistral and Llama 3 Instruct
- **GPU Offloading**: Configurable layer offloading via `n_gpu_layers`
- **Thread Safety**: Per-instance locking for concurrent inference
- **Model Caching**: Shared `WeakValueDictionary` cache avoids duplicate loads

This adapter implements the `serapeum.core.llms.LLM` completion interface with
`CompletionToChatMixin`, making it compatible with all Serapeum orchestrators and tools.

## Installation

### From Source

```bash
cd libs/providers/llama-cpp
uv sync --active
```

### From PyPI (when published)

```bash
pip install serapeum-llama-cpp
```

## Prerequisites

Before running the examples you need a GGUF model file. For a quick test you can
download the tiny **stories260K** model (~500 KB):

```bash
# Download the model
curl -L -o stories260K.gguf \
  https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```
or just copy and paste the URL into your browser, and the download will start.

Set the `LLAMA_MODEL_PATH` environment variable to the downloaded file:

```bash
export LLAMA_MODEL_PATH="/path/to/stories260K.gguf"
```

Or add it to a `.env` file in your project root and load it before running:

```bash
# .env
LLAMA_MODEL_PATH=/path/to/stories260K.gguf
```

```python
from dotenv import load_dotenv
load_dotenv()  # loads LLAMA_MODEL_PATH from .env
```

All examples below read the model path from this environment variable.

## Quick Start

```python
import os
from dotenv import load_dotenv
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

load_dotenv()
model_path = os.environ["LLAMA_MODEL_PATH"]

llm = LlamaCPP(
    model_path=model_path,
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("Once upon a time")
print(response.text)
```

## Model Sources

### Local File

```python
import os
from dotenv import load_dotenv
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

load_dotenv()

llm = LlamaCPP(
    model_path=os.environ["LLAMA_MODEL_PATH"],
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

### Direct URL

The model is downloaded, cached, and reused on subsequent runs:

```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    model_url="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)
```

### HuggingFace Hub

Downloads via `huggingface_hub` with automatic caching and SHA-256 verification:

```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    hf_model_id="TheBloke/Llama-2-13B-chat-GGUF",
    hf_filename="llama-2-13b-chat.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)
```

## Prompt Formatters

GGUF models require a specific chat template. Using the wrong formatter produces
garbage output. Choose the formatter that matches your model family:

```python
# Llama 3 Instruct / newer models
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

# Llama 2 / Mistral
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)
```

## Streaming

```python
import os
from dotenv import load_dotenv
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

load_dotenv()

llm = LlamaCPP(
    model_path=os.environ["LLAMA_MODEL_PATH"],
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

for chunk in llm.complete("Once upon a time", stream=True):
    print(chunk.delta, end="", flush=True)
print()
```

## Async

```python
import asyncio
import os
from dotenv import load_dotenv
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

load_dotenv()

llm = LlamaCPP(
    model_path=os.environ["LLAMA_MODEL_PATH"],
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

async def main():
    response = await llm.acomplete("Once upon a time")
    print(response.text)

asyncio.run(main())
```

## Configuration

```python
import os
from dotenv import load_dotenv
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

load_dotenv()

llm = LlamaCPP(
    model_path=os.environ["LLAMA_MODEL_PATH"],
    temperature=0.1,              # Sampling temperature (0.0–1.0)
    max_new_tokens=256,           # Maximum tokens to generate
    context_window=4096,          # Context window size
    n_gpu_layers=-1,              # GPU layers (-1 = all)
    stop=["</s>", "<|eot_id|>"], # Stop sequences
    verbose=False,                # Suppress llama.cpp output
    generate_kwargs={},           # Extra kwargs for Llama.__call__
    model_kwargs={},              # Extra kwargs for Llama.__init__
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

## Testing

```bash
# Unit and mock tests (no model required)
python -m pytest libs/providers/llama-cpp/tests -m "not e2e"

# End-to-end tests (requires a GGUF model)
# Set LLAMA_CPP_TEST_MODEL_PATH to your model file
python -m pytest libs/providers/llama-cpp/tests -m e2e
```

## Links

- **Homepage**: [github.com/Serapieum-of-alex/serapeum](https://github.com/Serapieum-of-alex/serapeum)
- **Documentation**: [serapieum-of-alex.github.io/serapeum](https://serapieum-of-alex.github.io/serapeum)
- **llama-cpp-python**: [github.com/abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
