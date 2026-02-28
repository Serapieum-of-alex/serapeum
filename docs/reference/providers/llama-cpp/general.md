# LlamaCPP LLM Integration

This directory contains comprehensive documentation explaining the complete workflow of the `LlamaCPP` class, from initialization to execution across various modes (completion, chat, streaming, async).

## Overview

The `LlamaCPP` class is a local LLM integration that provides:
1. **Local GGUF model inference** via llama-cpp-python
2. **Chat and completion APIs** with sync/async support
3. **Streaming responses** for real-time output
4. **Multiple model sources** — local path, URL download, or HuggingFace Hub
5. **Model caching** — shared memory across instances
6. **Thread-safe inference** — serialized calls to the C backend

## Example Usage

### Basic Completion

```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=8192,
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("Hello, how are you?")
print(response.text)
```

### With TextCompletionLLM

```python notest
from pydantic import BaseModel
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


class DummyModel(BaseModel):
    value: str


llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

text_llm = TextCompletionLLM(
    output_parser=PydanticParser(output_cls=DummyModel),
    prompt="Generate any value: {value}",
    llm=llm,
)

result = text_llm(value="input")
# Returns: DummyModel(value="input")
```

### Basic Chat

```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [Message(role=MessageRole.USER, content="Say hello.")]
response = llm.chat(messages)
print(response.message.content)
```

## Understanding the Workflow

### 1. [Execution Flow and Method Calls](./llama_cpp_sequence.md)
Shows the chronological flow of method calls and interactions across all usage patterns.

**Best for**:
- Understanding the order of operations
- Seeing how LlamaCPP loads and invokes models
- Debugging execution flow
- Understanding the completion-to-chat bridge

**Key Flows**:
- Initialization phase (model download / load)
- Direct completion calls
- Chat via CompletionToChatMixin
- Streaming execution
- Integration with TextCompletionLLM
- Async operations via thread pool

### 2. [Architecture and Class Relationships](./llama_cpp_class.md)
Illustrates the static structure, inheritance hierarchy, and relationships.

**Best for**:
- Understanding the architecture
- Seeing inheritance chain (BaseLLM → LLM → CompletionToChatMixin → LlamaCPP)
- Identifying class responsibilities
- Understanding integration points

**Key Classes**:
- `LlamaCPP`: Main LLM implementation
- `CompletionToChatMixin`: Bridges completion to chat interface
- `LLM`: High-level orchestration
- `BaseLLM`: Core interface
- `Llama`: Underlying llama-cpp-python model
- Response models: `ChatResponse`, `CompletionResponse`, `Message`

### 3. [Data Transformations and Validation](./llama_cpp_dataflow.md)
Tracks how data transforms through the system across different operation modes.

**Best for**:
- Understanding data transformations
- Identifying validation points
- Seeing error handling paths
- Understanding request/response formats

**Key Flows**:
- Initialization, model resolution, and loading
- Completion request building and response parsing
- Chat via prompt formatting bridge
- Streaming chunk processing
- Error handling pipelines

### 4. [Component Boundaries and Interactions](./llama_cpp_components.md)
Shows component boundaries, responsibilities, and interaction patterns.

**Best for**:
- Understanding system architecture
- Seeing component responsibilities
- Identifying interaction patterns
- Understanding the local inference stack

**Key Components**:
- User space (application code)
- LlamaCPP core (prompt formatting, context guard, generation)
- Model layer (Llama instance, model cache)
- Formatter layer (Llama 2 / Llama 3 prompt templates)
- Orchestrator layer (TextCompletionLLM)

### 5. [Lifecycle and State Management](./llama_cpp_state.md)
Depicts the lifecycle states, transitions, and state variables.

**Best for**:
- Understanding instance lifecycle
- Seeing state transitions
- Identifying error states
- Understanding concurrency considerations

**Key States**:
- Uninitialized → ModelLoading (initialization)
- ModelLoading → Ready (model loaded)
- Ready ↔ Processing* (request handling)
- Processing → Error → Ready (error handling)

### 6. [Usage Examples](./examples.md)
Comprehensive examples from real test cases.

**Best for**:
- Learning by example
- Understanding practical usage
- Seeing all API variants
- Integration patterns

**Key Examples**:
- Basic completion and chat
- Streaming operations
- Model loading from different sources
- Async operations
- Error handling
- Prompt formatters

## Core Capabilities

### 1. Completion API
```
Direct text generation:
- Single-prompt completions
- Pre-formatted prompt support
- Custom generation parameters (temperature, max_tokens, etc.)
- Stop token support
```

### 2. Chat API
```
Chat interface via CompletionToChatMixin:
- Single and multi-turn conversations
- System messages supported by formatters
- Converts messages → formatted prompt → completion → ChatResponse
```

### 3. Streaming
```
Real-time response generation:
- Stream completion responses
- Stream chat responses (via mixin)
- Chunk-by-chunk processing with delta content
```

### 4. Model Loading
```
Multiple model sources:
- Local GGUF file path
- Download from URL with progress bar
- Download from HuggingFace Hub
- Automatic caching and deduplication
```

### 5. Async Operations
```
Non-blocking execution via thread pool:
- Async completion (asyncio.to_thread)
- Async streaming (collect in thread, re-yield async)
- Async chat (via mixin bridge)
- Concurrent request handling
```

## Key Design Patterns

### 1. **Eager Initialization**
Model is loaded during `model_post_init`, not on first use:
```python notest
def model_post_init(self, __context):
    model_path = self._resolve_model_path()
    self._model = self._load_model(model_path)
```

### 2. **CompletionToChatMixin (Adapter Pattern)**
Chat is derived from completion, the inverse of Ollama:
```python notest
# CompletionToChatMixin provides chat() by:
# 1. Formatting messages with messages_to_prompt()
# 2. Calling complete() with the formatted string
# 3. Wrapping the result in a ChatResponse
```

### 3. **Model Cache (Flyweight Pattern)**
Models are cached in a module-level WeakValueDictionary:
```python notest
_MODEL_CACHE: WeakValueDictionary[tuple[str, str], Llama] = WeakValueDictionary()

def _load_model(self, model_path: Path) -> Llama:
    cache_key = (str(model_path), json.dumps(self.model_kwargs, sort_keys=True))
    # Double-checked locking for thread safety
    ...
```

### 4. **Thread Safety**
Inference is serialized per instance to prevent C-level races:
```python notest
with self._model_lock:
    response = self._model(prompt=prompt, **call_kwargs)
```

## Integration Architecture

```
User Application
    ↓
TextCompletionLLM
    ↓
LlamaCPP
    ↓
Prompt Formatter (Llama 2 / Llama 3)
    ↓
Llama (llama-cpp-python)
    ↓
GGUF Model File (local)
```

### TextCompletionLLM Integration
```
1. Formats prompt with variables
2. Checks is_chat_model → True
3. Calls LlamaCPP.chat()
4. Parses response with PydanticParser
5. Returns validated model instance
```

## Performance Considerations

1. **Model Caching**: WeakValueDictionary shares model across instances
2. **Thread Pool**: Async calls offloaded via `asyncio.to_thread`
3. **Context Guard**: Early rejection of oversized prompts before inference
4. **GPU Offloading**: `n_gpu_layers` parameter for GPU acceleration
5. **Streaming**: Reduces perceived latency for long responses
6. **Lock Granularity**: Per-instance lock for concurrent callers

## Configuration Options

### Model Source (one required)
- `model_path`: Path to a local GGUF file
- `model_url`: URL to download a GGUF file
- `hf_model_id` + `hf_filename`: HuggingFace Hub repository and filename

### Generation
- `temperature`: Sampling temperature (0.0–1.0, default: 0.75)
- `max_new_tokens`: Maximum tokens to generate (default: 256)
- `context_window`: Maximum context tokens (default: 3900)
- `stop`: List of stop token sequences (default: [])

### Model Loading
- `n_gpu_layers`: Layers to offload to GPU (0 = CPU only, -1 = all)
- `verbose`: Print llama.cpp verbose output (default: False)
- `model_kwargs`: Additional kwargs for `Llama()` constructor
- `generate_kwargs`: Additional kwargs for generation calls

### Prompt Formatting (required)
- `messages_to_prompt`: Callable to convert messages to prompt string
- `completion_to_prompt`: Callable to wrap a completion string in instruct format

## Error Handling

### Configuration Errors
```
ValueError: No model source provided (model_path, model_url, or hf_model_id)
ValueError: hf_filename required when hf_model_id is set
ValueError: model_path does not exist
ValueError: Missing prompt formatters (messages_to_prompt, completion_to_prompt)
```

### Model Loading Errors
```
RuntimeError: Download succeeded but file not found
ValueError: Content-Length too small (invalid model file)
ImportError: huggingface-hub not installed (for HF downloads)
requests.ConnectionError: Cannot reach download URL
```

### Inference Errors
```
ValueError: Prompt exceeds context_window
RuntimeError: llama-cpp-python backend error
```

## Prerequisites

### System Requirements
```bash
# Install llama-cpp-python (CPU)
pip install llama-cpp-python

# Or with GPU support (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Or with Metal support (macOS)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

### Python Requirements
```bash
# Install serapeum-llama-cpp
pip install serapeum-llama-cpp

# Or install from source
uv pip install -e libs/providers/llama-cpp
```

### Model Files
```bash
# Download a GGUF model (e.g., from HuggingFace)
# Option 1: Let LlamaCPP download automatically via model_url or hf_model_id
# Option 2: Download manually
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf
```

## Common Patterns

### Pattern 1: Reusable Instance
```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

# Create once (model loads here)
llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

# Reuse many times (no re-loading)
response1 = llm.complete("Hello!")
response2 = llm.complete("How are you?")
```

### Pattern 2: Streaming for Long Responses
```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

for chunk in llm.complete("Tell me a joke.", stream=True):
    print(chunk.delta, end="", flush=True)
```

### Pattern 3: GPU Acceleration
```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
    n_gpu_layers=-1,  # Offload all layers to GPU
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

### Pattern 4: Async for Concurrency
```python notest
import asyncio
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


async def main():
    # Wrap construction in to_thread to avoid blocking
    llm = await asyncio.to_thread(
        LlamaCPP,
        model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    response = await llm.acomplete("Say hello.")
    print(response.text)


asyncio.run(main())
```

## Troubleshooting

### Issue: Model Loading Takes Too Long
```
Solution: Use n_gpu_layers to offload to GPU, or use a smaller quantized model.
  llm = LlamaCPP(model_path="...", n_gpu_layers=-1, ...)
```

### Issue: Garbage Output
```
Solution: Ensure the prompt formatter matches the model family.
  Llama 2 / Mistral → formatters.llama2.messages_to_prompt
  Llama 3           → formatters.llama3.messages_to_prompt_v3_instruct
```

### Issue: Prompt Exceeds Context Window
```
Solution: Increase context_window or shorten the prompt.
  llm = LlamaCPP(model_path="...", context_window=8192, ...)
```

### Issue: ImportError for HuggingFace
```
Solution: Install the optional dependency.
  pip install huggingface-hub
```

### Issue: Blocking the Event Loop
```
Solution: Wrap construction in asyncio.to_thread.
  llm = await asyncio.to_thread(LlamaCPP, model_path="...", ...)
```

## Next Steps

1. **Start with [Examples](./examples.md)** for practical usage patterns
2. **Review [Sequence Diagrams](./llama_cpp_sequence.md)** to understand execution flow
3. **Study [Class Diagram](./llama_cpp_class.md)** to understand architecture
4. **Explore [Data Flow](./llama_cpp_dataflow.md)** to understand transformations
5. **Check [State Management](./llama_cpp_state.md)** for lifecycle details

## See Also

- [TextCompletionLLM](../../core/llms/orchestrators/text_completion_llm/general.md) - Structured completion orchestrator
- [Ollama Provider](../ollama/general.md) - Remote Ollama server integration
- [llama-cpp-python Documentation](https://github.com/abetlen/llama-cpp-python) - Backend library
