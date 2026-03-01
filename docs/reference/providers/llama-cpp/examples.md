# LlamaCPP Usage Examples

This guide provides comprehensive examples covering all possible ways to use the `LlamaCPP` class based on real test cases from the codebase.

## Prerequisites: Model File

LlamaCPP requires a local GGUF model file. You can:
- Download one manually from [HuggingFace](https://huggingface.co/models?search=gguf)
- Let LlamaCPP download one automatically via `model_url` or `hf_model_id`

**Example: set the model path via environment variable:**

```bash
export LLAMA_CPP_MODEL_PATH=/path/to/model.gguf
```

All examples below use a local model path. Replace it with your own.

---

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Initialization Patterns](#initialization-patterns)
3. [Completion Operations](#completion-operations)
4. [Chat Operations](#chat-operations)
5. [Streaming Operations](#streaming-operations)
6. [Async Operations](#async-operations)
7. [Integration with Orchestrators](#integration-with-orchestrators)
8. [Prompt Formatters](#prompt-formatters)

---

## Basic Usage

### Simple Completion

The most straightforward way to use `LlamaCPP`:

```python 
import os 
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

model_path = os.environ.get("LLAMA_MODEL_PATH")
llm = LlamaCPP(
    model_path=model_path,
    temperature=0.1,
    max_new_tokens=256,
    context_window=8192,
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("Say hello.")
print(response.text)  # "Hello! How are you?"
```

### Simple Chat

Using the chat API (via CompletionToChatMixin):

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [Message(role=MessageRole.USER, content="What is 2+2?")]
response = llm.chat(messages)
print(response.message.content)  # "4"
```

---

## Initialization Patterns

### 1. From Local Model Path

Load a GGUF file already on disk:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

### 2. From URL Download

Download and cache a GGUF model from a direct URL:

```python 
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    model_url="https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)
```

### 3. From HuggingFace Hub

Download from HuggingFace Hub using `huggingface_hub`:

```python 
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    hf_model_id="ggml-org/models",
    hf_filename="tinyllamas/stories260K.gguf",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)
```

### 4. Full Configuration

With all common parameters:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    temperature=0.2,
    max_new_tokens=512,
    context_window=8192,
    n_gpu_layers=-1,           # Offload all layers to GPU
    stop=["</s>", "<|eot_id|>"],
    verbose=False,
    generate_kwargs={"top_p": 0.9, "top_k": 40},
    model_kwargs={"n_threads": 8},
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

### 5. With Llama 2 Formatters

Using Llama 2 / Mistral-style prompt templates:

```python notest
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)

llm = LlamaCPP(
    model_path="/models/llama-2-13b-chat.Q4_0.gguf",
    temperature=0.1,
    max_new_tokens=256,
    context_window=4096,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)
```

---

## Completion Operations

### 1. Basic Completion

Simple text completion:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("The capital of France is")
print(response.text)  # "Paris"
```

### 2. Completion with Custom Parameters

Override default generation settings:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete(
    "Once upon a time",
    temperature=0.8,
    max_tokens=200,
)
print(response.text)
```

### 3. Pre-formatted Prompt

Pass an already-formatted prompt:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

# Build the prompt manually
prompt = completion_to_prompt_v3_instruct("Say hello.")

# Pass formatted=True to skip the automatic formatter
response = llm.complete(prompt, formatted=True)
print(response.text)
```

### 4. Completion with Stop Tokens

Stop generation at specific tokens:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    stop=["</s>", "<|eot_id|>", "<|end|>"],
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("Say hello.")
# Stop tokens are stripped from output
```

### 5. Raw Response Access

Access the underlying llama-cpp-python response:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

response = llm.complete("Say hello.")
print(response.raw)  # Full dict from llama-cpp-python
print(response.raw["choices"])  # List of generated choices
print(response.raw["usage"])    # Token usage statistics
```

---

## Chat Operations

Chat is provided by `CompletionToChatMixin`, which formats messages using
`messages_to_prompt` and delegates to `complete()`.

### 1. Single Turn Chat

Basic conversation:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [Message(role=MessageRole.USER, content="What is 2+2?")]
response = llm.chat(messages)
print(response.message.content)  # "4"
print(response.message.role)     # MessageRole.ASSISTANT
```

### 2. Multi-turn Conversation

With conversation history:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [
    Message(role=MessageRole.USER, content="My name is Alex."),
    Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alex!"),
    Message(role=MessageRole.USER, content="What is my name?"),
]

response = llm.chat(messages)
print(response.message.content)  # Should mention "Alex"
```

### 3. Chat with System Message

System message for context:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful math tutor."),
    Message(role=MessageRole.USER, content="What is the square root of 144?"),
]

response = llm.chat(messages)
print(response.message.content)  # "12"
```

---

## Streaming Operations

### 1. Stream Completion

Real-time streaming completion:

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

for chunk in llm.complete("Count from 1 to 5.", stream=True):
    print(chunk.delta, end="", flush=True)
    # "1" ", " "2" ", " "3" ", " "4" ", " "5"
```

### 2. Stream Chat

Real-time streaming chat:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

messages = [Message(role=MessageRole.USER, content="Tell me a joke.")]
for chunk in llm.chat(messages, stream=True):
    print(chunk.delta, end="", flush=True)
```

### 3. Processing Stream with Delta

Access incremental content:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

full_response = ""
for chunk in llm.complete("Say hello.", stream=True):
    delta = chunk.delta  # Incremental content
    if delta:
        full_response += delta
        print(delta, end="", flush=True)

print(f"\n\nFull response: {full_response}")

# Alternatively, the last chunk's .text contains the full accumulated text:
chunks = list(llm.complete("Say hello.", stream=True))
final_text = chunks[-1].text  # Same as joining all deltas
```

---

## Async Operations

LlamaCPP offloads CPU-bound inference to a thread pool via `asyncio.to_thread`,
so async calls do not block the event loop.

### 1. Async Completion

Non-blocking completion:

```python
import os
import asyncio
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


async def main():
    llm = await asyncio.to_thread(
        LlamaCPP,
        model_path=os.environ.get("LLAMA_MODEL_PATH"),
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    response = await llm.acomplete("Say hello.")
    print(response.text)


asyncio.run(main())
```

### 2. Async Chat

Non-blocking chat:

```python
import os
import asyncio
from serapeum.llama_cpp import LlamaCPP
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


async def main():
    llm = await asyncio.to_thread(
        LlamaCPP,
        model_path=os.environ.get("LLAMA_MODEL_PATH"),
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    messages = [Message(role=MessageRole.USER, content="Hello!")]
    response = await llm.achat(messages)
    print(response.message.content)


asyncio.run(main())
```

### 3. Async Streaming

Non-blocking streaming completion:

```python
import os
import asyncio
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


async def main():
    llm = await asyncio.to_thread(
        LlamaCPP,
        model_path=os.environ.get("LLAMA_MODEL_PATH"),
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    gen = await llm.acomplete("Count to 5", stream=True)
    async for chunk in gen:
        print(chunk.delta, end="", flush=True)


asyncio.run(main())
```

### 4. Concurrent Async Requests

Process multiple requests concurrently:

```python 
import os
import asyncio
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)


async def main():
    llm = await asyncio.to_thread(
        LlamaCPP,
        model_path=os.environ.get("LLAMA_MODEL_PATH"),
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

    prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]

    tasks = [llm.acomplete(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)

    for prompt, response in zip(prompts, responses):
        print(f"{prompt} -> {response.text}")


asyncio.run(main())
```

---

## Integration with Orchestrators

### 1. With TextCompletionLLM

Use LlamaCPP with `TextCompletionLLM` for structured outputs:

```python
import os
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
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

parser = PydanticParser(output_cls=DummyModel)

text_llm = TextCompletionLLM(
    output_parser=parser,
    prompt="Value: {value}",
    llm=llm,
)

result = text_llm(value="input")
print(result.value)  # "input"
```

---

## Prompt Formatters

LlamaCPP **requires** explicit prompt formatters. Using the wrong formatter
for your model produces garbage output.

### Available Formatters

#### Llama 2 / Mistral

```python 
from serapeum.llama_cpp.formatters.llama2 import (
    messages_to_prompt,
    completion_to_prompt,
)
```

Uses the `[INST]...[/INST]` format:
```
<s> [INST] <<SYS>> system prompt <</SYS>> user message [/INST] assistant reply </s>
```

#### Llama 3 Instruct

```python 
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)
```

Uses the `<|start_header_id|>...<|eot_id|>` format:
```
<|start_header_id|>system<|end_header_id|>
system prompt<|eot_id|>
<|start_header_id|>user<|end_header_id|>
user message<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

### Formatter Behavior

**messages_to_prompt**: Converts a list of `Message` objects to a single formatted
string. Handles system, user, and assistant roles, including multi-turn conversations.

**completion_to_prompt**: Wraps a plain string in the model's instruct format
with a default system prompt.

### Custom Formatters

You can write your own formatters for other model families:

```python 
from collections.abc import Sequence
from serapeum.core.llms import Message, MessageRole
from serapeum.llama_cpp import LlamaCPP


def my_messages_to_prompt(messages: Sequence[Message]) -> str:
    """Custom formatter for your model."""
    parts = []
    for msg in messages:
        if msg.role == MessageRole.SYSTEM:
            parts.append(f"[SYSTEM] {msg.content}")
        elif msg.role == MessageRole.USER:
            parts.append(f"[USER] {msg.content}")
        elif msg.role == MessageRole.ASSISTANT:
            parts.append(f"[ASSISTANT] {msg.content}")
    parts.append("[ASSISTANT]")
    return "\n".join(parts)


def my_completion_to_prompt(completion: str) -> str:
    """Custom formatter for completion."""
    return f"[SYSTEM] You are helpful.\n[USER] {completion}\n[ASSISTANT]"


llm = LlamaCPP(
    model_path="/models/my-model.gguf",
    messages_to_prompt=my_messages_to_prompt,
    completion_to_prompt=my_completion_to_prompt,
)
```

---

## Tokenization

LlamaCPP exposes the model's tokenizer for prompt length checking:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

# Get token IDs
tokens = llm.tokenize("Hello, world!")
print(tokens)  # [1, 15043, 29892, 3186, 29991]

# Count tokens
count = llm.count_tokens("Hello, world!")
print(count)  # 5
```

---

## Best Practices

### 1. Reuse LLM Instances

Create once, use many times — model loading is expensive:

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

# Good: Create once
llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

# Reuse for multiple calls (model stays loaded)
response1 = llm.complete("Hello!")
response2 = llm.complete("How are you?")

# Bad: Don't recreate for each call — model reloads every time
def process(prompt):
    llm = LlamaCPP(model_path="...", ...)  # Expensive!
    return llm.complete(prompt)
```

### 2. Match Formatter to Model

Always use the correct formatter for your model family:

```python 
from serapeum.llama_cpp import LlamaCPP
# Llama 2 / Mistral models
from serapeum.llama_cpp.formatters.llama2 import messages_to_prompt, completion_to_prompt

# Llama 3 Instruct models
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct, completion_to_prompt_v3_instruct
)
```

### 3. Handle Errors Gracefully

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

try:
    response = llm.complete("Hello")
except ValueError as e:
    print(f"Prompt too long: {e}")
except RuntimeError as e:
    print(f"Model error: {e}")
```

### 4. Use GPU for Performance

```python
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

# Offload all layers to GPU
llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    n_gpu_layers=-1,
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)
```

### 5. Use Temperature=0 for Deterministic Output

```python 
import os
from serapeum.llama_cpp import LlamaCPP
from serapeum.llama_cpp.formatters.llama3 import (
    messages_to_prompt_v3_instruct,
    completion_to_prompt_v3_instruct,
)

# Greedy decoding for reproducible results
llm = LlamaCPP(
    model_path=os.environ.get("LLAMA_MODEL_PATH"),
    temperature=0.0,
    messages_to_prompt=messages_to_prompt_v3_instruct,
    completion_to_prompt=completion_to_prompt_v3_instruct,
)

# Same prompt always produces same output
r1 = llm.complete("The capital of France is")
r2 = llm.complete("The capital of France is")
assert r1.text == r2.text
```

---

## See Also

- [Execution Flow and Method Calls](./llama_cpp_sequence.md) - Detailed sequence diagrams
- [Architecture and Class Relationships](./llama_cpp_class.md) - Class structure
- [Data Transformations and Validation](./llama_cpp_dataflow.md) - Data flow details
- [Component Boundaries and Interactions](./llama_cpp_components.md) - System components
- [Lifecycle and State Management](./llama_cpp_state.md) - State management
