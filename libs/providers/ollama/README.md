# Serapeum Ollama

Serapeum Ollama is the Ollama backend adapter for Serapeum Core. It implements
the `serapeum.core.llm.LLM` interface on top of the Ollama Python client,
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
cd libs/providers/serapeum-ollama
python -m pip install -e .
```

## Quick start

### Basic chat

```python
from serapeum.llms.ollama import Ollama
from serapeum.core.base.llms.types import Message, MessageRole

llm = Ollama(model="llama3.1", request_timeout=120)
messages = [Message(role=MessageRole.USER, content="Say pong.")]
response = llm.chat(messages)
print(response)
```

### Completion style usage

```python
from serapeum.llms.ollama import Ollama
from serapeum.core.prompts import PromptTemplate

llm = Ollama(model="llama3.1")
prompt = PromptTemplate("Hello, {name}!")
print(llm.predict(prompt, name="Serapeum"))
```

### JSON mode + structured outputs

```python
from pydantic import BaseModel
from serapeum.llms.ollama import Ollama
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

# result = llm.predict(prompt, text="Hello")  # requires running Ollama server
```

### Tool/function calling

```python
from pydantic import BaseModel, Field
from serapeum.llms.ollama import Ollama
from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM


class Album(BaseModel):
    name: str = Field(description="Album name")
    artist: str = Field(description="Artist name")


llm = Ollama(model="llama3.1", request_timeout=120, json_mode=True)
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create an album about {topic}.",
    llm=llm,
)

# result = tools_llm(topic="jazz")  # requires running Ollama server
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

## Notes

- Ollama must be running (`ollama serve`) before using this adapter.
- Tool calling behavior depends on the underlying model and Ollama version.
- For structured outputs, JSON mode improves reliability when the model supports it.

## Testing

From `libs/providers/serapeum-ollama/`:

```bash
python -m pytest
```

## Links

- Homepage: https://github.com/Serapieum-of-alex/serapeum
- Docs: https://serapeum.readthedocs.io/
- Changelog: https://github.com/Serapieum-of-alex/serapeum/HISTORY.rst
