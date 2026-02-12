# Serapeum Core

Serapeum Core is the provider-agnostic foundation for the Serapeum ecosystem. It
defines the core LLM abstractions, prompt templates, structured-output parsers,
and tool-execution utilities used by integration packages (OpenAI, Ollama, etc.).

If you are implementing a new LLM backend or want a consistent set of models and
helpers for prompts, output parsing, and tool calling, this is the package to use.

## Highlights

- Unified LLM interfaces for chat/completions, streaming, and async variants.
- Prompt templates for string and chat workflows with variable mapping utilities.
- Structured output parsing with Pydantic models and JSON schema hints.
- Tool metadata, schema generation, and safe tool execution utilities.
- Orchestrators for tool calling and structured outputs.

## Package layout

- `serapeum.core.base.llms`: Base interfaces and data models (messages, chunks,
  responses, metadata).
- `serapeum.core.llms`: High-level LLM class with predict/stream helpers and
  structured output utilities.
- `serapeum.core.prompts`: Prompt templates for text and chat models.
- `serapeum.core.output_parsers`: Parsers for structured outputs (e.g., Pydantic).
- `serapeum.core.tools`: Tool metadata, tool interfaces, and tool execution.
- `serapeum.core.llms.orchestrators`: Higher-level programs for structured outputs
  and function-calling orchestration.
- `serapeum.core.utils`: JSON/schema helpers and async utilities.
- `serapeum.core.configs`: Global configuration singleton (`Configs`).
- `serapeum.core.chat`: Agent response container with streaming helpers.

## Installation

From the repo:

```bash
cd serapeum-core
python -m pip install -e .
```

Python 3.11+ is required.

## Quick start

### 1) Build a minimal LLM implementation

```python
from serapeum.core.llms import LLM, CompletionResponse, Metadata
from serapeum.core.prompts import PromptTemplate


class EchoLLM(LLM):
  metadata = Metadata.model_construct(is_chat_model=False)

  def chat(self, messages, **kwargs):
    raise NotImplementedError()

  def stream_chat(self, messages, **kwargs):
    raise NotImplementedError()

  async def achat(self, messages, **kwargs):
    raise NotImplementedError()

  async def astream_chat(self, messages, **kwargs):
    raise NotImplementedError()

  def complete(self, prompt, formatted=False, **kwargs):
    return CompletionResponse(text=prompt, delta=prompt)

  def stream_complete(self, prompt, formatted=False, **kwargs):
    raise NotImplementedError()

  async def acomplete(self, prompt, formatted=False, **kwargs):
    return CompletionResponse(text=prompt, delta=prompt)

  async def astream_complete(self, prompt, formatted=False, **kwargs):
    raise NotImplementedError()


llm = EchoLLM()
prompt = PromptTemplate("Hello, {name}!")
result = llm.predict(prompt, name="Serapeum")
print(result)
```

### 2) Parse structured outputs with Pydantic

```python
from pydantic import BaseModel
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.prompts import PromptTemplate


class Greeting(BaseModel):
    message: str


parser = PydanticParser(output_cls=Greeting)
prompt = PromptTemplate(
    'Return JSON like {"message": "<text>"}. Text: {text}',
    output_parser=parser,
)

# With a real LLM backend, this returns a validated Greeting model.
# result = llm.predict(prompt, text="Hello")
```

### 3) Define tools and execute them safely

```python
from serapeum.core.tools import BaseTool, ToolMetadata, ToolOutput, ToolExecutor


class EchoTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(description="Echo input", name="echo")

    def __call__(self, input_values: dict) -> ToolOutput:
        return ToolOutput(tool_name="echo", content=input_values.get("input", ""))


tool = EchoTool()
executor = ToolExecutor()
output = executor.execute(tool, {"input": "hi"})
print(output.content)
```

## Core concepts

### LLM interface and orchestration

- `BaseLLM` defines the provider contract for chat/completion endpoints, streaming,
  and async variants.
- `LLM` builds on `BaseLLM` with helpers for prompt formatting, prediction, and
  structured output modes.
- `FunctionCallingLLM` extends `LLM` with tool-calling helper methods.

### Prompts

Use `PromptTemplate` for string prompts and `ChatPromptTemplate` for message
templates. Both support template variable mappings and optional output parsers.

### Output parsers

`PydanticParser` injects a compact JSON schema into prompts and parses LLM output
into validated Pydantic models. `output_parsers.utils` also provides markdown
JSON/code extraction helpers.

### Tools and schemas

- `ToolMetadata` produces OpenAI-style tool specs and JSON schema for tool inputs.
- `CallableTool` (in `serapeum.core.tools.callable_tool`) can wrap functions or
  Pydantic models into tool definitions.
- `ToolExecutor` runs tools safely with optional auto-unpacking and error
  normalization.

### Structured programs

- `TextCompletionLLM` runs a prompt + parser + LLM pipeline to return Pydantic
  outputs for completion-style models.
- `ToolOrchestratingLLM` uses function-calling models to produce structured
  outputs via tools.

## Configuration

`serapeum.core.configs.Configs` is a small global configuration holder. You can
set a default LLM instance and control structured output mode:

```python
from serapeum.core.configs import Configs
from serapeum.core.types import StructuredLLMMode


Configs.llm = llm
Configs.pydantic_program_mode = StructuredLLMMode.OPENAI
```

## Links

- Homepage: https://github.com/Serapieum-of-alex/serapeum
- Docs: https://serapeum.readthedocs.io/
- Changelog: https://github.com/Serapieum-of-alex/serapeum/HISTORY.rst
