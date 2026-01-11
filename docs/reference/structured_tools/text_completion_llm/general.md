# TextCompletionLLM Workflow

This directory contains comprehensive explaining the complete workflow of the `TextCompletionLLM` class, from initialization to execution and output parsing.

## Overview

The `TextCompletionLLM` is a structured text completion runner that orchestrates:
1. **Prompt formatting** with template variables
2. **LLM execution** (chat or completion mode)
3. **Output parsing** into validated Pydantic models

## Example Usage

```python
from pydantic import BaseModel
from serapeum.core.output_parsers.models import PydanticOutputParser
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
from serapeum.llms.ollama import Ollama

# Define the output schema
class ModelTest(BaseModel):
    hello: str

# Initialize components
LLM = Ollama(model="llama3.1", request_timeout=180)
output_parser = PydanticOutputParser(output_cls=ModelTest)

# Create TextCompletionLLM instance
text_llm = TextCompletionLLM(
    output_parser=output_parser,
    prompt="This is a test prompt with a {test_input}.",
    llm=LLM,
)

# Execute and get structured output
obj_output = text_llm(test_input="hello")
# Returns: ModelTest(hello="...")
```

## Available Diagrams

### 1. [Sequence Diagram](./text_completion_llm_sequence.md)
**Purpose**: Shows the chronological flow of method calls and interactions

**Best for**:
- Understanding the order of operations
- Seeing how objects communicate
- Debugging execution flow

**Key Sections**:
- Initialization phase (validation and component setup)
- Execution phase (prompt formatting and LLM invocation)
- Parsing phase (JSON extraction and Pydantic validation)

### 2. [Class Diagram](./text_completion_llm_class.md)
**Purpose**: Illustrates the static structure and relationships between classes

**Best for**:
- Understanding the architecture
- Seeing inheritance and composition
- Identifying class responsibilities

**Key Classes**:
- `TextCompletionLLM`: Main orchestrator
- `PydanticOutputParser`: Output validation
- `BasePromptTemplate` hierarchy: Prompt formatting
- `LLM` hierarchy: Model execution

### 3. [Data Flow Diagram](./text_completion_llm_dataflow.md)
**Purpose**: Tracks how data transforms through the system

**Best for**:
- Understanding data transformations
- Identifying validation points
- Seeing error handling paths

**Key Flows**:
- Initialization validation pipeline
- Chat model execution path
- Completion model execution path
- JSON parsing and validation

### 4. [Component Interaction Diagram](./text_completion_llm_components.md)
**Purpose**: Shows component boundaries and interaction patterns

**Best for**:
- Understanding system architecture
- Seeing component responsibilities
- Identifying interaction patterns

**Key Components**:
- User space (application code)
- TextCompletionLLM layer (orchestration)
- Prompt layer (template formatting)
- LLM layer (model execution)
- Parser layer (output validation)
- External service (Ollama server)

### 5. [State Diagram](./text_completion_llm_state.md)
**Purpose**: Depicts the lifecycle states and transitions

**Best for**:
- Understanding instance lifecycle
- Seeing state transitions
- Identifying error states

**Key States**:
- Initialization states (validation)
- Execution states (chat vs completion)
- Parsing states (JSON extraction and validation)
- Error states (initialization, execution, parsing)

## Workflow Summary

### Initialization Workflow
```
1. Create PydanticOutputParser with output schema (ModelTest)
2. Initialize TextCompletionLLM with:
   - output_parser: PydanticOutputParser
   - prompt: String or BasePromptTemplate
   - llm: Ollama instance
3. Validation occurs:
   - Parser type and output_cls extraction
   - LLM instance availability check
   - Prompt conversion to template if needed
4. Components stored in instance
5. Instance ready for execution
```

### Execution Workflow
```
1. User calls text_llm(test_input="hello")
2. Check LLM.metadata.is_chat_model:

   If Chat Model (True):
   a. Format prompt to messages with variables
   b. Extend messages with system prompts
   c. Call Ollama.chat() → HTTP POST /api/chat
   d. Extract message.content from ChatResponse

   If Completion Model (False):
   a. Format prompt to string with variables
   b. Extend prompt with system prompts
   c. Call Ollama.complete() → HTTP POST /api/generate
   d. Extract text from CompletionResponse

3. Parse output:
   a. Extract JSON string from raw text
   b. Validate JSON against Pydantic schema
   c. Create ModelTest instance
   d. Type check output

4. Return validated ModelTest instance
```

## Key Design Patterns

### 1. **Validation at Construction**
All components are validated during `__init__`, ensuring errors are caught early:
- Parser type checking
- LLM availability verification
- Prompt format conversion

### 2. **Adapter Pattern**
`TextCompletionLLM` adapts different prompt types and LLM modes:
- String prompts → PromptTemplate
- Chat models → format_messages
- Completion models → format string

### 3. **Strategy Pattern**
Output parsing strategy is injected via `PydanticOutputParser`:
- Customizable JSON extraction
- Flexible schema validation
- Extensible error handling

### 4. **Template Method Pattern**
`__call__` defines the algorithm skeleton:
1. Check model type
2. Format prompt (strategy varies)
3. Execute LLM (path varies)
4. Parse output (consistent)

## Error Handling

### Initialization Errors
- `ValueError`: Invalid parser type or prompt type
- `AssertionError`: No LLM provided and Configs.llm not set

### Execution Errors
- `ConnectionError`: Ollama server unreachable
- `TimeoutError`: Request exceeded timeout
- `HTTPError`: API error response

### Parsing Errors
- `ValueError`: Malformed JSON or type mismatch
- `ValidationError`: Schema validation failed

## Performance Considerations

1. **Reusable Instances**: `TextCompletionLLM` instances are reusable after initialization
2. **Async Support**: `acall()` method provides async execution
3. **Streaming Support**: LLM layer supports streaming responses
4. **Stateless Execution**: Each call creates independent transient state

## Related Documentation

- [Serapeum Core Documentation](../../reference/structured_tools/tools_llm.md)
- [Ollama Integration](../../ollama/)
- [Output Parsers](../../reference/structured_tools/)
- [Prompt Templates](../../reference/prompts/)

## Diagram Format

All diagrams are written in [Mermaid](https://mermaid.js.org/) format, which can be rendered by:
- GitHub (automatic rendering in markdown)
- VS Code (with Mermaid extension)
- MkDocs (with mermaid plugin)
- Online Mermaid Live Editor

## Version Information

- **Serapeum Version**: Based on current codebase structure
- **Test File**: `serapeum-integrations/llms/serapeum-ollama/tests/test_ollama_text_completion_llm.py`
- **Source Classes**: Located in `serapeum-core/src/serapeum/core/structured_tools/`

## Contributing

When updating these diagrams:
1. Keep them synchronized with code changes
2. Test rendering in multiple viewers
3. Update this README if adding new diagrams
4. Include practical examples in diagram notes
