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
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.llms.ollama import Ollama


# Define the output schema
class ModelTest(BaseModel):
    hello: str

# Initialize components
LLM = Ollama(model="llama3.1", request_timeout=180)
output_parser = PydanticParser(output_cls=ModelTest)

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

## Understanding the Workflow

### 1. [Execution Flow and Method Calls](./text_completion_llm_sequence.md)
Shows the chronological flow of method calls and interactions.

**Best for**:
- Understanding the order of operations
- Seeing how objects communicate
- Debugging execution flow

**Key Sections**:
- Initialization phase (validation and component setup)
- Execution phase (prompt formatting and LLM invocation)
- Parsing phase (JSON extraction and Pydantic validation)

### 2. [Architecture and Class Relationships](./text_completion_llm_class.md)
Illustrates the static structure and relationships between classes.

**Best for**:
- Understanding the architecture
- Seeing inheritance and composition
- Identifying class responsibilities

**Key Classes**:
- `TextCompletionLLM`: Main orchestrator
- `PydanticParser`: Output validation
- `BasePromptTemplate` hierarchy: Prompt formatting
- `LLM` hierarchy: Model execution

### 3. [Data Transformations and Validation](./text_completion_llm_dataflow.md)
Tracks how data transforms through the system.

**Best for**:
- Understanding data transformations
- Identifying validation points
- Seeing error handling paths

**Key Flows**:
- Initialization validation pipeline
- Chat model execution path
- Completion model execution path
- JSON parsing and validation

### 4. [Component Boundaries and Interactions](./text_completion_llm_components.md)
Shows component boundaries and interaction patterns.

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

### 5. [Lifecycle States and Transitions](./text_completion_llm_state.md)
Depicts the lifecycle states and transitions.

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
1. Create PydanticParser with output schema (ModelTest)
2. Initialize TextCompletionLLM with:
   - output_parser: PydanticParser
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
Output parsing strategy is injected via `PydanticParser`:
- Customizable JSON extraction
- Flexible schema validation
- Extensible error handling

### 4. **Template Method Pattern**
`__call__` defines the algorithm skeleton:
1. Check model type
2. Format prompt (strategy varies)
3. Execute LLM (path varies)
4. Parse output (consistent)

## Performance Considerations

1. **Reusable Instances**: `TextCompletionLLM` instances are reusable after initialization
2. **Async Support**: `acall()` method provides async execution
3. **Streaming Support**: LLM layer supports streaming responses
4. **Stateless Execution**: Each call creates independent transient state