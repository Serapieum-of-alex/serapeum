# ToolOrchestratingLLM Workflow

This directory contains comprehensive documentation explaining the complete workflow of the `ToolOrchestratingLLM` class, from initialization to execution and structured output extraction.

## Overview

The `ToolOrchestratingLLM` is a function-calling orchestrator that:
1. **Converts Pydantic models** into callable tools with JSON schemas
2. **Formats prompts** with template variables
3. **Executes function calling** via LLM with tool schemas
4. **Parses tool outputs** into validated Pydantic model instances

## Example Usage

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

# Define the output schema
class MockAlbum(BaseModel):
    title: str
    artist: str
    songs: List[MockSong]

class MockSong(BaseModel):
    title: str

# Initialize LLM with function calling support
llm = Ollama(model='llama3.1', request_timeout=80)

# Create ToolOrchestratingLLM instance
tools_llm = ToolOrchestratingLLM(
    output_cls=MockAlbum,
    prompt='This is a test album with {topic}',
    llm=llm,
)

# Execute and get structured output via function calling
obj_output = tools_llm(topic="songs")
# Returns: MockAlbum(title="hello", artist="world", songs=[...])
```

## Understanding the Workflow

### 1. [Execution Flow and Method Calls](./tool_orchestrating_llm_sequence.md)
Shows the chronological flow of method calls and interactions.

**Best for**:
- Understanding the order of operations
- Seeing how function calling works
- Debugging execution flow

**Key Sections**:
- Initialization phase (validation and component setup)
- Tool creation (CallableTool.from_model)
- Execution phase (predict_and_call with tools)
- Tool execution and output parsing

### 2. [Architecture and Class Relationships](./tool_orchestrating_llm_class.md)
Illustrates the static structure and relationships between classes.

**Best for**:
- Understanding the architecture
- Seeing inheritance and composition
- Identifying class responsibilities

**Key Classes**:
- `ToolOrchestratingLLM`: Main orchestrator
- `CallableTool`: Pydantic-to-tool converter
- `FunctionCallingLLM`: Abstract function-calling interface
- `Ollama`: Concrete LLM implementation
- `AgentChatResponse`: Response container with tool outputs

### 3. [Data Transformations and Validation](./tool_orchestrating_llm_dataflow.md)
Tracks how data transforms through the system.

**Best for**:
- Understanding data transformations
- Identifying validation points
- Seeing error handling paths

**Key Flows**:
- Initialization validation pipeline
- Tool schema generation
- Function calling request preparation
- Tool execution and validation
- Single vs. parallel output extraction

### 4. [Component Boundaries and Interactions](./tool_orchestrating_llm_components.md)
Shows component boundaries and interaction patterns.

**Best for**:
- Understanding system architecture
- Seeing component responsibilities
- Identifying interaction patterns

**Key Components**:
- User space (application code)
- ToolOrchestratingLLM layer (orchestration)
- Tool layer (CallableTool)
- Prompt layer (template formatting)
- LLM layer (function calling execution)
- Response layer (AgentChatResponse, ToolOutput)

### 5. [Lifecycle States and Transitions](./tool_orchestrating_llm_state.md)
Depicts the lifecycle states and transitions.

**Best for**:
- Understanding instance lifecycle
- Seeing state transitions
- Identifying error states

**Key States**:
- Initialization states (validation)
- Execution states (sync/async/streaming)
- Tool execution states (single/parallel)
- Parsing states (output extraction)

## Workflow Summary

### Initialization Workflow
```
1. Initialize ToolOrchestratingLLM with:
   - output_cls: Pydantic model (MockAlbum)
   - prompt: String or BasePromptTemplate
   - llm: Function-calling LLM (Ollama)
   - tool_choice: Optional tool selection strategy
   - allow_parallel_tool_calls: Single vs. multiple outputs

2. Validation occurs:
   - LLM must support function calling (is_function_calling_model=True)
   - Prompt conversion to template if needed

3. Components stored in instance

4. Instance ready for execution
```

### Execution Workflow
```
1. User calls tools_llm(topic="songs")

2. Convert Pydantic model to tool:
   a. CallableTool.from_model(MockAlbum)
   b. Extract JSON schema from model
   c. Create callable tool with validation

3. Format prompt:
   a. Apply template variables: topic="songs"
   b. Create messages: [Message(role=USER, content="...")]
   c. Extend with system prompts

4. Execute function calling:
   a. Prepare request with tool schemas
   b. Call predict_and_call([tool], messages, ...)
   c. HTTP POST to Ollama server with tools parameter
   d. LLM generates tool_calls in response

5. Execute tools:
   a. Parse tool_calls from response
   b. Extract tool arguments
   c. Validate arguments against Pydantic schema
   d. Create MockAlbum instance(s)
   e. Wrap in ToolOutput with raw_output

6. Create AgentChatResponse:
   a. Container with response text
   b. sources: List[ToolOutput]

7. Parse outputs:
   a. Extract raw_output from ToolOutput(s)
   b. Return MockAlbum or List[MockAlbum]

8. Return validated Pydantic instance(s)
```

## Key Design Patterns

### 1. **Function Calling Pattern**
Uses LLM's native function calling capabilities:
- Pydantic models → Tool schemas
- LLM generates tool_calls
- Tools execute with validation
- Structured outputs guaranteed

### 2. **Tool Abstraction**
Automatic conversion from Pydantic models to tools:
- JSON schema extraction
- Argument validation
- Output wrapping

### 3. **Flexible Output Modes**
Single or parallel tool calls:
- `allow_parallel_tool_calls=False` → Single model
- `allow_parallel_tool_calls=True` → List of models

### 4. **Multi-Modal Execution**
Supports multiple execution modes:
- Sync: `__call__`
- Async: `acall`
- Streaming: `stream_call`
- Async streaming: `astream_call`

## Comparison with TextCompletionLLM

| Feature | ToolOrchestratingLLM | TextCompletionLLM |
|---------|---------------------|-------------------|
| **Method** | Function calling | Text completion with parsing |
| **LLM Requirement** | Must support function calling | Any chat/completion model |
| **Schema Handling** | Native tool schemas | JSON in prompt |
| **Validation** | Before execution (by LLM) | After generation (by parser) |
| **Reliability** | Higher (structured by design) | Depends on LLM output quality |
| **Parallel Outputs** | Native support | Not supported |
| **Streaming** | Partial tool_calls | Not applicable |
| **Use Case** | When function calling available | Fallback for non-function-calling models |

## Performance Considerations

1. **Reusable Instances**: `ToolOrchestratingLLM` instances are reusable after initialization
2. **Async Support**: `acall()` method provides async execution
3. **Streaming Support**: `stream_call()` yields progressive updates
4. **Parallel Tool Calls**: Enable with `allow_parallel_tool_calls=True`
5. **Stateless Execution**: Each call creates independent transient state
6. **Tool Schema Caching**: Tool schemas are generated once per call

## Error Handling

### Initialization Errors
- `AssertionError`: No LLM provided
- `ValueError`: LLM doesn't support function calling

### Execution Errors
- Network/timeout errors from LLM
- Missing tool_calls in response
- Invalid tool arguments

### Validation Errors
- Pydantic validation fails on tool arguments
- Type mismatch in output

## When to Use ToolOrchestratingLLM

### Use When:
✅ Your LLM supports function calling (OpenAI, Ollama, etc.)
✅ You need guaranteed structured outputs
✅ You want parallel tool calls
✅ You need streaming with structured updates
✅ Reliability is critical

### Use TextCompletionLLM When:
❌ LLM doesn't support function calling
❌ You need simple text-to-JSON parsing
❌ Function calling overhead is unnecessary

## Next Steps

- [View Examples](./examples.md) - Comprehensive usage examples
- [Sequence Diagram](./tool_orchestrating_llm_sequence.md) - Detailed flow
- [Class Diagram](./tool_orchestrating_llm_class.md) - Architecture
- [Data Flow](./tool_orchestrating_llm_dataflow.md) - Transformations
- [Components](./tool_orchestrating_llm_components.md) - Interactions
- [State Machine](./tool_orchestrating_llm_state.md) - Lifecycle
