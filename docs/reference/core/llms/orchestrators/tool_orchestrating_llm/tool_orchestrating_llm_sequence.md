# Execution Flow and Method Calls

This diagram shows the complete workflow from initialization to execution of `ToolOrchestratingLLM`.

```mermaid
sequenceDiagram
    participant User
    participant ToolOrchestratingLLM
    participant CallableTool
    participant PromptTemplate
    participant Ollama
    participant LLMServer
    participant AgentChatResponse

    Note over User: Initialization Phase
    User->>ToolOrchestratingLLM: __init__(output_cls=MockAlbum, prompt, llm)
    activate ToolOrchestratingLLM

    ToolOrchestratingLLM->>ToolOrchestratingLLM: validate_llm(llm)
    Note over ToolOrchestratingLLM: Checks if llm is provided or uses Configs.llm<br/>Validates is_function_calling_model=True

    ToolOrchestratingLLM->>ToolOrchestratingLLM: validate_prompt(prompt)
    Note over ToolOrchestratingLLM: Converts string to PromptTemplate<br/>if not already BasePromptTemplate

    ToolOrchestratingLLM->>PromptTemplate: Create/validate prompt template
    activate PromptTemplate
    PromptTemplate-->>ToolOrchestratingLLM: template instance
    deactivate PromptTemplate

    Note over ToolOrchestratingLLM: Store: _output_cls, _llm, _prompt,<br/>_verbose, _allow_parallel_tool_calls,<br/>_tool_choice

    ToolOrchestratingLLM-->>User: tools_llm instance
    deactivate ToolOrchestratingLLM

    Note over User: Execution Phase
    User->>ToolOrchestratingLLM: __call__(topic="songs")
    activate ToolOrchestratingLLM

    ToolOrchestratingLLM->>CallableTool: from_model(MockAlbum)
    activate CallableTool
    Note over CallableTool: Convert Pydantic model to tool<br/>with JSON schema
    CallableTool-->>ToolOrchestratingLLM: tool instance
    deactivate CallableTool

    ToolOrchestratingLLM->>PromptTemplate: format_messages(topic="songs")
    activate PromptTemplate
    Note over PromptTemplate: Apply template variables:<br/>"This is a test album with songs"
    PromptTemplate-->>ToolOrchestratingLLM: List[Message]
    deactivate PromptTemplate

    ToolOrchestratingLLM->>Ollama: _extend_messages(messages)
    activate Ollama
    Note over Ollama: Add system prompts if configured
    Ollama-->>ToolOrchestratingLLM: extended messages
    deactivate Ollama

    ToolOrchestratingLLM->>Ollama: predict_and_call(tools=[tool], messages, ...)
    activate Ollama
    Note over Ollama: Prepare function calling request<br/>with tool schemas

    Ollama->>LLMServer: HTTP POST /api/chat
    activate LLMServer
    Note over LLMServer: Process chat with tools<br/>Generate tool calls
    LLMServer-->>Ollama: Response with tool_calls
    deactivate LLMServer

    Ollama->>Ollama: Parse tool_calls from response
    Note over Ollama: Extract tool arguments

    Ollama->>CallableTool: Execute tool with parsed arguments
    activate CallableTool
    CallableTool->>CallableTool: Validate args against schema
    CallableTool->>CallableTool: Create MockAlbum instance
    CallableTool-->>Ollama: ToolOutput with raw_output=MockAlbum
    deactivate CallableTool

    Ollama->>AgentChatResponse: Create response with sources
    activate AgentChatResponse
    AgentChatResponse-->>Ollama: response instance
    deactivate AgentChatResponse

    Ollama-->>ToolOrchestratingLLM: AgentChatResponse
    deactivate Ollama

    ToolOrchestratingLLM->>AgentChatResponse: parse_tool_outputs(allow_parallel=False)
    activate AgentChatResponse
    Note over AgentChatResponse: Extract raw_output from<br/>first ToolOutput in sources
    AgentChatResponse-->>ToolOrchestratingLLM: MockAlbum instance
    deactivate AgentChatResponse

    ToolOrchestratingLLM-->>User: MockAlbum(title="hello", artist="world", ...)
    deactivate ToolOrchestratingLLM
```

## Key Points

1. **Initialization validates all components** before storing them - LLM must support function calling
2. **Tool creation** converts Pydantic model to CallableTool with JSON schema
3. **Prompt formatting** applies template variables to create messages
4. **predict_and_call** orchestrates the function calling flow with the LLM
5. **Tool execution** happens automatically after LLM generates tool calls
6. **Response parsing** extracts structured Pydantic instances from ToolOutput

## Parallel Tool Calls

When `allow_parallel_tool_calls=True`:

```mermaid
sequenceDiagram
    participant User
    participant ToolOrchestratingLLM
    participant Ollama
    participant AgentChatResponse

    User->>ToolOrchestratingLLM: __call__(allow_parallel_tool_calls=True)
    activate ToolOrchestratingLLM

    ToolOrchestratingLLM->>Ollama: predict_and_call(..., allow_parallel=True)
    activate Ollama
    Note over Ollama: LLM generates multiple tool calls

    Ollama->>Ollama: Execute tool 1 → ToolOutput 1
    Ollama->>Ollama: Execute tool 2 → ToolOutput 2

    Ollama-->>ToolOrchestratingLLM: AgentChatResponse with multiple sources
    deactivate Ollama

    ToolOrchestratingLLM->>AgentChatResponse: parse_tool_outputs(allow_parallel=True)
    activate AgentChatResponse
    Note over AgentChatResponse: Extract all raw_outputs<br/>from sources list
    AgentChatResponse-->>ToolOrchestratingLLM: List[MockAlbum]
    deactivate AgentChatResponse

    ToolOrchestratingLLM-->>User: [MockAlbum(...), MockAlbum(...)]
    deactivate ToolOrchestratingLLM
```

## Async Execution Flow

The async flow (`acall`) follows the same pattern but uses:
- `apredict_and_call` instead of `predict_and_call`
- Async tool execution
- All operations are awaited

## Streaming Execution Flow

For `__call__(stream=True)`:
1. Uses `stream_chat_with_tools` instead of `predict_and_call`
2. Yields partial responses as `StreamingObjectProcessor` parses incremental tool calls
3. Maintains `cur_objects` state across chunks
4. Each yield contains progressively updated Pydantic instances
