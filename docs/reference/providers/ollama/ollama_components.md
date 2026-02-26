# Component Boundaries and Interactions

This diagram shows how components interact during the complete lifecycle of the Ollama LLM.

```mermaid
graph TB
    subgraph User Space
        UC[User Code]
        PM[Pydantic Models: DummyModel, Album]
    end

    subgraph Ollama Core
        OL[Ollama Instance]

        subgraph Configuration
            CFG[Configuration Fields]
            MD[Metadata]
        end

        subgraph Client Management
            CL[Client Property]
            ACL[AsyncClient Property]
            LINIT[Lazy Initialization]
        end

        subgraph Request Building
            BCRQ[_chat Request Builder]
            BCRQ_MSG[Message Converter]
            BCRQ_OPT[Options Builder]
            BCRQ_TOOL[Tool Converter]
        end

        subgraph Response Parsing
            PRSP[_chat_from_response]
            PRSP_MSG[Message Parser]
            PRSP_TOOL[Tool Calls Parser]
            PRSP_META[Metadata Extractor]
        end

        subgraph Stream Handling
            STRM[_chat_stream_from_response]
            ACC[Accumulator]
        end

        subgraph Tool Handling
            PREP[_prepare_chat_with_tools]
            VAL[_validate_chat_with_tools_response]
            FORCE[force_single_tool_call]
        end

        subgraph Decorators
            C2C[chat_to_completion_decorator]
            SC2C[stream_chat_to_completion]
            AC2C[achat_to_completion]
        end
    end

    subgraph External Client Layer
        CLI[ollama.Client]
        ACLI[ollama.AsyncClient]

        subgraph Client Operations
            CHAT_OP[chat method]
            GEN_OP[generate method]
        end
    end

    subgraph Ollama Server
        SRV[Ollama Server Process]

        subgraph API Endpoints
            EP_CHAT["/api/chat"]
            EP_GEN["/api/generate"]
        end

        subgraph Model Runtime
            MDL[Loaded Model: llama3.1]
            CTX[Context Window]
            INF[Inference Engine]
        end
    end

    subgraph Orchestrator Layer
        TCL[TextCompletionLLM]
        TOL[ToolOrchestratingLLM]

        subgraph Orchestrator Components
            PRS[PydanticParser]
            PTMP[PromptTemplate]
            CTOOL[CallableTool]
        end
    end

    subgraph Response Models
        CRESP[ChatResponse]
        CORESP[CompletionResponse]
        MSG[Message]
    end

    %% Initialization Flow
    UC -->|1. Initialize| OL
    OL -->|Store config| CFG
    OL -->|Create| MD
    MD -->|is_chat_model=True| OL
    MD -->|is_function_calling_model=True| OL

    %% Client lazy init
    OL -->|On first use| LINIT
    LINIT -->|Create if None| CL
    LINIT -->|Create if None| ACL
    CL -.->|Wraps| CLI
    ACL -.->|Wraps| ACLI

    %% Chat Flow
    UC -->|2a. chat(messages)| OL
    OL -->|Check tools| PREP
    PREP -->|Convert tools| BCRQ_TOOL
    BCRQ_TOOL -->|Merge| BCRQ

    OL -->|Build request| BCRQ
    BCRQ -->|Convert messages| BCRQ_MSG
    BCRQ_MSG -->|Add options| BCRQ_OPT
    BCRQ_OPT -->|Final payload| CL

    CL -->|chat(**request)| CLI
    CLI -->|HTTP POST| EP_CHAT
    EP_CHAT -->|Route to| MDL
    MDL -->|Use| CTX
    MDL -->|Run| INF
    INF -->|Generate| EP_CHAT
    EP_CHAT -->|Response dict| CLI
    CLI -->|Return| CL

    CL -->|Raw response| PRSP
    PRSP -->|Extract message| PRSP_MSG
    PRSP -->|Extract tool_calls| PRSP_TOOL
    PRSP -->|Extract metadata| PRSP_META
    PRSP_META -->|Create| CRESP
    CRESP -->|Contains| MSG
    CRESP -->|Return| UC

    %% Complete Flow (via decorator)
    UC -->|2b. complete(prompt)| C2C
    C2C -->|Wrap to Message| OL
    OL -->|Delegate chat| CL
    CL -->|ChatResponse| C2C
    C2C -->|Extract text| CORESP
    CORESP -->|Return| UC

    %% Stream Flow
    UC -->|2c. chat(messages, stream=True)| OL
    OL -->|stream=True| CLI
    CLI -->|Streaming POST| EP_CHAT
    EP_CHAT -.->|Chunk 1| CLI
    CLI -.->|Chunk 1| STRM
    STRM -.->|Accumulate| ACC
    ACC -.->|Yield| CRESP
    CRESP -.->|Yield| UC
    EP_CHAT -.->|Chunk N| CLI
    CLI -.->|Chunk N| STRM

    %% Tool calling flow
    UC -->|2d. generate_tool_calls(messages, tools)| OL
    OL -->|Prepare| PREP
    PREP -->|Convert to schema| BCRQ_TOOL
    OL -->|Call chat| CLI
    CLI -->|Response with tool_calls| PRSP
    PRSP -->|Parse| PRSP_TOOL
    PRSP_TOOL -->|Return to| VAL
    VAL -->|Check parallel| FORCE
    FORCE -->|Trim if needed| CRESP
    CRESP -->|Return| UC

    %% TextCompletionLLM Integration
    UC -->|3a. TextCompletionLLM(llm=Ollama)| TCL
    TCL -->|Store| OL
    TCL -->|Use| PTMP
    TCL -->|Use| PRS
    UC -->|Call| TCL
    TCL -->|Format prompt| PTMP
    PTMP -->|Messages| OL
    OL -->|chat| CRESP
    CRESP -->|message.content| PRS
    PRS -->|Parse JSON| PM
    PM -->|Return| UC

    %% ToolOrchestratingLLM Integration
    UC -->|3b. ToolOrchestratingLLM(llm=Ollama)| TOL
    TOL -->|Store| OL
    TOL -->|Create tool from| PM
    PM -->|Schema| CTOOL
    TOL -->|Use| PTMP
    UC -->|Call| TOL
    TOL -->|Format prompt| PTMP
    PTMP -->|Messages| OL
    TOL -->|Pass tools| CTOOL
    CTOOL -->|Convert| OL
    OL -->|generate_tool_calls| CRESP
    CRESP -->|tool_calls| TOL
    TOL -->|Execute tool| CTOOL
    CTOOL -->|Create| PM
    PM -->|Return| UC

    %% Styling
    classDef userClass fill:#e1f5ff,stroke:#01579b
    classDef ollamaClass fill:#e0f2f1,stroke:#004d40
    classDef configClass fill:#fff9c4,stroke:#f57f17
    classDef clientClass fill:#f3e5f5,stroke:#4a148c
    classDef parserClass fill:#e8f5e9,stroke:#1b5e20
    classDef serverClass fill:#efebe9,stroke:#3e2723
    classDef orchestratorClass fill:#fce4ec,stroke:#880e4f
    classDef responseClass fill:#fff3e0,stroke:#e65100

    class UC,PM userClass
    class OL ollamaClass
    class CFG,MD,BCRQ,BCRQ_MSG,BCRQ_OPT,BCRQ_TOOL,PRSP,PRSP_MSG,PRSP_TOOL,PRSP_META configClass
    class CL,ACL,LINIT,CLI,ACLI,CHAT_OP,GEN_OP clientClass
    class PREP,VAL,FORCE parserClass
    class SRV,EP_CHAT,EP_GEN,MDL,CTX,INF serverClass
    class TCL,TOL,PRS,PTMP,CTOOL orchestratorClass
    class CRESP,CORESP,MSG responseClass
```

## Component Interaction Patterns

### 1. Initialization Pattern
```
User Code
  └─→ Ollama.__init__
      ├─→ Store: model, base_url, temperature, request_timeout, json_mode, additional_kwargs
      ├─→ Create Metadata: is_chat_model=True, is_function_calling_model=True
      └─→ Set _client=None, _async_client=None (lazy init)
```

### 2. Client Lazy Initialization Pattern
```
User → Ollama.chat
  └─→ Access self.client property
      └─→ Check if self._client is None
          ├─→ If None: Create Client(host=base_url, timeout=request_timeout)
          └─→ Return self._client
```

### 3. Chat Request Pattern
```
User → Ollama.chat(messages, **kwargs)
  ├─→ _prepare_chat_with_tools (if tools present)
  │   └─→ For each tool:
  │       ├─→ Extract tool.metadata
  │       ├─→ Get fn_schema from metadata
  │       └─→ Build Ollama tool dict
  ├─→ _chat(messages, stream=False, **kwargs)
  │   ├─→ Build request dict:
  │   │   ├─→ model: self.model
  │   │   ├─→ messages: [msg.dict() for msg in messages]
  │   │   ├─→ options: {temperature, ...}
  │   │   ├─→ format: "json" if json_mode
  │   │   └─→ tools: converted tool dicts
  │   ├─→ Ensure client initialized
  │   ├─→ client.chat(**request)
  │   └─→ _chat_from_response(raw_response)
  │       ├─→ Extract message dict
  │       ├─→ Parse role, content, tool_calls
  │       ├─→ Create Message object
  │       ├─→ Extract metadata: model, times, tokens
  │       └─→ Create ChatResponse
  └─→ Return ChatResponse
```

### 4. Completion via Decorator Pattern
```
User → Ollama.complete(prompt, **kwargs)
  └─→ @chat_to_completion_decorator wrapper
      ├─→ Convert prompt to Message(role=USER, content=prompt)
      ├─→ Call self.chat([message], **kwargs)
      ├─→ Receive ChatResponse
      ├─→ Extract text = response.message.content
      └─→ Return CompletionResponse(text=text, raw=response.raw, ...)
```

### 5. Streaming Pattern
```
User → Ollama.chat(messages, stream=True, **kwargs)
  └─→ _chat(messages, stream=True, **kwargs)
      ├─→ Build request with stream=True
      ├─→ client.chat(**request) returns iterator
      └─→ For each chunk:
          ├─→ _chat_stream_from_response(chunk)
          │   ├─→ Extract delta content
          │   ├─→ Accumulate tool_calls
          │   └─→ Create ChatResponse with delta
          └─→ Yield ChatResponse
```

### 6. Tool Calling Pattern
```
User → Ollama.generate_tool_calls(messages, tools, **kwargs)
  ├─→ _prepare_chat_with_tools(messages, tools, **kwargs)
  │   └─→ For each tool in tools:
  │       ├─→ Get tool.metadata.fn_schema
  │       └─→ Build: {"type": "function", "function": {"name": ..., "parameters": schema}}
  ├─→ Merge tools into kwargs
  ├─→ Call self.chat(messages, **kwargs)
  ├─→ Receive ChatResponse with tool_calls
  ├─→ _validate_chat_with_tools_response(response, tools, **kwargs)
  │   └─→ If not allow_parallel_tool_calls:
  │       └─→ force_single_tool_call(response)
  └─→ Return ChatResponse
```

### 7. TextCompletionLLM Integration Pattern
```
User → TextCompletionLLM(output_parser=parser, prompt=prompt, llm=Ollama(...))
  └─→ TextCompletionLLM stores Ollama instance

User → text_llm(key="value")
  ├─→ Check llm.metadata.is_chat_model → True
  ├─→ Format prompt with variables → List[Message]
  ├─→ Ollama.chat(messages) → ChatResponse
  ├─→ Extract response.message.content
  ├─→ PydanticParser.parse(content) → DummyModel
  └─→ Return DummyModel instance
```

### 8. ToolOrchestratingLLM Integration Pattern
```
User → ToolOrchestratingLLM(output_cls=Album, prompt=prompt, llm=Ollama(...))
  ├─→ Convert Album Pydantic model to CallableTool
  └─→ Store Ollama instance

User → tools_llm(topic="rock")
  ├─→ Format prompt with topic → List[Message]
  ├─→ CallableTool.metadata.fn_schema → Album JSON schema
  ├─→ Ollama.generate_tool_calls(messages, tools=[album_tool])
  │   ├─→ _prepare_chat_with_tools converts tool to Ollama format
  │   ├─→ Server returns tool_calls with arguments
  │   └─→ Return ChatResponse with tool_calls
  ├─→ Extract tool_calls from response
  ├─→ For each tool_call:
  │   ├─→ Get function name and arguments
  │   ├─→ Execute CallableTool.call(**arguments)
  │   └─→ Creates Album instance from arguments
  └─→ Return Album instance (or list if parallel)
```

## Component State Management

### Ollama Instance State
```
Initialization:
  - model: str (immutable after init)
  - base_url: str (immutable after init)
  - request_timeout: float (immutable after init)
  - temperature: float (immutable after init)
  - json_mode: bool (immutable after init)
  - additional_kwargs: dict (immutable after init)
  - _client: Optional[Client] (mutable, lazy-initialized)
  - _async_client: Optional[AsyncClient] (mutable, lazy-initialized)

Runtime:
  - _client: None → Client instance (on first use)
  - _async_client: None → AsyncClient instance (on first async use)
```

### Request State (Per Call)
```
Input:
  - messages: List[Message]
  - tools: Optional[List[BaseTool]]
  - stream: bool
  - **kwargs: Additional options

Processing:
  - request_dict: Built from inputs
  - raw_response: dict from server
  - parsed_response: ChatResponse/CompletionResponse

Output:
  - ChatResponse or CompletionResponse
```

### Streaming State (Per Stream)
```
Initialization:
  - iterator: From client.chat(stream=True)

Per Chunk:
  - chunk_dict: Raw chunk from server
  - accumulated_content: Growing string
  - accumulated_tool_calls: Growing list
  - delta: New content in this chunk

Output:
  - Generator yielding ChatResponse objects
```

## Error Boundaries

### 1. Configuration Errors (Initialization)
```
Ollama.__init__
  └─→ Validate inputs
      ├─→ model: must be non-empty string
      ├─→ base_url: must be valid URL
      ├─→ temperature: must be in [0.0, 1.0]
      └─→ request_timeout: must be positive
```

### 2. Client Creation Errors (First Use)
```
client property
  └─→ Create Client(host, timeout)
      └─→ Catch: ValueError, ConnectionError
```

### 3. Request Errors (During Call)
```
client.chat(**request)
  └─→ Catch: TimeoutError, ConnectionError, HTTPError
      └─→ Wrap and re-raise with context
```

### 4. Parsing Errors (Response Processing)
```
_chat_from_response(raw_response)
  └─→ Extract required fields
      └─→ Catch: KeyError, TypeError
          └─→ Log warning and return default
```

### 5. Tool Validation Errors
```
_prepare_chat_with_tools(messages, tools)
  └─→ For each tool:
      └─→ Validate metadata.fn_schema exists
          └─→ Raise ValueError if missing
```

## Component Dependencies

### Ollama Depends On:
- `ollama.Client` and `ollama.AsyncClient` (external package)
- `serapeum.core.base.llms.types` (Message, ChatResponse, CompletionResponse, Metadata)
- `serapeum.core.llms.function_calling.FunctionCallingLLM` (base class)
- `serapeum.core.base.llms.utils` (decorators)
- `pydantic` (for configuration)

### Ollama Is Used By:
- `TextCompletionLLM` (as the LLM engine)
- `ToolOrchestratingLLM` (as the function-calling LLM)
- Direct user code (standalone usage)

### External Dependencies:
- **Ollama Server**: Must be running and accessible at base_url
- **Model**: Must be pulled and available on the server
