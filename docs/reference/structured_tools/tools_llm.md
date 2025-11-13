# ToolOrchestratingLLM — Comprehensive Diagrams and Documentation

This page documents `serapeum.core.structured_tools.tools_llm.ToolOrchestratingLLM` with class, component, sequence, activity, data-flow, state, and architecture diagrams. Diagrams include full type annotations, generic parameters, relationships, and both sync/async variants.

- Primary class: `ToolOrchestratingLLM(BasePydanticLLM[pydantic.BaseModel])`
- Key collaborators: `FunctionCallingLLM`, `BasePromptTemplate`/`PromptTemplate`, `CallableTool`, `StreamingObjectProcessor`, `AgentChatResponse`, `Configs`

Sections provide Mermaid diagrams and PlantUML alternatives where applicable.

---

## Class Hierarchy (UML)

```mermaid
classDiagram
    direction TB

    class ABC

    class BasePydanticLLM~Model~ {
        +output_cls() Type[Model]
        +__call__(*args: Any, **kwargs: Any) Model | List[Model]
        +acall(*args: Any, **kwargs: Any) Model | List[Model]
        +stream_call(*args: Any, **kwargs: Any) Generator[Model | List[Model]]
        +astream_call(*args: Any, **kwargs: Any) AsyncGenerator[Model | List[Model]]
    }

    class ToolOrchestratingLLM

    class LLM
    class FunctionCallingLLM
    LLM <|-- FunctionCallingLLM

    ABC <|-- BasePydanticLLM~Model~
    BasePydanticLLM~BaseModel~ <|-- ToolOrchestratingLLM

    class BasePromptTemplate
    class PromptTemplate
    BasePromptTemplate <|-- PromptTemplate

    class CallableTool
    class StreamingObjectProcessor
    class AgentChatResponse
    class Configs

    %% Relationships (composition/dependency)
    ToolOrchestratingLLM *-- BasePromptTemplate : _prompt
    ToolOrchestratingLLM *-- FunctionCallingLLM : _llm
    ToolOrchestratingLLM o-- CallableTool : from_model(output_cls)
    ToolOrchestratingLLM ..> StreamingObjectProcessor : uses (streaming)
    ToolOrchestratingLLM ..> AgentChatResponse : receives
    ToolOrchestratingLLM ..> Configs : validate_llm fallback
```

PlantUML alternative:

```plantuml
@startuml
set namespaceSeparator ::

abstract class ABC
class "pydantic.BaseModel" as BaseModel
class Model <<TypeVar: bound=BaseModel>>

abstract class BasePydanticLLM~Model~ {
  +{abstract} output_cls() : Type[Model]
  +{abstract} __call__(*args: Any, **kwargs: Any) : Model | List[Model]
  +acall(*args: Any, **kwargs: Any) : Model | List[Model]
  +stream_call(*args: Any, **kwargs: Any) : Generator[Model | List[Model]]
  +astream_call(*args: Any, **kwargs: Any) : AsyncGenerator[Model | List[Model]]
}

class ToolOrchestratingLLM
class LLM
class FunctionCallingLLM
LLM <|-- FunctionCallingLLM

ABC <|-- BasePydanticLLM~Model~
BasePydanticLLM~BaseModel~ <|-- ToolOrchestratingLLM

class BasePromptTemplate
class PromptTemplate
BasePromptTemplate <|-- PromptTemplate

class CallableTool
class StreamingObjectProcessor
class AgentChatResponse
class Configs

ToolOrchestratingLLM *-- BasePromptTemplate : _prompt
ToolOrchestratingLLM *-- FunctionCallingLLM : _llm
ToolOrchestratingLLM o-- CallableTool : from_model()
ToolOrchestratingLLM ..> StreamingObjectProcessor : uses
ToolOrchestratingLLM ..> AgentChatResponse : receives
ToolOrchestratingLLM ..> Configs : validate_llm()
@enduml
```

---

## Comprehensive Class Diagram (UML)

```mermaid
classDiagram
    direction LR

    class ToolOrchestratingLLM {
        - _output_cls: Type[Model]
        - _llm: FunctionCallingLLM
        - _prompt: BasePromptTemplate
        - _verbose: bool
        - _allow_parallel_tool_calls: bool
        - _tool_choice: Optional[str | Dict[str, Any]]

        +__init__(
            output_cls: Type[Model],
            prompt: BasePromptTemplate | str,
            llm: FunctionCallingLLM | None = None,
            tool_choice: str | Dict[str, Any] | None = None,
            allow_parallel_tool_calls: bool = False,
            verbose: bool = False
          ) -> None

        +output_cls() Type[BaseModel]
        +prompt() BasePromptTemplate
        +prompt(prompt: BasePromptTemplate) void

        +__call__(
            *args: Any,
            llm_kwargs: Dict[str, Any] | None = None,
            **kwargs: Any
          ) BaseModel | List[BaseModel]

        +acall(
            *args: Any,
            llm_kwargs: Dict[str, Any] | None = None,
            **kwargs: Any
          ) BaseModel | List[BaseModel]

        +stream_call(
            *args: Any,
            llm_kwargs: Dict[str, Any] | None = None,
            **kwargs: Any
          ) Generator[Model | List[Model], None, None]

        +astream_call(
            *args: Any,
            llm_kwargs: Dict[str, Any] | None = None,
            **kwargs: Any
          ) AsyncGenerator[Model | List[Model], None]

        +validate_prompt(prompt: BasePromptTemplate | str) BasePromptTemplate
        +validate_llm(llm: LLM) LLM
    }

    ToolOrchestratingLLM ..> CallableTool : uses from_model(output_cls)
    ToolOrchestratingLLM ..> FunctionCallingLLM : predict_and_call()/apredict_and_call()
    ToolOrchestratingLLM ..> FunctionCallingLLM : stream_chat_with_tools()/astream_chat_with_tools()
    ToolOrchestratingLLM ..> StreamingObjectProcessor : progressive parsing
    ToolOrchestratingLLM ..> AgentChatResponse : parse_tool_outputs()
    ToolOrchestratingLLM ..> PromptTemplate : via validate_prompt()
    ToolOrchestratingLLM ..> Configs : fallback LLM
```

PlantUML alternative:

```plantuml
@startuml
class ToolOrchestratingLLM {
  - _output_cls : Type[Model]
  - _llm : FunctionCallingLLM
  - _prompt : BasePromptTemplate
  - _verbose : bool
  - _allow_parallel_tool_calls : bool
  - _tool_choice : Optional[Union[str, Dict[str, Any]]]

  + __init__(output_cls: Type[Model], prompt: BasePromptTemplate|str, llm: FunctionCallingLLM|None = None, tool_choice: str|Dict[str, Any]|None = None, allow_parallel_tool_calls: bool = False, verbose: bool = False) : None
  + output_cls() : Type[pydantic.BaseModel]
  + prompt() : BasePromptTemplate
  + prompt(prompt: BasePromptTemplate) : None
  + __call__(*args: Any, llm_kwargs: Dict[str, Any]|None = None, **kwargs: Any) : BaseModel | List[BaseModel]
  + acall(*args: Any, llm_kwargs: Dict[str, Any]|None = None, **kwargs: Any) : BaseModel | List[BaseModel]
  + stream_call(*args: Any, llm_kwargs: Dict[str, Any]|None = None, **kwargs: Any) : Generator[Model | List[Model], None, None]
  + astream_call(*args: Any, llm_kwargs: Dict[str, Any]|None = None, **kwargs: Any) : AsyncGenerator[Model | List[Model], None]
  + {static} validate_prompt(prompt: BasePromptTemplate|str) : BasePromptTemplate
  + {static} validate_llm(llm: LLM) : LLM
}

ToolOrchestratingLLM ..> CallableTool : from_model()
ToolOrchestratingLLM ..> FunctionCallingLLM : predict_and_call()/apredict_and_call()
ToolOrchestratingLLM ..> FunctionCallingLLM : stream_chat_with_tools()/astream_chat_with_tools()
ToolOrchestratingLLM ..> StreamingObjectProcessor : process()
ToolOrchestratingLLM ..> AgentChatResponse : parse_tool_outputs()
ToolOrchestratingLLM ..> PromptTemplate : validate_prompt()
ToolOrchestratingLLM ..> Configs : validate_llm()
@enduml
```

---

## Component Diagram (UML)

```mermaid
flowchart LR
    subgraph Structured_Tools
      TOLL["ToolOrchestratingLLM\n(BasePydanticLLM[BaseModel])"]
      SOP["StreamingObjectProcessor"]
      CT["CallableTool"]
    end

    subgraph LLM_Layer
      FCLLM["FunctionCallingLLM : LLM"]
    end

    subgraph Prompts
      BPT["BasePromptTemplate"]
      PT["PromptTemplate"]
    end

    subgraph Chat
      ACR["AgentChatResponse"]
    end

    subgraph Configs
      CFG["Configs.llm fallback"]
    end

    BM["pydantic.BaseModel"]

    TOLL -- "composition" --> BPT
    BPT -- "inherits" --> PT
    TOLL -- "composition" --> FCLLM
    TOLL -- "uses" --> CT
    TOLL -- "uses" --> SOP
    FCLLM -- "returns" --> ACR
    ACR -- "parsed by" --> TOLL
    TOLL -- "fallback" --> CFG
    TOLL -. "output_cls: Type(BaseModel)" .-> BM
```

---

## Sequence Diagram — __call__

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant T as ToolOrchestratingLLM
    participant P as BasePromptTemplate
    participant L as FunctionCallingLLM
    participant Tool as CallableTool
    participant R as AgentChatResponse

    User->>T: __call__(llm_kwargs, **vars)
    T->>Tool: from_model(T._output_cls)
    Tool-->>T: tool: CallableTool
    T->>P: format_messages(llm=T._llm, **vars)
    P-->>T: messages: List[Message]
    T->>L: _extend_messages(messages)
    L-->>T: messages'
    T->>L: predict_and_call([tool], chat_history=messages', verbose, allow_parallel_tool_calls, **llm_kwargs)
    L-->>T: R: AgentChatResponse
    T->>R: parse_tool_outputs(allow_parallel_tool_calls)
    alt allow_parallel_tool_calls == True
        R-->>T: List[BaseModel]
    else
        R-->>T: BaseModel
    end
    T-->>User: BaseModel | List[BaseModel]
```

PlantUML alternative:

```plantuml
@startuml
autonumber
actor User
participant T as ToolOrchestratingLLM
participant P as BasePromptTemplate
participant L as FunctionCallingLLM
participant Tool as CallableTool
participant R as AgentChatResponse

User -> T: __call__(llm_kwargs, **vars)
T -> Tool: from_model(output_cls)
Tool --> T: CallableTool
T -> P: format_messages(llm, **vars)
P --> T: messages
T -> L: _extend_messages(messages)
L --> T: messages'
T -> L: predict_and_call([tool], chat_history=messages', verbose, allow_parallel_tool_calls, **llm_kwargs)
L --> T: AgentChatResponse
T -> R: parse_tool_outputs(allow_parallel_tool_calls)
alt parallel
  R --> T: List[BaseModel]
else
  R --> T: BaseModel
end
T --> User: BaseModel | List[BaseModel]
@enduml
```

---

## Sequence Diagram — Streaming (`stream_call`)

```mermaid
sequenceDiagram
    autonumber
    participant T as ToolOrchestratingLLM
    participant P as BasePromptTemplate
    participant L as FunctionCallingLLM
    participant Tool as CallableTool
    participant G as ChatResponse~gen~
    participant S as StreamingObjectProcessor

    T->>Tool: from_model(output_cls)
    Tool-->>T: CallableTool
    T->>P: format_messages(llm=T._llm, **vars)
    P-->>T: messages
    T->>L: _extend_messages(messages)
    L-->>T: messages'
    T->>L: stream_chat_with_tools([tool], chat_history=messages', verbose, allow_parallel_tool_calls, **llm_kwargs)
    L-->>T: G: Generator[ChatResponse]
    loop for partial_resp in G
        T->>S: process(partial_resp, cur_objects)
        S-->>T: objects: BaseModel | List[BaseModel]
        T->>T: cur_objects = list(objects)
        T-->>User: yield objects
    end
```

---

## Activity Diagram

```mermaid
flowchart TD
    A[Init __init__] --> B{validate_prompt}
    B -->|ok| C[store _prompt]
    B -->|error| E1[ValueError: invalid prompt]

    A --> D{validate_llm}
    D -->|ok & function-calling| F[store _llm]
    D -->|none| E2[AssertionError: llm must be provided or set]
    D -->|not function-calling| E3[ValueError: model does not support function-calling]

    C & F --> G[__call__/acall]
    G --> H[Prompt.format_messages]
    H --> I[CallableTool.from_model]
    I --> J[FunctionCallingLLM.predict_and_call / apredict_and_call]
    J --> K[AgentChatResponse.parse_tool_outputs]
    K -->|allow_parallel_tool_calls=True| O[List of BaseModel]
    K -->|False| P[BaseModel]

    C & F --> Q[stream_call/astream_call]
    Q --> H2[Prompt.format_messages]
    H2 --> J2[FunctionCallingLLM.stream_chat_with_tools / astream_chat_with_tools]
    J2 --> R[[for each ChatResponse chunk]]
    R --> S[StreamingObjectProcessor.process]
    S --> T[update cur_objects]
    T --> U[yield partial BaseModel or List of BaseModel]
    S -->|Exception| W[log warning and continue]
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    In1[Prompt variables kwargs] --> FMT[Format prompt messages]
    In2[llm_kwargs] --> CALL
    FMT --> TOOL[CallableTool.from_model]
    TOOL --> CALL[FunctionCallingLLM call]
    CALL -->|predict_and_call| RESP[AgentChatResponse]
    RESP --> PARSE[parse_tool_outputs allow_parallel_tool_calls]
    PARSE --> Out1[BaseModel]
    PARSE --> Out2[List of BaseModel]
```

---

## State Machine Diagram (Streaming Lifecycle)

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Streaming : start stream_call/astream_call
    state Streaming {
        [*] --> AwaitChunk
        AwaitChunk --> ProcessChunk : on ChatResponse
        ProcessChunk --> UpdateState : processor.process(partial, cur_objects)
        UpdateState --> Yield : yield objects
        Yield --> AwaitChunk
        ProcessChunk --> WarnOnError : on Exception
        WarnOnError --> AwaitChunk : continue
    }
    Streaming --> Completed : generator exhausted
    Completed --> [*]
```

---

## Architecture Overview (Block Diagram)

```mermaid
flowchart TB
    subgraph Prompts
      BPT[BasePromptTemplate / PromptTemplate]
    end

    subgraph Tools
      CT[CallableTool]
      SOP[StreamingObjectProcessor]
    end

    subgraph LLMs
      LLM[LLM]
      FCLLM[FunctionCallingLLM]
      LLM --> FCLLM
    end

    subgraph Structured_Tools
      TOLL[ToolOrchestratingLLM]
    end

    CFG[Configs.llm]

    BPT <--uses--> TOLL
    CT <--used by--> TOLL
    SOP <--used by--> TOLL
    TOLL --sync--> FCLLM:::sync
    TOLL --async--> FCLLM:::async
    TOLL --stream--> FCLLM:::stream
    CFG -.fallback .-> TOLL

    classDef sync stroke:#0b7,stroke-width:2px
    classDef async stroke:#07b,stroke-width:2px
    classDef stream stroke:#b70,stroke-width:2px
```

---

### Notes and Type Details

- Generics
  - `BasePydanticLLM[Model]` where `Model: TypeVar` bound to `pydantic.BaseModel`
  - `ToolOrchestratingLLM` is instantiated as `BasePydanticLLM[BaseModel]`

- Method return types
  - `__call__` / `acall`: `BaseModel | List[BaseModel]`
  - `stream_call`: `Generator[Model | List[Model], None, None]`
  - `astream_call`: `AsyncGenerator[Model | List[Model], None]`

- Key decisions and error paths
  - `validate_llm`: raises `AssertionError` if no LLM provided or configured; raises `ValueError` if not a function-calling model
  - `validate_prompt`: raises `ValueError` for invalid prompt types
  - `stream_call`/`astream_call`: raise `ValueError` if underlying LLM is not `FunctionCallingLLM`; warn and continue on streaming parse errors

- Parallel tool calls
  - `allow_parallel_tool_calls=True` may yield/return `List[BaseModel]` and affects parsing and selection in streaming

- Collaborators in method signatures
  - Parameters: `CallableTool`, `BasePromptTemplate`/`PromptTemplate`, `FunctionCallingLLM`, `StreamingObjectProcessor`
  - Return values: `BaseModel`, `List[BaseModel]`, `Generator`, `AsyncGenerator`; internally uses `AgentChatResponse`
