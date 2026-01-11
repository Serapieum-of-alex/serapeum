# Component Boundaries and Interactions

This diagram shows how components interact during the complete lifecycle.

```mermaid
graph TB
    subgraph User Space
        UC[User Code]
        MT[ModelTest Schema]
    end

    subgraph TextCompletionLLM Components
        TCL[TextCompletionLLM]

        subgraph Validators
            VP[validate_prompt]
            VL[validate_llm]
            VO[validate_output_parser_cls]
        end

        subgraph Stored State
            SP["_prompt: BasePromptTemplate"]
            SL["_llm: LLM"]
            SO["_output_parser: PydanticOutputParser"]
            SC["_output_cls: Type ModelTest"]
        end
    end

    subgraph Output Parser Layer
        POP[PydanticOutputParser]
        EJ[extract_json_str]
        VJ[model_validate_json]
    end

    subgraph Prompt Layer
        PT[PromptTemplate]
        CPT[ChatPromptTemplate]

        subgraph Prompt Operations
            PF[format]
            FM[format_messages]
            AV[Apply Variables]
        end
    end

    subgraph LLM Layer
        OL[Ollama]

        subgraph LLM Operations
            GM[_get_messages]
            GP[_get_prompt]
            EM[_extend_messages]
            EP[_extend_prompt]
        end

        subgraph API Methods
            CH[chat]
            CO[complete]
        end

        MD[metadata.is_chat_model]
    end

    subgraph External Service
        OS[Ollama Server]

        subgraph Endpoints
            EC["/api/chat"]
            EG["/api/generate"]
        end
    end

    subgraph Response Models
        CHR[ChatResponse]
        COR[CompletionResponse]
    end

    %% Initialization Flow
    UC -->|1. Create parser| POP
    POP -->|output_cls| MT

    UC -->|2. Initialize| TCL
    TCL -->|Validate| VO
    VO -->|Extract| MT

    TCL -->|Validate| VL
    VL -->|Check/Fallback| OL

    TCL -->|Validate/Convert| VP
    VP -->|Create if string| PT

    VO -->|Store| SO
    VO -->|Store| SC
    VL -->|Store| SL
    VP -->|Store| SP

    %% Execution Flow - Chat Path
    UC -->|3. Call with kwargs| TCL
    TCL -->|Check| MD

    MD -->|True| FM
    FM -->|Use kwargs| AV
    AV -->|Messages| GM
    GM -->|Add system| EM
    EM -->|Send| CH
    CH -->|HTTP POST| EC
    EC -->|Response| CHR
    CHR -->|message.content| EJ

    %% Execution Flow - Completion Path
    MD -->|False| PF
    PF -->|Use kwargs| AV
    AV -->|String| GP
    GP -->|Add system| EP
    EP -->|Send| CO
    CO -->|HTTP POST| EG
    EG -->|Response| COR
    COR -->|text| EJ

    %% Parsing Flow
    EJ -->|Extract| VJ
    VJ -->|Validate| MT
    MT -->|Instance| TCL
    TCL -->|Type check| UC

    %% Styling
    classDef userClass fill:#e1f5ff,stroke:#01579b
    classDef validatorClass fill:#fff9c4,stroke:#f57f17
    classDef stateClass fill:#f3e5f5,stroke:#4a148c
    classDef parserClass fill:#e8f5e9,stroke:#1b5e20
    classDef promptClass fill:#fce4ec,stroke:#880e4f
    classDef llmClass fill:#e0f2f1,stroke:#004d40
    classDef responseClass fill:#fff3e0,stroke:#e65100
    classDef externalClass fill:#efebe9,stroke:#3e2723

    class UC,MT userClass
    class VP,VL,VO validatorClass
    class SP,SL,SO,SC stateClass
    class POP,EJ,VJ parserClass
    class PT,CPT,PF,FM,AV promptClass
    class OL,GM,GP,EM,EP,CH,CO,MD llmClass
    class CHR,COR responseClass
    class OS,EC,EG externalClass
```

## Component Responsibilities Matrix

| Component | Initialization | Execution | Parsing |
|-----------|---------------|-----------|---------|
| **TextCompletionLLM** | Validates & stores all components | Routes to chat/complete | Type checks output |
| **PydanticOutputParser** | Stores output_cls schema | - | Extracts JSON & validates |
| **PromptTemplate** | Created/validated | Formats with variables | - |
| **Ollama (LLM)** | Validated/stored | Executes requests | - |
| **ModelTest** | Defines schema | - | Validates parsed data |

## Interaction Patterns

### 1. Initialization Pattern (Constructor)
```
User → TextCompletionLLM.__init__
  ├─→ validate_output_parser_cls
  │   └─→ Extract/Create parser & output_cls
  ├─→ validate_llm
  │   └─→ Use provided or fallback to Configs.llm
  └─→ validate_prompt
      └─→ Convert string to PromptTemplate if needed
```

### 2. Chat Model Execution Pattern
```
User → TextCompletionLLM.__call__
  ├─→ Check metadata.is_chat_model == True
  ├─→ PromptTemplate.format_messages(kwargs)
  ├─→ LLM._extend_messages
  ├─→ Ollama.chat → HTTP → ChatResponse
  ├─→ Extract message.content
  └─→ PydanticOutputParser.parse → ModelTest
```

### 3. Completion Model Execution Pattern
```
User → TextCompletionLLM.__call__
  ├─→ Check metadata.is_chat_model == False
  ├─→ PromptTemplate.format(kwargs)
  ├─→ LLM._extend_prompt
  ├─→ Ollama.complete → HTTP → CompletionResponse
  ├─→ Extract response.text
  └─→ PydanticOutputParser.parse → ModelTest
```

## State Management

### Immutable State (Post-Initialization)
- `_output_parser`: PydanticOutputParser instance
- `_output_cls`: Type[ModelTest]
- `_llm`: Ollama instance

### Mutable State
- `_prompt`: Can be updated via setter
- `_verbose`: Controls logging output

### Transient State (Per Call)
- `llm_kwargs`: Forwarded to LLM methods
- `**kwargs`: Template variables for prompt formatting
- `raw_output`: Intermediate text from LLM
- `parsed_output`: Final validated Pydantic model
