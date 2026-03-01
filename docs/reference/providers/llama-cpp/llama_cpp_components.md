# Component Boundaries and Interactions

This diagram shows how components interact during the complete lifecycle of the LlamaCPP LLM.

```mermaid
graph TB
    subgraph User Space
        UC[User Code]
        PM[Pydantic Models]
    end

    subgraph LlamaCPP Core
        LC[LlamaCPP Instance]

        subgraph Configuration
            CFG[Configuration Fields]
            MD[Metadata]
        end

        subgraph Validation
            VMP[_validate_model_path_exists]
            VMS[_check_model_source]
            VFM[_check_formatters]
            VPK[_prepare_kwargs]
        end

        subgraph Model Resolution
            RES[_resolve_model_path]
            FURL[_fetch_model_file]
            FHF[_fetch_model_file_hf]
        end

        subgraph Model Loading
            LOAD[_load_model]
            CACHE[_MODEL_CACHE]
            LOCK[_MODEL_CACHE_LOCK]
        end

        subgraph Context Guard
            GUARD[_guard_context]
            TOK[tokenize / count_tokens]
        end

        subgraph Completion Engine
            COMP[_complete]
            SCOMP[_stream_complete]
            KWARGS[generate_kwargs builder]
        end

        subgraph Chat Bridge
            MIXIN[CompletionToChatMixin]
            M2P[messages_to_prompt]
            C2P[completion_to_prompt]
        end

        subgraph Async Bridge
            ACOMP[acomplete]
            TOTHREAD[asyncio.to_thread]
        end
    end

    subgraph Formatter Layer
        subgraph Llama 2 Formatter
            L2M[messages_to_prompt]
            L2C[completion_to_prompt]
        end

        subgraph Llama 3 Formatter
            L3M[messages_to_prompt_v3_instruct]
            L3C[completion_to_prompt_v3_instruct]
        end
    end

    subgraph Model Layer
        LLAMA[Llama Instance]

        subgraph Inference
            GEN[__call__ / generate]
            VOCAB[Tokenizer / Vocabulary]
            CTX[Context Window]
        end
    end

    subgraph External Sources
        LOCAL[Local GGUF File]
        URL[Remote URL]
        HF[HuggingFace Hub]
    end

    subgraph Orchestrator Layer
        TCL[TextCompletionLLM]

        subgraph Orchestrator Components
            PRS[PydanticParser]
            PTMP[PromptTemplate]
        end
    end

    subgraph Response Models
        CRESP[ChatResponse]
        CORESP[CompletionResponse]
        MSG[Message]
    end

    %% Initialization Flow
    UC -->|1. Initialize| LC
    LC -->|Validate fields| VMP
    VMP -->|Validate cross-field| VMS
    VMS -->|Validate formatters| VFM
    VFM -->|Merge defaults| VPK
    VPK -->|Store config| CFG
    CFG -->|Create| MD

    %% Model Resolution
    LC -->|model_post_init| RES
    RES -->|model_path set| LOCAL
    RES -->|model_url set| FURL
    RES -->|hf_model_id set| FHF
    FURL -->|Download from| URL
    FHF -->|Download from| HF
    FURL -->|Save to| LOCAL
    FHF -->|Save to| LOCAL

    %% Model Loading
    RES -->|Path resolved| LOAD
    LOAD -->|Check cache| CACHE
    CACHE -.->|Thread-safe| LOCK
    LOAD -->|Cache miss| LLAMA
    LLAMA -->|Store in cache| CACHE

    %% Completion Flow
    UC -->|2a. complete(prompt)| LC
    LC -->|Format prompt| C2P
    C2P -.->|Uses| L2C
    C2P -.->|Uses| L3C
    LC -->|Check length| GUARD
    GUARD -->|Count tokens| TOK
    TOK -->|Use vocabulary| VOCAB
    LC -->|Build kwargs| KWARGS
    KWARGS -->|Non-stream| COMP
    KWARGS -->|Stream| SCOMP

    COMP -->|Acquire lock| LOCK
    COMP -->|Call| GEN
    GEN -->|Use| CTX
    GEN -->|Return dict| COMP
    COMP -->|Create| CORESP
    CORESP -->|Return| UC

    SCOMP -->|Acquire lock| LOCK
    SCOMP -->|Stream call| GEN
    GEN -.->|Yield chunks| SCOMP
    SCOMP -.->|Yield| CORESP
    CORESP -.->|Yield| UC

    %% Chat Flow (via mixin)
    UC -->|2b. chat(messages)| MIXIN
    MIXIN -->|Format messages| M2P
    M2P -.->|Uses| L2M
    M2P -.->|Uses| L3M
    M2P -->|Formatted string| LC
    LC -->|Delegate to complete| COMP
    COMP -->|CompletionResponse| MIXIN
    MIXIN -->|Wrap in| CRESP
    CRESP -->|Contains| MSG
    CRESP -->|Return| UC

    %% Async Flow
    UC -->|2c. acomplete(prompt)| ACOMP
    ACOMP -->|Offload to| TOTHREAD
    TOTHREAD -->|Call sync| COMP
    COMP -->|Return| TOTHREAD
    TOTHREAD -->|Return| ACOMP
    ACOMP -->|Return| UC

    %% TextCompletionLLM Integration
    UC -->|3. TextCompletionLLM(llm=LlamaCPP)| TCL
    TCL -->|Store| LC
    TCL -->|Use| PTMP
    TCL -->|Use| PRS
    UC -->|Call| TCL
    TCL -->|Format prompt| PTMP
    PTMP -->|Messages| LC
    LC -->|chat| CRESP
    CRESP -->|message.content| PRS
    PRS -->|Parse JSON| PM
    PM -->|Return| UC

    %% Styling
    classDef userClass fill:#e1f5ff,stroke:#01579b
    classDef llamaClass fill:#e0f2f1,stroke:#004d40
    classDef configClass fill:#fff9c4,stroke:#f57f17
    classDef modelClass fill:#f3e5f5,stroke:#4a148c
    classDef formatterClass fill:#e8f5e9,stroke:#1b5e20
    classDef sourceClass fill:#efebe9,stroke:#3e2723
    classDef orchestratorClass fill:#fce4ec,stroke:#880e4f
    classDef responseClass fill:#fff3e0,stroke:#e65100

    class UC,PM userClass
    class LC llamaClass
    class CFG,MD,VMP,VMS,VFM,VPK,GUARD,TOK,COMP,SCOMP,KWARGS,MIXIN,M2P,C2P,ACOMP,TOTHREAD configClass
    class LLAMA,GEN,VOCAB,CTX modelClass
    class RES,LOAD,CACHE,LOCK,FURL,FHF formatterClass
    class LOCAL,URL,HF sourceClass
    class TCL,PRS,PTMP orchestratorClass
    class CRESP,CORESP,MSG responseClass
    class L2M,L2C,L3M,L3C formatterClass
```

## Component Interaction Patterns

### 1. Initialization Pattern
```
User Code
  └─→ LlamaCPP.__init__
      ├─→ Validators (field_validator, model_validator):
      │   ├─→ _validate_model_path_exists: Check local file exists
      │   ├─→ _check_model_source: Ensure one source is set
      │   ├─→ _check_formatters: Ensure formatters are provided
      │   └─→ _prepare_kwargs: Merge n_ctx, verbose, n_gpu_layers into model_kwargs
      ├─→ Store: model_path, model_url, temperature, max_new_tokens, etc.
      ├─→ Create Metadata: is_chat_model=True, is_function_calling_model=False
      └─→ model_post_init:
          ├─→ _resolve_model_path: Download if needed, return local Path
          └─→ _load_model: Load Llama with caching
```

### 2. Model Resolution Pattern
```
_resolve_model_path()
  ├─→ If model_path is set:
  │   └─→ Return Path(model_path)
  ├─→ If hf_model_id is set:
  │   ├─→ Create cache directory
  │   ├─→ Call _fetch_model_file_hf(repo_id, filename, cache_dir)
  │   │   └─→ hf_hub_download → Path
  │   └─→ Update self.model_path, return Path
  └─→ Else (model_url or default):
      ├─→ Compute cache path from URL filename
      ├─→ If not cached:
      │   └─→ Call _fetch_model_file(url, path)
      │       └─→ requests.get(stream=True) with progress bar
      └─→ Update self.model_path, return Path
```

### 3. Model Loading Pattern (with Cache)
```
_load_model(model_path)
  ├─→ Compute cache_key = (str(path), json(model_kwargs))
  ├─→ With _MODEL_CACHE_LOCK: check cache
  │   └─→ result = _MODEL_CACHE.get(cache_key)
  ├─→ If cache miss:
  │   ├─→ loaded = Llama(model_path, **model_kwargs)
  │   └─→ With _MODEL_CACHE_LOCK: store if still missing
  └─→ Return result
```

### 4. Completion Request Pattern
```
User → LlamaCPP.complete(prompt, formatted, stream, **kwargs)
  ├─→ If not formatted:
  │   └─→ prompt = self.completion_to_prompt(prompt)
  ├─→ If stream:
  │   └─→ _stream_complete(prompt, **kwargs)
  │       ├─→ _guard_context(prompt) → count tokens, raise if too long
  │       ├─→ Build call_kwargs from generate_kwargs + defaults
  │       ├─→ Acquire _model_lock
  │       └─→ Yield CompletionResponse per chunk
  └─→ Else:
      └─→ _complete(prompt, **kwargs)
          ├─→ _guard_context(prompt)
          ├─→ Build call_kwargs
          ├─→ Acquire _model_lock
          ├─→ self._model(prompt=prompt, **call_kwargs)
          └─→ Return CompletionResponse(text=choices[0]["text"])
```

### 5. Chat via Mixin Pattern
```
User → LlamaCPP.chat(messages, stream, **kwargs)
  └─→ CompletionToChatMixin.chat(messages, stream, **kwargs)
      ├─→ formatted = self.messages_to_prompt(messages)
      ├─→ result = self.complete(formatted, formatted=True, stream, **kwargs)
      └─→ If stream:
      │   └─→ Convert each CompletionResponse chunk → ChatResponse chunk
      └─→ Else:
          └─→ Wrap CompletionResponse → ChatResponse
```

### 6. Async Pattern
```
User → LlamaCPP.acomplete(prompt, formatted, stream, **kwargs)
  ├─→ If stream:
  │   ├─→ chunks = await asyncio.to_thread(lambda: list(self.complete(..., stream=True)))
  │   └─→ Return async generator yielding chunks
  └─→ Else:
      └─→ await asyncio.to_thread(self.complete, prompt, formatted, stream=False, **kwargs)
```

### 7. TextCompletionLLM Integration Pattern
```
User → TextCompletionLLM(output_parser=parser, prompt=prompt, llm=LlamaCPP(...))
  └─→ TextCompletionLLM stores LlamaCPP instance

User → text_llm(key="value")
  ├─→ Check llm.metadata.is_chat_model → True
  ├─→ Format prompt with variables → List[Message]
  ├─→ LlamaCPP.chat(messages) → ChatResponse
  ├─→ Extract response.message.content
  ├─→ PydanticParser.parse(content) → Model instance
  └─→ Return Model instance
```

## Component State Management

### LlamaCPP Instance State
```
Initialization:
  - model_path: str (set during init or model_post_init)
  - model_url: Optional[str] (immutable after init)
  - hf_model_id: Optional[str] (immutable after init)
  - temperature: float (immutable after init)
  - max_new_tokens: int (immutable after init)
  - context_window: int (immutable after init)
  - n_gpu_layers: int (immutable after init)
  - stop: list[str] (immutable after init)
  - generate_kwargs: dict (immutable after init)
  - model_kwargs: dict (immutable after init)
  - _model: Llama (loaded during model_post_init)
  - _model_lock: threading.Lock (per-instance)

Runtime:
  - _model: Llama instance (loaded, never changes)
  - _model_lock: Acquired during each inference call
```

### Model Cache State (Module-level)
```
_MODEL_CACHE: WeakValueDictionary
  - Keys: (model_path_str, model_kwargs_json)
  - Values: Llama instances (weak references)
  - Entries auto-removed when all LlamaCPP refs are GC'd

_MODEL_CACHE_LOCK: threading.Lock
  - Protects read/write to _MODEL_CACHE
```

### Request State (Per Call)
```
Input:
  - prompt: str
  - formatted: bool
  - stream: bool
  - **kwargs: Additional generation options

Processing:
  - formatted_prompt: str (after completion_to_prompt if needed)
  - call_kwargs: dict (merged generate_kwargs + defaults + overrides)
  - raw_response: dict from Llama backend

Output:
  - CompletionResponse or CompletionResponseGen
```

## Error Boundaries

### 1. Configuration Errors (Initialization)
```
Validators:
  ├─→ _validate_model_path_exists: ValueError if path doesn't exist
  ├─→ _check_model_source: ValueError if no source specified
  ├─→ _check_model_source: ValueError if hf_model_id without hf_filename
  └─→ _check_formatters: ValueError if formatters missing
```

### 2. Model Resolution Errors (model_post_init)
```
_resolve_model_path:
  ├─→ _fetch_model_file: ValueError (Content-Length < 1 MB)
  ├─→ _fetch_model_file: requests.ConnectionError
  ├─→ _fetch_model_file: RuntimeError (download succeeded but file missing)
  ├─→ _fetch_model_file_hf: ImportError (huggingface-hub not installed)
  └─→ _fetch_model_file_hf: Network/auth errors from hf_hub_download
```

### 3. Inference Errors (During Call)
```
_guard_context:
  └─→ ValueError if prompt tokens exceed context_window

_complete / _stream_complete:
  └─→ RuntimeError from llama-cpp-python backend
```

## Component Dependencies

### LlamaCPP Depends On:
- `llama_cpp.Llama` (external package — model runtime)
- `serapeum.core.llms.LLM` (base class)
- `serapeum.core.llms.CompletionToChatMixin` (chat bridge)
- `serapeum.core.llms` types (Message, CompletionResponse, ChatResponse, Metadata)
- `serapeum.core.utils.base.get_cache_dir` (cache directory)
- `pydantic` (for configuration and validation)
- `requests` (for URL downloads)
- `huggingface_hub` (optional, for HF downloads)

### LlamaCPP Is Used By:
- `TextCompletionLLM` (as the LLM engine)
- Direct user code (standalone usage)

### External Dependencies:
- **GGUF Model File**: Must be available locally or downloadable
- **llama-cpp-python**: Must be compiled for the target platform (CPU/CUDA/Metal)
