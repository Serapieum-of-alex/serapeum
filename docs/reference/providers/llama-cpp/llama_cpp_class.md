# Architecture and Class Relationships

This diagram shows the class relationships and inheritance hierarchy for the `LlamaCPP` LLM implementation.

```mermaid
classDiagram
    class BaseLLM {
        <<abstract>>
        +metadata: Metadata
        +chat(messages, stream=false, **kwargs) ChatResponse | ChatResponseGen
        +achat(messages, stream=false, **kwargs) ChatResponse | ChatResponseAsyncGen
        +complete(prompt, stream=false, **kwargs) CompletionResponse | CompletionResponseGen
        +acomplete(prompt, stream=false, **kwargs) CompletionResponse | CompletionResponseAsyncGen
    }

    class LLM {
        +system_prompt: Optional[str]
        +messages_to_prompt: Callable
        +completion_to_prompt: Callable
        +output_parser: Optional[BaseParser]
        +structured_output_mode: StructuredOutputMode
        +_get_prompt(prompt, **kwargs) str
        +_get_messages(prompt, **kwargs) List[Message]
        +_parse_output(output) str
        +_extend_prompt(formatted_prompt) str
        +_extend_messages(messages) List[Message]
        +predict(prompt, **kwargs) str
        +stream(prompt, **kwargs) TokenGen
        +apredict(prompt, **kwargs) str
        +astream(prompt, **kwargs) TokenAsyncGen
        +parse(output_cls, prompt, **kwargs) Model
    }

    class CompletionToChatMixin {
        +chat(messages, stream=false, **kwargs) ChatResponse | ChatResponseGen
        +achat(messages, stream=false, **kwargs) ChatResponse | ChatResponseAsyncGen
        -_completion_to_chat(messages, **kwargs) ChatResponse
        -_stream_completion_to_chat(messages, **kwargs) ChatResponseGen
    }

    class LlamaCPP {
        +model_url: Optional[str]
        +model_path: Optional[str]
        +hf_model_id: Optional[str]
        +hf_filename: Optional[str]
        +temperature: float
        +max_new_tokens: int
        +context_window: int
        +n_gpu_layers: int
        +stop: list[str]
        +generate_kwargs: dict
        +model_kwargs: dict
        +verbose: bool
        -_model: Llama
        -_model_lock: threading.Lock
        +model_post_init(__context)
        +metadata: Metadata
        +complete(prompt, formatted, stream, **kwargs) CompletionResponse | CompletionResponseGen
        +acomplete(prompt, formatted, stream, **kwargs) CompletionResponse | CompletionResponseAsyncGen
        +tokenize(text) list[int]
        +count_tokens(text) int
        -_resolve_model_path() Path
        -_load_model(model_path) Llama
        -_complete(prompt, **kwargs) CompletionResponse
        -_stream_complete(prompt, **kwargs) CompletionResponseGen
        -_guard_context(prompt) None
        -_validate_model_path_exists(v) str | None
        -_check_model_source() LlamaCPP
        -_check_formatters() LlamaCPP
        -_prepare_kwargs(data) Any
    }

    class Llama {
        <<llama_cpp.Llama>>
        +__call__(prompt, **kwargs) dict
        +tokenize(text) list[int]
        +context_params: ContextParams
        +__init__(model_path, **kwargs)
    }

    class _MODEL_CACHE {
        <<WeakValueDictionary>>
        +get(key) Optional[Llama]
        +__setitem__(key, value)
    }

    class _MODEL_CACHE_LOCK {
        <<threading.Lock>>
        +acquire()
        +release()
    }

    class Metadata {
        +model_name: str
        +context_window: int
        +num_output: int
        +is_chat_model: bool
        +is_function_calling_model: bool
        +system_role: MessageRole
    }

    class Message {
        +role: MessageRole
        +content: str
        +additional_kwargs: dict
    }

    class MessageRole {
        <<enumeration>>
        SYSTEM
        USER
        ASSISTANT
        TOOL
    }

    class ChatResponse {
        +message: Message
        +raw: Optional[dict]
        +delta: Optional[str]
        +additional_kwargs: dict
    }

    class CompletionResponse {
        +text: str
        +raw: Optional[dict]
        +delta: Optional[str]
        +additional_kwargs: dict
    }

    class TextCompletionLLM {
        -_llm: LLM
        -_prompt: BasePromptTemplate
        -_output_parser: PydanticParser
        -_output_cls: Type[BaseModel]
        +__call__(**kwargs) BaseModel
        +acall(**kwargs) BaseModel
    }

    class BaseModel {
        <<pydantic>>
        +model_validate_json(json_data) BaseModel
        +model_json_schema() dict
    }

    class Llama2Formatter {
        <<module: formatters.llama2>>
        +messages_to_prompt(messages, system_prompt) str
        +completion_to_prompt(completion, system_prompt) str
    }

    class Llama3Formatter {
        <<module: formatters.llama3>>
        +messages_to_prompt_v3_instruct(messages, system_prompt) str
        +completion_to_prompt_v3_instruct(completion, system_prompt) str
    }

    %% Inheritance relationships
    BaseLLM <|-- LLM
    LLM <|-- LlamaCPP
    CompletionToChatMixin <|-- LlamaCPP

    %% Composition relationships
    LlamaCPP o-- Llama : uses (loaded in model_post_init)
    LlamaCPP ..> _MODEL_CACHE : caches model
    _MODEL_CACHE ..> _MODEL_CACHE_LOCK : thread-safe access
    LlamaCPP ..> Metadata : provides
    LlamaCPP ..> CompletionResponse : produces
    LlamaCPP ..> ChatResponse : produces (via mixin)
    LlamaCPP ..> Message : consumes (via mixin)

    %% Formatter relationships
    LlamaCPP ..> Llama2Formatter : uses (optional)
    LlamaCPP ..> Llama3Formatter : uses (optional)

    %% Message relationships
    Message o-- MessageRole : has
    ChatResponse o-- Message : contains

    %% Orchestrator relationships
    TextCompletionLLM o-- LlamaCPP : uses

    note for LlamaCPP "Main LLM implementation that:\n1. Loads GGUF models locally\n2. Supports completion and chat\n3. Handles streaming responses\n4. Provides sync/async interfaces\n5. Caches models across instances\n6. Thread-safe inference"
    note for CompletionToChatMixin "Bridges completion to chat:\n- Formats messages via messages_to_prompt\n- Delegates to complete()\n- Wraps result in ChatResponse"
    note for Llama "llama-cpp-python Llama class\nLoads and runs GGUF models"
    note for _MODEL_CACHE "WeakValueDictionary:\nShares Llama instances across\nLlamaCPP instances with same\nmodel_path + model_kwargs"
```

## Class Hierarchy

### Inheritance Chain
```
BaseLLM (abstract)
  └─→ LLM (adds prompting and structured outputs)
      └─→ CompletionToChatMixin (adds chat from completion)
          └─→ LlamaCPP (concrete implementation)
```

**Key difference from Ollama**: LlamaCPP does **not** inherit from `FunctionCallingLLM` — it does not support tool/function calling. Chat is provided through `CompletionToChatMixin` (completion → chat), whereas Ollama provides chat natively and derives completion from it.

## Component Responsibilities

### LlamaCPP
**Core LLM Implementation**
- **Model Management**: Downloads, caches, and loads GGUF model files
- **Completion**: Builds and executes text completion requests
- **Streaming**: Handles incremental response chunks
- **Context Guard**: Validates prompt length before inference
- **Tokenization**: Exposes model tokenizer for token counting
- **Thread Safety**: Serializes calls to the C backend with a lock
- **Configuration**: Manages model settings, temperature, max_tokens, etc.

### CompletionToChatMixin (Parent Mixin)
**Completion-to-Chat Bridge**
- **Chat Interface**: Provides `chat()` and `achat()` by delegating to `complete()`
- **Message Formatting**: Uses `messages_to_prompt` to format messages as a string
- **Response Wrapping**: Converts `CompletionResponse` to `ChatResponse`
- **Streaming Bridge**: Bridges streaming completion to streaming chat

### LLM (Grandparent Class)
**High-Level Orchestration**
- **Prompt Management**: Extends prompts with system messages
- **Message Formatting**: Converts between formats
- **Structured Outputs**: Forces Pydantic model outputs via `parse`
- **Parser Integration**: Applies output parsers to responses

### BaseLLM (Root Class)
**Core Interface**
- **Standard Methods**: Defines chat, complete, and their variants
- **Sync/Async**: Requires both synchronous and asynchronous implementations
- **Streaming**: Requires streaming variants of all methods
- **Metadata**: Requires metadata property for capabilities

### Llama (llama-cpp-python)
**Model Runtime**
- **Model Loading**: Loads GGUF files into memory
- **Inference**: Runs token generation on CPU/GPU
- **Tokenization**: Provides vocabulary-based tokenization
- **Streaming**: Supports streaming generation via iterator

### Model Cache
**Shared Memory**
- **WeakValueDictionary**: Models are shared across LlamaCPP instances
- **Automatic Cleanup**: Entries are removed when all references are garbage-collected
- **Thread-Safe**: Protected by a module-level lock

### Formatter Modules
**Prompt Templates**
- **Llama 2 / Mistral**: `[INST]...[/INST]` format
- **Llama 3 Instruct**: `<|start_header_id|>...<|eot_id|>` format
- **Pluggable**: Users can provide custom formatters

### Message/ChatResponse/CompletionResponse
**Data Models**
- **Message**: Represents a single chat message with role and content
- **ChatResponse**: Wraps assistant response with metadata
- **CompletionResponse**: Wraps text completion with metadata

## Design Patterns

### 1. Eager Initialization (vs. Ollama's Lazy Init)
```python notest
def model_post_init(self, __context):
    model_path = self._resolve_model_path()
    self._model = self._load_model(model_path)
```

### 2. Flyweight Pattern (Model Cache)
```python notest
_MODEL_CACHE: WeakValueDictionary[tuple[str, str], Llama] = WeakValueDictionary()

def _load_model(self, model_path: Path) -> Llama:
    cache_key = (str(model_path), json.dumps(self.model_kwargs, sort_keys=True))
    # Check cache → load if missing → store in cache
    ...
```

### 3. CompletionToChatMixin (Adapter Pattern)
```python notest
# Chat is derived from completion:
# 1. messages_to_prompt(messages) → formatted string
# 2. complete(formatted_string) → CompletionResponse
# 3. Wrap in ChatResponse
```

### 4. Thread Safety (Monitor Pattern)
```python notest
with self._model_lock:
    response = self._model(prompt=prompt, **call_kwargs)
```

## Integration Points

### With TextCompletionLLM
```
TextCompletionLLM uses LlamaCPP for:
  - Checking is_chat_model via metadata
  - Calling chat() or complete()
  - Getting raw text responses for parsing
```

### With External Packages
```
LlamaCPP depends on:
  - llama-cpp-python (Llama model runtime)
  - pydantic (for configuration and validation)
  - serapeum.core (for base classes and types)
  - requests (for URL model downloads)
  - huggingface-hub (optional, for HF downloads)
```
