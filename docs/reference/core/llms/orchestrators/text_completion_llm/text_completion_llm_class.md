# Architecture and Class Relationships

This diagram shows the class relationships and structure for `TextCompletionLLM`.

```mermaid
classDiagram
    class BasePydanticLLM~BaseModel~ {
        <<abstract>>
        +output_cls: Type[BaseModel]
        +__call__(**kwargs) BaseModel
        +acall(**kwargs) BaseModel
    }

    class TextCompletionLLM~BaseModel~ {
        +__init__(output_parser, prompt, output_cls, llm, verbose)
        +prompt: BasePromptTemplate
        +output_cls: Type[BaseModel]
        +__call__(llm_kwargs, **kwargs) BaseModel
        +acall(llm_kwargs, **kwargs) BaseModel
        -_output_parser: BaseParser
        -_output_cls: Type[BaseModel]
        -_llm: LLM
        -_prompt: BasePromptTemplate
        -_verbose: bool
        +_validate_prompt(prompt) BasePromptTemplate
        +_validate_llm(llm) LLM
        +_validate_output_parser_cls(parser, cls) Tuple
    }

    class BaseParser {
        <<abstract>>
        +parse(output: str) Any
        +format(query: str) str
        +format_messages(messages) List[Message]
    }

    class PydanticParser~Model~ {
        -_output_cls: Type[Model]
        -_excluded_schema_keys_from_format: List
        -_pydantic_format_tmpl: str
        +__init__(output_cls, excluded_schema_keys, pydantic_format_tmpl)
        +output_cls: Type[Model]
        +format_string: str
        +get_format_string(escape_json) str
        +parse(text: str) Any
        +format(query: str) str
    }

    class BasePromptTemplate {
        <<abstract>>
        +metadata: Dict[str, Any]
        +template_vars: List[str]
        +kwargs: Dict[str, str]
        +output_parser: Optional[BaseParser]
        +template_var_mappings: Optional[Dict]
        +function_mappings: Optional[Dict]
        +partial_format(**kwargs) BasePromptTemplate
        +format(llm, **kwargs) str
        +format_messages(llm, **kwargs) List[Message]
        +get_template(llm) str
    }

    class PromptTemplate {
        +template: str
        +__init__(template, prompt_type, output_parser, metadata, ...)
        +partial_format(**kwargs) PromptTemplate
        +format(llm, completion_to_prompt, **kwargs) str
        +format_messages(llm, **kwargs) List[Message]
        +get_template(llm) str
    }

    class ChatPromptTemplate {
        +message_templates: List[Message]
        +__init__(message_templates, prompt_type, output_parser, ...)
        +from_messages(message_templates, **kwargs) ChatPromptTemplate
        +partial_format(**kwargs) ChatPromptTemplate
        +format(llm, messages_to_prompt, **kwargs) str
        +format_messages(llm, **kwargs) List[Message]
        +get_template(llm) str
    }

    class BaseLLM {
        <<abstract>>
        +metadata: Metadata
        +chat(messages, **kwargs) ChatResponse
        +stream_chat(messages, **kwargs) ChatResponseGen
        +achat(messages, **kwargs) ChatResponse
        +astream_chat(messages, **kwargs) ChatResponseAsyncGen
        +complete(prompt, **kwargs) CompletionResponse
        +stream_complete(prompt, **kwargs) CompletionResponseGen
        +acomplete(prompt, **kwargs) CompletionResponse
        +astream_complete(prompt, **kwargs) CompletionResponseAsyncGen
    }

    class LLM {
        +system_prompt: Optional[str]
        +messages_to_prompt: MessagesToPromptCallable
        +completion_to_prompt: CompletionToPromptCallable
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

    class Ollama {
        +model: str
        +base_url: str
        +request_timeout: int
        +temperature: float
        +metadata: Metadata
        +__init__(model, base_url, request_timeout, ...)
        +chat(messages, **kwargs) ChatResponse
        +achat(messages, **kwargs) ChatResponse
        +complete(prompt, **kwargs) CompletionResponse
        +acomplete(prompt, **kwargs) CompletionResponse
        -_chat_request(messages, stream, **kwargs) dict
        -_complete_request(prompt, stream, **kwargs) dict
    }

    class BaseModel {
        <<pydantic>>
        +model_validate_json(json_data) BaseModel
        +model_json_schema() dict
    }

    class ModelTest {
        +hello: str
    }

    BasePydanticLLM <|-- TextCompletionLLM
    BaseLLM <|-- LLM
    LLM <|-- Ollama
    BaseParser <|-- PydanticParser
    BasePromptTemplate <|-- PromptTemplate
    BasePromptTemplate <|-- ChatPromptTemplate
    BaseModel <|-- ModelTest

    TextCompletionLLM o-- PydanticParser : uses
    TextCompletionLLM o-- BasePromptTemplate : uses
    TextCompletionLLM o-- LLM : uses
    TextCompletionLLM ..> ModelTest : produces
    PydanticParser o-- ModelTest : validates against
    PromptTemplate --|> BasePromptTemplate
    ChatPromptTemplate --|> BasePromptTemplate

    note for TextCompletionLLM "Main orchestrator that:\n1. Validates parser, prompt, LLM\n2. Formats prompts with variables\n3. Calls LLM (chat or complete)\n4. Parses output to Pydantic model"
    note for PydanticParser "Extracts JSON from text\nand validates against schema"
    note for Ollama "Concrete LLM implementation\nfor Ollama server"
```

## Class Responsibilities

### TextCompletionLLM
- **Orchestrates** the complete workflow from prompt to structured output
- **Validates** all components during initialization
- **Routes** to chat or completion based on LLM metadata
- **Ensures** type safety of output

### PydanticParser
- **Extracts** JSON from raw LLM output
- **Validates** JSON against Pydantic schema
- **Formats** prompts with schema hints

### Ollama (LLM)
- **Executes** requests to Ollama server
- **Supports** both chat and completion modes
- **Handles** streaming and async operations

### PromptTemplate/ChatPromptTemplate
- **Formats** prompts with variables
- **Supports** both string and message-based templates
- **Manages** template variables and mappings
