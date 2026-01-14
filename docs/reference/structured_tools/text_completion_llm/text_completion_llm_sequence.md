# Execution Flow and Method Calls

This diagram shows the complete workflow from initialization to execution of `TextCompletionLLM`.

```mermaid
sequenceDiagram
    participant User
    participant PydanticParser
    participant TextCompletionLLM
    participant PromptTemplate
    participant Ollama
    participant LLMChat/Complete

    Note over User: Initialization Phase
    User->>PydanticParser: Create parser with ModelTest
    activate PydanticParser
    PydanticParser-->>User: parser instance
    deactivate PydanticParser

    User->>TextCompletionLLM: __init__(output_parser, prompt, llm)
    activate TextCompletionLLM

    TextCompletionLLM->>TextCompletionLLM: validate_output_parser_cls(parser, None)
    Note over TextCompletionLLM: Validates parser is PydanticParser<br/>Extracts output_cls (ModelTest)

    TextCompletionLLM->>TextCompletionLLM: validate_llm(llm)
    Note over TextCompletionLLM: Ensures LLM instance is provided<br/>or falls back to Configs.llm

    TextCompletionLLM->>TextCompletionLLM: validate_prompt(prompt)
    Note over TextCompletionLLM: Converts string to PromptTemplate<br/>if needed

    TextCompletionLLM->>PromptTemplate: Create/validate prompt template
    activate PromptTemplate
    PromptTemplate-->>TextCompletionLLM: template instance
    deactivate PromptTemplate

    TextCompletionLLM-->>User: text_llm instance
    deactivate TextCompletionLLM

    Note over User: Execution Phase
    User->>TextCompletionLLM: __call__(test_input="hello")
    activate TextCompletionLLM

    TextCompletionLLM->>Ollama: Check metadata.is_chat_model
    activate Ollama
    Ollama-->>TextCompletionLLM: True/False
    deactivate Ollama

    alt is_chat_model == True
        TextCompletionLLM->>PromptTemplate: format_messages(llm, test_input="hello")
        activate PromptTemplate
        PromptTemplate-->>TextCompletionLLM: List[Message]
        deactivate PromptTemplate

        TextCompletionLLM->>Ollama: _extend_messages(messages)
        activate Ollama
        Note over Ollama: Add system prompts if configured
        Ollama-->>TextCompletionLLM: extended messages
        deactivate Ollama

        TextCompletionLLM->>Ollama: chat(messages, **llm_kwargs)
        activate Ollama
        Ollama->>LLMChat/Complete: HTTP request to Ollama server
        activate LLMChat/Complete
        LLMChat/Complete-->>Ollama: JSON response
        deactivate LLMChat/Complete
        Ollama-->>TextCompletionLLM: ChatResponse
        deactivate Ollama

        TextCompletionLLM->>TextCompletionLLM: Extract message.content
        Note over TextCompletionLLM: raw_output = chat_response.message.content
    else is_chat_model == False
        TextCompletionLLM->>PromptTemplate: format(llm, test_input="hello")
        activate PromptTemplate
        PromptTemplate-->>TextCompletionLLM: formatted prompt string
        deactivate PromptTemplate

        TextCompletionLLM->>Ollama: complete(formatted_prompt, **llm_kwargs)
        activate Ollama
        Ollama->>LLMChat/Complete: HTTP request to Ollama server
        activate LLMChat/Complete
        LLMChat/Complete-->>Ollama: JSON response
        deactivate LLMChat/Complete
        Ollama-->>TextCompletionLLM: CompletionResponse
        deactivate Ollama

        TextCompletionLLM->>TextCompletionLLM: Extract text
        Note over TextCompletionLLM: raw_output = response.text
    end

    TextCompletionLLM->>PydanticParser: parse(raw_output)
    activate PydanticParser
    PydanticParser->>PydanticParser: extract_json_str(raw_output)
    Note over PydanticParser: Extracts JSON from text

    PydanticParser->>PydanticParser: model_validate_json(json_str)
    Note over PydanticParser: Validates against ModelTest schema

    PydanticParser-->>TextCompletionLLM: ModelTest instance
    deactivate PydanticParser

    TextCompletionLLM->>TextCompletionLLM: Validate isinstance(output, ModelTest)
    Note over TextCompletionLLM: Raises ValueError if type mismatch

    TextCompletionLLM-->>User: ModelTest instance
    deactivate TextCompletionLLM
```

## Key Points

1. **Initialization validates all components** before storing them
2. **Prompt formatting** adapts based on whether the LLM is a chat model or completion model
3. **Output parsing** extracts JSON and validates against the Pydantic schema
4. **Type checking** ensures the parsed output matches the expected output class
