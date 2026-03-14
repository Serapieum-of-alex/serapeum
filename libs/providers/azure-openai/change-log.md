## serapeum-azure-openai-0.1.0 (2026-03-14)


- fix(ci): add openai and azure-openai to pypi-release tag pattern (#62)
- The pypi-release workflow_run trigger resolves the package name by                            
  matching the most recent git tag against a hardcoded grep pattern.                            
  The pattern only included core, ollama, and llama-cpp — causing                               
  releases for serapeum-openai to silently fall back to the most                                
  recent matching tag (serapeum-llama-cpp).                                                     
                                                                                                
  - Add openai|azure-openai to the grep alternation in pypi-release                             
  - Add serapeum-openai and serapeum-azure-openai to the                                        
    workflow_dispatch options list                                                              
  - Reorder github-release options for consistency
-   Closes #63
- build: sync uv.lock after serapeum-openai version bump
- refactor(api)!: unify provider API surface across core, OpenAI, Azure, and Ollama (#48)
- - Move shared logic (tool-call extraction, response validation) from                                               
    individual providers into FunctionCallingLLM base class
  - Rename `structured_predict` to `parse` across all providers
  - Rename `LikelihoodScore` to `LogProb` for accuracy
  - Replace `ContentBlock` discriminated union with `ChunkType`
  - Store tool calls in `Message.chunks` instead of `additional_kwargs`
  - Merge `force_single_tool_call` into `ChatResponse`
  - Add `ToolCallBlock.get_arguments()` and move `ToolCallArguments`
    to `core.base.llms.types`
  - Remove `Message.from_str` and `MessageList.from_list` constructors;
    convert `Message` constructor to a Pydantic validator
  - Rename `_validate_chat_with_tools_response` to `_validate_response`
    and lift it to base class
  - Refactor OpenAI provider: rename classes, restructure Responses API,
    reorganize tests into completions/ and responses/ directories
  - Add Azure OpenAI README, docstrings, and e2e/responses test suites
  - Add timeout to `requests.get` calls and resolve Bandit findings
  - Remove Claude code-review CI workflows
  - Fix install-groups parameter in all test workflows
-   BREAKING CHANGE: `structured_predict` renamed to `parse`;
  `LikelihoodScore` renamed to `LogProb`; `ContentBlock` replaced by
  `ChunkType`; `Message.from_str` and `MessageList.from_list` removed;
  tool calls now stored in `Message.chunks` instead of
  `additional_kwargs`; `force_single_tool_call` merged into
  `ChatResponse`; OpenAI class names updated
- ref: #50, #51, #52, #53, #54, #55, #56, #57, #58
- feat: add OpenAI and Azure OpenAI providers with retry framework (#15)
- - Add serapeum-openai provider with chat completions and responses API
    support, structured outputs, tool calling, and streaming
  - Add serapeum-azure-openai provider extending OpenAI with Azure
    endpoint, deployment, and Entra ID authentication support
  - Add core retry framework with sync/async/streaming wrappers and
    per-provider retryable-exception classifiers (core, ollama, llama-cpp,
    openai)
  - Refactor OpenAI class into composable base classes (Client,
    ModelMetadata, StructuredOutput) under llm/base/ with factory methods
    for subclass client creation
  - Add parsers subpackage splitting formatters, chat parsers, and
    response parsers into dedicated modules
  - Rename core abstractions/mixins to abstractions/adapters and merge
    stream methods into chat/complete with a stream parameter
  - Harden existing providers: forbid extra attributes, rename
    request_timeout to timeout, disable SDK retries in favour of @retry
    decorator
  - Add CI workflows for openai and azure-openai test suites
  - Add conftest SDK pre-import guards for all providers to prevent
    namespace collisions during doctest collection
  - Add streaming object processor utilities for incremental structured
    output parsing in orchestrators
-   BREAKING CHANGE: `serapeum.core.llms.abstractions.mixins` renamed to
  `serapeum.core.llms.abstractions.adapters`; `ChatToCompletionMixin`
  renamed to `ChatToCompletion`, `CompletionToChatMixin` renamed to
  `CompletionToChat`; `stream_chat`/`astream_chat`/`stream_complete`/
  `astream_complete` removed in favour of `stream=True` parameter on
  `chat`/`achat`/`complete`/`acomplete`.
-   Closes #42, #43, #44, #45, #46
- build: sync uv.lock after serapeum-llama-cpp version bump
- fix(release): add missing change log file (#40)
- ci(release): add `serapeum-llama-cpp` to the github-release workflow (#39)
- fix(ollama,ci): stabilize e2e tests and split CI into separate workflows (#37)
- fix(ollama,ci): stabilize e2e tests and split CI into separate workflows
-   - Fix streaming completion tests to assert on final chunk only, not
    every chunk — cloud models emit empty-content chunks that cause
    str(r).strip() to return "" on intermediate responses
  - Add function_calling pytest marker to all ToolOrchestratingLLM and
    structured-predict tests; exclude them from cloud CI with
    -m "e2e and not function_calling" since Ollama Cloud does not
    support the tools API reliably
  - Change streaming count assertion from == 2 to >= 1 (count is
    model-dependent, not a framework contract)
  - Skip embedding e2e tests in CI (no embedding models on Ollama Cloud)
  - Split test-core.yml and introduce test-ollama.yml so Ollama Cloud
    failures no longer block core CI
  - Update cloud model default from qwen3-next:80b to mistral-large-3:675b
  - Register function_calling marker in pyproject.toml
-  ref: #38
- feat(llama-cpp): add serapeum-llama-cpp provider package (#12)
- feat(llama-cpp): add serapeum-llama-cpp provider package
-   - Add new `serapeum-llama-cpp` provider package under                                                       
    `libs/providers/llama-cpp/` with full src layout and namespace
    package `serapeum.llama_cpp`
  - Implement `LlamaCPP` class inheriting from `LLM` +               
    `CompletionToChatMixin` for running quantised GGUF models locally
  - Add `CompletionToChatMixin` to core, bridging completion-based
    providers into the chat interface automatically
  - Add model formatters for Llama 2 and Llama 3 prompt templates
    under `serapeum.llama_cpp.formatters`
  - Add utility helpers: GGUF model file fetching from URL or
    HuggingFace Hub, caching, download progress, and timeout handling
  - Add `n_gpu_layers`, `stop`, `tokenize()`, `count_tokens()`, and
    context-window methods to `LlamaCPP`
  - Add error handling for empty choices, stalled downloads, missing
    headers, and non-serialisable `model_kwargs`
  - Add HuggingFace Hub integration as an optional download backend
  - Add comprehensive unit, mock, integration, and e2e test suites
    (~2 600 lines across formatters, llm, and utils)
  - Add dedicated CI workflow `test-llama-cpp.yml`; split core tests
    into a separate `test-core.yml`; remove the old monolithic
    `test.yml`
  - Extend core `CompletionResponse` / `BaseLLM` types to support the
    completion-to-chat bridge
  - Use lazy `__getattr__` in provider `__init__` modules to prevent
    circular-import issues when third-party SDK names collide with
    namespace sub-packages
  - Add full MkDocs reference documentation for the llama-cpp provider
- ref: #35
- build: bump up ollama (#33)
- refactor(core,ollama)!: unify streaming API and rename methods for consistency (#31)
- refactor(core,ollama)!: unify streaming API and rename methods for consistency
- BREAKING CHANGE: Major API refactoring consolidating streaming methods and
renaming public APIs for improved consistency and developer experience.
- ## Key Changes
- ### Stream Parameter Unification
- Merge `stream_chat()`/`astream_chat()` into `chat(stream=True)`/`achat(stream=True)`
- Merge `stream_complete()`/`astream_complete()` into `complete(stream=True)`/`acomplete(stream=True)`
- Remove all separate `stream_*` methods from `BaseLLM` and subclasses
- Add `stream: bool = False` parameter to all primary methods
- ### Method Renaming
- Rename `structured_predict()` → `parse()` for concise terminology
- Rename `chat_with_tools()` → `generate_tool_calls()` for clearer intent
- Rename `predict_and_call()` → `invoke_callable()` for clarity
- Rename `stream_structured_predict()` → `parse(stream=True)`
- Rename `stream_call()`/`astream_call()` → `__call__(stream=True)`/`acall(stream=True)`
- ### Parameter Renaming
- Rename `output_tool` → `schema` in `ToolOrchestratingLLM` (more accurate naming)
- Rename `output_cls` → `schema` in `LLM.parse()` for consistency
- ### Signature Alignment
- Align `BaseLLM.chat()` abstract method signature with subclass implementations
- Add `ABC` inheritance to `BaseLLM` for proper abstract base class behavior
- Update return types to `ChatResponse | ChatResponseGen` for stream support
- ### Documentation & Tests
- Update 24 documentation files with new API examples
- Add comprehensive tests: `test_ollama_structured_predict.py` (1148 lines)
- Add unit tests: `test_parse_unit.py` (876 lines)
- Update all existing tests to use new API
- Update README files for core and ollama packages
- ## Migration Guide
- ```python
# Streaming methods
llm.stream_chat(messages)          → llm.chat(messages, stream=True)
llm.astream_chat(messages)         → llm.achat(messages, stream=True)
llm.stream_complete(prompt)        → llm.complete(prompt, stream=True)
- # Method renames
llm.structured_predict(output_cls=Model)  → llm.parse(schema=Model)
llm.chat_with_tools(tools)                → llm.generate_tool_calls(tools)
llm.predict_and_call(tools)               → llm.invoke_callable(tools)
- # Orchestrator
ToolOrchestratingLLM(output_tool=Model)   → ToolOrchestratingLLM(schema=Model)
orchestrator.stream_call(**kwargs)        → orchestrator(**kwargs, stream=True)
- ref: #32
- feat(core)!: add markdown doc testing and refactor orchestrator API (#29)
- feat(core)!: add markdown doc testing and refactor orchestrator API
-   - Add pytest-markdown-docs to validate code blocks in documentation files
  - Configure markdown testing as pre-commit hook and CI workflow
  - Rename `output_cls` parameter to `output_tool` in ToolOrchestratingLLM
  - Change ToolOrchestratingLLM to accept keyword-only arguments only
  - Add custom ToolCallError exception for structured error handling
  - Update Ollama provider and examples to use Ollama Cloud API
  - Add python-dotenv dependency for environment variable management
-   BREAKING CHANGE: ToolOrchestratingLLM now requires keyword-only arguments, and the `output_cls` parameter has been renamed to `output_tool`.  
- ref: #30
- fix(ci): pypi-release (#28)
- ci:finalize release to pypi.org (#27)
- ci(release): wire pypi-release to trigger on github-release completion (#24)
- ci(release): wire pypi-release to trigger on github-release completion
  
  - Rename github-release workflow (required for workflow_run reference)                                                                                                            
  - Replace release: event with workflow_run trigger on github-release                                                                                                   
  - Add workflow_dispatch inputs for manual publish (package + registry)
  - Resolve package name from most recent serapeum-{pkg}-* tag on auto runs
  - Delegate build and publish to composite pypi action
  - Move checkout step from composite action into caller workflow
  - Add update_changelog_on_bump = true to core and ollama commitizen configs
- ref: #25
- fix(deps): remove httpx dependency and resolve build issues (#19)
- fix(deps): remove httpx dependency and resolve build issues
-   - Replace httpx with stdlib urllib.request in tests
  - Add pydantic and numpy version constraints
  - Skip E2E tests when serapeum-ollama unavailable
  - Fix sdist packaging configuration
- feat(embeddings): add embedding support with Ollama provider implementation (#11)
- feat(embeddings): add embedding support with Ollama provider implementation
                                                                                                                                                                                                                              
  - Add BaseEmbedding abstraction in serapeum-core with sync/async interfaces                                                                                                                                                 
  - Implement OllamaEmbedding with support for query, text, and batch operations                                                                                                                                              
  - Add embedding data models (BaseNode, TextNode, ImageNode, LinkedNodes, NodeInfo)                                                                                                                                          
  - Refactor provider structure to move ollama from serapeum.llms.ollama to serapeum.ollama                                                                                                                                   
  - Add comprehensive test suite (unit, e2e, pydantic validation) with 3200+ test lines
  - Add pytest-xdist and pytest-benchmark for parallel testing and benchmarking
  - Update documentation and READMEs for embedding functionality
- refactor!: restructure core and providers packages and APIs (#17)
- refactor!: restructure core and providers packages and APIs
- - move core into `libs/core` and providers into `libs/providers/ollama`
- rename multiple modules/types (models→types, prompts/utils→format, tools utils)
- merge structured_tools into llms and rename StructuredLLM to StructuredOutputLLM
- update build config, CI, docs, and tests for new layout
- BREAKING CHANGE: public import paths and several module/class names were renamed or relocated; update imports to the new libs-based structure.
- fix: improve schema guidance and tool call handling (#8)
- fix: improve schema guidance and tool call handling
- - Move schema helpers into core.utils.schemas and reuse in parsers
- Generate concise required-field descriptions in tool metadata
- Guard empty tool_calls in Ollama and adjust tests/models
- feat(core): implement initial core architecture and foundational LLM utility features (#3)
- **feat(core): implement initial core architecture and foundational LLM utility features**
- - Design and implement core modules for LLM orchestration, function conversion, prompt validation, and tool execution
- Introduce key classes: FunctionConverter, SyncAsyncConverter, ToolExecutor, ToolCallArguments, Docstring, Schema, ArgumentCoercer, StreamingObjectProcessor, MessageList, and FlexibleModel
- Establish foundational LLM abstractions: BaseLLM, FunctionCallingLLM, ToolOrchestratingLLM, TextCompletionLLM, and related parser classes
- Provide robust support for synchronous/asynchronous function handling, tool calling, argument coercion, and schema validation
- Integrate comprehensive unit and integration tests for all major components and workflows
- Add extensive documentation, architecture diagrams, and usage examples for core modules and LLM integration
- Configure project metadata, dependencies (`numpy`, `filetype`, `requests`), and development workflows for Python 3.11/3.12 compatibility
- Set up modern packaging, namespace structure, and CI/CD with uv and mkdocs for documentation
- ---
- **Package Insights:**
- The `serapeum-core` package is designed as a utility library for LLM-based applications, supporting generative AI, chatbots, RAG, and NLP workflows.
- It provides a modular, extensible foundation for building, orchestrating, and testing LLM-driven tools and pipelines.
- The architecture is production-ready, with robust testing, documentation, and modern Python packaging best practices.
- ref: #5
