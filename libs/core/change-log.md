## serapeum-core-0.4.0 (2026-03-02)


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

## serapeum-core-0.3.0 (2026-02-26)


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

## serapeum-core-0.2.0 (2026-02-23)


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
