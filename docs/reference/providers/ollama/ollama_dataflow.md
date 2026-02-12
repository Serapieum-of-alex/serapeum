# Data Transformations and Validation

This diagram shows how data flows and transforms through the Ollama LLM system.

```mermaid
flowchart TD
    Start([User Code]) --> Init{Initialize Ollama}

    Init --> SetConfig[Set Configuration]
    SetConfig --> StoreModel[Store model name]
    StoreModel --> StoreURL[Store base_url]
    StoreURL --> StoreTimeout[Store request_timeout]
    StoreTimeout --> StoreTemp[Store temperature]
    StoreTemp --> StoreJSON[Store json_mode flag]
    StoreJSON --> StoreKwargs[Store additional_kwargs]

    StoreKwargs --> CreateMetadata[Create Metadata]
    CreateMetadata --> SetChatFlag[Set is_chat_model=True]
    SetChatFlag --> SetFnCallFlag[Set is_function_calling_model]
    SetFnCallFlag --> SetContext[Set context_window]
    SetContext --> Ready([Ollama Instance Ready])

    Ready --> CallType{Call Type?}

    CallType -->|chat| ChatPath[Chat Path]
    CallType -->|complete| CompletePath[Complete Path]
    CallType -->|chat_with_tools| ToolsPath[Tools Path]
    CallType -->|stream_chat| StreamPath[Stream Path]

    %% Chat Path
    ChatPath --> ValidateMessages{Messages Valid?}
    ValidateMessages -->|No| Error1[Raise ValueError]
    ValidateMessages -->|Yes| BuildChatReq[Build Chat Request]

    BuildChatReq --> AddModel[Add model name]
    AddModel --> ConvertMessages[Convert Messages to dicts]
    ConvertMessages --> AddOptions[Add options: temperature, etc.]
    AddOptions --> CheckJSON{json_mode?}
    CheckJSON -->|True| AddFormat[Add format: json]
    CheckJSON -->|False| AddKeepAlive
    AddFormat --> AddKeepAlive[Add keep_alive]

    AddKeepAlive --> EnsureClient[Ensure client initialized]
    EnsureClient --> CheckClient{Client exists?}
    CheckClient -->|No| CreateClient[Create Client with base_url, timeout]
    CheckClient -->|Yes| SendRequest
    CreateClient --> SendRequest[Send HTTP POST /api/chat]

    SendRequest --> ReceiveRaw[Receive raw dict response]
    ReceiveRaw --> ParseResponse[_chat_from_response]

    ParseResponse --> ExtractMsg[Extract message dict]
    ExtractMsg --> ParseRole[Parse role: assistant]
    ParseRole --> ParseContent[Parse content: str]
    ParseContent --> CheckTools{tool_calls present?}
    CheckTools -->|Yes| ParseToolCalls[Parse tool_calls array]
    CheckTools -->|No| CreateMessage1
    ParseToolCalls --> CreateMessage1[Create Message object]

    CreateMessage1 --> ExtractMeta[Extract metadata: model, times, tokens]
    ExtractMeta --> CreateChatResp[Create ChatResponse]
    CreateChatResp --> ReturnChat([Return ChatResponse])

    %% Complete Path
    CompletePath --> Decorator[@chat_to_completion_decorator]
    Decorator --> WrapPrompt[Wrap prompt in Message]
    WrapPrompt --> SetRole[role=USER, content=prompt]
    SetRole --> CallChat[Delegate to chat method]
    CallChat --> ChatPath
    ReturnChat --> UnwrapDecorator[Decorator unwraps response]
    UnwrapDecorator --> ExtractText[Extract message.content as text]
    ExtractText --> CreateCompleteResp[Create CompletionResponse]
    CreateCompleteResp --> ReturnComplete([Return CompletionResponse])

    %% Tools Path
    ToolsPath --> PrepareTools[_prepare_chat_with_tools]
    PrepareTools --> ConvertToolsLoop[For each tool in tools]
    ConvertToolsLoop --> ExtractToolMeta[Extract tool.metadata]
    ExtractToolMeta --> GetSchema[Get fn_schema from metadata]
    GetSchema --> BuildToolDict[Build Ollama tool dict]
    BuildToolDict --> AddToolType[Add type: function]
    AddToolType --> AddToolFunc[Add function: name, description, parameters]

    AddToolFunc --> MergeKwargs[Merge tools into kwargs]
    MergeKwargs --> CallChatWithTools[Call chat with tools kwarg]
    CallChatWithTools --> SendRequestTools[HTTP POST with tools array]
    SendRequestTools --> ReceiveToolResp[Receive response with tool_calls]
    ReceiveToolResp --> ValidateTools[_validate_chat_with_tools_response]
    ValidateTools --> CheckParallel{allow_parallel?}
    CheckParallel -->|No| ForceSingle[force_single_tool_call]
    CheckParallel -->|Yes| ReturnToolResp
    ForceSingle --> ReturnToolResp([Return ChatResponse with tools])

    %% Stream Path
    StreamPath --> BuildStreamReq[Build chat request with stream=True]
    BuildStreamReq --> SendStreamReq[HTTP POST /api/chat streaming]
    SendStreamReq --> StreamLoop{For each chunk}

    StreamLoop --> ReceiveChunk[Receive chunk dict]
    ReceiveChunk --> ParseChunk[_chat_stream_from_response]
    ParseChunk --> ExtractDelta[Extract message delta]
    ExtractDelta --> AccumContent[Accumulate content]
    AccumContent --> CheckToolChunk{tool_calls in chunk?}
    CheckToolChunk -->|Yes| AccumTools[Accumulate tool_calls]
    CheckToolChunk -->|No| CreateStreamResp
    AccumTools --> CreateStreamResp[Create ChatResponse with delta]
    CreateStreamResp --> YieldResp[Yield ChatResponse]
    YieldResp --> CheckDone{done=True?}
    CheckDone -->|No| StreamLoop
    CheckDone -->|Yes| EndStream([Stream Complete])

    %% Error paths
    Error1 --> ErrorEnd([Raise Exception])

    %% Styling
    style Start fill:#e1f5ff
    style Ready fill:#e1f5ff
    style ReturnChat fill:#c8e6c9
    style ReturnComplete fill:#c8e6c9
    style ReturnToolResp fill:#c8e6c9
    style EndStream fill:#c8e6c9
    style Error1 fill:#ffcdd2
    style ErrorEnd fill:#ffcdd2
    style SendRequest fill:#fff9c4
    style SendRequestTools fill:#fff9c4
    style SendStreamReq fill:#fff9c4
```

## Data Transformation Examples

### 1. Initialization
```python
Input:
  Ollama(model="llama3.1", base_url="http://localhost:11434", request_timeout=180)

Transformations:
  1. Store configuration:
     - model = "llama3.1"
     - base_url = "http://localhost:11434"
     - request_timeout = 180
     - temperature = 0.75 (default)
     - json_mode = False (default)

  2. Create metadata:
     - model_name = "llama3.1"
     - is_chat_model = True
     - is_function_calling_model = True
     - context_window = 3900
     - num_output = 256

Output:
  Ollama instance with lazy-initialized client
```

### 2. Chat Request
```python
Input:
  messages = [Message(role=MessageRole.USER, content="Say 'pong'.")]
  kwargs = {"temperature": 0.2}

Transformations:
  1. Convert messages to dicts:
     [{"role": "user", "content": "Say 'pong'."}]

  2. Build request payload:
     {
       "model": "llama3.1",
       "messages": [{"role": "user", "content": "Say 'pong'."}],
       "options": {"temperature": 0.2},
       "stream": False,
       "keep_alive": None
     }

  3. HTTP POST to /api/chat

  4. Raw response:
     {
       "model": "llama3.1",
       "created_at": "2025-01-22T...",
       "message": {
         "role": "assistant",
         "content": "Pong!"
       },
       "done": True,
       "total_duration": 1234567890,
       "prompt_eval_count": 10,
       "eval_count": 2
     }

  5. Parse to ChatResponse:
     ChatResponse(
       message=Message(
         role=MessageRole.ASSISTANT,
         content="Pong!",
         additional_kwargs={}
       ),
       raw={...},
       additional_kwargs={
         "model": "llama3.1",
         "created_at": "...",
         "total_duration": 1234567890,
         "prompt_eval_count": 10,
         "eval_count": 2
       }
     )

Output:
  ChatResponse with assistant message
```

### 3. Complete Request (via Decorator)
```python
Input:
  prompt = "Say 'pong'."
  kwargs = {}

Transformations:
  1. Decorator wraps prompt:
     Message(role=MessageRole.USER, content="Say 'pong'.")

  2. Delegates to chat([message], **kwargs)
     [Follows Chat Request flow above]

  3. Decorator extracts text:
     text = chat_response.message.content  # "Pong!"

  4. Creates CompletionResponse:
     CompletionResponse(
       text="Pong!",
       raw={...},
       additional_kwargs={...}
     )

Output:
  CompletionResponse with text
```

### 4. Chat with Tools
```python
Input:
  messages = [Message(role=MessageRole.USER, content="Create album about rock")]
  tools = [CallableTool(fn=create_album, metadata=ToolMetadata(...))]
  kwargs = {}

Transformations:
  1. Convert each tool to Ollama format:
     {
       "type": "function",
       "function": {
         "name": "create_album",
         "description": "Create an album with title and songs",
         "parameters": {
           "type": "object",
           "properties": {
             "title": {"type": "string"},
             "artist": {"type": "string"},
             "songs": {"type": "array", "items": {"type": "string"}}
           },
           "required": ["title", "artist", "songs"]
         }
       }
     }

  2. Build request with tools:
     {
       "model": "llama3.1",
       "messages": [...],
       "tools": [<converted tool dicts>],
       "stream": False
     }

  3. HTTP POST to /api/chat

  4. Raw response with tool_calls:
     {
       "message": {
         "role": "assistant",
         "content": "",
         "tool_calls": [
           {
             "function": {
               "name": "create_album",
               "arguments": {
                 "title": "Rock Legends",
                 "artist": "Various Artists",
                 "songs": ["Song 1", "Song 2"]
               }
             }
           }
         ]
       },
       ...
     }

  5. Parse tool_calls in message:
     Message(
       role=MessageRole.ASSISTANT,
       content="",
       additional_kwargs={
         "tool_calls": [
           {
             "function": {
               "name": "create_album",
               "arguments": {...}
             }
           }
         ]
       }
     )

  6. If not allow_parallel, force_single_tool_call:
     Keep only first tool call

Output:
  ChatResponse with tool_calls in message.additional_kwargs
```

### 5. Streaming Chat
```python
Input:
  messages = [Message(role=MessageRole.USER, content="Count to 3")]
  stream = True

Transformations:
  1. Build request with stream=True

  2. HTTP POST returns chunk iterator

  3. For each chunk:
     Chunk 1: {"message": {"content": "1"}, "done": False}
       → ChatResponse(message=Message(content="1"), delta="1")
       → Yield

     Chunk 2: {"message": {"content": ", 2"}, "done": False}
       → ChatResponse(message=Message(content=", 2"), delta=", 2")
       → Yield

     Chunk 3: {"message": {"content": ", 3"}, "done": True}
       → ChatResponse(message=Message(content=", 3"), delta=", 3")
       → Yield

Output:
  Generator yielding ChatResponse objects
```

## Validation Points

### 1. Configuration Validation
- **model**: Must be non-empty string
- **base_url**: Must be valid URL format
- **temperature**: Must be float in range [0.0, 1.0]
- **request_timeout**: Must be positive float

### 2. Message Validation
- **messages**: Must be non-empty list
- **role**: Must be valid MessageRole enum
- **content**: Must be string (can be empty for tool calls)

### 3. Tool Validation
- **tools**: Must be list of BaseTool
- **tool.metadata**: Must have name, description, fn_schema
- **fn_schema**: Must be valid JSON schema dict

### 4. Response Validation
- **HTTP status**: Must be 200, else raise error
- **JSON parsing**: Must be valid JSON
- **Required fields**: Must have message/text in response
- **tool_calls format**: Must match expected structure

## Error Handling

### Network Errors
```
Request → Timeout → Raise RequestException with timeout info
Request → Connection Error → Raise ConnectionError with URL
Request → HTTP Error → Raise HTTPError with status code
```

### Parsing Errors
```
Response → Invalid JSON → Raise JSONDecodeError
Response → Missing fields → Raise KeyError
Response → Invalid tool_calls → Log warning, return empty list
```

### Configuration Errors
```
Invalid model → Raise ValueError
Invalid URL → Raise ValueError
Missing required field → Raise TypeError
```

## Data Flow Summary

```
User Input
  ↓
Configuration/Validation
  ↓
Request Building (convert to Ollama format)
  ↓
Client Initialization (lazy)
  ↓
HTTP Request (sync/async)
  ↓
Raw Response (dict)
  ↓
Response Parsing (to typed models)
  ↓
Validation/Post-processing
  ↓
Typed Response (ChatResponse/CompletionResponse)
  ↓
User Output
```
