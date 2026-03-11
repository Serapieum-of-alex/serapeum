# Serapeum Azure OpenAI Provider

**Azure OpenAI integration for the Serapeum LLM framework**

The `serapeum-azure-openai` package extends `serapeum-openai` to work with
Azure OpenAI Service deployments. It provides:

- **Azure Authentication**: API key and Microsoft Entra ID (Azure AD)
  support
- **Deployment Management**: Routes requests to your named Azure
  deployments
- **Full Feature Parity**: Inherits all chat, streaming, tool calling, and
  structured output capabilities from the OpenAI provider

The `AzureOpenAI` class extends `serapeum.openai.OpenAI`, so all features
from the OpenAI provider work transparently with Azure deployments.

## Installation

### From Source

```bash
cd libs/providers/azure-openai
uv sync --active
```

### From PyPI (when published)

```bash
pip install serapeum-azure-openai
```

## Prerequisites

1. An Azure OpenAI resource with at least one model deployment
2. Either an API key or Microsoft Entra ID credentials

Set the required environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE.openai.azure.com/"
export OPENAI_API_VERSION="2024-02-01"
```

## Quick Start

### API Key Authentication

```python
from serapeum.azure_openai import AzureOpenAI
from serapeum.core.llms import Message, MessageRole

llm = AzureOpenAI(
    engine="my-gpt4o-deployment",
    model="gpt-4o",
    api_key="your-azure-api-key",
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
    api_version="2024-02-01",
)

messages = [
    Message(role=MessageRole.USER, content="Explain quantum computing in one sentence.")
]
response = llm.chat(messages)
print(response.message.content)
```

### Microsoft Entra ID (Azure AD) Authentication

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from serapeum.azure_openai import AzureOpenAI

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

llm = AzureOpenAI(
    engine="my-gpt4o-deployment",
    model="gpt-4o",
    azure_ad_token_provider=token_provider,
    use_azure_ad=True,
    azure_endpoint="https://YOUR_RESOURCE.openai.azure.com/",
    api_version="2024-02-01",
)
```

## Features

Since `AzureOpenAI` extends the OpenAI provider, all features from
`serapeum-openai` are available. See the
[OpenAI provider README](../openai/README.md) for full usage examples of:

- Streaming (sync and async)
- Tool calling
- Structured outputs with Pydantic models
- Completion-style usage with prompt templates

The only difference is initialization: use `AzureOpenAI` with your
deployment name and Azure endpoint instead of `OpenAI` with an API key.

## Configuration

```python
from serapeum.azure_openai import AzureOpenAI

llm = AzureOpenAI(
    engine="my-deployment",                # Required: Azure deployment name
    model="gpt-4o",                        # Required: model name
    azure_endpoint="https://...",          # Azure resource endpoint
    api_key="...",                         # Azure API key (or env var)
    api_version="2024-02-01",             # API version
    use_azure_ad=False,                    # Use Entra ID authentication
    azure_ad_token_provider=None,          # Custom token provider callable
    temperature=0.7,                       # Sampling temperature
    max_tokens=1024,                       # Max tokens to generate
    timeout=60.0,                          # Request timeout
    additional_kwargs={},                  # Extra API parameters
)
```

The `engine` parameter accepts several aliases: `deployment_name`,
`deployment_id`, `deployment`, or `azure_deployment`.

## Package Layout

```
src/serapeum/azure_openai/
├── __init__.py    # Lazy imports (AzureOpenAI, SyncAzureOpenAI, AsyncAzureOpenAI)
├── llm.py         # AzureOpenAI class (extends OpenAI)
└── utils.py       # Azure AD token refresh, alias resolution
```

## Testing

```bash
# All tests (from repo root)
python -m pytest libs/providers/azure-openai/tests

# Skip end-to-end tests
python -m pytest libs/providers/azure-openai/tests -m "not e2e"
```

## Links

- **Homepage**:
  [github.com/Serapieum-of-alex/serapeum](https://github.com/Serapieum-of-alex/serapeum)
- **Documentation**:
  [serapieum-of-alex.github.io/serapeum](https://serapieum-of-alex.github.io/serapeum)
- **Azure OpenAI Service**:
  [learn.microsoft.com/azure/ai-services/openai](https://learn.microsoft.com/azure/ai-services/openai)
