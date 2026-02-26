# TextCompletionLLM Usage Examples

This guide provides comprehensive examples covering all possible ways to use `TextCompletionLLM`.

!!! note "Model compatibility"
    Despite the name, `TextCompletionLLM` works with **both** chat/instruct models and raw
    text-completion models. Internally it routes to `llm.chat()` when
    `llm.metadata.is_chat_model` is `True`, and to `llm.complete()` otherwise.

    **Streaming is not supported** by this class. If you need incremental results,
    use [`ToolOrchestratingLLM`](../tool_orchestrating_llm/examples.md) with `stream=True` instead.

## Table of Contents

1. [Prerequisites: Ollama Cloud API Key](#prerequisites-ollama-cloud-api-key)
2. [Basic Usage](#basic-usage)
3. [Initialization Patterns](#initialization-patterns)
4. [Prompt Formats](#prompt-formats)
5. [Execution Modes](#execution-modes)
6. [Advanced Usage](#advanced-usage)
7. [Error Handling](#error-handling)

---

## Prerequisites: Ollama Cloud API Key

The examples in this guide use the [Ollama Cloud](https://ollama.com/cloud) inference API, which requires an API key.

**Steps to create your API key:**

1. Create an account at [ollama.com](https://ollama.com) (or sign in if you already have one)
2. Navigate to [ollama.com/settings/keys](https://ollama.com/settings/keys)
3. Click **Generate** to create a new API key
4. Copy the key immediately — it will not be shown again

**Set the environment variable:**

```bash
export OLLAMA_API_KEY=your_api_key_here
```

Or add it to your `.env` file:

```
OLLAMA_API_KEY=your_api_key_here
```

**Loading the `.env` file in Python:**

Install [`python-dotenv`](https://pypi.org/project/python-dotenv/):

```bash
pip install python-dotenv
```

Then load it at the top of your script:

```python notest
from dotenv import load_dotenv

load_dotenv()  # loads variables from .env into os.environ
```

All examples below read the key via `os.environ.get("OLLAMA_API_KEY")`.

---

## Basic Usage

### Simple String Prompt with Variables

The most straightforward way to use `TextCompletionLLM`:

```python
import os
from pydantic import BaseModel
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama


# Define your output schema
class Greeting(BaseModel):
    message: str
    language: str

# Initialize the LLM
llm = Ollama(
    model="ministral-3:14b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    request_timeout=180
)

# Create output parser
output_parser = PydanticParser(output_cls=Greeting)

# Create TextCompletionLLM with string prompt
text_llm = TextCompletionLLM(
    output_parser=output_parser,
    prompt="Generate a greeting in {language} for {name}. Return as JSON.",
    llm=llm,
)

# Execute with variables
result = text_llm(language="dutch", name="Ahmed")
print(result.message)  # "Hallo Ahmed"
print(result.language)  # "Dutch"
```

---

## Initialization Patterns

### 1. With Explicit Output Parser

Provide a fully configured `PydanticParser`:

```python
import os
from pydantic import BaseModel, Field
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama


class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price in USD")
    in_stock: bool = Field(description="Availability status")

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)
output_parser = PydanticParser(output_cls=Product)

text_llm = TextCompletionLLM(
    output_parser=output_parser,
    prompt="Extract product information from: {text}",
    llm=llm,
)

result = text_llm(text="iPhone 15 costs $999 and is available")
# Returns: Product(name="iPhone 15", price=999.0, in_stock=True)
```

### 2. With Output Class Only (Auto-creates Parser)

Let `TextCompletionLLM` create the parser for you:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Person(BaseModel):
    name: str
    age: int

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

# Parser is created automatically from output_cls
text_llm = TextCompletionLLM(
    output_parser=None,  # Will be auto-created
    prompt="Extract person info from: {bio}",
    output_cls=Person,  # Parser created from this
    llm=llm,
)

result = text_llm(bio="John Smith is 30 years old")
# Returns: Person(name="John Smith", age=30)
```

### 3. Using Global LLM from Configs

Set a default LLM for the entire application:

```python
import os
from pydantic import BaseModel
from serapeum.core.configs.configs import Configs
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

# Set global LLM
Configs.llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

class Task(BaseModel):
    id: str
    priority: int

# No need to pass llm parameter
text_llm = TextCompletionLLM(
    output_cls=Task,
    prompt="Create a task from: {description}",
    # llm=None uses Configs.llm by default
)

result = text_llm(description="Fix critical bug in authentication")
# Returns: Task(id="Fix critical bug in authentication", priority=1)
```

---

## Prompt Formats

### 1. String Prompt (Auto-converted to PromptTemplate)

Simple string prompts are automatically wrapped in `PromptTemplate`:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Summary(BaseModel):
    summary: str
    word_count: int

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Summary,
    prompt="Generate an essay with a maximum number of words {max_words} about the topic: {text}",  # String prompt
    llm=llm,
)

result = text_llm(
    text="AI",
    max_words=50
)
```

### 2. PromptTemplate Object

Use `PromptTemplate` for more control:

```python
import os
from pydantic import BaseModel
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Sentiment(BaseModel):
    sentiment: str
    confidence: float

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

# Create explicit PromptTemplate
prompt_template = PromptTemplate(
    "Analyze sentiment of: {review}"
)

text_llm = TextCompletionLLM(
    output_cls=Sentiment,
    prompt=prompt_template,
    llm=llm,
)

result = text_llm(review="This product is amazing!")
# Returns: Sentiment(sentiment="positive", confidence=1.0)
```

### 3. ChatPromptTemplate with Messages

Use structured message templates for chat models:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import Message, MessageRole
from serapeum.core.prompts import ChatPromptTemplate
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Translation(BaseModel):
    translated_text: str
    source_language: str
    target_language: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

# Create message templates
messages = [
    Message(
        role=MessageRole.SYSTEM,
        content="You are a professional translator."
    ),
    Message(
        role=MessageRole.USER,
        content="Translate to {target_lang}: {text}"
    ),
]

prompt = ChatPromptTemplate(message_templates=messages)

text_llm = TextCompletionLLM(
    output_cls=Translation,
    prompt=prompt,
    llm=llm,
)

result = text_llm(target_lang="French", text="Hello, world!")
# Returns: Translation(translated_text="Bonjour, monde!", ...)
```

## Execution Modes

### 1. Synchronous Execution

Standard blocking execution:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Answer(BaseModel):
    answer: str
    reasoning: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Answer,
    prompt="Answer this question: {question}",
    llm=llm,
)

# Synchronous call using __call__
result = text_llm(question="What is the capital of France?")
print(result.answer)  # "Paris"
```

### 2. Asynchronous Execution

Non-blocking async execution:

```python
import os
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Analysis(BaseModel):
    result: str
    confidence: float

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Analysis,
    prompt="Analyze: {data}",
    llm=llm,
)

async def analyze_data(data: str) -> Analysis:
    # Asynchronous call using acall
    result = await text_llm.acall(data=data)
    return result

# Run async function
result = asyncio.run(analyze_data("Sample data"))
```

### 3. Batch Processing

Process multiple inputs efficiently:

```python
import os
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Category(BaseModel):
    category: str
    subcategory: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Category,
    prompt="Categorize this item: {item}",
    llm=llm,
)

async def categorize_batch(items: list[str]) -> list[Category]:
    tasks = [text_llm.acall(item=item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

items = ["Laptop", "Apple", "T-shirt", "Novel"]
categories = asyncio.run(categorize_batch(items))
for item, cat in zip(items, categories):
    print(f"{item}: {cat.category}/{cat.subcategory}")
```

### 4. Passing LLM-specific Parameters

Forward parameters directly to the LLM:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Story(BaseModel):
    title: str
    content: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Story,
    prompt="Write a {genre} story about {topic}",
    llm=llm,
)

# Pass LLM-specific kwargs
result = text_llm(
    llm_kwargs={
        "temperature": 0.8,     # Higher temperature for creativity
        "top_p": 0.9,
        "max_tokens": 500,
    },
    genre="sci-fi",
    topic="time travel"
)
```

---

## Advanced Usage

### 1. Dynamic Prompt Updates

Change the prompt at runtime:

```python
import os
from pydantic import BaseModel
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Response(BaseModel):
    response: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Response,
    prompt="Default prompt: {input}",
    llm=llm,
)

# Use with initial prompt
result1 = text_llm(input="test")

# Update prompt dynamically
text_llm.prompt = PromptTemplate("Updated prompt: {input}")

# Use with new prompt
result2 = text_llm(input="test")
```

### 2. Reusable Instance Pattern

Create once, use many times:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Entity(BaseModel):
    name: str
    type: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

# Create reusable instance
entity_extractor = TextCompletionLLM(
    output_cls=Entity,
    prompt="Extract the main entity from: {text}",
    llm=llm,
)

# Reuse multiple times
texts = [
    "Apple Inc. announced new products",
    "Paris is the capital of France",
    "Python is a programming language",
]

for text in texts:
    entity = entity_extractor(text=text)
    print(f"{entity.name} ({entity.type})")
```

### 3. Complex Nested Models

Use deeply nested Pydantic models:

```python
import os
from pydantic import BaseModel, Field
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Address(BaseModel):
    street: str
    city: str
    country: str

class Contact(BaseModel):
    email: str
    phone: str

class Person(BaseModel):
    name: str
    age: int
    address: Address
    contact: Contact

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Person,
    prompt="Extract person information from: {bio}",
    llm=llm,
)

bio = """
John Doe, 30 years old
Lives at 123 Main St, New York, USA
Email: john@example.com
Phone: +1-555-0123
"""

result = text_llm(bio=bio)
print(result.name)              # "John Doe"
print(result.address.city)      # "New York"
print(result.contact.email)     # "john@example.com"
```

### 4. Optional and Union Types

Handle optional fields and union types:

```python
import os
from typing import Optional, Union
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Event(BaseModel):
    title: str
    date: str
    location: Optional[str] = None
    attendees: Optional[int] = None
    type: Union[str, None] = "general"

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Event,
    prompt="Extract event details from: {text}",
    llm=llm,
)

# Works with partial information
result = text_llm(text="Python Conference on March 15")
print(result.title)      # "Python Conference"
print(result.date)       # "March 15"
print(result.location)   # None (optional field)
```

### 5. List and Array Fields

Extract lists of items:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time: int  # in minutes

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Recipe,
    prompt="Extract recipe from: {text}",
    llm=llm,
)

recipe_text = """
Chocolate Chip Cookies
Ingredients: flour, butter, sugar, eggs, chocolate chips
Steps: Mix dry ingredients, cream butter and sugar, add eggs, fold in chips, bake
Prep time: 30 minutes
"""

result = text_llm(text=recipe_text)
print(result.name)
print(f"Ingredients: {', '.join(result.ingredients)}")
print(f"Steps: {len(result.steps)}")
```

---

## Error Handling

### 1. Handling Validation Errors

Catch and handle Pydantic validation errors:

```python
import os
from pydantic import BaseModel, ValidationError
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class StrictModel(BaseModel):
    count: int  # Must be integer
    ratio: float  # Must be float

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=StrictModel,
    prompt="Extract numbers from: {text}",
    llm=llm,
)

try:
    result = text_llm(text="Invalid data")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Handle invalid response format
except ValueError as e:
    print(f"Type mismatch: {e}")
    # Handle type checking errors
```

### 2. Handling Missing LLM

Gracefully handle missing LLM configuration:

```python
from pydantic import BaseModel
from serapeum.core.configs.configs import Configs
from serapeum.core.llms import TextCompletionLLM

class Data(BaseModel):
    value: str

# Clear global LLM
Configs.llm = None

try:
    text_llm = TextCompletionLLM(
        output_cls=Data,
        prompt="Process: {input}",
        llm=None,  # No LLM provided
    )
except AssertionError as e:
    print("LLM must be provided or set in Configs")
    # Provide fallback or configuration instructions
```

### 3. Handling Type Mismatches

Handle cases where parser returns wrong type:

```python
import os
from pydantic import BaseModel
from serapeum.core.output_parsers import BaseParser
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama


class ExpectedModel(BaseModel):
    value: str

class WrongModel(BaseModel):
    other: str

# Custom parser that returns wrong type
class FaultyParser(BaseParser):

    def parse(self, output: str):
        return WrongModel(other=output)  # Wrong type!

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_parser=FaultyParser(),
    output_cls=ExpectedModel,
    prompt="Process: {input}",
    llm=llm,
)

try:
    result = text_llm(input="test")
except ValueError as e:
    print(f"Type check failed: {e}")
    # Parser returned wrong type
```

### 4. Retry Logic

Implement retry logic for robustness:

```python
import os
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Result(BaseModel):
    data: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=Result,
    prompt="Process: {input}",
    llm=llm,
)

async def call_with_retry(
    text_llm: TextCompletionLLM,
    max_retries: int = 3,
    **kwargs
) -> Result:
    """Call TextCompletionLLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return await text_llm.acall(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Use with retry
result = asyncio.run(call_with_retry(text_llm, input="test"))
```

---

## Best Practices

### 1. Model Validation

Always define clear Pydantic models with validation:

```python
import os
from pydantic import BaseModel, Field, field_validator
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class ValidatedData(BaseModel):
    email: str = Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(ge=0, le=150)
    score: float = Field(ge=0.0, le=1.0)

    @field_validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

text_llm = TextCompletionLLM(
    output_cls=ValidatedData,
    prompt="Extract data from: {text}",
    llm=llm,
)
```

### 2. Clear Prompt Instructions

Provide clear instructions for JSON output:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Output(BaseModel):
    result: str

llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)

# Good: Clear instructions
text_llm = TextCompletionLLM(
    output_cls=Output,
    prompt="""
    Analyze the following text and return ONLY valid JSON.
    Do not include any explanation or markdown formatting.

    Text: {text}

    Return format: {{"result": "your analysis here"}}
    """,
    llm=llm,
)
```

### 3. Instance Reuse

Create instances once and reuse them:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import TextCompletionLLM
from serapeum.ollama import Ollama

class Classification(BaseModel):
    category: str
    confidence: float

# Create once
llm = Ollama(model="ministral-3:14b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=180)
classifier = TextCompletionLLM(
    output_cls=Classification,
    prompt="Classify: {text}",
    llm=llm,
)
texts = ["Apple", "Banana", "Pear"]

# Reuse many times - this is efficient!
for text in texts:
    result = classifier(text=text)
```

---

## See Also

- [Execution Flow and Method Calls](./text_completion_llm_sequence.md) - Detailed sequence diagram
- [Architecture and Class Relationships](./text_completion_llm_class.md) - Class structure
- [Data Transformations and Validation](./text_completion_llm_dataflow.md) - Data flow details
- [Component Boundaries and Interactions](./text_completion_llm_components.md) - System components
- [Lifecycle States and Transitions](./text_completion_llm_state.md) - State management
