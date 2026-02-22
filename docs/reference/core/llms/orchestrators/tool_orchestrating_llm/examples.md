# ToolOrchestratingLLM Usage Examples

This guide provides comprehensive examples covering all possible ways to use `ToolOrchestratingLLM`.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Initialization Patterns](#initialization-patterns)
3. [Prompt Formats](#prompt-formats)
4. [Execution Modes](#execution-modes)
5. [Parallel Tool Calls](#parallel-tool-calls)
6. [Advanced Usage](#advanced-usage)
7. [Error Handling](#error-handling)

---

## Basic Usage

### Simple String Prompt with Variables

The most straightforward way to use `ToolOrchestratingLLM`:

```python
import os
from typing import List
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

# Define your output schema
class Song(BaseModel):
    title: str
    duration: int  # in seconds

class Album(BaseModel):
    title: str
    artist: str
    songs: List[Song]

# Initialize the LLM with function calling support
llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

# Create ToolOrchestratingLLM with string prompt
tools_llm = ToolOrchestratingLLM(
    output_cls=Album,
    prompt="Create an album about {topic} with {num_songs} songs.",
    llm=llm,
)

# Execute with variables - LLM uses function calling to return structured data
result = tools_llm(topic="space exploration", num_songs=3)
print(result.title)  # "Journey Through the Cosmos"
print(result.artist)  # "The Astronomers"
print(len(result.songs))  # 3
```

---

## Initialization Patterns

### 1. With Explicit LLM

Provide a fully configured function-calling LLM:

```python
import os
from typing import List
from pydantic import BaseModel, Field
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Task(BaseModel):
    id: str = Field(description="Task identifier")
    description: str = Field(description="Task description")
    priority: int = Field(description="Priority level 1-5", ge=1, le=5)
    subtasks: List[str] = Field(description="List of subtasks")

# Initialize Ollama with function calling support
llm = Ollama(
    model="qwen3.5:397b",
    api_key=os.environ.get("OLLAMA_API_KEY"),
    request_timeout=80,
    temperature=0.7,
)

tools_llm = ToolOrchestratingLLM(
    output_cls=Task,
    prompt="Break down this project into tasks: {project}",
    llm=llm,
)

result = tools_llm(project="Build a web application")
# Returns: Task with properly structured data via function calling
```

### 2. Using Global LLM from Configs

Set a default function-calling LLM for the entire application:

```python
import os
from pydantic import BaseModel
from serapeum.core.configs.configs import Configs
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

# Set global LLM
Configs.llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

class Entity(BaseModel):
    name: str
    type: str
    properties: dict

# No need to pass llm parameter
tools_llm = ToolOrchestratingLLM(
    output_cls=Entity,
    prompt="Extract entity from: {text}",
    # llm=None uses Configs.llm by default
)

result = tools_llm(text="Apple Inc. is a technology company")
# Returns: Entity(name="Apple Inc.", type="company", ...)
```

### 3. With Tool Choice Strategy

Control which tool the LLM should use:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Response(BaseModel):
    answer: str
    confidence: float

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

# Force the LLM to use the tool
tools_llm = ToolOrchestratingLLM(
    output_cls=Response,
    prompt="Answer: {question}",
    llm=llm,
    tool_choice="auto",  # or "required" to force tool use
)

result = tools_llm(question="What is Python?")
```

---

## Prompt Formats

### 1. String Prompt (Auto-converted to PromptTemplate)

Simple string prompts are automatically wrapped in `PromptTemplate`:

```python
import os
from typing import List
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    steps: List[str]
    prep_time: int

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Recipe,
    prompt="Create a {cuisine} recipe for {dish}",  # String prompt
    llm=llm,
)

result = tools_llm(cuisine="Italian", dish="pasta")
```

### 2. PromptTemplate Object

Use `PromptTemplate` for more control:

```python
import os
from pydantic import BaseModel
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Analysis(BaseModel):
    sentiment: str
    topics: list[str]
    summary: str

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

# Create explicit PromptTemplate
prompt_template = PromptTemplate(
    "Analyze this text: {text}\n\nProvide sentiment, topics, and summary."
)

tools_llm = ToolOrchestratingLLM(
    output_cls=Analysis,
    prompt=prompt_template,
    llm=llm,
)

result = tools_llm(text="AI is transforming industries worldwide")
```

### 3. ChatPromptTemplate with Messages

Use structured message templates for complex prompts:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import Message, MessageRole, ToolOrchestratingLLM
from serapeum.core.prompts import ChatPromptTemplate
from serapeum.ollama import Ollama

class CodeReview(BaseModel):
    issues: list[str]
    suggestions: list[str]
    rating: int  # 1-10

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

# Create message templates
messages = [
    Message(
        role=MessageRole.SYSTEM,
        content="You are an expert code reviewer."
    ),
    Message(
        role=MessageRole.USER,
        content="Review this {language} code:\n\n{code}"
    ),
]

prompt = ChatPromptTemplate(message_templates=messages)

tools_llm = ToolOrchestratingLLM(
    output_cls=CodeReview,
    prompt=prompt,
    llm=llm,
)

result = tools_llm(language="Python", code="def foo(): pass")
```

---

## Execution Modes

### 1. Synchronous Execution

Standard blocking execution:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Summary(BaseModel):
    main_points: list[str]
    conclusion: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Summary,
    prompt="Summarize: {text}",
    llm=llm,
)

# Synchronous call using __call__
result = tools_llm(text="Long article text here...")
print(result.main_points)
print(result.conclusion)
```

### 2. Asynchronous Execution

Non-blocking async execution:

```python
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Classification(BaseModel):
    category: str
    subcategory: str
    confidence: float

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Classification,
    prompt="Classify: {item}",
    llm=llm,
)

async def classify_item(item: str) -> Classification:
    # Asynchronous call using acall
    result = await tools_llm.acall(item=item)
    return result

# Run async function
result = asyncio.run(classify_item("Laptop computer"))
print(f"{result.category} > {result.subcategory}")
```

### 3. Batch Processing with Async

Process multiple inputs concurrently:

```python
import asyncio
from typing import List
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class EntityExtraction(BaseModel):
    entities: List[str]
    entity_types: List[str]

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=EntityExtraction,
    prompt="Extract entities from: {text}",
    llm=llm,
)

async def extract_batch(texts: List[str]) -> List[EntityExtraction]:
    tasks = [tools_llm.acall(text=text) for text in texts]
    results = await asyncio.gather(*tasks)
    return results

texts = [
    "Apple Inc. is in California",
    "Microsoft was founded by Bill Gates",
    "Paris is the capital of France"
]
results = asyncio.run(extract_batch(texts))
for text, result in zip(texts, results):
    print(f"{text}: {result.entities}")
```

### 4. Streaming Execution

Stream progressive updates:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Article(BaseModel):
    title: str
    sections: list[str]
    word_count: int

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Article,
    prompt="Write an article about {topic}",
    llm=llm,
)

# Stream results as they arrive
for partial_article in tools_llm.stream_call(topic="AI"):
    print(f"Current title: {partial_article.title}")
    print(f"Sections so far: {len(partial_article.sections)}")
    # Display progressive updates in UI
```

### 5. Async Streaming

Async version of streaming:

```python
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Report(BaseModel):
    title: str
    findings: list[str]
    recommendations: list[str]

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Report,
    prompt="Generate report on {subject}",
    llm=llm,
)

async def stream_report(subject: str):
    stream = await tools_llm.astream_call(subject=subject)
    async for partial_report in stream:
        print(f"Findings: {len(partial_report.findings)}")
        print(f"Recommendations: {len(partial_report.recommendations)}")

asyncio.run(stream_report("Market analysis"))
```

### 6. Passing LLM-specific Parameters

Forward parameters directly to the LLM:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Story(BaseModel):
    title: str
    plot: str
    characters: list[str]

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Story,
    prompt="Write a {genre} story",
    llm=llm,
)

# Pass LLM-specific kwargs
result = tools_llm(
    llm_kwargs={
        "temperature": 0.9,  # Higher for creativity
        "top_p": 0.95,
        "max_tokens": 1000,
    },
    genre="science fiction"
)
```

---

## Parallel Tool Calls

### Single Output (Default)

By default, only one tool call is expected:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Product(BaseModel):
    name: str
    price: float
    description: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Product,
    prompt="Extract product info: {text}",
    llm=llm,
    allow_parallel_tool_calls=False,  # Default
)

# Returns single Product instance
result = tools_llm(text="iPhone 15 costs $999")
print(type(result))  # <class 'Product'>
```

### Multiple Outputs (Parallel)

Enable parallel tool calls to receive multiple objects:

```python
from typing import List
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Item(BaseModel):
    name: str
    category: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Item,
    prompt="Extract all items from this list: {text}",
    llm=llm,
    allow_parallel_tool_calls=True,  # Enable parallel calls
)

# Returns List[Item]
results = tools_llm(text="apples, laptops, books, phones")
print(type(results))  # <class 'list'>
print(len(results))   # 4
for item in results:
    print(f"{item.name}: {item.category}")
```

### Parallel with Streaming

Stream multiple objects as they're generated:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Question(BaseModel):
    question: str
    difficulty: str

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Question,
    prompt="Generate 5 questions about {topic}",
    llm=llm,
    allow_parallel_tool_calls=True,
)

# Stream list of questions as they arrive
for questions_so_far in tools_llm.stream_call(topic="Python"):
    if isinstance(questions_so_far, list):
        print(f"Questions generated: {len(questions_so_far)}")
        # Show latest question
        if questions_so_far:
            latest = questions_so_far[-1]
            print(f"  Latest: {latest.question}")
```

---

## Advanced Usage

### 1. Dynamic Prompt Updates

Change the prompt at runtime:

```python
from pydantic import BaseModel
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Response(BaseModel):
    answer: str
    reasoning: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Response,
    prompt="Answer briefly: {question}",
    llm=llm,
)

# Use with initial prompt
result1 = tools_llm(question="What is AI?")

# Update prompt dynamically
tools_llm.prompt = PromptTemplate("Answer in detail: {question}")

# Use with new prompt
result2 = tools_llm(question="What is AI?")
```

### 2. Reusable Instance Pattern

Create once, use many times:

```python
import os
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Sentiment(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    keywords: list[str]

llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"), request_timeout=80)

# Create reusable instance
sentiment_analyzer = ToolOrchestratingLLM(
    output_cls=Sentiment,
    prompt="Analyze sentiment of: {text}",
    llm=llm,
)

# Reuse multiple times
reviews = [
    "This product is amazing!",
    "Terrible experience, very disappointed",
    "It's okay, nothing special",
]

for review in reviews:
    analysis = sentiment_analyzer(text=review)
    print(f"{review[:20]}... → {analysis.sentiment} ({analysis.confidence})")
```

### 3. Complex Nested Models

Use deeply nested Pydantic models:

```python
from typing import List
from pydantic import BaseModel, Field
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Author(BaseModel):
    name: str
    email: str

class Comment(BaseModel):
    author: Author
    text: str
    upvotes: int

class Article(BaseModel):
    title: str
    content: str
    author: Author
    tags: List[str]
    comments: List[Comment]

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Article,
    prompt="Create a blog article about {topic} with comments",
    llm=llm,
)

result = tools_llm(topic="Machine Learning")
print(result.title)
print(result.author.name)
print(f"Tags: {', '.join(result.tags)}")
print(f"Comments: {len(result.comments)}")
for comment in result.comments:
    print(f"  - {comment.author.name}: {comment.text[:50]}...")
```

### 4. Using with Verbose Mode

Enable detailed logging:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Data(BaseModel):
    result: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Data,
    prompt="Process: {input}",
    llm=llm,
    verbose=True,  # Enable verbose logging
)

# Will log detailed information about tool calls
result = tools_llm(input="test data")
```

### 5. Custom Tool Choice

Control tool selection:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class Output(BaseModel):
    data: str

llm = Ollama(model="llama3.1", request_timeout=80)

# Force the LLM to always use the tool
tools_llm = ToolOrchestratingLLM(
    output_cls=Output,
    prompt="Generate output for: {input}",
    llm=llm,
    tool_choice="required",  # Force tool use
)

result = tools_llm(input="test")
```

---

## Using Regular Functions with ToolOrchestratingLLM

`ToolOrchestratingLLM` now supports both Pydantic models and regular Python functions as `output_cls`. When you pass a function, the system automatically detects it and creates the appropriate tool.

### 1. Using Regular Functions

Pass regular Python functions directly as `output_cls`:

```python
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

def calculate_statistics(numbers: list[float], operation: str) -> dict[str, float]:
    """Calculate statistics on a list of numbers.

    Args:
        numbers: List of numbers to analyze
        operation: Type of operation (mean, sum, max, min)

    Returns:
        Dictionary with the result
    """
    if operation == "mean":
        result = sum(numbers) / len(numbers)
    elif operation == "sum":
        result = sum(numbers)
    elif operation == "max":
        result = max(numbers)
    elif operation == "min":
        result = min(numbers)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return {
        "operation": operation,
        "result": result,
        "count": len(numbers)
    }

# Use function directly with ToolOrchestratingLLM
llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=calculate_statistics,  # Pass function directly!
    prompt="Calculate the mean of these numbers: {text}",
    llm=llm,
)

# Call with input
result = tools_llm(text="10, 20, 30, 40, 50")

print(f"Operation: {result['operation']}")
print(f"Result: {result['result']}")
print(f"Count: {result['count']}")
```

### 2. Using Functions with Regular Classes

Wrap regular Python classes in functions and use with ToolOrchestratingLLM:

```python
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.ollama import Ollama

class EmailValidator:
    """Regular Python class for email validation."""

    def __init__(self, email: str, check_mx: bool = False):
        self.email = email
        self.check_mx = check_mx
        self.is_valid = self._validate()

    def _validate(self) -> bool:
        """Simple email validation."""
        return "@" in self.email and "." in self.email.split("@")[1]

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "is_valid": self.is_valid,
            "check_mx": self.check_mx
        }

def validate_email(email: str, check_mx: bool = False) -> dict:
    """Validate an email address.

    Args:
        email: Email address to validate
        check_mx: Whether to check MX records (not implemented)

    Returns:
        Validation result dictionary
    """
    validator = EmailValidator(email, check_mx)
    return validator.to_dict()

# Use function with ToolOrchestratingLLM
llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=validate_email,  # Pass function that uses the class
    prompt="Validate this email: {email_text}",
    llm=llm,
)

result = tools_llm(email_text="user@example.com")

print(f"Email: {result['email']}")
print(f"Valid: {result['is_valid']}")
```

### 3. Factory Functions with Dataclasses

Use factory functions that return dataclass instances:

```python
from dataclasses import dataclass
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

@dataclass
class Product:
    """Regular dataclass (not Pydantic)."""
    name: str
    price: float
    category: str
    in_stock: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "price": self.price,
            "category": self.category,
            "in_stock": self.in_stock
        }

def create_product(name: str, price: float, category: str, in_stock: bool = True) -> dict:
    """Create a product from parameters.

    Args:
        name: Product name
        price: Product price in USD
        category: Product category
        in_stock: Whether product is in stock

    Returns:
        Product data as dictionary
    """
    product = Product(name, price, category, in_stock)
    return product.to_dict()

# Use factory function with ToolOrchestratingLLM
llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=create_product,  # Pass factory function
    prompt="Create a product entry for: {product_info}",
    llm=llm,
)

result = tools_llm(product_info="Laptop, $999, Electronics, available")

print(f"Product: {result['name']}")
print(f"Price: ${result['price']}")
print(f"Category: {result['category']}")
```

### 4. Async Functions

Use async functions directly with ToolOrchestratingLLM:

```python
import asyncio
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

async def fetch_user_data(user_id: int, include_posts: bool = False) -> dict:
    """Asynchronously fetch user data.

    Args:
        user_id: User ID to fetch
        include_posts: Whether to include user posts

    Returns:
        User data dictionary
    """
    # Simulate async API call
    await asyncio.sleep(0.1)

    user_data = {
        "user_id": user_id,
        "username": f"user_{user_id}",
        "email": f"user{user_id}@example.com"
    }

    if include_posts:
        user_data["posts"] = [
            {"id": 1, "title": "First post"},
            {"id": 2, "title": "Second post"}
        ]

    return user_data

# Use async function with ToolOrchestratingLLM
async def main():
    llm = Ollama(model="llama3.1", request_timeout=80)

    tools_llm = ToolOrchestratingLLM(
        output_cls=fetch_user_data,  # Pass async function
        prompt="Fetch data for user ID {user_id_text} with their posts",
        llm=llm,
    )

    # Use acall for async execution
    result = await tools_llm.acall(user_id_text="42")

    print(f"User: {result['username']}")
    print(f"Email: {result['email']}")
    if "posts" in result:
        print(f"Posts: {len(result['posts'])}")

# Run async example
asyncio.run(main())
```

### 5. Lambda Functions

Use lambda functions for simple transformations:

```python
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

# Simple lambda function for temperature conversion
convert_temp = lambda celsius: {
    "celsius": celsius,
    "fahrenheit": (celsius * 9/5) + 32,
    "kelvin": celsius + 273.15
}

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=convert_temp,  # Pass lambda function
    prompt="Convert {temperature} degrees Celsius",
    llm=llm,
)

result = tools_llm(temperature="25")

print(f"Celsius: {result['celsius']}")
print(f"Fahrenheit: {result['fahrenheit']}")
print(f"Kelvin: {result['kelvin']}")
```

**Note**: Lambda functions work but have limitations:
- No docstring for the LLM to understand the function
- Parameters aren't well-documented
- Better to use regular functions with proper documentation for complex cases

### 6. Functions with Complex Return Types

Use functions that return complex data structures:

```python
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

def analyze_text(text: str, language: str = "en") -> dict:
    """Analyze text and return basic metrics.

    Args:
        text: Text to analyze
        language: Language code

    Returns:
        Analysis metrics including word count, character count, and average word length
    """
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "language": language,
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=analyze_text,  # Pass function with complex return
    prompt="Analyze this text: {text_input}",
    llm=llm,
)

result = tools_llm(text_input="Hello world, this is a test message.")

print(f"Word count: {result['word_count']}")
print(f"Character count: {result['char_count']}")
print(f"Average word length: {result['avg_word_length']:.2f}")
```

### Important Notes

**Advantages of using functions as output_cls:**
1. Works with existing Python functions - no need to convert to Pydantic
2. Full access to all `ToolOrchestratingLLM` features (streaming, async, etc.)
3. Automatic tool creation and orchestration
4. Simple and direct - just pass your function

**When to use functions vs Pydantic models:**
- **Use functions** when:
  - You have existing functions you want to reuse
  - You need simple dict/list returns
  - You're prototyping quickly
  - Working with legacy code

- **Use Pydantic models** when:
  - You need strict validation of outputs
  - You want better type safety and IDE support
  - Building production systems with clear schemas
  - Need automatic documentation from models

**Both approaches work equally well** with `ToolOrchestratingLLM` - choose based on your needs!

---

## Error Handling

### 1. Handling LLM Validation Errors

Catch initialization errors:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class Data(BaseModel):
    value: str

# Create LLM that doesn't support function calling (hypothetically)
# llm = SomeLLM(...)  # without function calling support

try:
    tools_llm = ToolOrchestratingLLM(
        output_cls=Data,
        prompt="Process: {input}",
        llm=None,  # No LLM provided
    )
except AssertionError as e:
    print("LLM must be provided or set in Configs")
except ValueError as e:
    print(f"LLM doesn't support function calling: {e}")
```

### 2. Handling Tool Execution Errors

Handle runtime errors:

```python
from pydantic import BaseModel, ValidationError
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class StrictData(BaseModel):
    number: int  # Must be integer
    ratio: float  # Must be float

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=StrictData,
    prompt="Extract numbers from: {text}",
    llm=llm,
)

try:
    result = tools_llm(text="Some text with invalid data")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # LLM generated invalid tool arguments
except ValueError as e:
    print(f"Error: {e}")
    # Other errors (network, timeout, etc.)
```

### 3. Retry Logic

Implement retry logic for robustness:

```python
import asyncio
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class Result(BaseModel):
    data: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Result,
    prompt="Process: {input}",
    llm=llm,
)

async def call_with_retry(
    tools_llm: ToolOrchestratingLLM,
    max_retries: int = 3,
    **kwargs
) -> Result:
    """Call ToolOrchestratingLLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return await tools_llm.acall(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Use with retry
result = asyncio.run(call_with_retry(tools_llm, input="test"))
```

### 4. Handling Missing Tool Calls

Handle cases where LLM doesn't generate tool calls:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class Output(BaseModel):
    result: str

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=Output,
    prompt="Generate output",
    llm=llm,
    tool_choice="required",  # Force tool use to prevent this
)

try:
    result = tools_llm()
except ValueError as e:
    print(f"No tool calls generated: {e}")
    # Try with different prompt or parameters
```

---

## Best Practices

### 1. Clear Model Definitions

Always define clear Pydantic models with descriptions:

```python
from pydantic import BaseModel, Field
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class WellDefinedModel(BaseModel):
    """A well-documented model for structured output."""

    name: str = Field(description="The name of the entity")
    category: str = Field(description="Category classification")
    confidence: float = Field(
        description="Confidence score",
        ge=0.0,
        le=1.0
    )
    tags: list[str] = Field(
        description="Relevant tags",
        default_factory=list
    )

llm = Ollama(model="llama3.1", request_timeout=80)

tools_llm = ToolOrchestratingLLM(
    output_cls=WellDefinedModel,
    prompt="Extract information from: {text}",
    llm=llm,
)
```

### 2. Use Function Calling Compatible Models

Ensure your LLM supports function calling:

```python
from serapeum.llms.ollama import Ollama

# Good: Models that support function calling
good_models = [
    "llama3.1",
    "llama3.2",
    "mistral",
    # Check Ollama docs for function calling support
]

llm = Ollama(model="llama3.1", request_timeout=80)
# llm.metadata.is_function_calling_model should be True
```

### 3. Instance Reuse

Create instances once and reuse them:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class Classification(BaseModel):
    category: str
    confidence: float

# Create once
llm = Ollama(model="llama3.1", request_timeout=80)
classifier = ToolOrchestratingLLM(
    output_cls=Classification,
    prompt="Classify: {text}",
    llm=llm,
)

# Reuse many times - this is efficient!
texts = ["text1", "text2", "text3"]
for text in texts:
    result = classifier(text=text)
```

### 4. Use Parallel Calls for Lists

When extracting multiple items, use parallel tool calls:

```python
from pydantic import BaseModel
from serapeum.core.llms import ToolOrchestratingLLM
from serapeum.llms.ollama import Ollama

class Item(BaseModel):
    name: str
    type: str

llm = Ollama(model="llama3.1", request_timeout=80)

# Good: Enable parallel for extracting multiple items
tools_llm = ToolOrchestratingLLM(
    output_cls=Item,
    prompt="Extract ALL items from: {text}",
    llm=llm,
    allow_parallel_tool_calls=True,  # Enable for lists
)

# Returns List[Item] with all extracted items
results = tools_llm(text="apples, oranges, bananas")
```

---

## See Also

- [General Overview](./general.md) - Complete workflow explanation
- [Execution Flow and Method Calls](./tool_orchestrating_llm_sequence.md) - Detailed sequence diagram
- [Architecture and Class Relationships](./tool_orchestrating_llm_class.md) - Class structure
- [Data Transformations and Validation](./tool_orchestrating_llm_dataflow.md) - Data flow details
- [Component Boundaries and Interactions](./tool_orchestrating_llm_components.md) - System components
- [Lifecycle States and Transitions](./tool_orchestrating_llm_state.md) - State management
