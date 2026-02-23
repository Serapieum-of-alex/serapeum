# Testing Documentation Code Examples

Serapeum uses [`pytest-markdown-docs`](https://github.com/modal-labs/pytest-markdown-docs) to run Python code fences in markdown files as pytest tests. This ensures all documentation examples stay correct and up to date.

## Setup

The plugin is included in the dev dependencies. Install it with:

```bash
uv sync --dev
```

## Running Doc Tests

The `docs/` directory is excluded from the default pytest paths, so you must pass the file explicitly:

```bash
# Run all code blocks in a single file
uv run pytest --markdown-docs docs/reference/providers/ollama/examples.md

# Run all code blocks across the entire docs directory
uv run pytest --markdown-docs docs/

# Verbose output
uv run pytest -v --markdown-docs docs/reference/providers/ollama/examples.md
```

## Listing Code Blocks in a File

To see all testable code blocks and their identifiers:

```bash
uv run pytest --collect-only --markdown-docs docs/reference/providers/ollama/examples.md
```

This outputs each block with its fence number and line:

```
<MarkdownInlinePythonItem [CodeFence#1][line:52]>
<MarkdownInlinePythonItem [CodeFence#2][line:70]>
...
```

## Running a Specific Code Block

Use the node ID shown by `--collect-only`:

```bash
uv run pytest -v --markdown-docs "docs/reference/providers/ollama/examples.md::[CodeFence#1][line:52]"
```

The line number corresponds to the opening ` ``` ` of the code fence in the file, making it easy to cross-reference with your editor.

## Controlling Which Blocks Are Tested

Use info string modifiers on the opening code fence:

| Modifier | Effect |
|---|---|
| `notest` | Skip this block entirely |
| `continuation` | Share state with the previous block (imports, variables carry over) |
| `fixture:<name>` | Inject a pytest fixture into the block's scope |
| `retry:N` | Retry the block up to N times on failure |

**Example — skip a block:**

````markdown
```python notest
# This illustrative snippet will not be executed
llm = Ollama(model="...", api_key="sk-...")
```
````

**Example — continuation across blocks:**

````markdown
```python
from serapeum.ollama import Ollama
llm = Ollama(model="qwen3.5:397b", api_key=os.environ.get("OLLAMA_API_KEY"))
```

```python continuation
# `llm` is still in scope from the block above
response = llm.complete("Hello")
```
````

## Environment Variables

Code blocks that call the Ollama cloud API require `OLLAMA_API_KEY` to be set. Load it from a `.env` file before running:

```bash
# The dotenv package is included in dev dependencies
uv run pytest --markdown-docs docs/reference/providers/ollama/examples.md
```

In your `.env` file at the repo root:

```
OLLAMA_API_KEY=your_api_key_here
```

Tests that require a live API call should be marked as `e2e` (see [Contributing](./contributing.md)) or use `notest` if they are purely illustrative.
