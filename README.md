# Serapeum

Serapeum is a modular LLM toolkit. The repo contains a core package with shared
LLM abstractions plus provider integrations (Ollama, others) and supporting
docs, examples, and prompts.

## What is in this repo

- `serapeum-core/`: Provider-agnostic core (LLM interfaces, prompts, parsers,
  tools, structured outputs).
- `serapeum-integrations/`: Provider adapters (e.g. Ollama).
- `docs/`: Architecture and reference docs (MkDocs).
- `examples/`: Usage examples and notebooks.
- `prompts/`: Prompt templates and assets.
- `scripts/`: Utility scripts for development and maintenance.
- `skills/`: Codex skills used for repo automation.

## Packages

- `serapeum-core` (Python package)
  - Shared LLM models and interfaces.
  - Prompt templates and output parsers.
  - Tool schemas and execution utilities.
- `serapeum-ollama` (Python package under `libs/providers/ollama/`)
  - Ollama-backed LLM adapter.
  - Tool calling and structured output support when available.

Each package has its own README with details and examples.

## Quick start (editable installs)

From the repo root:

```bash
python -m pip install -e libs/core
python -m pip install -e libs/providers/serapeum-ollama
```

## Development setup

Install dev dependencies per package:

```bash
python -m pip install -e serapeum-core[dev]
python -m pip install -e libs/providers/serapeum-ollama[dev]
```

## Testing

Run tests per package:

```bash
cd serapeum-core
python -m pytest
```

```bash
cd libs/providers/serapeum-ollama
python -m pytest
```

Markers are defined in each package `pyproject.toml`.

## Links

- Homepage: https://github.com/Serapieum-of-alex/serapeum
- Docs: https://serapieum-of-alex.github.io/serapeum
- Changelog: https://github.com/Serapieum-of-alex/serapeum/HISTORY.rst
- Security: `SECURITY.md`
