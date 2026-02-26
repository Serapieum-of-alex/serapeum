# Serapeum

[![Documentations](https://img.shields.io/badge/Documentations-blue?logo=github&logoColor=white)](https://serapieum-of-alex.github.io/serapeum/main/)
[![Python Versions](https://img.shields.io/pypi/pyversions/statista.svg)](https://pypi.org/project/serapeum/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://serapieum-of-alex.github.io/serapeum/latest/)
[![codecov](https://codecov.io/gh/Serapieum-of-alex/serapeum/branch/main/graph/badge.svg?token=GQKhcj2pFK)](https://codecov.io/gh/Serapieum-of-alex/serapeum)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub last commit](https://img.shields.io/github/last-commit/Serapieum-of-alex/serapeum)](https://github.com/Serapieum-of-alex/serapeum/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/Serapieum-of-alex/serapeum)](https://github.com/Serapieum-of-alex/serapeum/issues)

[//]: # ([![GitHub stars]&#40;https://img.shields.io/github/stars/Serapieum-of-alex/serapeum&#41;]&#40;https://github.com/Serapieum-of-alex/serapeum/stargazers&#41;)

[//]: # ([![GitHub forks]&#40;https://img.shields.io/github/forks/Serapieum-of-alex/serapeum&#41;]&#40;https://github.com/Serapieum-of-alex/serapeum/network/members&#41;)

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
python -m pip install -e libs/providers/ollama
```

## Development setup

Install dev dependencies per package:

```bash
python -m pip install -e serapeum-core[dev]
python -m pip install -e libs/providers/ollama[dev]
```

## Testing

Run tests per package:

```bash
cd serapeum-core
python -m pytest
```

```bash
cd libs/providers/ollama
python -m pytest
```

Markers are defined in each package `pyproject.toml`.

## Links

- Homepage: https://github.com/Serapieum-of-alex/serapeum
- Docs: https://serapieum-of-alex.github.io/serapeum
- Changelog: https://github.com/Serapieum-of-alex/serapeum/HISTORY.rst
- Security: `SECURITY.md`
