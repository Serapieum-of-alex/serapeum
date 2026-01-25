# Installation

This page explains how to install serapeum and its dependencies using uv (recommended) or plain pip. All information here is aligned with the project’s `pyproject.toml`.

Package name: serapeum
Current version: 0.1.0
Supported Python versions: 3.11–3.12 (requires Python >=3.11,<4.0)

## Dependencies

### Core runtime (from `[project.dependencies]`)
- ollama >= 0.5.4
- numpy
- filetype >= 1.2.0
- requests >= 2.32.5

### Development and documentation groups (from `[dependency-groups]`)
These are not pip extras. They are declared as uv/pip groups in `pyproject.toml`:
- dev: pytest, pytest-cov, pre-commit, pre-commit-hooks, pytest-asyncio, nest-asyncio, nbval
- docs: mkdocs, mkdocs-material, mkdocstrings, mkdocstrings-python, mike, mkdocs-jupyter, mkdocs-autorefs, mkdocs-macros-plugin, mkdocs-table-reader-plugin, mkdocs-mermaid2-plugin, jupyter, notebook<7, commitizen, mkdocs-panzoom-plugin

See also: `docs/uv-usage.md` for more uv tips.

## Recommended: install with uv
uv is a fast Python package manager. If you don’t have it yet, see `docs/uv-usage.md` for install options.

From the project root:

```bash
# 1) Create a virtual environment (once)
uv venv

# 2) Activate it
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 3) Install the package (editable/local)
uv pip install -e .

# 4) Optionally add development or docs tools
uv pip install --group dev
uv pip install --group docs

# 5) Run tests
uv run pytest -q
```

To install directly from GitHub instead of a local checkout:

```bash
uv pip install "git+https://github.com/Serapieum-of-alex/serapeum.git"
# or a specific tag (example: v0.1.0)
uv pip install "git+https://github.com/Serapieum-of-alex/serapeum.git@v0.1.0"
```

## Alternative: install with pip
If you prefer plain pip/venv:

```bash
# 1) Create and activate a venv
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

# 2) Install the package (editable/local)
pip install -e .

# 3) (Optional) Install dev/docs tools
# Note: dev/docs are defined as groups, not pip extras. With pip,
# install the needed packages manually according to pyproject.toml.
# Example (partial):
pip install pytest pytest-cov pre-commit pytest-asyncio nbval
```

Install from GitHub with pip:

```bash
pip install "git+https://github.com/Serapieum-of-alex/serapeum.git"
# or a specific tag
pip install "git+https://github.com/Serapieum-of-alex/serapeum.git@v0.1.0"
```

## Install from source (clone and install locally)
```bash
git clone https://github.com/Serapieum-of-alex/serapeum.git
cd serapeum
# using uv (recommended for speed)
uv pip install -e .
# or using pip
pip install -e .
```

## Quick check
After installation, open Python and run:

```python
import serapeum
print(serapeum.__version__)
```

## Notes
- Tested Python versions: 3.11–3.12.
- Project homepage: https://github.com/Serapieum-of-alex/serapeum
- Documentation: https://serapeum.readthedocs.io/
