# Installation and Usage Guide

This page explains how to install Serapeum and its dependencies, using the recommended [uv](https://github.com/astral-sh/uv) package manager, or plain pip. It also provides troubleshooting tips and common commands for development.

## Project Information
- **Package name:** serapeum
- **Current version:** 0.1.0
- **Supported Python versions:** 3.11–3.12 (requires Python >=3.11,<4.0)

### Dependencies
- **Core runtime:** ollama >= 0.5.4, numpy, filetype >= 1.2.0, requests >= 2.32.5
- **Development group:** pytest, pytest-cov, pre-commit, pre-commit-hooks, pytest-asyncio, nest-asyncio, nbval
- **Docs group:** mkdocs, mkdocs-material, mkdocstrings, mkdocstrings-python, mike, mkdocs-jupyter, mkdocs-autorefs, mkdocs-macros-plugin, mkdocs-table-reader-plugin, mkdocs-mermaid2-plugin, jupyter, notebook<7, commitizen, mkdocs-panzoom-plugin

## Installing uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip (alternative)
pip install uv
```

## Setting Up the Project (Recommended: uv)

1. **Create a virtual environment**
   ```bash
   uv venv
   ```
   This creates a `.venv` directory in your project root.

2. **Activate the virtual environment**
   ```bash
   # On macOS/Linux
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   ```

3. **Install the package (editable/local)**
   ```bash
   uv pip install -e .
   ```

4. **Optionally add development or docs tools**
   ```bash
   uv pip install --group dev
   uv pip install --group docs
   ```

5. **Run tests**
   ```bash
   uv run pytest -q
   ```

6. **Sync dependencies from `pyproject.toml`**
   ```bash
   uv sync --active
   ```

### Install from GitHub
```bash
uv pip install "git+https://github.com/Serapieum-of-alex/serapeum.git"
# or a specific tag (example: v0.1.0)
uv pip install "git+https://github.com/Serapieum-of-alex/serapeum.git@v0.1.0"
```

## Alternative: Using pip

1. **Create and activate a venv**
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

2. **Install the package (editable/local)**
   ```bash
   pip install -e .
   ```

3. **(Optional) Install dev/docs tools**
   - Note: dev/docs are defined as groups, not pip extras. With pip, install the needed packages manually according to `pyproject.toml`.
   - Example (partial):
     ```bash
     pip install pytest pytest-cov pre-commit pytest-asyncio nbval
     ```

4. **Install from GitHub with pip**
   ```bash
   pip install "git+https://github.com/Serapieum-of-alex/serapeum.git"
   # or a specific tag
   pip install "git+https://github.com/Serapieum-of-alex/serapeum.git@v0.1.0"
   ```

## Common uv Commands

- **Install a new package:**
  ```bash
  uv pip install <package-name>
  uv pip install <package-name> --group dev  # as dev dependency
  ```
- **Update a package:**
  ```bash
  uv pip install --upgrade <package-name>
  ```
- **Run Python scripts:**
  ```bash
  uv run python script.py
  uv run pytest
  ```

## Syncing Provider Dev Dependencies

Each workspace member (provider package/subpackage) has its own dev dependency group. The root-level `uv sync --dev --active` 
only installs root-level dev dependencies. 

To install a specific provider's dev dependencies (e.g. type stubs, test plugins), sync that package explicitly:

```bash
# Sync dev deps for a specific sub-package
uv sync --package serapeum-llama-cpp --dev --active
uv sync --package serapeum-ollama --dev --active
```

For example, `serapeum-llama-cpp` declares `types-requests` and `types-tqdm` in its dev group for mypy type checking. These are only installed when you sync that package:

```bash
# Install llama-cpp dev deps (includes type stubs for mypy)
uv sync --package serapeum-llama-cpp --dev --active

# Then run mypy
python -m mypy libs/providers/llama-cpp
```

## Troubleshooting

- **Virtual environment not activating:**
  Make sure you've created the virtual environment first:
  ```bash
  uv venv
  ```
- **Package installation fails:**
  Try clearing the cache:
  ```bash
  uv cache clean
  ```
- **ImportError after installation:**
  Ensure you've installed the package in editable mode:
  ```bash
  uv pip install -e .
  ```

## Quick Check
After installation, open Python and run:
```python
import serapeum
print(serapeum.__version__)
```

## Additional Resources
- [uv Documentation](https://github.com/astral-sh/uv)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- Project homepage: https://github.com/Serapieum-of-alex/serapeum
- Documentation: https://serapeum.readthedocs.io/
