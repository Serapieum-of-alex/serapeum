# Using uv Package Manager

This project uses [uv](https://github.com/astral-sh/uv) as the package manager. uv is an extremely fast Python package installer and resolver written in Rust.

## Installation

### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

## Common Commands

### Create a Virtual Environment

```bash
uv venv
```

This creates a `.venv` directory in your project root.

### Activate the Virtual Environment

```bash
# On macOS and Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### Install Dependencies

```bash
# Install all project dependencies
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Sync Dependencies

to sync from `pyproject.toml`:

```bash
uv sync --active
```

### Install a New Package

```bash
# Add a new dependency
uv pip install <package-name>

# Add a dev dependency
uv pip install <package-name> --group dev
```

### Update Dependencies

```bash
uv pip install --upgrade <package-name>
```

### Run Python Scripts

```bash
# Run Python with uv
uv run python script.py

# Run pytest
uv run pytest
```

## Troubleshooting

### Virtual environment not activating

Make sure you've created the virtual environment first:
```bash
uv venv
```

### Package installation fails

Try clearing the cache:
```bash
uv cache clean
```

### ImportError after installation

Ensure you've installed the package in editable mode:
```bash
uv pip install -e .
```

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
