---
title: Taskfile Usage
---

# Taskfile usage

This project uses a Taskfile (`Taskfile.yml`) to standardize build and install
steps for the core package and libs. Tasks are run with the `task` CLI.

## Prerequisites

- Install Task (go-task): https://taskfile.dev/installation/
- Install `uv` (used by build and install tasks).
- A working Python environment you want to target.

## Quick start

List available tasks:

```bash
task -l
```

Build everything:

```bash
task build:all
```

Build and install all wheels into the default Python:

```bash
task install:all
```

## Task variables

The Taskfile defines a few variables you can override at runtime:

- `PYTHON`: Optional path to a Python executable, used for `uv pip` commands.
- `DIST_DIR`: Output directory for built wheels (default: `dist`).
- `CORE_LIB_DIR`: Path to the core library package.
- `PLUGIN_OLLAMA_DIR`: Path to the Ollama plugin package.

Override variables on the command line:

```bash
task install:all PYTHON=C:\path\to\python.exe
task build:all DIST_DIR=dist-artifacts
```

## Common workflows

Build only the core packages:

```bash
task build:core-lib
task build:core
```

Build and install just the Ollama plugin:

```bash
task build:plugin:ollama
task install:plugin:ollama
```

Uninstall all packages from the target environment:

```bash
task uninstall:all
```

Remove built wheels:

```bash
task clean:dist
```

## Task reference

| Task | Description |
| --- | --- |
| `build:core` | Build wheel for the root project (`serapeum`). |
| `build:core-lib` | Build wheel for `serapeum-core`. |
| `build:plugin:ollama` | Build wheel for the Ollama plugin. |
| `build:all` | Build wheels for core and libs. |
| `install:core` | Install `serapeum-core` wheel into the target env. |
| `install:plugin:ollama` | Install the Ollama plugin wheel into the target env. |
| `install:all` | Build and install all wheels into the target env. |
| `uninstall:all` | Uninstall `serapeum`, `serapeum-core`, and the plugin. |
| `clean:dist` | Remove the wheel output directory. |

## Notes

- Install tasks are non-editable installs from the built wheels in `DIST_DIR`.
- If you change `DIST_DIR`, make sure the install tasks point to the same
  directory you built into.
