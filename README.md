# omanage - Ollama Model Manager

A CLI tool and Python API to manage Ollama model storage by moving blob files between filesystems.

## Overview

`omanage` helps you manage Ollama model storage by:
- Tracking model blobs and their locations
- Freezing models (moving blobs to remote storage)
- Thawing models (moving blobs back to base storage)
- Compressing models when frozen for space savings

## Features

- **Config Management**: Set and view configuration for base and remote storage paths
- **Model Index**: Track all installed models with their blob information
- **Freeze/Thaw**: Move model blobs between storage locations
- **Compression**: Optional gzip compression when freezing models
- **Verification**: Verify model files exist in expected locations
- **Python API**: Use as a Python module in your own applications

## Installation

```bash
pip install -e .
```

Or run directly:
```bash
python -m omanage [command]
```

## Usage as a CLI Tool

### Commands

| Command | Description |
|-----|----|
| `config` | Show or set configuration options |
| `help` | Show help message |
| `list` | List all models with their status |
| `init` | Initialize model index from Ollama |
| `refresh` | Refresh model index from Ollama |
| `freeze <model>` | Move a model's blob to remote storage |
| `thaw <model>` | Move a model's blob back to base storage |
| `export <model>` | Copy a model's blob to remote storage without deleting the source |
| `import <model>` | Copy a model's blob from remote storage without deleting the source |
| `verify` | Verify model file locations match index |

### Usage Examples

#### Initialize model index
```bash
omanage init
```

#### List all models
```bash
omanage list
```

#### Freeze a model
```bash
omanage freeze llama3:8b
```

#### Thaw a model
```bash
omanage thaw llama3:8b
```

#### Export a model (copy to remote storage, keep local source)
```bash
omanage export llama3:8b
```

#### Import a model (copy from remote storage, keep remote source)
```bash
omanage import llama3:8b
```

#### Set configuration
```bash
omanage config --set baseStorage=/mnt/storage/ollama
```

## Usage as a Python API

### Basic Usage

```python
from pathlib import Path
from omanage import OmanageAPI, OllamaNotInstalledError, StorageNotConfiguredError

# Initialize the API
api = OmanageAPI(Path.cwd())

try:
    # Initialize model index
    models = api.initialize()
    print(f"Initialized {len(models)} models")
    
    # List all models
    models = api.list_models()
    for name, meta in models.items():
        print(f"{name}: frozen={meta['frozen']}, compressed={meta['compressed']}")
    
    # Freeze a model with compression
    result = api.freeze_model("llama3:8b", compress=True)
    print(f"Froze model: {result}")
    
    # Thaw a model
    result = api.thaw_model("llama3:8b")
    print(f"Thawed model: {result}")
    
    # Verify files
    verification = api.verify()
    print(f"Status: {verification['status']}")
    
except OllamaNotInstalledError as e:
    print(f"Error: {e}")
except StorageNotConfiguredError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

### Exception Types

| Exception | Description |
|-----------|-------------|
| `OmanageAPIError` | Base exception for all API errors |
| `OllamaNotInstalledError` | Ollama CLI is not installed or not in PATH |
| `ModelNotFoundError` | Model not found in index |
| `StorageNotConfiguredError` | Storage paths not configured |
| `FileOperationError` | File operation failed |

### API Reference

#### `OmanageAPI(project_dir: Path = None)`

Create a new API instance.

**Parameters:**
- `project_dir` - Path to the project directory containing `.omanage.conf`. If None, uses current working directory.

#### `initialize(model_name: Optional[str] = None) -> List[Dict[str, Any]]`

Initialize the model index from Ollama.

**Parameters:**
- `model_name` - If specified, only initialize this model. Otherwise, initialize all models.

**Returns:** List of dictionaries with 'name' key for each initialized model.

#### `list_models() -> Dict[str, Dict[str, Any]]`

Get all models in the index.

**Returns:** Dictionary mapping model names to their metadata.

#### `get_model(model_name: str) -> Optional[Dict[str, Any]]`

Get metadata for a specific model.

**Parameters:**
- `model_name` - Name of the model to retrieve.

**Returns:** Model metadata dictionary, or None if not found.

#### `freeze_model(model_name: str, compress: bool = False) -> Dict[str, Any]`

Freeze a model by moving its blob to remote storage.

**Parameters:**
- `model_name` - Name of the model to freeze.
- `compress` - If True, compress the blob during the move.

**Returns:** Dictionary with 'success', 'model', 'blob_sha', 'compressed' keys.

#### `thaw_model(model_name: str) -> Dict[str, Any]`

Thaw a model by moving its blob back to base storage.

**Parameters:**
- `model_name` - Name of the model to thaw.

**Returns:** Dictionary with 'success', 'model', 'blob_sha', 'decompressed' keys.

#### `verify() -> Dict[str, Any]`

Verify model files exist in expected locations.

**Returns:** Dictionary with 'status', 'total_models', 'missing', 'mismatched' keys.

#### `get_installed_models() -> List[Dict[str, str]]`

Get list of models installed in Ollama.

**Returns:** List of dictionaries with 'name' key for each installed model.

## Configuration

Configuration is stored in `.omanage.conf` in the project directory:

```json
{
  "ollamaBinary": "ollama",
  "baseStorage": "/path/to/ollama/storage",
  "remoteStorage": "/path/to/remote/storage"
}
```

## Model Index

Model metadata is stored in `.omanage.index.json`:

```json
{
  "models": {
    "llama3:8b": {
      "blobSha": "sha256-abcdef123456...",
      "blobName": "blobfile",
      "frozen": false,
      "compressed": false
    }
  }
}
```

## License

MIT

## Note

This application was coded with assistance from Cline, kimi-k2.5, and Qwen3-coder-next:q8_0.

This application has no affiliation whatsoever with Ollama.  The name "Ollama" is owned by its owner.  It is only used here as a descriptor for what it does, and no other affiliation or use of the mark is claimed.

The Ollama application must be installed for this application to work (or have meaning).  See https://ollama.com.

If this application helps you, please let me know.