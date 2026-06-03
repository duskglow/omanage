# omanage - Ollama Model Manager

A CLI tool to manage Ollama model storage by moving blob files between filesystems.

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

## Installation

```bash
pip install -e .
```

Or run directly:
```bash
python -m omanage [command]
```

## Commands

| Command | Description |
|---------|-------------|
| `config` | Show or set configuration options |
| `help` | Show help message |
| `list` | List all models with their status |
| `init` | Initialize model index from Ollama |
| `refresh` | Refresh model index from Ollama |
| `freeze <model>` | Move a model's blob to remote storage |
| `thaw <model>` | Move a model's blob back to base storage |
| `verify` | Verify model file locations match index |

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

## Usage Examples

### Initialize model index
```bash
omanage init
```

### List all models
```bash
omanage list
```

### Freeze a model
```bash
omanage freeze llama3:8b
```

### Thaw a model
```bash
omanage thaw llama3:8b
```

### Set configuration
```bash
omanage config --set baseStorage=/mnt/storage/ollama
```

## License

MIT