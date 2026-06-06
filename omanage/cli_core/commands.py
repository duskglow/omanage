"""Command handler functions for omanage CLI."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ..config import ConfigManager, ConfigError
from ..index import IndexManager
from ..utils import (
    ProgressBar,
    compress_file,
    decompress_file,
    detect_compression,
    validate_model_name,
    InvalidModelNameError,
    ValidationError,
)
from ..api import OmanageAPI, OmanageAPIError, ModelNotFoundError, StorageNotConfiguredError, FileOperationError


class CliError(Exception):
    """CLI-related errors."""
    pass


def get_ollama_models() -> list:
    """
    Get list of installed Ollama models.
    
    Returns:
        List of model dictionaries with 'name' key
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        
        models = []
        for line in lines[1:]:  # Skip header line
            if line.strip():
                parts = line.split()
                if parts:
                    models.append({"name": parts[0]})
        
        return models
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        raise CliError("Ollama not found. Please install Ollama first.")


def get_model_blob_info(model_name: str) -> Optional[dict]:
    """
    Get blob information for a model from its modelfile.
    
    Args:
        model_name: Name of the model to query
        
    Returns:
        Dictionary with 'blobSha' and 'blobName' keys, or None if not found
    """
    try:
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith("FROM "):
                from_path = line[5:].strip()
                blob_name = Path(from_path).name
                return {
                    "blobSha": blob_name,
                    "blobName": blob_name
                }
        
        return None
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        raise CliError("Ollama not found. Please install Ollama first.")


def get_blob_path(model_meta: dict, frozen: bool, config: ConfigManager) -> Path:
    """
    Get the expected path for a model's blob file.
    
    Args:
        model_meta: Model metadata from index
        frozen: Whether the model is frozen
        config: ConfigManager instance
        
    Returns:
        Path to the blob file
    """
    config.load()
    
    if frozen:
        remote_storage = config.get('remoteStorage')
        if not remote_storage:
            raise CliError("remoteStorage not configured")
        return Path(remote_storage) / model_meta['blobName']
    else:
        base_storage = config.get('baseStorage')
        if not base_storage:
            raise CliError("baseStorage not configured")
        return Path(base_storage) / model_meta['blobName']


# Command functions

def cmd_config(args: argparse.Namespace) -> int:
    """Handle the config command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    
    # Load existing config or create default
    config.load()
    
    # Handle --set option
    if args.set:
        key, value = args.set.split('=', 1)
        try:
            config.set(key, value)
            config.save()
            print(f"Set config['{key}'] = {value}")
        except (ConfigError, ValidationError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Handle --get option
    if args.get:
        value = config.get(args.get)
        if value is not None:
            print(f"{args.get} = {value}")
        else:
            print(f"Config key '{args.get}' not found")
            return 1
    
    # Show all config if no --set or --get
    if not args.set and not args.get:
        print("Current configuration:")
        for key, value in config.config.items():
            print(f"  {key} = {value}")
    
    return 0


def cmd_help(args: argparse.Namespace) -> int:
    """Handle the help command."""
    print("""
 Ollama Model Manager - omanage CLI tool

Usage: omanage <command> [options]

Commands:
  config      Show or set configuration options
  help        Show this help message
  list        List all models with their status
  init        Initialize model index from Ollama
  refresh     Refresh model index from Ollama
  freeze      Move a model's blob to remote storage
  thaw        Move a model's blob back to base storage
  verify      Verify model file locations match index

Use 'omanage <command> --help' for more information about a command.
""")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Handle the list command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return 1
    
    models = index.list_models()
    
    if not models:
        print("No models in index. Run 'omanage init' to populate.")
        return 0
    
    # Print header
    print(f"{'Model Name':<30} {'Blob SHA':<64} {'Status':<10} {'Compressed'}")
    print("-" * 114)
    
    # Print each model
    for model_name, metadata in models.items():
        blob_sha = metadata.get('blobSha', 'N/A')
        frozen = metadata.get('frozen', False)
        compressed = metadata.get('compressed', False)
        status = "Frozen" if frozen else "Thawed"
        compressed_str = "Yes" if compressed else "No"
        
        print(f"{model_name:<30} {blob_sha:<64} {status:<10} {compressed_str}")
    
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Handle the init command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Initialize config if needed
    config.initialize()
    
    # Initialize index if needed
    index.initialize()
    
    # Get models from Ollama
    try:
        models = get_ollama_models()
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not models:
        print("No models found in Ollama. Run 'ollama list' to see installed models.")
        return 0
    
    # Filter by model name if specified
    target_model = getattr(args, 'model_name', None)
    if target_model:
        validate_model_name(target_model)
        models = [m for m in models if m['name'] == target_model]
        if not models:
            print(f"Model '{target_model}' not found in Ollama.")
            return 1
    
    print(f"Processing {len(models)} model(s)...")
    
    for model in models:
        model_name = model['name']
        print(f"  Processing {model_name}...")
        
        # Get blob info
        blob_info = get_model_blob_info(model_name)
        if blob_info:
            index.set_model(
                model_name=model_name,
                blob_sha=blob_info['blobSha'],
                blob_name=blob_info['blobName'],
                frozen=False,
                compressed=False
            )
            print(f"    Added blob: {blob_info['blobSha']}")
        else:
            print(f"    Warning: Could not extract blob info for {model_name}", file=sys.stderr)
    
    # Save index
    index.save()
    print(f"\nInitialized {len(models)} model(s) in index.")
    
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Handle the refresh command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return 1
    
    # Get models from Ollama
    try:
        models = get_ollama_models()
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not models:
        print("No models found in Ollama.")
        return 0
    
    # Filter by model name if specified
    target_model = getattr(args, 'model_name', None)
    if target_model:
        validate_model_name(target_model)
        models = [m for m in models if m['name'] == target_model]
        if not models:
            print(f"Model '{target_model}' not found in Ollama.")
            return 1
    
    print(f"Refreshing {len(models)} model(s)...")
    
    for model in models:
        model_name = model['name']
        print(f"  Refreshing {model_name}...")
        
        # Get current metadata if exists
        current_meta = index.get_model(model_name)
        
        # Get blob info
        blob_info = get_model_blob_info(model_name)
        if blob_info:
            # Preserve frozen/compressed status if model already in index
            frozen = current_meta.get('frozen', False) if current_meta else False
            compressed = current_meta.get('compressed', False) if current_meta else False
            
            index.set_model(
                model_name=model_name,
                blob_sha=blob_info['blobSha'],
                blob_name=blob_info['blobName'],
                frozen=frozen,
                compressed=compressed
            )
            print(f"    Updated blob: {blob_info['blobSha']}")
        else:
            print(f"    Warning: Could not extract blob info for {model_name}", file=sys.stderr)
    
    # Save index
    index.save()
    print(f"\nRefreshed {len(models)} model(s).")
    
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    """Handle the freeze command."""
    config_dir = Path.cwd()
    model_name = args.model_name
    compress = args.compress
    
    # Validate model name
    try:
        validate_model_name(model_name)
    except InvalidModelNameError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Use API for freezing
    api = OmanageAPI(config_dir)
    
    try:
        result = api.freeze_model(model_name, compress)
        
        if result['success']:
            print(f"\nModel '{model_name}' frozen successfully.")
            return 0
        else:
            print(f"Model '{model_name}' freeze failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_thaw(args: argparse.Namespace) -> int:
    """Handle the thaw command."""
    config_dir = Path.cwd()
    model_name = args.model_name
    
    # Validate model name
    try:
        validate_model_name(model_name)
    except InvalidModelNameError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Use API for thawing
    api = OmanageAPI(config_dir)
    
    try:
        result = api.thaw_model(model_name)
        
        if result['success']:
            print(f"\nModel '{model_name}' thawed successfully.")
            return 0
        else:
            print(f"Model '{model_name}' thaw failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the verify command."""
    config_dir = Path.cwd()
    api = OmanageAPI(config_dir)
    
    try:
        result = api.verify()
        
        if result['status'] == 'ok':
            print("All files verified successfully.")
            return 0
        elif result['status'] == 'error':
            print(f"Verification failed: {result.get('error', 'Unknown error')}")
            return 1
        else:
            print(f"Verification complete with issues.")
            return 1
            
    except Exception as e:
        print(f"Error during verification: {e}", file=sys.stderr)
        return 1