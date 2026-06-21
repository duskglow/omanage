"""Command handler functions for omanage CLI."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from ..config import ConfigManager, ConfigError
from ..index import IndexManager
from ..utils import (
    validate_model_name,
    InvalidModelNameError,
    ValidationError,
)
from ..api import OmanageAPI, OmanageAPIError, ModelNotFoundError, StorageNotConfiguredError, FileOperationError, ModelAlreadyFrozenError, ModelAlreadyThawedError
from ..api_core.subprocess_utils import (
    get_ollama_models as api_get_ollama_models,
    get_model_blob_info as api_get_model_blob_info,
    SubprocessError,
)


class CliError(Exception):
    """CLI-related errors."""
    pass


def get_ollama_models() -> list:
    """
    Get list of installed Ollama models using secure subprocess wrapper.
    
    Returns:
        List of model dictionaries with 'name' key
    """
    return api_get_ollama_models()


def get_model_blob_info(model_name: str) -> Optional[dict]:
    """
    Get blob information for a model from its modelfile using secure subprocess wrapper.
    
    Args:
        model_name: Name of the model to query
        
    Returns:
        Dictionary with 'blobSha' and 'blobName' keys, or None if not found
    """
    return api_get_model_blob_info(model_name)


# Command functions

def cmd_config(args: argparse.Namespace) -> int:
    """Handle the config command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    
    # Load existing config or create default
    config.load()
    
    # Handle --set option
    if args.set:
        if '=' not in args.set:
            print("Error: Config value must be in KEY=VALUE format", file=sys.stderr)
            return 1
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
  export      Copy a model's blob to remote storage without deleting the source
  import      Copy a model's blob from remote storage without deleting the source
  verify      Verify model file locations match index

Use 'omanage <command> --help' for more information about a command.
""")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Handle the list command."""
    config_dir = Path.cwd()
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except (OSError, ValueError) as e:
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
    target_model = getattr(args, 'model_name', None)
    
    api = OmanageAPI(config_dir)
    
    try:
        initialized = api.initialize(model_name=target_model)
    except (CliError, SubprocessError, OmanageAPIError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    if not initialized:
        print("No models found in Ollama or remote storage.")
        return 0
    
    print(f"Processing {len(initialized)} model(s)...")
    for model in initialized:
        status = "frozen" if model.get('frozen', False) else "thawed"
        print(f"  {model['name']}: {model['blobSha']} ({status})")
    print(f"\nInitialized {len(initialized)} model(s) in index.")
    
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Handle the refresh command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except (OSError, ValueError) as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return 1
    
    # Load config to check storage paths for state detection
    try:
        config.load()
    except (OSError, ValueError, ConfigError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1
    
    base_storage = config.get('baseStorage')
    remote_storage = config.get('remoteStorage')
    
    # Get models from Ollama
    try:
        models = get_ollama_models()
    except (CliError, SubprocessError, OmanageAPIError) as e:
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
    
    def _detect_frozen(blob_name: str) -> Tuple[bool, bool]:
        """Detect whether a model blob is on remote storage."""
        if not base_storage or not remote_storage:
            return False, False
        base_blob = Path(base_storage) / blob_name
        remote_blob = Path(remote_storage) / blob_name
        if base_blob.exists():
            return False, False
        if remote_blob.exists():
            from omanage.utils import detect_compression
            return True, detect_compression(remote_blob)
        return False, False
    
    try:
        for model in models:
            model_name = model['name']
            print(f"  Refreshing {model_name}...")
            
            # Get blob info
            blob_info = get_model_blob_info(model_name)
            if blob_info:
                # Detect actual storage state; fall back to existing index state if detection unavailable
                frozen, compressed = _detect_frozen(blob_info['blobName'])
                
                index.set_model(
                    model_name=model_name,
                    blob_sha=blob_info['blobSha'],
                    blob_name=blob_info['blobName'],
                    frozen=frozen,
                    compressed=compressed
                )
                status = "frozen" if frozen else "thawed"
                print(f"    Updated blob: {blob_info['blobSha']} ({status})")
            else:
                print(f"    Warning: Could not extract blob info for {model_name}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nOperation cancelled. Saving partial progress...", file=sys.stderr)
    finally:
        # Save index (always, even on partial progress or unexpected errors)
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
    except ModelAlreadyFrozenError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
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
    except ModelAlreadyThawedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_export(args: argparse.Namespace) -> int:
    """Handle the export command."""
    config_dir = Path.cwd()
    model_name = args.model_name
    compress = args.compress
    
    # Validate model name
    try:
        validate_model_name(model_name)
    except InvalidModelNameError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Use API for exporting
    api = OmanageAPI(config_dir)
    
    try:
        result = api.export_model(model_name, compress)
        
        if result['success']:
            print(f"\nModel '{model_name}' exported successfully.")
            return 0
        else:
            print(f"Model '{model_name}' export failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ModelAlreadyFrozenError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    """Handle the import command."""
    config_dir = Path.cwd()
    model_name = args.model_name
    
    # Validate model name
    try:
        validate_model_name(model_name)
    except InvalidModelNameError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Use API for importing
    api = OmanageAPI(config_dir)
    
    try:
        result = api.import_model(model_name)
        
        if result['success']:
            print(f"\nModel '{model_name}' imported successfully.")
            return 0
        else:
            print(f"Model '{model_name}' import failed: {result.get('message', 'Unknown error')}")
            return 1
            
    except ModelNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ModelAlreadyThawedError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except StorageNotConfiguredError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileOperationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
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
            error_msg = result.get('error', 'Unknown error')
            print(f"Verification failed: {error_msg}")
            if result.get('missing'):
                print("\nMissing files:")
                for item in result['missing']:
                    print(f"  - {item['model']}: {item['path']}")
                    if 'issue' in item:
                        print(f"    Issue: {item['issue']}")
            return 1
        elif result['status'] == 'mismatch':
            print("Verification complete with mismatches.")
            if result.get('missing'):
                print("\nMissing files:")
                for item in result['missing']:
                    print(f"  - {item['model']}: {item['path']}")
                    if 'issue' in item:
                        print(f"    Issue: {item['issue']}")
            if result.get('mismatched'):
                print("\nMismatched files:")
                for item in result['mismatched']:
                    print(f"  - {item['model']}: {item['issue']}")
            return 1
        else:
            print(f"Verification complete with unexpected status: {result['status']}")
            return 1
            
    except (OSError, ValueError) as e:
        print(f"Error during verification: {e}", file=sys.stderr)
        return 1
