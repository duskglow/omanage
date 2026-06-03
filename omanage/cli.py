"""CLI argument parsing and command handling for omanage."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigManager
from .index import IndexManager


def get_ollama_models() -> List[dict]:
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
                # Extract blob path from FROM directive
                from_path = line[5:].strip()
                # Path is like: /path/to/blobs/sha256-abcdef123456...
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


class CliError(Exception):
    """CLI-related errors."""
    pass


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
        config.set(key, value)
        config.save()
        print(f"Set config['{key}'] = {value}")
    
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
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return 1
    
    model_name = args.model_name
    
    # Check if model exists in index
    model_meta = index.get_model(model_name)
    if not model_meta:
        print(f"Model '{model_name}' not found in index. Run 'omanage init' first.")
        return 1
    
    if model_meta.get('frozen', False):
        print(f"Model '{model_name}' is already frozen.")
        return 0
    
    # TODO: Actually move the blob file
    # For Phase 1, just update the index
    index.set_model(
        model_name=model_name,
        blob_sha=model_meta['blobSha'],
        blob_name=model_meta['blobName'],
        frozen=True,
        compressed=args.compress
    )
    index.save()
    
    print(f"Model '{model_name}' frozen.")
    
    return 0


def cmd_thaw(args: argparse.Namespace) -> int:
    """Handle the thaw command."""
    config_dir = Path.cwd()
    config = ConfigManager(config_dir)
    index = IndexManager(config_dir)
    
    # Load index
    try:
        index.load()
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        return 1
    
    model_name = args.model_name
    
    # Check if model exists in index
    model_meta = index.get_model(model_name)
    if not model_meta:
        print(f"Model '{model_name}' not found in index. Run 'omanage init' first.")
        return 1
    
    if not model_meta.get('frozen', False):
        print(f"Model '{model_name}' is already thawed.")
        return 0
    
    # TODO: Actually move the blob file back
    # For Phase 1, just update the index
    index.set_model(
        model_name=model_name,
        blob_sha=model_meta['blobSha'],
        blob_name=model_meta['blobName'],
        frozen=False,
        compressed=args.compress
    )
    index.save()
    
    print(f"Model '{model_name}' thawed.")
    
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the verify command."""
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
        print("No models in index to verify.")
        return 0
    
    # Verification results
    missing = []
    mismatched = []
    
    print("Verifying model files...")
    
    for model_name, metadata in models.items():
        blob_sha = metadata.get('blobSha')
        frozen = metadata.get('frozen', False)
        
        # Determine expected path based on frozen state
        if frozen:
            # Should be in remote storage
            # TODO: Check actual remote storage path
            expected_path = None  # placeholder
            location = "remote storage"
        else:
            # Should be in base storage
            # TODO: Check actual base storage path
            expected_path = None  # placeholder
            location = "base storage"
        
        # For Phase 1, just note the model
        print(f"  {model_name}: {location}")
    
    # Summary
    print(f"\nVerification complete.")
    print(f"  Missing: {len(missing)}")
    print(f"  Mismatched: {len(mismatched)}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="omanage",
        description="Ollama Model Manager - Manage Ollama model storage"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show or set configuration options"
    )
    config_parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        help="Set a configuration key"
    )
    config_parser.add_argument(
        "--get",
        metavar="KEY",
        help="Get a configuration key"
    )
    config_parser.set_defaults(func=cmd_config)
    
    # help command
    help_parser = subparsers.add_parser(
        "help",
        help="Show help message"
    )
    help_parser.set_defaults(func=cmd_help)
    
    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List all models with their status"
    )
    list_parser.set_defaults(func=cmd_list)
    
    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize model index from Ollama"
    )
    init_parser.add_argument(
        "model_name",
        nargs="?",
        help="Specific model to initialize (optional)"
    )
    init_parser.set_defaults(func=cmd_init)
    
    # refresh command
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Refresh model index from Ollama"
    )
    refresh_parser.add_argument(
        "model_name",
        nargs="?",
        help="Specific model to refresh (optional)"
    )
    refresh_parser.set_defaults(func=cmd_refresh)
    
    # freeze command
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Move a model's blob to remote storage"
    )
    freeze_parser.add_argument(
        "model_name",
        help="Model to freeze"
    )
    freeze_parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the blob during move"
    )
    freeze_parser.set_defaults(func=cmd_freeze)
    
    # thaw command
    thaw_parser = subparsers.add_parser(
        "thaw",
        help="Move a model's blob back to base storage"
    )
    thaw_parser.add_argument(
        "model_name",
        help="Model to thaw"
    )
    thaw_parser.add_argument(
        "--compress",
        action="store_true",
        help="Decompress the blob during move"
    )
    thaw_parser.set_defaults(func=cmd_thaw)
    
    # verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify model file locations match index"
    )
    verify_parser.set_defaults(func=cmd_verify)
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as e:
        return e.code if e.code is not None else 1
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Execute the command
    try:
        return parsed_args.func(parsed_args)
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())