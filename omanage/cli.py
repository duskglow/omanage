"""CLI argument parsing and command handling for omanage."""

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigManager
from .index import IndexManager
from .utils import ProgressBar, compress_file, decompress_file, detect_compression


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
        # Blob should be in remote storage
        remote_storage = config.get('remoteStorage')
        if not remote_storage:
            raise CliError("remoteStorage not configured")
        return Path(remote_storage) / model_meta['blobName']
    else:
        # Blob should be in base storage
        base_storage = config.get('baseStorage')
        if not base_storage:
            raise CliError("baseStorage not configured")
        return Path(base_storage) / model_meta['blobName']


def _parse_model_name(model_name: str) -> tuple:
    """
    Parse a model name into its components.
    
    Args:
        model_name: Model name in format "name:tag" or "name"
        
    Returns:
        Tuple of (model, tag) where tag defaults to 'latest' if not specified
        
    Raises:
        ValueError: If model_name is empty or invalid
    """
    if not model_name:
        raise ValueError("Model name cannot be empty")
    
    # Validate model name contains only allowed characters
    if not re.match(r'^[a-zA-Z0-9_\-:]+$', model_name):
        raise ValueError(f"Invalid model name: {model_name}")
    
    if ':' in model_name:
        model_parts = model_name.split(':', 1)
        model = model_parts[0]
        tag = model_parts[1]
    else:
        # If no tag specified, use 'latest'
        model = model_name
        tag = 'latest'
    
    return model, tag


def _get_manifest_paths(model: str, tag: str, config: ConfigManager) -> tuple:
    """
    Get the manifest paths for a model based on its storage location.
    
    Args:
        model: Model name (without tag)
        tag: Model tag
        config: ConfigManager instance
        
    Returns:
        Tuple of (base_manifest_path, remote_manifest_path)
    """
    base_storage = config.get('baseStorage')
    remote_storage = config.get('remoteStorage')
    
    if not base_storage:
        raise CliError("baseStorage not configured")
    if not remote_storage:
        raise CliError("remoteStorage not configured")
    
    # Manifest path components: registry.ollama.ai/library/<model>/<tag>
    manifest_dir = f"registry.ollama.ai/library/{model}"
    
    base_manifest_path = Path(base_storage) / "manifests" / manifest_dir / tag
    remote_manifest_path = Path(remote_storage) / "manifests" / manifest_dir / tag
    
    return base_manifest_path, remote_manifest_path


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
    compress = args.compress
    
    # Check if model exists in index
    model_meta = index.get_model(model_name)
    if not model_meta:
        print(f"Model '{model_name}' not found in index. Run 'omanage init' first.")
        return 1
    
    if model_meta.get('frozen', False):
        print(f"Model '{model_name}' is already frozen.")
        return 0
    
    # Get blob paths
    base_storage = config.get('baseStorage')
    remote_storage = config.get('remoteStorage')
    
    if not base_storage:
        print("Error: baseStorage not configured. Run 'omanage config --set baseStorage=<path>'", file=sys.stderr)
        return 1
    if not remote_storage:
        print("Error: remoteStorage not configured. Run 'omanage config --set remoteStorage=<path>'", file=sys.stderr)
        return 1
    
    # Parse model name to get model and tag
    try:
        model, tag = _parse_model_name(model_name)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Get manifest paths
    base_manifest_path, remote_manifest_path = _get_manifest_paths(model, tag, config)
    manifest_name = tag
    
    source_path = Path(base_storage) / model_meta['blobName']
    dest_path = Path(remote_storage) / model_meta['blobName']
    
    # Check if manifest file exists
    manifest_exists = base_manifest_path.exists()
    
    if not source_path.exists():
        print(f"Error: Blob file not found at {source_path}", file=sys.stderr)
        return 1
    
    # Check if destination already exists
    if dest_path.exists():
        print(f"Error: Destination already exists at {dest_path}", file=sys.stderr)
        return 1
    
    # Check if manifest destination exists
    if manifest_exists and remote_manifest_path.exists():
        print(f"Error: Manifest destination already exists at {remote_manifest_path}", file=sys.stderr)
        return 1
    
    # Ensure remote storage directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get file size for progress bar
    file_size = source_path.stat().st_size
    
    print(f"Freezing {model_name}...")
    print(f"  Source: {source_path}")
    print(f"  Destination: {dest_path}")
    
    try:
        if compress:
            print("  Compressing with gzip...")
            with ProgressBar(file_size, "  Compression") as pb:
                compress_file(source_path, dest_path, pb)
        else:
            with ProgressBar(file_size, "  Moving") as pb:
                with source_path.open('rb') as src, dest_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                pb.update(file_size)
        
        # Verify destination exists
        if not dest_path.exists():
            print("Error: Destination file not created", file=sys.stderr)
            return 1
        
        # Remove source file after successful copy
        source_path.unlink()
        print(f"  Removed source file: {source_path}")
        
        # Handle manifest file if it exists
        if manifest_exists:
            # Ensure manifest destination directory exists
            remote_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy manifest file
            print(f"  Freezing manifest: {base_manifest_path}")
            shutil.copy2(base_manifest_path, remote_manifest_path)
            base_manifest_path.unlink()
            print(f"  Removed manifest: {base_manifest_path}")
        
        # Update index
        index.set_model(
            model_name=model_name,
            blob_sha=model_meta['blobSha'],
            blob_name=model_meta['blobName'],
            frozen=True,
            compressed=compress,
            manifest_name=manifest_name
        )
        index.save()
        
        print(f"\nModel '{model_name}' frozen successfully.")
        
    except Exception as e:
        # Clean up partial file on error
        if dest_path.exists():
            dest_path.unlink()
        if manifest_exists and remote_manifest_path.exists():
            remote_manifest_path.unlink()
        print(f"Error freezing model: {e}", file=sys.stderr)
        return 1
    
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
    
    # Get blob paths
    config.load()
    base_storage = config.get('baseStorage')
    remote_storage = config.get('remoteStorage')
    
    if not base_storage:
        print("Error: baseStorage not configured. Run 'omanage config --set baseStorage=<path>'", file=sys.stderr)
        return 1
    if not remote_storage:
        print("Error: remoteStorage not configured. Run 'omanage config --set remoteStorage=<path>'", file=sys.stderr)
        return 1
    
    # Get manifest name from model name (e.g., "phi3:mini" -> manifest is "mini" in "registry.ollama.ai/library/phi3/")
    if ':' in model_name:
        model_parts = model_name.split(':', 1)
        model = model_parts[0]
        tag = model_parts[1]
    else:
        model = model_name
        tag = 'latest'
    
    # Manifest path components: registry.ollama.ai/library/<model>/<tag>
    manifest_dir = f"registry.ollama.ai/library/{model}"
    manifest_name = tag
    
    base_manifest_path = Path(base_storage).parent / "manifests" / manifest_dir / manifest_name
    remote_manifest_path = Path(remote_storage).parent / "manifests" / manifest_dir / manifest_name
    
    source_path = Path(remote_storage) / model_meta['blobName']
    dest_path = Path(base_storage) / model_meta['blobName']
    
    # Check if manifest file exists in remote storage
    manifest_exists = remote_manifest_path.exists()
    
    if not source_path.exists():
        print(f"Error: Blob file not found at {source_path}", file=sys.stderr)
        return 1
    
    # Check if destination already exists
    if dest_path.exists():
        print(f"Error: Destination already exists at {dest_path}", file=sys.stderr)
        return 1
    
    # Check if manifest destination already exists
    if manifest_exists and base_manifest_path.exists():
        print(f"Error: Manifest destination already exists at {base_manifest_path}", file=sys.stderr)
        return 1
    
    # Ensure base storage directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get file size for progress bar
    file_size = source_path.stat().st_size
    
    # Auto-detect if compressed by checking magic bytes
    is_compressed = detect_compression(source_path)
    should_decompress = model_meta.get('compressed', False) or is_compressed
    
    print(f"Thawing {model_name}...")
    print(f"  Source: {source_path}")
    print(f"  Destination: {dest_path}")
    print(f"  Detected compressed: {is_compressed}")
    
    try:
        if should_decompress:
            print("  Decompressing with gzip...")
            with ProgressBar(file_size, "  Decompression") as pb:
                decompress_file(source_path, dest_path, pb)
            
            # Update index to indicate not compressed (file is now decompressed)
            compressed = False
        else:
            with ProgressBar(file_size, "  Moving") as pb:
                with source_path.open('rb') as src, dest_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                pb.update(file_size)
            
            # Update index with compressed status from metadata
            compressed = model_meta.get('compressed', False)
        
        # Verify destination exists
        if not dest_path.exists():
            print("Error: Destination file not created", file=sys.stderr)
            return 1
        
        # Remove source file after successful copy
        source_path.unlink()
        print(f"  Removed source file: {source_path}")
        
        # Handle manifest file if it exists
        if manifest_exists:
            # Ensure manifest destination directory exists
            base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy manifest file
            print(f"  Thawing manifest: {remote_manifest_path}")
            shutil.copy2(remote_manifest_path, base_manifest_path)
            remote_manifest_path.unlink()
            print(f"  Removed manifest: {remote_manifest_path}")
        
        # Update index
        index.set_model(
            model_name=model_name,
            blob_sha=model_meta['blobSha'],
            blob_name=model_meta['blobName'],
            frozen=False,
            compressed=compressed,
            manifest_name=manifest_name
        )
        index.save()
        
        print(f"\nModel '{model_name}' thawed successfully.")
        
    except Exception as e:
        # Clean up partial file on error
        if dest_path.exists():
            dest_path.unlink()
        if manifest_exists and base_manifest_path.exists():
            base_manifest_path.unlink()
        print(f"Error thawing model: {e}", file=sys.stderr)
        return 1
    
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
    
    # Load config
    config.load()
    
    models = index.list_models()
    
    # Verify config is set up
    base_storage = config.get('baseStorage')
    remote_storage = config.get('remoteStorage')
    
    if not base_storage:
        print("Error: baseStorage not configured. Run 'omanage config --set baseStorage=<path>'", file=sys.stderr)
        return 1
    if not remote_storage:
        print("Error: remoteStorage not configured. Run 'omanage config --set remoteStorage=<path>'", file=sys.stderr)
        return 1
    
    base_path = Path(base_storage)
    remote_path = Path(remote_storage)
    
    if not base_path.exists():
        print(f"Warning: baseStorage path does not exist: {base_path}", file=sys.stderr)
    if not remote_path.exists():
        print(f"Warning: remoteStorage path does not exist: {remote_path}", file=sys.stderr)
    
    if not models:
        print("No models in index to verify.")
        return 0
    
    # Verification results
    missing = []
    mismatched = []
    missing_manifests = []
    mismatched_manifests = []
    
    # Get manifest directory path
    base_manifest_base = Path(base_storage).parent / "manifests"
    remote_manifest_base = Path(remote_storage).parent / "manifests"
    
    print("Verifying model files...")
    print(f"  Base storage: {base_path}")
    print(f"  Remote storage: {remote_path}")
    print()
    
    # Track blob files found in base storage
    base_files = set()
    remote_files = set()
    
    # Get actual files in storage directories
    if base_path.exists():
        base_files = {f.name for f in base_path.iterdir() if f.is_file()}
    if remote_path.exists():
        remote_files = {f.name for f in remote_path.iterdir() if f.is_file()}
    
    # Track manifest files for extra file detection
    base_manifest_files = set()
    remote_manifest_files = set()
    
    if base_manifest_base.exists():
        for root, dirs, files in os.walk(base_manifest_base):
            for f in files:
                base_manifest_files.add(os.path.relpath(os.path.join(root, f), base_manifest_base))
    
    if remote_manifest_base.exists():
        for root, dirs, files in os.walk(remote_manifest_base):
            for f in files:
                remote_manifest_files.add(os.path.relpath(os.path.join(root, f), remote_manifest_base))
    
    for model_name, metadata in models.items():
        blob_name = metadata.get('blobName')
        frozen = metadata.get('frozen', False)
        compressed = metadata.get('compressed', False)
        
        # Expected path based on frozen state
        if frozen:
            expected_path = remote_path / blob_name
            expected_location = "remote storage"
            actual_location = None
            
            if expected_path.exists():
                # Check if compressed status matches
                if compressed and not detect_compression(expected_path):
                    # File exists but compression status doesn't match
                    mismatched.append({
                        'model': model_name,
                        'expected': expected_location,
                        'actual': "uncompressed in remote storage",
                        'path': expected_path
                    })
                elif not compressed and detect_compression(expected_path):
                    # File exists but compression status doesn't match
                    mismatched.append({
                        'model': model_name,
                        'expected': expected_location,
                        'actual': "compressed in remote storage",
                        'path': expected_path
                    })
                else:
                    actual_location = expected_location
            else:
                missing.append({
                    'model': model_name,
                    'location': expected_location,
                    'path': expected_path
                })
        else:
            expected_path = base_path / blob_name
            expected_location = "base storage"
            actual_location = None
            
            if expected_path.exists():
                if compressed:
                    mismatched.append({
                        'model': model_name,
                        'expected': expected_location,
                        'actual': "compressed in base storage (should be thawed)",
                        'path': expected_path
                    })
                else:
                    actual_location = expected_location
            else:
                missing.append({
                    'model': model_name,
                    'location': expected_location,
                    'path': expected_path
                })
        
        # Print status for this model
        if actual_location:
            status = "✓ OK"
        elif any(m['model'] == model_name for m in missing):
            status = "✗ MISSING"
        else:
            status = "✗ MISMATCH"
        
        print(f"  {status}: {model_name}")
        print(f"         Blob: {blob_name}")
        print(f"         Expected: {expected_location}")
        if expected_path.exists():
            print(f"         Size: {expected_path.stat().st_size} bytes")
        print()
        
        # Verify manifest file
        manifest_name = metadata.get('manifestName')
        if manifest_name:
            # Parse model name to get model and tag
            if ':' in model_name:
                model_parts = model_name.split(':', 1)
                manifest_model = model_parts[0]
                manifest_tag = model_parts[1]
            else:
                manifest_model = model_name
                manifest_tag = 'latest'
            
            manifest_dir_path = f"registry.ollama.ai/library/{manifest_model}"
            
            if frozen:
                manifest_path = remote_manifest_base / manifest_dir_path / manifest_tag
                expected_manifest_location = "remote storage manifest"
            else:
                manifest_path = base_manifest_base / manifest_dir_path / manifest_tag
                expected_manifest_location = "base storage manifest"
            
            if manifest_path.exists():
                if actual_location is None:
                    # Manifest exists but blob doesn't - mismatch
                    mismatched_manifests.append({
                        'model': model_name,
                        'expected': expected_manifest_location,
                        'actual': f"exists but blob is missing",
                        'path': manifest_path
                    })
            else:
                # Manifest is missing
                missing_manifests.append({
                    'model': model_name,
                    'location': expected_manifest_location,
                    'path': manifest_path
                })
    
    # Check for models in index that don't exist in Ollama anymore (stale entries)
    try:
        installed_models = get_ollama_models()
        installed_names = {m['name'] for m in installed_models}
        index_names = set(models.keys())
        
        # Models in index but not in Ollama (might have been deleted)
        stale = index_names - installed_names
        if stale:
            print("  Note: Models in index but not in Ollama:")
            for m in sorted(stale):
                print(f"    - {m} (may have been deleted from Ollama)")
            print()
    except CliError:
        # Ollama not available, skip this check
        pass
    
    # Summary
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"  Total models in index: {len(models)}")
    print(f"  Missing files: {len(missing)}")
    print(f"  Mismatched: {len(mismatched)}")
    
    if missing:
        print(f"\n  Missing files:")
        for item in missing:
            print(f"    - {item['model']}: {item['path']}")
    
    if mismatched:
        print(f"\n  Mismatched:")
        for item in mismatched:
            print(f"    - {item['model']}: expected {item['expected']}, got {item['actual']}")
    
    # Check for files in storage that aren't in index
    extra_in_base = base_files - {m['blobName'] for m in models.values()}
    extra_in_remote = remote_files - {m['blobName'] for m in models.values()}
    
    if extra_in_base:
        print(f"\n  Files in base storage not in index: {len(extra_in_base)}")
        for f in sorted(extra_in_base)[:5]:
            print(f"    - {f}")
    
    if extra_in_remote:
        print(f"\n  Files in remote storage not in index: {len(extra_in_remote)}")
        for f in sorted(extra_in_remote)[:5]:
            print(f"    - {f}")
    
    print()
    
    # Return error code if there are issues
    if missing or mismatched:
        return 1
    
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