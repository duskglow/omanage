"""Python API for omanage - Programmatic access to Ollama model management."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .api_core.subprocess_utils import (
    SubprocessError,
    run_ollama_command,
    get_ollama_models as get_ollama_models_secure,
    get_model_blob_info as get_model_blob_info_secure,
)

from .config import ConfigManager, ConfigError
from .index import IndexManager, OmanageIndexError
from .utils import (
    OmanageError,
    ValidationError,
    ProgressBar,
    compress_file,
    decompress_file,
    detect_compression,
    validate_model_name,
    validate_path_traversal,
    parse_model_name,
    PathTraversalError,
    InvalidModelNameError,
    CHUNK_SIZE,
)
from .api_core.errors import (
    OmanageAPIError,
    OllamaNotInstalledError,
    ModelNotFoundError,
    StorageNotConfiguredError,
    FileOperationError,
    ModelAlreadyFrozenError,
    ModelAlreadyThawedError,
)
from .api_core.locking import _FileLock
from .api_core.file_utils import (
    atomic_copy_with_temp,
    create_secure_tempfile,
    transfer_manifest_file,
)

# Constants for magic values
LOCK_FILE_SUFFIX = '.lock'
MANIFEST_BASE_DIR = "manifests"
MANIFEST_REGISTRY_PATH = "registry.ollama.ai/library"


class OmanageAPI:
    """
    Python API for managing Ollama models.
    
    This class provides programmatic access to model management operations
    including initialization, listing, freezing, thawing, and verification
    of Ollama models.
    
    Example usage:
        from omanage import OmanageAPI
        
        api = OmanageAPI(Path.cwd())
        api.initialize()
        models = api.list_models()
        api.freeze_model("llama3:8b", compress=True)
    """
    
    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize the API with a project directory.
        
        Args:
            project_dir: Path to the project directory containing .omanage.conf
                        If None, uses current working directory.
        """
        self.config_dir = project_dir or Path.cwd()
        self.config = ConfigManager(self.config_dir)
        self.index = IndexManager(self.config_dir)
    
    def initialize(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Initialize the model index from Ollama and remote storage.
        
        Args:
            model_name: If specified, only initialize this model. Otherwise,
                       initialize all models.
        
        Returns:
            List of dictionaries with 'name' key for each initialized model.
        
        Raises:
            OllamaNotInstalledError: If Ollama is not installed.
        """
        self.config.initialize()
        self.index.initialize()
        
        models = self._get_ollama_models()
        
        # Also discover models that exist only on remote storage (e.g., an NFS
        # share mounted on a new machine where Ollama has not yet seen them).
        remote_models = self._scan_remote_storage_models()
        
        # Merge remote-only models, using Ollama's view when a model is known locally
        known_names = {m['name'] for m in models}
        for remote_model in remote_models:
            if remote_model['name'] not in known_names:
                if model_name and remote_model['name'] != model_name:
                    continue
                models.append(remote_model)
        
        if model_name:
            validate_model_name(model_name)
            models = [m for m in models if m['name'] == model_name]
            if not models:
                raise ModelNotFoundError(f"Model '{model_name}' not found in Ollama or remote storage.")
        
        initialized = []
        for model in models:
            model_name = model['name']
            blob_info = self._get_model_blob_info(model_name)
            if blob_info:
                frozen, compressed = self._detect_storage_state(blob_info['blobName'])
            elif 'blobSha' in model and 'blobName' in model:
                # Remote-only model discovered by manifest scan
                blob_info = {
                    'blobSha': model['blobSha'],
                    'blobName': model['blobName']
                }
                frozen, compressed = True, False
                # Try to detect compression from the remote blob if available
                self.config.load()
                remote_storage = self.config.get('remoteStorage')
                if remote_storage:
                    remote_blob = Path(remote_storage) / model['blobName']
                    if remote_blob.exists():
                        compressed = detect_compression(remote_blob)
            else:
                continue
            
            self.index.set_model(
                model_name=model_name,
                blob_sha=blob_info['blobSha'],
                blob_name=blob_info['blobName'],
                frozen=frozen,
                compressed=compressed
            )
            initialized.append({
                'name': model_name,
                'blobSha': blob_info['blobSha'],
                'frozen': frozen
            })
        
        self.index.save()
        return initialized
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all models in the index.
        
        Returns:
            Dictionary mapping model names to their metadata.
        """
        self.index.load()
        return self.index.list_models()
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific model.
        
        Args:
            model_name: Name of the model to retrieve.
        
        Returns:
            Model metadata dictionary, or None if not found.
        """
        validate_model_name(model_name)
        self.index.load()
        return self.index.get_model(model_name)
    
    def freeze_model(self, model_name: str, compress: bool = False) -> Dict[str, Any]:
        """
        Freeze a model by moving its blob to remote storage.
        
        Args:
            model_name: Name of the model to freeze.
            compress: If True, compress the blob during the move.
        
        Returns:
            Dictionary with operation results including:
            - 'success': bool indicating success
            - 'model': model name
            - 'blob_sha': SHA of the moved blob
            - 'compressed': whether compression was used
        
        Raises:
            ModelNotFoundError: If model not in index.
            StorageNotConfiguredError: If storage paths not configured.
            FileOperationError: If file operation fails.
            InvalidModelNameError: If model name is invalid.
        """
        validate_model_name(model_name)
        self.index.load()
        
        model_meta = self.index.get_model(model_name)
        if not model_meta:
            raise ModelNotFoundError(f"Model '{model_name}' not found in index. Run initialize() first.")
        
        # Validate blob name from index to prevent path traversal from tampered index
        blob_name = model_meta.get('blobName', '')
        if not blob_name or not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
            raise FileOperationError(f"Invalid blob name in index for model '{model_name}': {blob_name}")
        
        if model_meta.get('frozen', False):
            raise ModelAlreadyFrozenError(f"Model '{model_name}' is already frozen")
        
        self.config.load()
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')
        
        if not base_storage:
            raise StorageNotConfiguredError("baseStorage not configured")
        if not remote_storage:
            raise StorageNotConfiguredError("remoteStorage not configured")
        
        # Validate storage paths exist
        base_path = Path(base_storage)
        remote_path = Path(remote_storage)
        
        if not base_path.exists():
            raise FileOperationError(f"Base storage path does not exist: {base_path}")
        if not remote_path.exists():
            raise FileOperationError(f"Remote storage path does not exist: {remote_path}")
        
        source_path = Path(base_storage) / model_meta['blobName']
        dest_path = Path(remote_storage) / model_meta['blobName']
        
        # Validate paths don't traverse outside expected directories
        validate_path_traversal(source_path, base_path, "source path")
        validate_path_traversal(dest_path, remote_path, "destination path")
        
        if not source_path.exists():
            raise FileOperationError(f"Blob file not found: {source_path}")
        
        # Use file locking to prevent race conditions (cross-platform)
        dest_lock_path = dest_path.with_suffix(LOCK_FILE_SUFFIX)
        
        # Check for symlinks before file operations
        if source_path.is_symlink():
            raise FileOperationError(f"Symlinks are not allowed in storage: {source_path}")
        
        with _FileLock(dest_lock_path) as lock:
            # Re-check after acquiring lock
            if dest_path.exists():
                raise FileOperationError(f"Destination already exists: {dest_path}")
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_size = source_path.stat().st_size
            
            try:
                if compress:
                    with ProgressBar(file_size, "Compressing") as pb:
                        compress_file(source_path, dest_path, pb)
                else:
                    with ProgressBar(file_size, "Moving") as pb:
                        with source_path.open('rb') as src, dest_path.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        pb.update(file_size)
                
                if not dest_path.exists():
                    raise FileOperationError("Destination file not created")
                
                # Handle manifest file if it exists
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )
                
                if base_manifest_path.exists():
                    # Use transfer_manifest_file for atomic transfer with rollback handling
                    result = transfer_manifest_file(base_manifest_path, remote_manifest_path, delete_source=True)
                    if not result:
                        raise FileOperationError(f"Failed to transfer manifest file from {base_manifest_path} to {remote_manifest_path}")
                
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=True,
                    compressed=compress,
                    manifest_name=tag
                )
                self.index.save()
                
                # ONLY delete source after all metadata is safely persisted
                source_path.unlink()
                
                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'compressed': compress
                }
                
            except (OSError, ValueError) as e:
                # Rollback: remove destination but leave source intact
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except OSError:
                        pass
                raise FileOperationError(f"Error freezing model: {e}")
    
    def thaw_model(self, model_name: str) -> Dict[str, Any]:
        """
        Thaw a model by moving its blob back to base storage.
        
        Args:
            model_name: Name of the model to thaw.
        
        Returns:
            Dictionary with operation results including:
            - 'success': bool indicating success
            - 'model': model name
            - 'blob_sha': SHA of the moved blob
            - 'decompressed': whether decompression occurred
        
        Raises:
            ModelNotFoundError: If model not in index.
            StorageNotConfiguredError: If storage paths not configured.
            FileOperationError: If file operation fails.
            InvalidModelNameError: If model name is invalid.
        """
        validate_model_name(model_name)
        self.index.load()
        
        model_meta = self.index.get_model(model_name)
        if not model_meta:
            raise ModelNotFoundError(f"Model '{model_name}' not found in index. Run initialize() first.")
        
        # Validate blob name from index to prevent path traversal from tampered index
        blob_name = model_meta.get('blobName', '')
        if not blob_name or not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
            raise FileOperationError(f"Invalid blob name in index for model '{model_name}': {blob_name}")
        
        if not model_meta.get('frozen', False):
            raise ModelAlreadyThawedError(f"Model '{model_name}' is already thawed")
        
        self.config.load()
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')
        
        if not base_storage:
            raise StorageNotConfiguredError("baseStorage not configured")
        if not remote_storage:
            raise StorageNotConfiguredError("remoteStorage not configured")
        
        # Validate storage paths exist
        base_path = Path(base_storage)
        remote_path = Path(remote_storage)
        
        if not base_path.exists():
            raise FileOperationError(f"Base storage path does not exist: {base_path}")
        if not remote_path.exists():
            raise FileOperationError(f"Remote storage path does not exist: {remote_path}")
        
        source_path = Path(remote_storage) / model_meta['blobName']
        dest_path = Path(base_storage) / model_meta['blobName']
        
        # Validate paths don't traverse outside expected directories
        validate_path_traversal(source_path, remote_path, "source path")
        validate_path_traversal(dest_path, base_path, "destination path")
        
        if not source_path.exists():
            raise FileOperationError(f"Blob file not found: {source_path}")
        
        # Use file locking to prevent race conditions (cross-platform)
        dest_lock_path = dest_path.with_suffix(LOCK_FILE_SUFFIX)
        
        # Check for symlinks before file operations
        if source_path.is_symlink():
            raise FileOperationError(f"Symlinks are not allowed in storage: {source_path}")
        
        with _FileLock(dest_lock_path) as lock:
            # Re-check after acquiring lock
            if dest_path.exists():
                raise FileOperationError(f"Destination already exists: {dest_path}")
            
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_size = source_path.stat().st_size
            is_compressed = detect_compression(source_path) or model_meta.get('compressed', False)
            
            try:
                if is_compressed:
                    with ProgressBar(file_size, "Decompressing") as pb:
                        decompress_file(source_path, dest_path, pb)
                else:
                    with ProgressBar(file_size, "Moving") as pb:
                        with source_path.open('rb') as src, dest_path.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                        pb.update(file_size)
                
                if not dest_path.exists():
                    raise FileOperationError("Destination file not created")
                
                # Handle manifest file if it exists
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )
                
                if remote_manifest_path.exists():
                    # Use transfer_manifest_file for atomic transfer with rollback handling
                    result = transfer_manifest_file(remote_manifest_path, base_manifest_path, delete_source=True)
                    if not result:
                        raise FileOperationError(f"Failed to transfer manifest file from {remote_manifest_path} to {base_manifest_path}")
                
                # After successful thaw, the blob is always decompressed in base storage
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=False,
                    compressed=False,
                    manifest_name=tag
                )
                self.index.save()
                
                # ONLY delete source after all metadata is safely persisted
                source_path.unlink()
                
                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'decompressed': is_compressed
                }
                
            except (OSError, ValueError) as e:
                # Rollback: remove destination but leave source intact
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except OSError:
                        pass
                raise FileOperationError(f"Error thawing model: {e}")
    
    def export_model(self, model_name: str, compress: bool = False) -> Dict[str, Any]:
        """
        Export a model by copying its blob to remote storage without deleting the source.

        Behaves exactly like freeze_model() except the source blob and manifest are
        left intact in base storage. This is useful for creating a copy on an NFS
        partition that can later be imported elsewhere.

        Args:
            model_name: Name of the model to export.
            compress: If True, compress the blob during the copy.

        Returns:
            Dictionary with operation results including:
            - 'success': bool indicating success
            - 'model': model name
            - 'blob_sha': SHA of the copied blob
            - 'compressed': whether compression was used

        Raises:
            ModelNotFoundError: If model not in index.
            StorageNotConfiguredError: If storage paths not configured.
            FileOperationError: If file operation fails.
            InvalidModelNameError: If model name is invalid.
        """
        validate_model_name(model_name)
        self.index.load()

        model_meta = self.index.get_model(model_name)
        if not model_meta:
            raise ModelNotFoundError(f"Model '{model_name}' not found in index. Run initialize() first.")

        # Validate blob name from index to prevent path traversal from tampered index
        blob_name = model_meta.get('blobName', '')
        if not blob_name or not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
            raise FileOperationError(f"Invalid blob name in index for model '{model_name}': {blob_name}")

        if model_meta.get('frozen', False):
            raise ModelAlreadyFrozenError(f"Model '{model_name}' is already frozen")

        self.config.load()
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')

        if not base_storage:
            raise StorageNotConfiguredError("baseStorage not configured")
        if not remote_storage:
            raise StorageNotConfiguredError("remoteStorage not configured")

        # Validate storage paths exist
        base_path = Path(base_storage)
        remote_path = Path(remote_storage)

        if not base_path.exists():
            raise FileOperationError(f"Base storage path does not exist: {base_path}")
        if not remote_path.exists():
            raise FileOperationError(f"Remote storage path does not exist: {remote_path}")

        source_path = Path(base_storage) / model_meta['blobName']
        dest_path = Path(remote_storage) / model_meta['blobName']

        # Validate paths don't traverse outside expected directories
        validate_path_traversal(source_path, base_path, "source path")
        validate_path_traversal(dest_path, remote_path, "destination path")

        if not source_path.exists():
            raise FileOperationError(f"Blob file not found: {source_path}")

        # Use file locking to prevent race conditions (cross-platform)
        dest_lock_path = dest_path.with_suffix(LOCK_FILE_SUFFIX)

        # Check for symlinks before file operations
        if source_path.is_symlink():
            raise FileOperationError(f"Symlinks are not allowed in storage: {source_path}")

        with _FileLock(dest_lock_path) as lock:
            # Re-check after acquiring lock
            if dest_path.exists():
                raise FileOperationError(f"Destination already exists: {dest_path}")

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            file_size = source_path.stat().st_size

            try:
                if compress:
                    with ProgressBar(file_size, "Compressing") as pb:
                        compress_file(source_path, dest_path, pb)
                else:
                    with ProgressBar(file_size, "Copying") as pb:
                        atomic_copy_with_temp(source_path, dest_path, progress_callback=pb.update)

                if not dest_path.exists():
                    raise FileOperationError("Destination file not created")

                # Copy manifest file if it exists (do not move/delete source)
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )

                if base_manifest_path.exists():
                    result = transfer_manifest_file(
                        base_manifest_path,
                        remote_manifest_path,
                        delete_source=False,
                        copy_only=True
                    )
                    if not result:
                        raise FileOperationError(f"Failed to copy manifest file from {base_manifest_path} to {remote_manifest_path}")

                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=True,
                    compressed=compress,
                    manifest_name=tag
                )
                self.index.save()

                # Do NOT delete source — export leaves the original in place

                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'compressed': compress
                }

            except (OSError, ValueError) as e:
                # Rollback: remove destination but leave source intact
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except OSError:
                        pass
                raise FileOperationError(f"Error exporting model: {e}")

    def import_model(self, model_name: str) -> Dict[str, Any]:
        """
        Import a model by copying its blob from remote storage without deleting the source.

        Behaves exactly like thaw_model() except the source blob and manifest in remote
        storage are left intact. This is useful for pulling a model from an NFS partition
        onto a new machine while keeping the shared copy available.

        Args:
            model_name: Name of the model to import.

        Returns:
            Dictionary with operation results including:
            - 'success': bool indicating success
            - 'model': model name
            - 'blob_sha': SHA of the copied blob
            - 'decompressed': whether decompression occurred

        Raises:
            ModelNotFoundError: If model not in index.
            StorageNotConfiguredError: If storage paths not configured.
            FileOperationError: If file operation fails.
            InvalidModelNameError: If model name is invalid.
        """
        validate_model_name(model_name)
        self.index.load()

        model_meta = self.index.get_model(model_name)
        if not model_meta:
            raise ModelNotFoundError(f"Model '{model_name}' not found in index. Run initialize() first.")

        # Validate blob name from index to prevent path traversal from tampered index
        blob_name = model_meta.get('blobName', '')
        if not blob_name or not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
            raise FileOperationError(f"Invalid blob name in index for model '{model_name}': {blob_name}")

        if not model_meta.get('frozen', False):
            raise ModelAlreadyThawedError(f"Model '{model_name}' is already thawed")

        self.config.load()
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')

        if not base_storage:
            raise StorageNotConfiguredError("baseStorage not configured")
        if not remote_storage:
            raise StorageNotConfiguredError("remoteStorage not configured")

        # Validate storage paths exist
        base_path = Path(base_storage)
        remote_path = Path(remote_storage)

        if not base_path.exists():
            raise FileOperationError(f"Base storage path does not exist: {base_path}")
        if not remote_path.exists():
            raise FileOperationError(f"Remote storage path does not exist: {remote_path}")

        source_path = Path(remote_storage) / model_meta['blobName']
        dest_path = Path(base_storage) / model_meta['blobName']

        # Validate paths don't traverse outside expected directories
        validate_path_traversal(source_path, remote_path, "source path")
        validate_path_traversal(dest_path, base_path, "destination path")

        if not source_path.exists():
            raise FileOperationError(f"Blob file not found: {source_path}")

        # Use file locking to prevent race conditions (cross-platform)
        dest_lock_path = dest_path.with_suffix(LOCK_FILE_SUFFIX)

        # Check for symlinks before file operations
        if source_path.is_symlink():
            raise FileOperationError(f"Symlinks are not allowed in storage: {source_path}")

        with _FileLock(dest_lock_path) as lock:
            # Re-check after acquiring lock
            if dest_path.exists():
                raise FileOperationError(f"Destination already exists: {dest_path}")

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            file_size = source_path.stat().st_size
            is_compressed = detect_compression(source_path) or model_meta.get('compressed', False)

            try:
                if is_compressed:
                    with ProgressBar(file_size, "Decompressing") as pb:
                        decompress_file(source_path, dest_path, pb)
                else:
                    with ProgressBar(file_size, "Copying") as pb:
                        atomic_copy_with_temp(source_path, dest_path, progress_callback=pb.update)

                if not dest_path.exists():
                    raise FileOperationError("Destination file not created")

                # Copy manifest file if it exists (do not move/delete source)
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )

                if remote_manifest_path.exists():
                    result = transfer_manifest_file(
                        remote_manifest_path,
                        base_manifest_path,
                        delete_source=False,
                        copy_only=True
                    )
                    if not result:
                        raise FileOperationError(f"Failed to copy manifest file from {remote_manifest_path} to {base_manifest_path}")

                # After successful import, the blob is always decompressed in base storage
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=False,
                    compressed=False,
                    manifest_name=tag
                )
                self.index.save()

                # Do NOT delete source — import leaves the original in place

                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'decompressed': is_compressed
                }

            except (OSError, ValueError) as e:
                # Rollback: remove destination but leave source intact
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except OSError:
                        pass
                raise FileOperationError(f"Error importing model: {e}")

    def verify(self) -> Dict[str, Any]:
        """
        Verify model files exist in expected locations.
        
        Returns:
            Dictionary with verification results including:
            - 'status': 'ok', 'missing', or 'mismatch'
            - 'total_models': total models in index
            - 'missing': list of missing files
            - 'mismatched': list of mismatched files
        """
        self.index.load()
        self.config.load()
        
        models = self.index.list_models()
        
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')
        
        if not base_storage or not remote_storage:
            return {
                'status': 'error',
                'error': 'Storage paths not configured'
            }
        
        base_path = Path(base_storage)
        remote_path = Path(remote_storage)
        
        if not base_path.exists():
            return {
                'status': 'error',
                'error': f'Base storage path does not exist: {base_path}'
            }
        if not remote_path.exists():
            return {
                'status': 'error',
                'error': f'Remote storage path does not exist: {remote_path}'
            }
        
        missing = []
        mismatched = []
        
        for model_name, metadata in models.items():
            blob_name = metadata.get('blobName')
            frozen = metadata.get('frozen', False)
            compressed = metadata.get('compressed', False)
            
            if not blob_name:
                missing.append({
                    'model': model_name,
                    'path': 'N/A',
                    'issue': 'Missing blobName in index'
                })
                continue
            
            if frozen:
                expected_path = remote_path / blob_name
            else:
                expected_path = base_path / blob_name
            
            # Validate path doesn't traverse
            try:
                validate_path_traversal(expected_path, remote_path if frozen else base_path, "expected path")
            except PathTraversalError:
                missing.append({
                    'model': model_name,
                    'path': str(expected_path),
                    'issue': 'Path traversal detected'
                })
                continue
            
            if not expected_path.exists():
                missing.append({
                    'model': model_name,
                    'path': str(expected_path)
                })
            else:
                # Check compression status
                is_compressed = detect_compression(expected_path)
                if compressed and not is_compressed:
                    mismatched.append({
                        'model': model_name,
                        'issue': 'Expected compressed but found uncompressed'
                    })
                elif not compressed and is_compressed:
                    mismatched.append({
                        'model': model_name,
                        'issue': 'Expected uncompressed but found compressed'
                    })
        
        if missing or mismatched:
            status = 'error' if missing else 'mismatch'
        else:
            status = 'ok'
        
        return {
            'status': status,
            'total_models': len(models),
            'missing': missing,
            'mismatched': mismatched
        }
    
    def get_installed_models(self) -> List[Dict[str, str]]:
        """
        Get list of models installed in Ollama.
        
        Returns:
            List of dictionaries with 'name' key for each installed model.
        
        Raises:
            OllamaNotInstalledError: If Ollama is not installed.
        """
        return self._get_ollama_models()
    
    def get_model_blob_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """
        Get blob information for a model from its modelfile.
        
        Args:
            model_name: Name of the model.
        
        Returns:
            Dictionary with 'blobSha' and 'blobName' keys, or None if not found.
        
        Raises:
            OllamaNotInstalledError: If Ollama is not installed.
            InvalidModelNameError: If model name is invalid.
        """
        validate_model_name(model_name)
        return self._get_model_blob_info(model_name)
    
    def _get_ollama_models(self) -> List[Dict[str, str]]:
        """Get list of installed Ollama models using secure subprocess wrapper."""
        # Use configured binary path if available
        ollama_binary = self.config.get("ollamaBinary")
        try:
            return get_ollama_models_secure(ollama_binary=ollama_binary)
        except SubprocessError:
            raise OllamaNotInstalledError("Ollama not found. Please install Ollama first.")
    
    def _get_model_blob_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """Get blob information for a model from its modelfile using secure subprocess wrapper."""
        # Use configured binary path if available
        ollama_binary = self.config.get("ollamaBinary")
        try:
            result = get_model_blob_info_secure(model_name, ollama_binary=ollama_binary)
            if result is None:
                return None
            return result
        except SubprocessError as e:
            err = str(e).lower()
            # A specific "model not found" response means Ollama is installed but
            # doesn't know this model yet (e.g., remote-only models). Return None
            # so callers can fall back to other sources (manifest scan, index).
            if "not found" in err and "model" in err:
                return None
            raise OllamaNotInstalledError("Ollama not found. Please install Ollama first.")
    
    def _scan_remote_storage_models(self) -> List[Dict[str, str]]:
        """
        Scan remote storage manifests for models not known to local Ollama.

        Walks the remote manifests directory (e.g.,
        <remoteStorage>/../manifests/registry.ollama.ai/library/<model>/<tag>)
        and extracts model names plus their model-weight blob digest.

        Supports both layout conventions:
          - remoteStorage points to the blobs directory (manifests sibling)
          - remoteStorage points to the parent models directory (manifests child)

        Returns:
            List of dictionaries with 'name', 'blobSha', and 'blobName' keys for
            each remote-only model whose model blob exists in remote storage.
        """
        import logging
        logger = logging.getLogger(__name__)

        self.config.load()
        remote_storage = self.config.get('remoteStorage')
        if not remote_storage:
            logger.debug("remoteStorage not configured, skipping remote manifest scan")
            return []

        remote_storage_path = Path(remote_storage).resolve()

        # Try both common layout conventions
        candidate_roots = [
            remote_storage_path.parent / MANIFEST_BASE_DIR / MANIFEST_REGISTRY_PATH,
            remote_storage_path / MANIFEST_BASE_DIR / MANIFEST_REGISTRY_PATH,
        ]

        # Blobs may live directly under remoteStorage or under remoteStorage/blobs/
        candidate_blob_dirs = [
            remote_storage_path,
            remote_storage_path / "blobs",
        ]

        discovered: List[Dict[str, str]] = []
        seen_manifests: set = set()

        for remote_manifest_root in candidate_roots:
            if not remote_manifest_root.exists():
                logger.debug("Remote manifest root does not exist: %s", remote_manifest_root)
                continue

            logger.info("Scanning remote manifests under %s", remote_manifest_root)

            for manifest_path in remote_manifest_root.rglob('*'):
                if not manifest_path.is_file():
                    continue
                if manifest_path in seen_manifests:
                    continue
                seen_manifests.add(manifest_path)

                try:
                    # Reconstruct model name from manifest path
                    relative = manifest_path.relative_to(remote_manifest_root)
                    parts = relative.parts
                    if len(parts) < 2:
                        logger.debug("Skipping manifest with insufficient path depth: %s", manifest_path)
                        continue

                    tag = parts[-1]
                    model = '/'.join(parts[:-1])
                    model_name = f"{model}:{tag}"

                    # Validate model name components
                    try:
                        validate_model_name(model_name)
                    except InvalidModelNameError:
                        logger.debug("Skipping manifest with invalid model name '%s': %s", model_name, manifest_path)
                        continue

                    # Parse manifest JSON
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    layers = manifest.get('layers', [])
                    if not isinstance(layers, list):
                        logger.debug("Skipping manifest without layers array: %s", manifest_path)
                        continue

                    model_digest = self._extract_model_blob_digest(layers, candidate_blob_dirs)

                    if not model_digest or not model_digest.startswith('sha256:'):
                        logger.debug("Could not determine model blob digest for manifest: %s", manifest_path)
                        continue

                    blob_name = model_digest.replace('sha256:', 'sha256-', 1)
                    blob_sha = blob_name

                    # Validate blob name to prevent traversal
                    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
                        logger.debug("Invalid blob name derived from manifest: %s", blob_name)
                        continue

                    # Only report if the referenced blob actually exists in remote storage
                    remote_blob = self._find_remote_blob(blob_name, candidate_blob_dirs)
                    if remote_blob is None:
                        logger.debug("Referenced blob missing from remote storage: %s", blob_name)
                        continue

                    logger.info("Discovered remote-only model %s -> %s", model_name, blob_name)
                    discovered.append({
                        'name': model_name,
                        'blobSha': blob_sha,
                        'blobName': blob_name,
                    })

                except (OSError, json.JSONDecodeError, ValueError) as e:
                    logger.debug("Skipping unreadable or malformed manifest %s: %s", manifest_path, e)
                    continue

        return discovered

    def _find_remote_blob(self, blob_name: str, candidate_dirs: List[Path]) -> Optional[Path]:
        """
        Find an existing blob file in one of the candidate remote blob directories.

        Args:
            blob_name: Name of the blob file.
            candidate_dirs: List of directories to check.

        Returns:
            Path to the existing blob, or None if not found in any candidate directory.
        """
        for directory in candidate_dirs:
            blob_path = directory / blob_name
            if blob_path.exists():
                return blob_path
        return None

    def _extract_model_blob_digest(self, layers: List[Any], candidate_blob_dirs: List[Path]) -> Optional[str]:
        """
        Determine the model-weight blob digest from a manifest's layers.

        First tries the explicit Ollama model media type. If that is absent,
        falls back to the largest existing blob referenced by any layer digest,
        which is almost always the model weights file.

        Args:
            layers: Manifest layers list.
            candidate_blob_dirs: List of directories where blobs may exist.

        Returns:
            The selected blob digest (sha256:...), or None if no suitable blob found.
        """
        # Preferred: explicit model weights media type
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            media_type = layer.get('mediaType', '')
            if media_type == 'application/vnd.ollama.image.model':
                digest = layer.get('digest', '')
                if digest.startswith('sha256:'):
                    return digest

        # Fallback: largest existing blob among all sha256 digests
        best_digest: Optional[str] = None
        best_size = -1
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            digest = layer.get('digest', '')
            if not digest.startswith('sha256:'):
                continue
            blob_name = digest.replace('sha256:', 'sha256-', 1)
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', blob_name):
                continue
            blob_path = self._find_remote_blob(blob_name, candidate_blob_dirs)
            if blob_path is None:
                continue
            try:
                size = blob_path.stat().st_size
            except OSError:
                continue
            if size > best_size:
                best_size = size
                best_digest = digest

        return best_digest

    def _detect_storage_state(self, blob_name: str) -> Tuple[bool, bool]:
        """
        Detect whether a model blob is currently in base or remote storage.

        Args:
            blob_name: Name of the blob file.

        Returns:
            Tuple of (frozen, compressed) where frozen=True means the blob is in
            remote storage and compressed=True means the remote blob is gzipped.
            Defaults to (False, False) if storage paths are not configured.
        """
        self.config.load()
        base_storage = self.config.get('baseStorage')
        remote_storage = self.config.get('remoteStorage')

        if not base_storage or not remote_storage:
            return False, False

        base_path = Path(base_storage)

        if (base_path / blob_name).exists():
            return False, False

        remote_candidates = [
            Path(remote_storage) / blob_name,
            Path(remote_storage) / "blobs" / blob_name,
        ]
        for remote_blob in remote_candidates:
            if remote_blob.exists():
                return True, detect_compression(remote_blob)

        # Not found in either location; default to thawed
        return False, False

    def _get_manifest_paths(self, model: str, tag: str, base_storage: str, remote_storage: str) -> Tuple[Path, Path]:
        """
        Get the manifest paths for a model based on its storage location.
        
        Args:
            model: Model name (without tag)
            tag: Model tag
            base_storage: Base storage path
            remote_storage: Remote storage path
            
        Returns:
            Tuple of (base_manifest_path, remote_manifest_path)
        """
        # Validate model and tag names
        validate_model_name(model)
        if tag:
            validate_model_name(tag)
        
        # Use resolve() to get absolute paths for security
        base_storage_path = Path(base_storage).resolve()
        remote_storage_path = Path(remote_storage).resolve()
        
        # Build manifest paths relative to the storage directory's parent
        # Ollama convention: blobs/ and manifests/ are siblings under models/
        # e.g., baseStorage=/home/user/.ollama/models/blobs/ -> manifests at /home/user/.ollama/models/manifests/
        manifest_dir = Path(MANIFEST_REGISTRY_PATH) / model
        
        base_manifest_path = base_storage_path.parent / MANIFEST_BASE_DIR / manifest_dir / tag
        remote_manifest_path = remote_storage_path.parent / MANIFEST_BASE_DIR / manifest_dir / tag
        
        # Validate paths don't traverse outside expected directories
        validate_path_traversal(base_manifest_path, base_storage_path.parent, "base manifest path")
        validate_path_traversal(remote_manifest_path, remote_storage_path.parent, "remote manifest path")
        
        return base_manifest_path, remote_manifest_path
