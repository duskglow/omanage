"""Python API for omanage - Programmatic access to Ollama model management."""

import os
import shutil
import tempfile
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
    PROGRESS_UPDATE_INTERVAL,
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
    safe_manifest_transfer,
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
    
    def __init__(self, project_dir: Path = None):
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
        Initialize the model index from Ollama.
        
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
        
        if not models:
            return []
        
        if model_name:
            validate_model_name(model_name)
            models = [m for m in models if m['name'] == model_name]
            if not models:
                raise ModelNotFoundError(f"Model '{model_name}' not found in Ollama.")
        
        initialized = []
        for model in models:
            model_name = model['name']
            blob_info = self._get_model_blob_info(model_name)
            if blob_info:
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=blob_info['blobSha'],
                    blob_name=blob_info['blobName'],
                    frozen=False,
                    compressed=False
                )
                initialized.append({
                    'name': model_name,
                    'blobSha': blob_info['blobSha']
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
        lock = _FileLock(dest_lock_path)
        
        try:
            lock.acquire()
            
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
                
                source_path.unlink()
                
                # Handle manifest file if it exists
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )
                
                if base_manifest_path.exists():
                    # Use safe_manifest_transfer for atomic transfer with rollback handling
                    try:
                        safe_manifest_transfer(base_manifest_path, remote_manifest_path, delete_source=True)
                    except FileOperationError as e:
                        raise FileOperationError(f"Failed to transfer manifest file: {e}")
                
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=True,
                    compressed=compress,
                    manifest_name=tag
                )
                self.index.save()
                
                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'compressed': compress
                }
                
            except Exception as e:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except:
                        pass
                raise FileOperationError(f"Error freezing model: {e}")
                
        finally:
            lock.release()
    
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
        lock = _FileLock(dest_lock_path)
        
        try:
            lock.acquire()
            
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
                
                source_path.unlink()
                
                # Handle manifest file if it exists
                model, tag = parse_model_name(model_name)
                base_manifest_path, remote_manifest_path = self._get_manifest_paths(
                    model, tag, base_storage, remote_storage
                )
                
                if remote_manifest_path.exists():
                    # Use safe_manifest_transfer for atomic transfer with rollback handling
                    try:
                        safe_manifest_transfer(remote_manifest_path, base_manifest_path, delete_source=True)
                    except FileOperationError as e:
                        raise FileOperationError(f"Failed to transfer manifest file: {e}")
                
                self.index.set_model(
                    model_name=model_name,
                    blob_sha=model_meta['blobSha'],
                    blob_name=model_meta['blobName'],
                    frozen=False,
                    compressed=is_compressed,
                    manifest_name=tag
                )
                self.index.save()
                
                return {
                    'success': True,
                    'model': model_name,
                    'blob_sha': model_meta['blobSha'],
                    'decompressed': is_compressed
                }
                
            except Exception as e:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except:
                        pass
                raise FileOperationError(f"Error thawing model: {e}")
                
        finally:
            lock.release()
    
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
        try:
            return get_ollama_models_secure()
        except SubprocessError:
            raise OllamaNotInstalledError("Ollama not found. Please install Ollama first.")
    
    def _get_model_blob_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """Get blob information for a model from its modelfile using secure subprocess wrapper."""
        try:
            result = get_model_blob_info_secure(model_name)
            if result is None:
                return None
            return result
        except SubprocessError:
            raise OllamaNotInstalledError("Ollama not found. Please install Ollama first.")
    
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
        
        # Build manifest paths relative to the storage directory
        manifest_dir = Path(MANIFEST_REGISTRY_PATH) / model
        
        # Create manifest paths within the storage directory hierarchy
        # Use the storage directory itself as the base, not its parent
        base_manifest_path = base_storage_path / MANIFEST_BASE_DIR / manifest_dir / tag
        remote_manifest_path = remote_storage_path / MANIFEST_BASE_DIR / manifest_dir / tag
        
        # Validate paths don't traverse outside expected directories
        validate_path_traversal(base_manifest_path, base_storage_path, "base manifest path")
        validate_path_traversal(remote_manifest_path, remote_storage_path, "remote manifest path")
        
        return base_manifest_path, remote_manifest_path
