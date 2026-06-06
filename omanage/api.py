"""Python API for omanage - Programmatic access to Ollama model management."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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


# Constants for magic values
LOCK_FILE_SUFFIX = '.lock'
MANIFEST_BASE_DIR = "manifests"
MANIFEST_REGISTRY_PATH = "registry.ollama.ai/library"

# Cross-platform file locking
# fcntl is Unix-only, use a dummy context manager on Windows
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


class _DummyLock:
    """Dummy lock context manager for Windows (fcntl not available)."""
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._lock_file = None
    
    def __enter__(self):
        # Create lock file for Windows (advisory locking)
        if not self.lock_path.exists():
            self.lock_path.touch()
        self._lock_file = self.lock_path.open('w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            self._lock_file.close()
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except:
                pass
        return False


class OmanageAPIError(OmanageError):
    """Base exception for omanage API errors."""
    pass


class OllamaNotInstalledError(OmanageAPIError):
    """Ollama CLI is not installed or not in PATH."""
    pass


class ModelNotFoundError(OmanageAPIError):
    """Model not found in index."""
    pass


class StorageNotConfiguredError(OmanageAPIError):
    """Storage paths not configured."""
    pass


class FileOperationError(OmanageAPIError):
    """File operation failed."""
    pass


class ModelAlreadyFrozenError(OmanageAPIError):
    """Model is already in frozen state."""
    pass


class ModelAlreadyThawedError(OmanageAPIError):
    """Model is already in thawed state."""
    pass


class _FileLock:
    """Cross-platform file locking context manager."""
    
    def __init__(self, lock_path: Path, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_file = None
    
    def acquire(self) -> bool:
        """Acquire the lock."""
        import time
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if not self.lock_path.exists():
                try:
                    # Create lock file atomically
                    self.lock_path.touch(exist_ok=False)
                    self._lock_file = self.lock_path.open('w')
                    if HAS_FCNTL and self._lock_file:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
                    return True
                except (FileExistsError, OSError):
                    time.sleep(0.1)
                    continue
        
        return False
    
    def release(self) -> None:
        """Release the lock."""
        if self._lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            except:
                pass
            self._lock_file.close()
            self._lock_file = None
        
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except:
                pass
    
    def __enter__(self) -> '_FileLock':
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False


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
                    remote_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create a temp file for atomic operation
                    temp_manifest_path = dest_path.parent / f"{dest_path.name}.manifest.tmp"
                    shutil.copy2(base_manifest_path, temp_manifest_path)
                    temp_manifest_path.rename(remote_manifest_path)
                    base_manifest_path.unlink()
                
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
                    base_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create a temp file for atomic operation
                    temp_manifest_path = base_path / f"{base_path.name}.manifest.tmp"
                    shutil.copy2(remote_manifest_path, temp_manifest_path)
                    temp_manifest_path.rename(base_manifest_path)
                    remote_manifest_path.unlink()
                
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
        """Get list of installed Ollama models."""
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
            raise OllamaNotInstalledError("Ollama not found. Please install Ollama first.")
    
    def _get_model_blob_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """Get blob information for a model from its modelfile."""
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
        
        # Build manifest paths
        manifest_dir = Path(MANIFEST_REGISTRY_PATH) / model
        
        base_manifest_path = Path(base_storage).parent / MANIFEST_BASE_DIR / manifest_dir / tag
        remote_manifest_path = Path(remote_storage).parent / MANIFEST_BASE_DIR / manifest_dir / tag
        
        # Validate paths don't traverse
        base_storage_path = Path(base_storage).resolve()
        validate_path_traversal(base_manifest_path, base_storage_path.parent, "base manifest path")
        validate_path_traversal(remote_manifest_path, Path(remote_storage).resolve().parent, "remote manifest path")
        
        return base_manifest_path, remote_manifest_path