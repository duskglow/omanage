"""API core subpackage for omanage."""

from .errors import (
    OmanageAPIError,
    OllamaNotInstalledError,
    ModelNotFoundError,
    StorageNotConfiguredError,
    FileOperationError,
    ModelAlreadyFrozenError,
    ModelAlreadyThawedError,
)
from .locking import _FileLock
from .subprocess_utils import (
    SubprocessError,
    run_ollama_command,
    get_ollama_models,
    get_model_blob_info,
)
from .file_utils import (
    atomic_copy_with_lock,
    atomic_move_with_lock,
    atomic_copy_with_temp,
    create_secure_tempfile,
    safe_manifest_transfer,
    safe_delete,
)

__all__ = [
    'OmanageAPIError',
    'OllamaNotInstalledError',
    'ModelNotFoundError',
    'StorageNotConfiguredError',
    'FileOperationError',
    'ModelAlreadyFrozenError',
    'ModelAlreadyThawedError',
    '_FileLock',
    'SubprocessError',
    'run_ollama_command',
    'get_ollama_models',
    'get_model_blob_info',
    'atomic_copy_with_lock',
    'atomic_move_with_lock',
    'atomic_copy_with_temp',
    'create_secure_tempfile',
    'safe_manifest_transfer',
    'safe_delete',
]