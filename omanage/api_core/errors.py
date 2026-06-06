"""Exception classes for omanage API."""

from ..utils import OmanageError


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