"""Ollama Model Manager - omanage CLI tool and Python API."""

__version__ = "0.2.0"
__author__ = "Russell Miller"

from .utils import (
    OmanageError,
    ValidationError,
    PathTraversalError,
    InvalidModelNameError,
    ProgressBar,
)
from .index import OmanageIndexError
from .config import ConfigError
from .api import OmanageAPI
from .api_core.errors import OmanageAPIError, OllamaNotInstalledError, ModelNotFoundError, StorageNotConfiguredError, FileOperationError, ModelAlreadyFrozenError, ModelAlreadyThawedError

__all__ = [
    '__version__',
    '__author__',
    'OmanageError',
    'ValidationError',
    'PathTraversalError',
    'InvalidModelNameError',
    'ProgressBar',
    'OmanageIndexError',
    'ConfigError',
    'OmanageAPI',
    'OmanageAPIError',
    'OllamaNotInstalledError',
    'ModelNotFoundError',
    'StorageNotConfiguredError',
    'FileOperationError',
    'ModelAlreadyFrozenError',
    'ModelAlreadyThawedError',
]
