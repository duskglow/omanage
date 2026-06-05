"""Ollama Model Manager - omanage CLI tool and Python API."""

__version__ = "0.1.0"
__author__ = "Russell Miller"

from .utils import OmanageError, ValidationError, ProgressBar
from .index import OmanageIndexError
from .config import ConfigError
from .api import (
    OmanageAPI,
    OmanageAPIError,
    OllamaNotInstalledError,
    ModelNotFoundError,
    StorageNotConfiguredError,
    FileOperationError
)

__all__ = [
    '__version__',
    '__author__',
    'OmanageError',
    'ValidationError',
    'ProgressBar',
    'OmanageIndexError',
    'ConfigError',
    'OmanageAPI',
    'OmanageAPIError',
    'OllamaNotInstalledError',
    'ModelNotFoundError',
    'StorageNotConfiguredError',
    'FileOperationError',
]