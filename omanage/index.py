"""Index file handling for omanage."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigManager
from .utils import validate_model_name, InvalidModelNameError


class OmanageIndexError(Exception):
    """Index-related errors."""
    pass


# SHA256 regex pattern for blob validation
# Ollama may report blob identifiers as either a bare 64-char hex digest or
# with a "sha256-" prefix (e.g., sha256-1b2b95e2...).
_SHA256_PATTERN = re.compile(r'^(?:sha256-)?[a-fA-F0-9]{64}$', re.IGNORECASE)
# Maximum file size for index JSON (10MB)
_MAX_INDEX_SIZE = 10 * 1024 * 1024


def validate_blob_sha(blob_sha: str) -> None:
    """Validate that a blob SHA has the correct format."""
    if not blob_sha or not _SHA256_PATTERN.match(blob_sha):
        raise OmanageIndexError(f"Invalid blob SHA format: {blob_sha}")


# Regex for safe blob names (prevents path traversal via blobName)
_BLOB_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')


def validate_blob_name(blob_name: str) -> None:
    """Validate that a blob name contains only safe characters."""
    if not blob_name or not _BLOB_NAME_PATTERN.match(blob_name):
        raise OmanageIndexError(
            f"Invalid blob name format: '{blob_name}'. "
            f"Blob names can only contain letters, numbers, underscores, hyphens, and dots."
        )


class IndexManager:
    """Manages the .omanage.index.json model metadata index."""
    
    INDEX_FILE_NAME = ".omanage.index.json"
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the index manager.
        
        Args:
            config_dir: Directory to look for index file. If None, uses current working directory.
        """
        self.config_dir = config_dir or Path.cwd()
        self.index_file = self.config_dir / self.INDEX_FILE_NAME
        self._index: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        """
        Load index from file if it exists.
        
        Returns:
            Index dictionary
        """
        if not self.index_file.exists():
            self._index = {"models": {}}
        else:
            try:
                # Check file size to prevent memory exhaustion from malicious files
                file_size = self.index_file.stat().st_size
                if file_size > _MAX_INDEX_SIZE:
                    raise OmanageIndexError(
                        f"Index file too large: {file_size} bytes (max {_MAX_INDEX_SIZE})"
                    )
                
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                
                # Schema validation: ensure correct structure and types
                if not isinstance(raw, dict):
                    raise OmanageIndexError("Index file must contain a JSON object")
                if "models" not in raw:
                    raw["models"] = {}
                if not isinstance(raw["models"], dict):
                    raise OmanageIndexError("Index 'models' field must be an object")
                
                # Validate each model entry has required string fields
                for model_name, meta in raw["models"].items():
                    if not isinstance(meta, dict):
                        raise OmanageIndexError(f"Model '{model_name}' metadata must be an object")
                    for field in ("blobSha", "blobName"):
                        if field in meta and not isinstance(meta[field], str):
                            raise OmanageIndexError(
                                f"Model '{model_name}' field '{field}' must be a string"
                            )
                    for field in ("frozen", "compressed"):
                        if field in meta and not isinstance(meta[field], bool):
                            raise OmanageIndexError(
                                f"Model '{model_name}' field '{field}' must be a boolean"
                            )
                
                self._index = raw
            except json.JSONDecodeError as e:
                raise OmanageIndexError(f"Invalid JSON in index file: {e}") from e
        
        self._loaded = True
        return self._index
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by name."""
        if not self._loaded:
            self.load()
        return self._index["models"].get(model_name)
    
    def set_model(self, model_name: str, blob_sha: str, blob_name: str, 
                  frozen: bool = False, compressed: bool = False,
                  manifest_name: Optional[str] = None) -> None:
        """
        Set or update model metadata.
        
        Args:
            model_name: Name of the model
            blob_sha: SHA256 hash of the blob
            blob_name: Name of the blob file
            frozen: Whether the model is frozen
            compressed: Whether the blob is compressed
            manifest_name: Name of the manifest file (without path)
        """
        if not self._loaded:
            self.load()
        
        # Validate model name to prevent injection of arbitrary keys
        validate_model_name(model_name)
        
        # Validate blob_sha format
        validate_blob_sha(blob_sha)
        
        # Validate blob_name format for defense-in-depth
        validate_blob_name(blob_name)
        
        self._index["models"][model_name] = {
            "blobSha": blob_sha,
            "blobName": blob_name,
            "frozen": frozen,
            "compressed": compressed,
            "manifestName": manifest_name
        }
    
    def remove_model(self, model_name: str) -> bool:
        """
        Remove a model from the index.
        
        Returns:
            True if model was removed, False if it didn't exist
        """
        if not self._loaded:
            self.load()
        
        if model_name in self._index["models"]:
            del self._index["models"][model_name]
            return True
        return False
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all models in the index."""
        if not self._loaded:
            self.load()
        return self._index["models"]
    
    def save(self) -> None:
        """Save index to file."""
        if not self._loaded:
            self.load()
        
        # Write index with pretty formatting
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2)
        
        # Enforce restrictive permissions (owner read/write only)
        try:
            os.chmod(self.index_file, 0o600)
        except OSError:
            pass
    
    def exists(self) -> bool:
        """Check if index file exists."""
        return self.index_file.exists()
    
    def initialize(self) -> None:
        """Initialize index file with empty models if it doesn't exist."""
        if not self.exists():
            self._index = {"models": {}}
            self.save()
    
    @property
    def index(self) -> Dict[str, Any]:
        """Get the full index dictionary."""
        if not self._loaded:
            self.load()
        return self._index