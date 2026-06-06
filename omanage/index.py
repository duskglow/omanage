"""Index file handling for omanage."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigManager


class OmanageIndexError(Exception):
    """Index-related errors."""
    pass


# SHA256 regex pattern for blob validation
_SHA256_PATTERN = re.compile(r'^[a-fA-F0-9]{64}$')
# Maximum file size for index JSON (10MB)
_MAX_INDEX_SIZE = 10 * 1024 * 1024


def validate_blob_sha(blob_sha: str) -> None:
    """Validate that a blob SHA has the correct format."""
    if not blob_sha or not _SHA256_PATTERN.match(blob_sha):
        raise OmanageIndexError(f"Invalid blob SHA format: {blob_sha}")


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
                
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
                    # Ensure models key exists
                    if "models" not in self._index:
                        self._index["models"] = {}
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
        
        # Validate blob_sha format
        validate_blob_sha(blob_sha)
        
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
        with open(self.index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
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
    
