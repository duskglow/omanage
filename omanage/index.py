"""Index file handling for omanage."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigManager


class OmanageIndexError(Exception):
    """Index-related errors."""
    pass


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
                with open(self.index_file, 'r') as f:
                    self._index = json.load(f)
                    # Ensure models key exists
                    if "models" not in self._index:
                        self._index["models"] = {}
            except json.JSONDecodeError as e:
                raise OmanageIndexError(f"Invalid JSON in index file: {e}")
        
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
    
    def get_manifest_path(self, model_meta: dict, frozen: bool, config: ConfigManager) -> Path:
        """
        Get the expected path for a model's manifest file.
        
        Args:
            model_meta: Model metadata from index
            frozen: Whether the model is frozen
            config: ConfigManager instance
            
        Returns:
            Path to the manifest file
        """
        config.load()
        
        if frozen:
            # Manifest should be in remote storage
            remote_storage = config.get('remoteStorage')
            if not remote_storage:
                raise OmanageIndexError("remoteStorage not configured")
            manifest_name = model_meta.get('manifestName')
            if not manifest_name:
                raise OmanageIndexError("manifestName not found in model metadata")
            return Path(remote_storage) / manifest_name
        else:
            # Manifest should be in base storage
            base_storage = config.get('baseStorage')
            if not base_storage:
                raise OmanageIndexError("baseStorage not configured")
            manifest_name = model_meta.get('manifestName')
            if not manifest_name:
                raise OmanageIndexError("manifestName not found in model metadata")
            return Path(base_storage) / manifest_name


# Keep the class definition but with new name (removed duplicate)
