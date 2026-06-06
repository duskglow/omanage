"""Configuration file handling for omanage."""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ValidationError, PathTraversalError


class ConfigManager:
    """Manages the .omanage.conf configuration file."""
    
    CONFIG_FILE_NAME = ".omanage.conf"
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "ollamaBinary": "ollama",
        "baseStorage": "",
        "remoteStorage": ""
    }
    
    # Valid configuration keys
    VALID_KEYS = {"ollamaBinary", "baseStorage", "remoteStorage"}
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the config manager.
        
        Args:
            config_dir: Directory to look for config file. If None, uses current working directory.
        """
        self.config_dir = config_dir or Path.cwd()
        self.config_file = self.config_dir / self.CONFIG_FILE_NAME
        self._config: Dict[str, Any] = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file if it exists.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_file.exists():
            self._config = self.DEFAULT_CONFIG.copy()
        else:
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
                    # Merge with defaults for any missing keys
                    for key, value in self.DEFAULT_CONFIG.items():
                        if key not in self._config:
                            self._config[key] = value
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in config file: {e}")
        
        self._loaded = True
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if not self._loaded:
            self.load()
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value with validation.
        
        Args:
            key: Configuration key to set
            value: Value to set
            
        Raises:
            ConfigError: If key is invalid or value is invalid
            ValidationError: If path validation fails
        """
        if not self._loaded:
            self.load()
        
        # Validate key
        if key not in self.VALID_KEYS:
            raise ConfigError(f"Invalid configuration key: '{key}'. Valid keys are: {', '.join(sorted(self.VALID_KEYS))}")
        
        # Validate and normalize path values
        if key in ("baseStorage", "remoteStorage"):
            if value:
                path = Path(value)
                if not path.exists():
                    raise ConfigError(f"Storage path does not exist: {value}")
                # Normalize to absolute path
                self._config[key] = str(path.resolve())
            else:
                self._config[key] = ""
        
        # Validate ollama binary
        if key == "ollamaBinary":
            if value:
                # Try to find the binary in PATH
                binary_path = shutil.which(value)
                if not binary_path:
                    raise ConfigError(f"Ollama binary not found in PATH: {value}")
                self._config[key] = value
            else:
                self._config[key] = "ollama"
        
        self._config[key] = value
    
    def save(self) -> None:
        """Save configuration to file."""
        if not self._loaded:
            self.load()
        
        # Write config with pretty formatting
        with open(self.config_file, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def exists(self) -> bool:
        """Check if config file exists."""
        return self.config_file.exists()
    
    def initialize(self) -> None:
        """Initialize config file with defaults if it doesn't exist."""
        if not self.exists():
            self._config = self.DEFAULT_CONFIG.copy()
            self.save()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        if not self._loaded:
            self.load()
        return self._config


class ConfigError(Exception):
    """Configuration-related errors."""
    pass


class ValidationError(OSError):
    """Validation error for invalid configuration values."""
    pass