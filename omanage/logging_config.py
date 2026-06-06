"""Logging configuration for omanage."""

import logging
import sys
from typing import Optional

# Module-level logger cache
_loggers: dict = {}

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the given module name.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Only add handler if none exists to avoid duplicates
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        logger.addHandler(handler)
    
    _loggers[name] = logger
    return logger