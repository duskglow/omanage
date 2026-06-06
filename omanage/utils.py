"""Utility functions for omanage."""

import gzip
import os
import re
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    'ProgressBar',
    'OmanageError',
    'ValidationError',
    'compress_file',
    'decompress_file',
    'detect_compression',
    'validate_file_not_empty',
    'move_with_progress',
    'validate_model_name',
    'validate_path_traversal',
    'SUPPORTED_MODEL_NAME_CHARS',
    'CHUNK_SIZE',
    'PROGRESS_UPDATE_INTERVAL',
]


# Constants for magic values
SUPPORTED_MODEL_NAME_CHARS = r'^[a-zA-Z0-9_\-:]+$'
MAX_MODEL_NAME_LENGTH = 256
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file operations
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds between progress updates


class OmanageError(Exception):
    """Base exception for omanage-related errors."""
    pass


class ValidationError(OmanageError):
    """Validation error for invalid file operations."""
    pass


class PathTraversalError(ValidationError):
    """Path traversal attempt detected."""
    pass


class InvalidModelNameError(ValidationError):
    """Invalid model name provided."""
    pass


class ProgressBar:
    """A simple progress bar for file operations."""
    
    # Class constant for progress bar width
    DEFAULT_BAR_WIDTH = 40
    
    def __init__(self, total_size: int, title: str = "Progress"):
        """
        Initialize the progress bar.
        
        Args:
            total_size: Total size of the operation in bytes
            title: Title to display above the progress bar
        
        Raises:
            ValueError: If total_size is negative
        """
        if total_size < 0:
            raise ValueError("total_size must be non-negative")
        self.total_size = total_size
        self.title = title
        self.bytes_transferred = 0
        self.start_time = None
        self.last_update = 0
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = time.time()
        self._update(0)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - ensures final update."""
        if exc_type is None:
            self._update(self.total_size)
        return False
    
    def update(self, size: int) -> None:
        """
        Update progress with additional bytes.
        
        Args:
            size: Number of bytes transferred since last update
        """
        self.bytes_transferred += size
        self._update(self.bytes_transferred)
    
    def finish(self) -> None:
        """Finish the progress bar."""
        self._update(self.total_size)
    
    def _update(self, current: int) -> None:
        """Update the display (internal)."""
        # Limit update frequency
        if time.time() - self.last_update < PROGRESS_UPDATE_INTERVAL:
            return
        self.last_update = time.time()
        
        # Calculate percentage and speed
        if self.total_size > 0:
            percent = (current / self.total_size) * 100
        else:
            percent = 100
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        # Use threshold to avoid division by zero and extreme speed values
        speed = current / elapsed if elapsed >= 0.01 else 0
        
        # Format progress bar using class constant
        bar_width = self.DEFAULT_BAR_WIDTH
        filled = int(bar_width * current / self.total_size) if self.total_size > 0 else bar_width
        bar = '#' * filled + '-' * (bar_width - filled)
        
        # Format size strings
        total_str = self._format_size(self.total_size)
        current_str = self._format_size(current)
        speed_str = self._format_size(speed) + '/s'
        
        # Display (overwrites previous line)
        # Safely check TTY status, handling closed file descriptors
        try:
            is_tty = sys.stdout.isatty() if not sys.stdout.closed else False
        except (ValueError, OSError):
            is_tty = False
        
        if is_tty:
            print(f'\r{self.title}: |{bar}| {percent:6.1f}% {current_str}/{total_str} {speed_str}', end='', flush=True)
        else:
            # Non-TTY: print periodically
            if int(percent) % 20 == 0:
                print(f'{self.title}: {percent:6.1f}% complete')
    
    def _format_size(self, size: int) -> str:
        """Format size in bytes to human-readable string."""
        current_size = float(size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if current_size < 1024:
                return f"{current_size:.1f} {unit}"
            current_size /= 1024
        return f"{current_size:.1f} PB"


def compress_file(source_path: Path, dest_path: Path, progress_bar: Optional[ProgressBar] = None) -> None:
    """
    Compress a file using gzip.
    
    Args:
        source_path: Path to source file
        dest_path: Path to compressed output
        progress_bar: Optional progress bar to update
    """
    with open(source_path, 'rb') as f_in:
        with gzip.open(dest_path, 'wb') as f_out:
            if progress_bar:
                # Read and compress in chunks for progress tracking
                while True:
                    chunk = f_in.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    progress_bar.update(len(chunk))
            else:
                # Simple copy without progress
                shutil.copyfileobj(f_in, f_out)


def decompress_file(source_path: Path, dest_path: Path, progress_bar: Optional[ProgressBar] = None) -> None:
    """
    Decompress a gzip file.
    
    Args:
        source_path: Path to compressed source file
        dest_path: Path to decompressed output
        progress_bar: Optional progress bar to update
        
    Raises:
        ValidationError: If the file is not a valid gzip archive.
    """
    try:
        with gzip.open(source_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                if progress_bar:
                    # Read and decompress in chunks
                    while True:
                        chunk = f_in.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        progress_bar.update(len(chunk))
                else:
                    # Simple copy without progress
                    shutil.copyfileobj(f_in, f_out)
    except (gzip.BadGzipFile, OSError, zlib.error) as e:
        raise ValidationError(f"Failed to decompress {source_path}: not a valid gzip file") from e


def validate_file_not_empty(source_path: Path) -> None:
    """
    Validate that a file is not empty (zero-length).
    
    Args:
        source_path: Path to the file to validate
        
    Raises:
        ValidationError: If the file is empty (zero-length)
    """
    file_size = source_path.stat().st_size
    if file_size == 0:
        raise ValidationError(f"Cannot process zero-length file: {source_path}")


def detect_compression(source_path: Path) -> bool:
    """
    Detect if a file is gzip compressed by checking magic bytes.
    
    Performs a more thorough check by validating the full gzip header
    (magic bytes + compression method) to reduce false positives.
    
    Args:
        source_path: Path to file to check
        
    Returns:
        True if file appears to be a valid gzip compressed file
    """
    try:
        with open(source_path, 'rb') as f:
            header = f.read(3)
            if len(header) < 3:
                return False
            # Gzip magic bytes: 0x1f 0x8b, compression method: 0x08 (deflate)
            return header[0:2] == b'\x1f\x8b' and header[2] == 0x08
    except (IOError, OSError):
        return False


def move_with_progress(src: Path, dst: Path, title: str = "Moving") -> None:
    """
    Move a file with progress bar.
    
    Uses atomic copy utilities to ensure data integrity and proper
    cleanup on failure.
    
    Args:
        src: Source file path
        dst: Destination file path
        title: Title for progress bar
    """
    from .api_core.file_utils import atomic_copy_with_temp
    
    total_size = src.stat().st_size
    
    with ProgressBar(total_size, title) as pb:
        # Use atomic copy for data integrity
        atomic_copy_with_temp(src, dst)
        pb.update(total_size)
        
        # Remove source after successful copy
        try:
            src.unlink()
        except OSError as e:
            # Do NOT delete the destination — the copy succeeded and it's the only
            # remaining copy if source deletion failed. The inconsistency is
            # recoverable (user can manually delete source later).
            raise OSError(f"Failed to remove source file after copy: {e}")


def validate_model_name(model_name: str) -> None:
    """
    Validate a model name contains only safe characters.
    
    Args:
        model_name: Model name to validate
        
    Raises:
        InvalidModelNameError: If model name contains invalid characters
        ValueError: If model name is empty or None
    """
    if not model_name:
        raise InvalidModelNameError("Model name cannot be empty")
    
    if len(model_name) > MAX_MODEL_NAME_LENGTH:
        raise InvalidModelNameError(
            f"Model name too long ({len(model_name)} characters, max {MAX_MODEL_NAME_LENGTH})"
        )
    
    if not re.match(SUPPORTED_MODEL_NAME_CHARS, model_name):
        raise InvalidModelNameError(
            f"Invalid model name: '{model_name}'. "
            f"Model names can only contain letters, numbers, underscores, hyphens, and colons."
        )


def validate_path_traversal(path: Path, base_path: Path, description: str = "path") -> Path:
    """
    Validate that a path does not traverse outside its expected base directory.
    
    Uses absolute() instead of resolve() to avoid following symlinks during validation,
    preventing TOCTOU race conditions where a symlink target is changed between
    validation and file operations.
    
    Args:
        path: The path to validate
        base_path: The base directory the path should be within
        description: Description of what is being validated (for error messages)
        
    Returns:
        The validated path if valid
        
    Raises:
        PathTraversalError: If the path attempts to traverse outside base_path
    """
    try:
        # Use absolute() instead of resolve() to avoid following symlinks
        # This prevents TOCTOU attacks where a symlink is changed between check and use
        resolved_path = path.absolute()
        resolved_base = base_path.absolute()
        
        # Normalize paths for comparison
        resolved_path = Path(os.path.normpath(resolved_path))
        resolved_base = Path(os.path.normpath(resolved_base))
        
        # Check if path is within base_path using proper path containment check
        try:
            resolved_path.relative_to(resolved_base)
        except ValueError:
            raise PathTraversalError(
                f"Path traversal attempt detected for {description}: "
                f"'{path}' resolves to '{resolved_path}', which is outside '{resolved_base}'"
            )
        
        return resolved_path
    except (OSError, RuntimeError) as e:
        # Handle cases where absolute() might fail
        raise PathTraversalError(
            f"Could not validate {description} path '{path}': {e}"
        )


def parse_model_name(model_name: str) -> Tuple[str, str]:
    """
    Parse a model name into its components (model, tag).
    
    Args:
        model_name: Model name in format "name:tag" or "name"
        
    Returns:
        Tuple of (model, tag) where tag defaults to 'latest' if not specified
        
    Raises:
        InvalidModelNameError: If model_name is empty or invalid
    """
    if not model_name:
        raise InvalidModelNameError("Model name cannot be empty")
    
    if ':' in model_name:
        # Reject multiple colons to prevent ambiguous parsing
        if model_name.count(':') > 1:
            raise InvalidModelNameError("Model name can contain at most one colon")
        model_parts = model_name.split(':', 1)
        model = model_parts[0]
        tag = model_parts[1]
    else:
        model = model_name
        tag = 'latest'
    
    # Validate model and tag components separately
    if not re.match(r'^[a-zA-Z0-9_\-]+$', model):
        raise InvalidModelNameError(f"Invalid model component: '{model}'")
    
    if not re.match(r'^[a-zA-Z0-9_\-]+$', tag):
        raise InvalidModelNameError(f"Invalid tag component: '{tag}'")
    
    return model, tag


def sanitize_for_path_component(value: str) -> str:
    """
    Sanitize a string for use in a file path component.
    
    Args:
        value: String to sanitize
        
    Returns:
        Sanitized string safe for use in file paths
    """
    # Replace any characters that aren't safe for filenames
    safe_value = re.sub(r'[^a-zA-Z0-9_\-]', '_', value)
    return safe_value