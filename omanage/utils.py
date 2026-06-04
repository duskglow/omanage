"""Utility functions for omanage."""

import gzip
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional


__all__ = [
    'ProgressBar',
    'OmanageError',
    'ValidationError',
    'compress_file',
    'decompress_file',
    'detect_compression',
    'validate_file_not_empty',
    'move_with_progress',
]


class OmanageError(Exception):
    """Base exception for omanage-related errors."""
    pass


class ValidationError(OmanageError):
    """Validation error for invalid file operations."""
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
        if time.time() - self.last_update < 0.1:
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
        if sys.stdout.isatty():
            print(f'\r{self.title}: |{bar}| {percent:6.1f}% {current_str}/{total_str} {speed_str}', end='', flush=True)
        else:
            # Non-TTY: print periodically
            if int(percent) % 20 == 0:
                print(f'{self.title}: {percent:6.1f}% complete')
    
    def _format_size(self, size: int) -> str:
        """Format size in bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


def compress_file(source_path: Path, dest_path: Path, progress_bar: Optional[ProgressBar] = None) -> None:
    """
    Compress a file using gzip.
    
    Args:
        source_path: Path to source file
        dest_path: Path to compressed output
        progress_bar: Optional progress bar to update
    """
    source_size = source_path.stat().st_size
    
    with open(source_path, 'rb') as f_in:
        with gzip.open(dest_path, 'wb') as f_out:
            if progress_bar:
                # Read and compress in chunks for progress tracking
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f_in.read(chunk_size)
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
    """
    source_size = source_path.stat().st_size
    
    with gzip.open(source_path, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            if progress_bar:
                # Read and decompress in chunks
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    progress_bar.update(len(chunk))
            else:
                # Simple copy without progress
                shutil.copyfileobj(f_in, f_out)


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
    
    Args:
        source_path: Path to file to check
        
    Returns:
        True if file appears to be gzip compressed
    """
    try:
        with open(source_path, 'rb') as f:
            magic_bytes = f.read(2)
            # Gzip magic bytes: 0x1f 0x8b
            return magic_bytes == b'\x1f\x8b'
    except (IOError, OSError):
        return False


def move_with_progress(src: Path, dst: Path, title: str = "Moving") -> None:
    """
    Move a file with progress bar.
    
    Args:
        src: Source file path
        dst: Destination file path
        title: Title for progress bar
    """
    total_size = src.stat().st_size
    
    with ProgressBar(total_size, title) as pb:
        # Copy with progress
        with open(src, 'rb') as f_in:
            with open(dst, 'wb') as f_out:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pb.update(len(chunk))
        
        # Remove source after successful copy
        src.unlink()