"""Secure file utilities for omanage - atomic operations and safe file handling."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from ..utils import validate_path_traversal, PathTraversalError, CHUNK_SIZE
from .errors import FileOperationError


def atomic_copy_with_lock(src: Path, dst: Path) -> None:
    """
    Copy a file atomically using exclusive creation (O_EXCL flag).
    
    This function provides atomic file creation to prevent TOCTOU (Time-of-Check-Time-of-Use)
    vulnerabilities by using os.open with O_CREAT | O_EXCL flags.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Raises:
        FileOperationError: If the copy operation fails or the destination already exists.
    """
    src = src.resolve()
    dst = dst.resolve()
    
    # Validate destination is within expected directory
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Use os.open with O_CREAT | O_EXCL for atomic creation
    # This fails if the file already exists, preventing TOCTOU
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(dst, flags, 0o600)
        with os.fdopen(fd, 'wb') as dst_file:
            with src.open('rb') as src_file:
                while True:
                    chunk = src_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_file.write(chunk)
        
        # Verify copy integrity
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size
        if src_size != dst_size:
            dst.unlink()  # Clean up partial copy
            raise FileOperationError(
                f"Copy failed: size mismatch (src={src_size}, dst={dst_size})"
            )
            
    except FileExistsError:
        raise FileOperationError(f"Destination already exists: {dst}")
    except OSError as e:
        raise FileOperationError(f"Atomic copy failed: {e}")


def atomic_move_with_lock(src: Path, dst: Path) -> None:
    """
    Move a file atomically with proper cleanup on failure.
    
    This function attempts to use os.rename() for atomic moves (same filesystem),
    falling back to copy+delete if necessary (cross-filesystem).
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Raises:
        FileOperationError: If the move operation fails.
    """
    src = src.resolve()
    dst = dst.resolve()
    
    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try rename first (atomic on same filesystem)
        src.rename(dst)
    except OSError:
        # Fall back to copy+delete if rename fails (cross-filesystem)
        try:
            atomic_copy_with_lock(src, dst)
            src.unlink()
        except OSError as e:
            raise FileOperationError(f"Atomic move failed: {e}")


def atomic_copy_with_temp(src: Path, dst: Path) -> None:
    """
    Copy a file using a temporary file for atomicity.
    
    This function creates a temporary file in the same directory as the destination,
    copies the content, then renames the temp file to the destination atomically.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Raises:
        FileOperationError: If the copy operation fails.
    """
    src = src.resolve()
    dst = dst.resolve()
    
    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in the same directory as destination
    temp_dir = dst.parent
    fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=str(temp_dir))
    
    try:
        # Close the file descriptor (we'll reopen for writing)
        os.close(fd)
        
        # Copy content to temp file
        with src.open('rb') as src_file:
            with open(temp_path, 'wb') as dst_file:
                while True:
                    chunk = src_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_file.write(chunk)
        
        # Verify copy integrity
        src_size = src.stat().st_size
        temp_size = Path(temp_path).stat().st_size
        if src_size != temp_size:
            raise FileOperationError(
                f"Copy failed: size mismatch (src={src_size}, temp={temp_size})"
            )
        
        # Atomic rename from temp to destination
        os.rename(temp_path, dst)
        
    except Exception as e:
        # Clean up temp file on failure
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass
        raise FileOperationError(f"Atomic copy with temp failed: {e}")


def create_secure_tempfile(
    suffix: str = '',
    prefix: str = '',
    directory: Optional[Path] = None
) -> Path:
    """
    Create a secure temporary file with restrictive permissions.
    
    This function creates a temporary file with 0o600 (owner read/write only)
    permissions to prevent other users from accessing sensitive data.
    
    Args:
        suffix: Suffix for the temporary file name
        prefix: Prefix for the temporary file name
        directory: Directory to create the temp file in (uses system temp if None)
        
    Returns:
        Path to the created temporary file
        
    Raises:
        FileOperationError: If temp file creation fails.
    """
    try:
        dir_arg = str(directory) if directory else None
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir_arg)
        try:
            os.fchmod(fd, 0o600)  # Owner read/write only
        except OSError:
            os.close(fd)
            os.unlink(path)
            raise
        os.close(fd)
        return Path(path)
    except OSError as e:
        raise FileOperationError(f"Failed to create secure temp file: {e}")


def transfer_manifest_file(
    source: Path,
    dest: Path,
    delete_source: bool = True
) -> bool:
    """
    Transfer a manifest file with transactional guarantees.
    
    This function attempts to move the manifest file atomically, falling back to
    copy+delete if needed (cross-filesystem), with proper cleanup on failure.
    
    Args:
        source: Source manifest file path
        dest: Destination manifest file path
        delete_source: If True, delete the source file after successful transfer
        
    Returns:
        True if transfer was successful, False if source doesn't exist
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to move first (atomic on same filesystem)
        source.rename(dest)
        return True
    except FileNotFoundError:
        # Source doesn't exist
        return False
    except OSError:
        # Fall back to copy+delete if rename fails (cross-filesystem)
        pass
    
    # Copy with temp file for atomicity using secure tempfile
    fd, temp_path_str = tempfile.mkstemp(suffix='.tmp', dir=str(dest.parent))
    temp_path = Path(temp_path_str)
    try:
        os.close(fd)
        with source.open('rb') as src_file:
            with open(temp_path, 'wb') as dst_file:
                while True:
                    chunk = src_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst_file.write(chunk)
        
        # Verify copy integrity
        src_size = source.stat().st_size
        temp_size = temp_path.stat().st_size
        if src_size != temp_size:
            raise FileOperationError(
                f"Manifest transfer failed: size mismatch (src={src_size}, temp={temp_size})"
            )
        
        # Atomic rename
        temp_path.rename(dest)
        
        if delete_source:
            source.unlink()
        return True
        
    except OSError as e:
        # Clean up temp file on failure
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise FileOperationError(f"Manifest transfer failed: {e}")


def safe_delete(path: Path, missing_ok: bool = True) -> bool:
    """
    Safely delete a file with error handling.
    
    Args:
        path: Path to the file to delete
        missing_ok: If True, don't raise an error if the file doesn't exist
        
    Returns:
        True if file was deleted or didn't exist (when missing_ok=True), False otherwise
        
    Raises:
        FileOperationError: If deletion fails and missing_ok is False
    """
    try:
        if path.exists():
            path.unlink()
            return True
        return bool(missing_ok)
    except OSError as e:
        if missing_ok:
            return False
        raise FileOperationError(f"Failed to delete file: {e}")
