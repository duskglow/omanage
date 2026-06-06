"""Cross-platform file locking utilities for omanage."""

import os
import time
from pathlib import Path
from typing import Optional

# fcntl is Unix-only, check availability at import time
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# msvcrt is Windows-only, check availability at import time
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class _FileLock:
    """Cross-platform file locking context manager."""

    def __init__(self, lock_path: Path, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_fd: Optional[int] = None

    def acquire(self) -> bool:
        """Acquire the lock with timeout.

        Returns:
            True if lock acquired, False on timeout.
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            try:
                # Use atomic file creation with O_EXCL to prevent TOCTOU race
                # This fails immediately if the lock file already exists
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self._lock_fd = fd

                # Apply platform-specific locking on the file descriptor
                if HAS_FCNTL:
                    # Unix: Use advisory flock for inter-process locking
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif HAS_MSVCRT:
                    # Windows: Use msvcrt.locking for file locking
                    try:
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    except OSError:
                        # If locking fails, clean up and retry
                        os.close(fd)
                        self._lock_fd = None
                        try:
                            self.lock_path.unlink()
                        except OSError:
                            pass
                        time.sleep(0.1)
                        continue

                return True

            except FileExistsError:
                # Lock file already exists - another process holds the lock
                time.sleep(0.1)
                continue
            except OSError:
                # Other filesystem errors - retry after brief delay
                time.sleep(0.1)
                continue

        return False

    def release(self) -> None:
        """Release the lock and clean up the lock file."""
        if self._lock_fd is not None:
            try:
                if HAS_FCNTL:
                    try:
                        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                elif HAS_MSVCRT:
                    try:
                        msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
            finally:
                try:
                    os.close(self._lock_fd)
                except OSError:
                    pass
                self._lock_fd = None

        # Clean up lock file
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except OSError:
                # Lock file may be held by another process or already removed
                pass

    def __enter__(self) -> '_FileLock':
        if not self.acquire():
            from .errors import FileOperationError
            raise FileOperationError(
                f"Failed to acquire lock: {self.lock_path}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False