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


class _DummyLock:
    """Dummy lock context manager for Windows (fcntl not available)."""
    
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._lock_file = None
    
    def __enter__(self):
        # Create lock file for Windows (advisory locking)
        if not self.lock_path.exists():
            self.lock_path.touch()
        self._lock_file = self.lock_path.open('w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_file:
            self._lock_file.close()
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except:
                pass
        return False


class _FileLock:
    """Cross-platform file locking context manager."""
    
    def __init__(self, lock_path: Path, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_file = None
    
    def acquire(self) -> bool:
        """Acquire the lock."""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if not self.lock_path.exists():
                try:
                    # Create lock file atomically
                    self.lock_path.touch(exist_ok=False)
                    self._lock_file = self.lock_path.open('w')
                    if HAS_FCNTL:
                        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
                    return True
                except (FileExistsError, OSError):
                    time.sleep(0.1)
                    continue
        
        return False
    
    def release(self) -> None:
        """Release the lock."""
        if self._lock_file:
            try:
                if HAS_FCNTL:
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            except:
                pass
            self._lock_file.close()
            self._lock_file = None
        
        if self.lock_path.exists():
            try:
                self.lock_path.unlink()
            except:
                pass
    
    def __enter__(self) -> '_FileLock':
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.release()
        return False