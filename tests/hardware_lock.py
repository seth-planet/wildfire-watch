"""Hardware lock utilities for exclusive hardware access in tests.

This module provides locking mechanisms to ensure tests that require
exclusive hardware access (Coral TPU, Hailo, etc.) don't run in parallel.
"""

import os
import time
import fcntl
import tempfile
from pathlib import Path
from contextlib import contextmanager
import pytest


class HardwareLock:
    """File-based lock for hardware resources."""
    
    def __init__(self, resource_name: str, timeout: float = 30.0):
        self.resource_name = resource_name
        self.timeout = timeout
        self.lock_dir = Path(tempfile.gettempdir()) / "wildfire_test_locks"
        self.lock_dir.mkdir(exist_ok=True)
        self.lock_file = self.lock_dir / f"{resource_name}.lock"
        self.lock_fd = None
        
    def acquire(self):
        """Acquire the hardware lock with timeout."""
        start_time = time.time()
        
        # Open lock file
        self.lock_fd = open(self.lock_file, 'w')
        
        while True:
            try:
                # Try to acquire exclusive lock
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID for debugging
                self.lock_fd.write(f"{os.getpid()}\n")
                self.lock_fd.flush()
                return True
            except IOError:
                # Lock is held by another process
                if time.time() - start_time > self.timeout:
                    self.release()
                    raise TimeoutError(f"Failed to acquire {self.resource_name} lock after {self.timeout}s")
                time.sleep(0.1)
    
    def release(self):
        """Release the hardware lock."""
        if self.lock_fd:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                self.lock_fd.close()
            except:
                pass
            self.lock_fd = None
    
    def __enter__(self):
        self.acquire()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


@pytest.fixture
def coral_tpu_lock():
    """Fixture for exclusive Coral TPU access."""
    with HardwareLock("coral_tpu", timeout=60.0) as lock:
        yield lock


@pytest.fixture
def hailo_lock():
    """Fixture for exclusive Hailo device access."""
    with HardwareLock("hailo", timeout=60.0) as lock:
        yield lock


@pytest.fixture
def gpu_lock():
    """Fixture for exclusive GPU access."""
    with HardwareLock("gpu", timeout=60.0) as lock:
        yield lock


@pytest.fixture
def camera_lock():
    """Fixture for exclusive camera hardware access."""
    with HardwareLock("camera", timeout=30.0) as lock:
        yield lock


@contextmanager
def hardware_resource(resource_name: str, timeout: float = 30.0):
    """Context manager for exclusive hardware access.
    
    Usage:
        with hardware_resource("coral_tpu"):
            # Use Coral TPU exclusively
            pass
    """
    lock = HardwareLock(resource_name, timeout)
    try:
        lock.acquire()
        yield lock
    finally:
        lock.release()