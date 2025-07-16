"""
Hardware lock utilities for test isolation.

Provides exclusive access to hardware resources (Coral TPU, Hailo) during testing
to prevent conflicts when tests run in parallel.
"""
import os
import time
import fcntl
import contextlib
from typing import Optional


@contextlib.contextmanager
def hardware_lock(hardware_name: str, timeout: int = 1800):
    """
    Context manager for hardware exclusive access.
    
    Args:
        hardware_name: Name of hardware resource (e.g., 'coral_tpu', 'hailo')
        timeout: Maximum time to wait for lock in seconds (default: 30 minutes)
    
    Yields:
        None
        
    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_file = f"/tmp/wildfire_watch_{hardware_name}_lock"
    lock_fd = None
    
    try:
        # Create lock file
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        
        # Try to acquire lock with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write process info to lock file
                os.write(lock_fd, f"PID:{os.getpid()}\nTime:{time.time()}\n".encode())
                yield
                return
            except BlockingIOError:
                time.sleep(0.1)
        
        # Timeout reached
        raise TimeoutError(f"Could not acquire {hardware_name} lock within {timeout} seconds")
        
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                os.unlink(lock_file)
            except:
                pass


@contextlib.contextmanager
def coral_tpu_lock(timeout: int = 1800):
    """
    Acquire exclusive lock for Coral TPU access.
    
    Args:
        timeout: Maximum time to wait for lock in seconds
        
    Yields:
        None
    """
    with hardware_lock("coral_tpu", timeout):
        yield


@contextlib.contextmanager
def hailo_lock(timeout: int = 1800):
    """
    Acquire exclusive lock for Hailo accelerator access.
    
    Args:
        timeout: Maximum time to wait for lock in seconds
        
    Yields:
        None
    """
    with hardware_lock("hailo", timeout):
        yield


def requires_hardware_lock(hardware_name: str, timeout: int = 1800):
    """
    Decorator for test methods that require exclusive hardware access.
    
    Args:
        hardware_name: Name of hardware resource
        timeout: Maximum time to wait for lock
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with hardware_lock(hardware_name, timeout):
                return func(*args, **kwargs)
        return wrapper
    return decorator