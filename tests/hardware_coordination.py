#!/usr/bin/env python3.12
"""
Hardware Test Coordination for Wildfire Watch

This module provides lock-based coordination for hardware tests to prevent
concurrent access to exclusive hardware resources like Coral TPU and Hailo.

Key Features:
- File-based locking for cross-process coordination
- Timeout handling for stuck locks
- Hardware-specific lock management
- Integration with pytest-xdist for parallel testing
"""

import os
import time
import json
import fcntl
import signal
import logging
import tempfile
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Lock directory
LOCK_DIR = Path(tempfile.gettempdir()) / "wildfire_test_locks"
LOCK_DIR.mkdir(exist_ok=True)

# Hardware types that require exclusive access
EXCLUSIVE_HARDWARE = ['coral_tpu', 'hailo', 'rpi_gpio']

# Lock timeout defaults (seconds)
DEFAULT_LOCK_TIMEOUT = 300  # 5 minutes
STALE_LOCK_THRESHOLD = 600  # 10 minutes


class HardwareLockError(Exception):
    """Exception raised when hardware lock cannot be acquired"""
    pass


class HardwareLock:
    """
    File-based lock for hardware resource coordination
    """
    
    def __init__(self, hardware_type: str, worker_id: str = 'master'):
        self.hardware_type = hardware_type
        self.worker_id = worker_id
        self.lock_file = LOCK_DIR / f"{hardware_type}.lock"
        self.lock_info_file = LOCK_DIR / f"{hardware_type}.info"
        self.lock_fd = None
        self.is_locked = False
        self.lock_start_time = None
        
    def acquire(self, timeout: float = DEFAULT_LOCK_TIMEOUT, retry_interval: float = 0.5) -> bool:
        """
        Acquire the hardware lock
        
        Args:
            timeout: Maximum time to wait for lock
            retry_interval: Time between lock attempts
            
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        
        # Check for stale locks first
        self._check_and_clear_stale_lock()
        
        while time.time() - start_time < timeout:
            try:
                # Try to acquire lock
                self.lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Lock acquired, write info
                self.is_locked = True
                self.lock_start_time = time.time()
                self._write_lock_info()
                
                logger.info(f"Worker {self.worker_id} acquired {self.hardware_type} lock")
                return True
                
            except (IOError, OSError):
                # Lock is held by another process
                if self.lock_fd:
                    self.lock_fd.close()
                    self.lock_fd = None
                
                # Check who has the lock
                lock_info = self._read_lock_info()
                if lock_info:
                    logger.debug(
                        f"Worker {self.worker_id} waiting for {self.hardware_type} lock "
                        f"(held by {lock_info.get('worker_id', 'unknown')})"
                    )
                
                time.sleep(retry_interval)
        
        # Timeout reached
        logger.warning(
            f"Worker {self.worker_id} failed to acquire {self.hardware_type} lock "
            f"after {timeout}s"
        )
        return False
    
    def release(self):
        """Release the hardware lock"""
        if not self.is_locked or not self.lock_fd:
            return
            
        try:
            # Release the lock
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            
            # Remove lock info file
            try:
                self.lock_info_file.unlink()
            except FileNotFoundError:
                pass
                
            duration = time.time() - self.lock_start_time if self.lock_start_time else 0
            logger.info(
                f"Worker {self.worker_id} released {self.hardware_type} lock "
                f"(held for {duration:.1f}s)"
            )
            
        except Exception as e:
            logger.error(f"Error releasing {self.hardware_type} lock: {e}")
        finally:
            self.is_locked = False
            self.lock_start_time = None
    
    def _write_lock_info(self):
        """Write lock information for debugging"""
        info = {
            'worker_id': self.worker_id,
            'pid': os.getpid(),
            'timestamp': time.time(),
            'start_time': datetime.now().isoformat(),
        }
        
        try:
            with open(self.lock_info_file, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write lock info: {e}")
    
    def _read_lock_info(self) -> Optional[Dict[str, Any]]:
        """Read lock information"""
        try:
            with open(self.lock_info_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _check_and_clear_stale_lock(self):
        """Check for and clear stale locks from dead processes"""
        lock_info = self._read_lock_info()
        if not lock_info:
            return
            
        # Check if lock is stale
        lock_time = lock_info.get('timestamp', 0)
        if time.time() - lock_time > STALE_LOCK_THRESHOLD:
            # Check if process is still alive
            pid = lock_info.get('pid')
            if pid and not self._is_process_alive(pid):
                logger.warning(
                    f"Clearing stale {self.hardware_type} lock from dead process {pid}"
                )
                try:
                    self.lock_file.unlink()
                    self.lock_info_file.unlink()
                except FileNotFoundError:
                    pass
    
    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still alive"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def __enter__(self):
        """Context manager entry"""
        if not self.acquire():
            raise HardwareLockError(f"Failed to acquire {self.hardware_type} lock")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class HardwareCoordinator:
    """
    Centralized hardware test coordination
    """
    
    def __init__(self, worker_id: str = 'master'):
        self.worker_id = worker_id
        self.active_locks: Dict[str, HardwareLock] = {}
        self._lock = threading.Lock()
        
    def acquire_hardware(self, hardware_types: List[str], timeout: float = DEFAULT_LOCK_TIMEOUT):
        """
        Acquire locks for multiple hardware types
        
        Args:
            hardware_types: List of hardware types to lock
            timeout: Timeout for acquiring all locks
        """
        acquired_locks = []
        
        try:
            for hw_type in hardware_types:
                if hw_type not in EXCLUSIVE_HARDWARE:
                    continue
                    
                lock = HardwareLock(hw_type, self.worker_id)
                if lock.acquire(timeout):
                    with self._lock:
                        self.active_locks[hw_type] = lock
                    acquired_locks.append(hw_type)
                else:
                    # Failed to acquire lock, release all acquired locks
                    for acquired_hw in acquired_locks:
                        self.release_hardware([acquired_hw])
                    raise HardwareLockError(
                        f"Failed to acquire {hw_type} lock for worker {self.worker_id}"
                    )
                    
        except Exception as e:
            # Clean up any acquired locks on error
            for acquired_hw in acquired_locks:
                self.release_hardware([acquired_hw])
            raise
    
    def release_hardware(self, hardware_types: List[str]):
        """Release locks for hardware types"""
        with self._lock:
            for hw_type in hardware_types:
                if hw_type in self.active_locks:
                    lock = self.active_locks.pop(hw_type)
                    lock.release()
    
    def release_all(self):
        """Release all held locks"""
        with self._lock:
            hw_types = list(self.active_locks.keys())
        self.release_hardware(hw_types)


# Global coordinator instance per worker
_coordinator_instances: Dict[str, HardwareCoordinator] = {}
_coordinator_lock = threading.Lock()


def get_hardware_coordinator(worker_id: str = 'master') -> HardwareCoordinator:
    """Get or create hardware coordinator for worker"""
    with _coordinator_lock:
        if worker_id not in _coordinator_instances:
            _coordinator_instances[worker_id] = HardwareCoordinator(worker_id)
        return _coordinator_instances[worker_id]


@contextmanager
def hardware_lock(hardware_type: str, worker_id: str = 'master', timeout: float = DEFAULT_LOCK_TIMEOUT):
    """
    Context manager for single hardware lock
    
    Args:
        hardware_type: Type of hardware to lock
        worker_id: Worker ID for coordination
        timeout: Lock acquisition timeout
    """
    coordinator = get_hardware_coordinator(worker_id)
    
    try:
        coordinator.acquire_hardware([hardware_type], timeout)
        yield
    finally:
        coordinator.release_hardware([hardware_type])


@contextmanager
def multi_hardware_lock(hardware_types: List[str], worker_id: str = 'master', 
                       timeout: float = DEFAULT_LOCK_TIMEOUT):
    """
    Context manager for multiple hardware locks
    
    Args:
        hardware_types: List of hardware types to lock
        worker_id: Worker ID for coordination
        timeout: Lock acquisition timeout
    """
    coordinator = get_hardware_coordinator(worker_id)
    
    try:
        coordinator.acquire_hardware(hardware_types, timeout)
        yield
    finally:
        coordinator.release_hardware(hardware_types)


# ─────────────────────────────────────────────────────────────
# pytest Integration
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure hardware coordination for pytest"""
    # Register cleanup hook
    config._hardware_coordinator = None


def pytest_sessionstart(session):
    """Initialize hardware coordination at session start"""
    worker_id = getattr(session.config, 'workerinput', {}).get('workerid', 'master')
    session.config._hardware_coordinator = get_hardware_coordinator(worker_id)
    
    # Clean up any stale locks at session start
    for hw_type in EXCLUSIVE_HARDWARE:
        lock = HardwareLock(hw_type, worker_id)
        lock._check_and_clear_stale_lock()


def pytest_sessionfinish(session, exitstatus):
    """Clean up hardware locks at session end"""
    if hasattr(session.config, '_hardware_coordinator'):
        session.config._hardware_coordinator.release_all()


def pytest_runtest_setup(item):
    """Acquire hardware locks before test execution"""
    # Check for hardware markers
    markers = [marker.name for marker in item.iter_markers()]
    hardware_needed = []
    
    for hw_type in EXCLUSIVE_HARDWARE:
        if hw_type in markers:
            hardware_needed.append(hw_type)
    
    if hardware_needed:
        worker_id = getattr(item.config, 'workerinput', {}).get('workerid', 'master')
        coordinator = get_hardware_coordinator(worker_id)
        
        # Store coordinator on item for cleanup
        item._hardware_coordinator = coordinator
        item._hardware_locked = hardware_needed
        
        # Acquire locks
        try:
            coordinator.acquire_hardware(hardware_needed)
        except HardwareLockError as e:
            pytest.skip(f"Hardware lock unavailable: {e}")


def pytest_runtest_teardown(item, nextitem):
    """Release hardware locks after test execution"""
    if hasattr(item, '_hardware_coordinator') and hasattr(item, '_hardware_locked'):
        item._hardware_coordinator.release_hardware(item._hardware_locked)


# ─────────────────────────────────────────────────────────────
# Lock Status Utilities
# ─────────────────────────────────────────────────────────────

def get_lock_status() -> Dict[str, Any]:
    """Get current status of all hardware locks"""
    status = {}
    
    for hw_type in EXCLUSIVE_HARDWARE:
        lock_file = LOCK_DIR / f"{hw_type}.lock"
        info_file = LOCK_DIR / f"{hw_type}.info"
        
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    info['locked'] = True
                    status[hw_type] = info
            except:
                status[hw_type] = {'locked': True, 'error': 'Cannot read lock info'}
        else:
            status[hw_type] = {'locked': False}
    
    return status


def clear_all_locks(force: bool = False):
    """
    Clear all hardware locks
    
    Args:
        force: Force clear even if processes are alive
    """
    cleared = 0
    
    for hw_type in EXCLUSIVE_HARDWARE:
        lock = HardwareLock(hw_type, 'cleanup')
        lock_info = lock._read_lock_info()
        
        if lock_info:
            pid = lock_info.get('pid')
            if force or (pid and not lock._is_process_alive(pid)):
                try:
                    lock.lock_file.unlink()
                    lock.lock_info_file.unlink()
                    cleared += 1
                    logger.info(f"Cleared {hw_type} lock")
                except FileNotFoundError:
                    pass
    
    return cleared


if __name__ == "__main__":
    # Example usage and status display
    print("Hardware Lock Status:")
    print("=" * 50)
    
    status = get_lock_status()
    for hw_type, info in status.items():
        if info['locked']:
            print(f"{hw_type}: LOCKED by {info.get('worker_id', 'unknown')} "
                  f"(PID: {info.get('pid', 'unknown')})")
        else:
            print(f"{hw_type}: Available")
    
    print("\nLock directory:", LOCK_DIR)