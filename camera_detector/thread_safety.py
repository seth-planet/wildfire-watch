#!/usr/bin/env python3.12
"""Thread Safety Enhancements for Camera Detector

This module provides thread-safe wrappers and utilities for the camera detector
to ensure proper synchronization of shared state access.
"""

import threading
import time
import logging
from typing import Dict, Set, Optional, List, Any, Callable
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class ThreadSafeDict:
    """Thread-safe dictionary wrapper with atomic operations"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe get with optional default"""
        with self._lock:
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Thread-safe set"""
        with self._lock:
            self._data[key] = value
    
    def pop(self, key: str, default: Any = None) -> Any:
        """Thread-safe pop"""
        with self._lock:
            return self._data.pop(key, default)
    
    def update(self, other: Dict[str, Any]):
        """Thread-safe update from dict"""
        with self._lock:
            self._data.update(other)
    
    def items(self) -> List[tuple]:
        """Get thread-safe copy of items"""
        with self._lock:
            return list(self._data.items())
    
    def values(self) -> List[Any]:
        """Get thread-safe copy of values"""
        with self._lock:
            return list(self._data.values())
    
    def keys(self) -> List[str]:
        """Get thread-safe copy of keys"""
        with self._lock:
            return list(self._data.keys())
    
    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock:
            return len(self._data)
    
    def __contains__(self, key: str) -> bool:
        """Thread-safe contains check"""
        with self._lock:
            return key in self._data
    
    def clear(self):
        """Thread-safe clear"""
        with self._lock:
            self._data.clear()
    
    def copy(self) -> Dict[str, Any]:
        """Get thread-safe copy of entire dict"""
        with self._lock:
            return self._data.copy()

class ThreadSafeSet:
    """Thread-safe set wrapper"""
    
    def __init__(self):
        self._data: Set[Any] = set()
        self._lock = threading.RLock()
    
    def add(self, item: Any):
        """Thread-safe add"""
        with self._lock:
            self._data.add(item)
    
    def remove(self, item: Any):
        """Thread-safe remove (raises KeyError if not found)"""
        with self._lock:
            self._data.remove(item)
    
    def discard(self, item: Any):
        """Thread-safe discard (no error if not found)"""
        with self._lock:
            self._data.discard(item)
    
    def __contains__(self, item: Any) -> bool:
        """Thread-safe contains check"""
        with self._lock:
            return item in self._data
    
    def __len__(self) -> int:
        """Thread-safe length"""
        with self._lock:
            return len(self._data)
    
    def clear(self):
        """Thread-safe clear"""
        with self._lock:
            self._data.clear()
    
    def copy(self) -> Set[Any]:
        """Get thread-safe copy"""
        with self._lock:
            return self._data.copy()
    
    def update(self, other: Set[Any]):
        """Thread-safe update from another set"""
        with self._lock:
            self._data.update(other)

@dataclass
class ThreadSafeCounter:
    """Thread-safe counter with atomic increment/decrement"""
    
    _value: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    
    def increment(self, amount: int = 1) -> int:
        """Atomically increment and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Atomically decrement and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value
    
    def set(self, value: int):
        """Set value"""
        with self._lock:
            self._value = value
    
    def reset(self):
        """Reset to zero"""
        with self._lock:
            self._value = 0

class ThreadSafeStateManager:
    """Manages thread-safe state transitions"""
    
    def __init__(self, initial_state: str = "INIT"):
        self._state = initial_state
        self._lock = threading.RLock()
        self._state_listeners: List[Callable] = []
        self._valid_transitions: Dict[str, Set[str]] = {}
    
    def add_transition(self, from_state: str, to_states: Set[str]):
        """Define valid state transitions"""
        with self._lock:
            self._valid_transitions[from_state] = to_states
    
    def transition_to(self, new_state: str) -> bool:
        """Attempt state transition
        
        Returns:
            True if transition was successful
        """
        with self._lock:
            current = self._state
            
            # Check if transition is valid
            valid_states = self._valid_transitions.get(current, set())
            if new_state not in valid_states and valid_states:
                logger.warning(f"Invalid transition: {current} -> {new_state}")
                return False
            
            # Perform transition
            self._state = new_state
            
            # Notify listeners
            for listener in self._state_listeners:
                try:
                    listener(current, new_state)
                except Exception as e:
                    logger.error(f"State listener error: {e}")
            
            return True
    
    def get_state(self) -> str:
        """Get current state"""
        with self._lock:
            return self._state
    
    def add_listener(self, callback: Callable[[str, str], None]):
        """Add state change listener"""
        with self._lock:
            self._state_listeners.append(callback)

class ThreadSafeCameraRegistry:
    """Thread-safe registry for camera management"""
    
    def __init__(self):
        self._cameras = ThreadSafeDict()
        self._ip_to_id = ThreadSafeDict()
        self._mac_to_id = ThreadSafeDict()
        self._lock = threading.RLock()
    
    def add_camera(self, camera: Any) -> bool:
        """Add camera with all mappings atomically
        
        Returns:
            True if camera was added (new), False if updated
        """
        with self._lock:
            is_new = camera.id not in self._cameras
            
            # Update all mappings atomically
            self._cameras.set(camera.id, camera)
            self._ip_to_id.set(camera.ip, camera.id)
            if camera.mac:
                self._mac_to_id.set(camera.mac, camera.id)
            
            return is_new
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove camera and all mappings atomically
        
        Returns:
            True if camera was removed
        """
        with self._lock:
            camera = self._cameras.get(camera_id)
            if not camera:
                return False
            
            # Remove all mappings atomically
            self._cameras.pop(camera_id)
            self._ip_to_id.pop(camera.ip, None)
            if camera.mac:
                self._mac_to_id.pop(camera.mac, None)
            
            return True
    
    def get_by_id(self, camera_id: str) -> Optional[Any]:
        """Get camera by ID"""
        return self._cameras.get(camera_id)
    
    def get_by_ip(self, ip: str) -> Optional[Any]:
        """Get camera by IP"""
        camera_id = self._ip_to_id.get(ip)
        return self._cameras.get(camera_id) if camera_id else None
    
    def get_by_mac(self, mac: str) -> Optional[Any]:
        """Get camera by MAC"""
        camera_id = self._mac_to_id.get(mac)
        return self._cameras.get(camera_id) if camera_id else None
    
    def get_all(self) -> List[Any]:
        """Get all cameras"""
        return self._cameras.values()
    
    def update_ip(self, camera_id: str, old_ip: str, new_ip: str) -> bool:
        """Update camera IP atomically
        
        Returns:
            True if update was successful
        """
        with self._lock:
            camera = self._cameras.get(camera_id)
            if not camera or camera.ip != old_ip:
                return False
            
            # Update mappings atomically
            self._ip_to_id.pop(old_ip, None)
            self._ip_to_id.set(new_ip, camera_id)
            camera.ip = new_ip
            
            return True
    
    def count(self) -> int:
        """Get camera count"""
        return len(self._cameras)
    
    def count_online(self) -> int:
        """Count online cameras"""
        return sum(1 for camera in self._cameras.values() if camera.online)

def synchronized(lock_attr: str = '_lock'):
    """Decorator to synchronize method with a lock
    
    Args:
        lock_attr: Name of the lock attribute on the instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_attr)
            with lock:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

def with_timeout(timeout: float):
    """Decorator to add timeout to blocking operations
    
    Args:
        timeout: Timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                logger.error(f"{func.__name__} timed out after {timeout}s")
                raise TimeoutError(f"Operation timed out after {timeout}s")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator

@contextmanager
def multi_lock(*locks):
    """Context manager to acquire multiple locks in order
    
    This prevents deadlocks by always acquiring locks in the same order.
    
    Args:
        *locks: Lock objects to acquire
    """
    # Sort locks by id() to ensure consistent ordering
    sorted_locks = sorted(locks, key=id)
    
    acquired = []
    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        # Release in reverse order
        for lock in reversed(acquired):
            lock.release()

class PeriodicTask:
    """Thread-safe periodic task runner"""
    
    def __init__(self, interval: float, function: Callable, 
                 args: tuple = (), kwargs: dict = None):
        """Initialize periodic task
        
        Args:
            interval: Seconds between executions
            function: Function to call
            args: Function arguments
            kwargs: Function keyword arguments
        """
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self._timer = None
        self._lock = threading.RLock()
        self._running = False
    
    def start(self):
        """Start periodic execution"""
        with self._lock:
            if not self._running:
                self._running = True
                self._schedule_next()
    
    def stop(self):
        """Stop periodic execution"""
        with self._lock:
            self._running = False
            if self._timer:
                self._timer.cancel()
                self._timer = None
    
    def _run(self):
        """Execute the function and schedule next run"""
        try:
            self.function(*self.args, **self.kwargs)
        except Exception as e:
            logger.error(f"Periodic task error: {e}", exc_info=True)
        finally:
            with self._lock:
                if self._running:
                    self._schedule_next()
    
    def _schedule_next(self):
        """Schedule next execution"""
        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()

# Example usage in camera detector
def make_thread_safe_camera_detector(detector_class):
    """Decorator to make a camera detector class thread-safe
    
    This wraps methods that access shared state with proper locking.
    """
    
    # Methods that need synchronization
    synchronized_methods = [
        'add_camera', 'remove_camera', 'update_camera',
        'get_camera', 'get_all_cameras', 'get_online_cameras',
        'increment_discovery_count', 'update_discovery_state'
    ]
    
    # Wrap each method
    for method_name in synchronized_methods:
        if hasattr(detector_class, method_name):
            original_method = getattr(detector_class, method_name)
            wrapped_method = synchronized('lock')(original_method)
            setattr(detector_class, method_name, wrapped_method)
    
    return detector_class