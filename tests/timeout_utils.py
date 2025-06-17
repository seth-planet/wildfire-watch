#!/usr/bin/env python3.12
"""
Timeout utilities for Wildfire Watch tests.

Provides decorators and context managers for handling expected long timeouts
in test scenarios where infrastructure setup is inherently slow.
"""

import time
import logging
import functools
import threading
from typing import Callable, Any, Optional, Union
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Timeout Decorators
# ─────────────────────────────────────────────────────────────

def expect_long_timeout(timeout_seconds: int = 1800, reason: str = "Infrastructure setup"):
    """
    Decorator to mark tests that are expected to have long timeouts.
    
    Args:
        timeout_seconds: Expected timeout in seconds (default: 30 minutes)
        reason: Reason for the long timeout (for logging)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting long-timeout test: {func.__name__} (expected: {timeout_seconds}s, reason: {reason})")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Long-timeout test completed: {func.__name__} ({duration:.1f}s)")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Long-timeout test failed: {func.__name__} ({duration:.1f}s) - {e}")
                raise
                
        # Mark the function for pytest
        wrapper._timeout_expected = True
        wrapper._timeout_seconds = timeout_seconds
        wrapper._timeout_reason = reason
        return wrapper
    return decorator

def mqtt_infrastructure_test(timeout_seconds: int = 900):
    """
    Decorator specifically for tests that require MQTT infrastructure.
    
    Args:
        timeout_seconds: Expected timeout in seconds (default: 15 minutes)
    """
    return expect_long_timeout(
        timeout_seconds=timeout_seconds,
        reason="MQTT broker and infrastructure setup"
    )

def integration_test(timeout_seconds: int = 1200):
    """
    Decorator for integration tests that may require multiple services.
    
    Args:
        timeout_seconds: Expected timeout in seconds (default: 20 minutes)
    """
    return expect_long_timeout(
        timeout_seconds=timeout_seconds,
        reason="Multi-service integration setup"
    )

# ─────────────────────────────────────────────────────────────
# Context Managers for Timeout Handling
# ─────────────────────────────────────────────────────────────

@contextmanager
def timeout_context(name: str, expected_duration: Optional[float] = None):
    """
    Context manager for timing operations and handling expected long durations.
    
    Args:
        name: Name of the operation being timed
        expected_duration: Expected duration in seconds (for logging)
    """
    start_time = time.time()
    logger.info(f"Starting {name}...")
    
    try:
        yield
        duration = time.time() - start_time
        
        if expected_duration and duration > expected_duration:
            logger.warning(f"{name} took longer than expected: {duration:.2f}s > {expected_duration:.2f}s")
        else:
            logger.info(f"{name} completed in {duration:.2f}s")
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{name} failed after {duration:.2f}s: {e}")
        raise

@contextmanager
def mqtt_setup_context():
    """Context manager specifically for MQTT setup operations."""
    with timeout_context("MQTT setup", expected_duration=30.0):
        yield

@contextmanager
def service_startup_context(service_name: str):
    """Context manager for service startup operations."""
    with timeout_context(f"{service_name} startup", expected_duration=60.0):
        yield

# ─────────────────────────────────────────────────────────────
# Timeout Monitoring and Reporting
# ─────────────────────────────────────────────────────────────

class TimeoutMonitor:
    """Monitor and report on timeout patterns during test execution."""
    
    def __init__(self):
        self.operation_times = {}
        self.slow_operations = []
        self._lock = threading.Lock()
    
    def record_operation(self, name: str, duration: float, threshold: float = 10.0):
        """Record an operation duration and flag if it's slow."""
        with self._lock:
            if name not in self.operation_times:
                self.operation_times[name] = []
            
            self.operation_times[name].append(duration)
            
            if duration > threshold:
                self.slow_operations.append((name, duration, time.time()))
                logger.warning(f"Slow operation detected: {name} ({duration:.2f}s)")
    
    def get_average_time(self, name: str) -> Optional[float]:
        """Get average time for an operation."""
        with self._lock:
            times = self.operation_times.get(name, [])
            return sum(times) / len(times) if times else None
    
    def report_summary(self):
        """Report summary of all monitored operations."""
        with self._lock:
            logger.info("=== Timeout Monitor Summary ===")
            
            for name, times in self.operation_times.items():
                avg_time = sum(times) / len(times)
                max_time = max(times)
                count = len(times)
                logger.info(f"{name}: avg={avg_time:.2f}s, max={max_time:.2f}s, count={count}")
            
            if self.slow_operations:
                logger.info(f"Slow operations detected: {len(self.slow_operations)}")
                for name, duration, timestamp in self.slow_operations[-5:]:  # Show last 5
                    logger.info(f"  {name}: {duration:.2f}s")

# Global timeout monitor instance
timeout_monitor = TimeoutMonitor()

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def wait_with_timeout(condition_func: Callable[[], bool], 
                     timeout: float = 30.0, 
                     interval: float = 0.5,
                     description: str = "condition") -> bool:
    """
    Wait for a condition to be true with timeout and logging.
    
    Args:
        condition_func: Function that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        description: Description of what we're waiting for
        
    Returns:
        True if condition was met, False if timeout occurred
    """
    start_time = time.time()
    logger.info(f"Waiting for {description} (timeout: {timeout}s)")
    
    while time.time() - start_time < timeout:
        try:
            if condition_func():
                duration = time.time() - start_time
                logger.info(f"{description} achieved in {duration:.2f}s")
                return True
        except Exception as e:
            logger.debug(f"Error checking {description}: {e}")
        
        time.sleep(interval)
    
    duration = time.time() - start_time
    logger.warning(f"Timeout waiting for {description} after {duration:.2f}s")
    return False

def log_test_timing(test_name: str, start_time: float):
    """Log test timing information."""
    duration = time.time() - start_time
    timeout_monitor.record_operation(f"test_{test_name}", duration)
    
    if duration > 60:  # 1 minute
        logger.warning(f"Long test: {test_name} ({duration:.1f}s)")
    elif duration > 10:  # 10 seconds
        logger.info(f"Test completed: {test_name} ({duration:.1f}s)")

# ─────────────────────────────────────────────────────────────
# Integration with pytest
# ─────────────────────────────────────────────────────────────

def pytest_runtest_setup(item):
    """pytest hook to set up timeout monitoring."""
    if hasattr(item.function, '_timeout_expected'):
        timeout_seconds = getattr(item.function, '_timeout_seconds', 1800)
        reason = getattr(item.function, '_timeout_reason', 'Unknown')
        logger.info(f"Test expects long timeout: {item.name} ({timeout_seconds}s) - {reason}")

def pytest_runtest_teardown(item, nextitem):
    """pytest hook to record test timing."""
    if hasattr(item, '_test_start_time'):
        duration = time.time() - item._test_start_time
        timeout_monitor.record_operation(f"test_{item.name}", duration)

def pytest_sessionfinish(session, exitstatus):
    """pytest hook to report timeout summary."""
    timeout_monitor.report_summary()

# ─────────────────────────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example of how to use the timeout utilities
    
    @mqtt_infrastructure_test(timeout_seconds=600)  # 10 minutes
    def example_mqtt_test():
        """Example test that expects long MQTT setup time."""
        with mqtt_setup_context():
            # Simulate MQTT broker setup
            time.sleep(2)
            
        # Test logic here
        pass
    
    @integration_test(timeout_seconds=900)  # 15 minutes
    def example_integration_test():
        """Example integration test with multiple services."""
        with service_startup_context("camera_detector"):
            time.sleep(1)
            
        with service_startup_context("fire_consensus"):
            time.sleep(1)
            
        # Integration test logic here
        pass
    
    # Run examples
    example_mqtt_test()
    example_integration_test()
    
    # Report summary
    timeout_monitor.report_summary()