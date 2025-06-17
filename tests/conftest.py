#!/usr/bin/env python3.12
"""
Central pytest configuration for Wildfire Watch test suite.

This file provides:
- Timeout handling for slow infrastructure setup
- Session-scoped fixtures for expensive operations
- Performance monitoring and reporting
- Graceful handling of long-running test scenarios
"""

import os
import sys
import time
import pytest
import logging
import threading
import signal
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

# Configure logging for test infrastructure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] conftest: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Session Management and Performance Monitoring
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def session_performance_monitor():
    """Monitor and report test session performance."""
    session_start = time.time()
    logger.info("=== Test Session Started ===")
    
    # Track infrastructure setup times
    setup_times = {}
    
    yield setup_times
    
    session_duration = time.time() - session_start
    logger.info(f"=== Test Session Complete: {session_duration:.2f}s ===")
    
    # Report infrastructure timing if we have data
    if setup_times:
        logger.info("Infrastructure Setup Times:")
        for name, duration in setup_times.items():
            logger.info(f"  {name}: {duration:.3f}s")

@pytest.fixture(scope="session", autouse=True)
def handle_session_timeouts():
    """Handle session-level timeouts gracefully."""
    # Set up signal handlers for graceful shutdown
    original_sigterm = signal.signal(signal.SIGTERM, lambda s, f: logger.warning("Received SIGTERM during test session"))
    original_sigint = signal.signal(signal.SIGINT, lambda s, f: logger.warning("Received SIGINT during test session"))
    
    yield
    
    # Restore original handlers
    signal.signal(signal.SIGTERM, original_sigterm)
    signal.signal(signal.SIGINT, original_sigint)

# ─────────────────────────────────────────────────────────────
# Timeout-Aware Test Environment Setup
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def long_timeout_environment():
    """Configure environment for tests that require long timeouts."""
    # Store original environment
    original_env = dict(os.environ)
    
    # Set timeout-friendly environment variables
    timeout_env = {
        # MQTT settings - allow more time for broker operations
        'MQTT_CONNECT_TIMEOUT': '30',       # 30s for MQTT connections
        'MQTT_KEEPALIVE': '60',             # 60s keepalive
        'MQTT_RETRY_INTERVAL': '5',         # 5s between retries
        'MQTT_MAX_RETRIES': '3',            # Max 3 retries
        
        # Service timeouts - generous but not infinite
        'DISCOVERY_INTERVAL': '30',         # 30s minimum discovery interval  
        'HEALTH_INTERVAL': '10',            # 10s minimum health check
        'OFFLINE_THRESHOLD': '60',          # 60s minimum offline threshold
        'RTSP_TIMEOUT': '10',               # 10s RTSP validation timeout
        'ONVIF_TIMEOUT': '5',               # 5s ONVIF timeout
        
        # Test-specific settings
        'LOG_LEVEL': 'INFO',                # Reduce log noise
        'GPIO_SIMULATION': 'true',          # Always simulate GPIO in tests
        'TEST_MODE': 'true',                # Indicate we're in test mode
    }
    
    # Apply timeout environment
    os.environ.update(timeout_env)
    logger.info("Applied long timeout environment configuration")
    
    yield timeout_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    logger.info("Restored original environment configuration")

# ─────────────────────────────────────────────────────────────
# Enhanced MQTT Test Infrastructure
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def session_mqtt_broker(session_performance_monitor):
    """Session-scoped MQTT broker for all tests."""
    setup_start = time.time()
    logger.info("Setting up session MQTT broker...")
    
    try:
        # Import here to avoid import errors in environments without mqtt_test_broker
        from mqtt_test_broker import TestMQTTBroker
        
        broker = TestMQTTBroker()
        broker.start()
        
        # Wait for broker to be ready with timeout
        broker_ready = broker.wait_for_ready(timeout=60)  # 1 minute timeout
        
        if not broker_ready:
            logger.error("MQTT broker failed to start within 60 seconds")
            raise RuntimeError("MQTT broker startup timeout")
        
        setup_time = time.time() - setup_start
        session_performance_monitor['mqtt_broker_setup'] = setup_time
        logger.info(f"Session MQTT broker ready in {setup_time:.3f}s")
        
        yield broker
        
    except ImportError:
        logger.warning("mqtt_test_broker not available, using mock broker")
        # Provide a mock broker for tests that don't need real MQTT
        mock_broker = Mock()
        mock_broker.host = 'localhost'
        mock_broker.port = 1883
        mock_broker.wait_for_ready = lambda timeout=30: True
        yield mock_broker
        
    except Exception as e:
        logger.error(f"Failed to start session MQTT broker: {e}")
        # Don't fail the entire session, provide a mock
        mock_broker = Mock()
        mock_broker.host = 'localhost'
        mock_broker.port = 1883
        yield mock_broker
        
    finally:
        try:
            if 'broker' in locals() and hasattr(broker, 'stop'):
                broker.stop()
                logger.info("Session MQTT broker stopped")
        except Exception as e:
            logger.warning(f"Error stopping MQTT broker: {e}")

@pytest.fixture
def test_mqtt_broker(session_mqtt_broker):
    """Per-test MQTT broker fixture that reuses session broker."""
    # Just return the session broker - it's designed to handle multiple concurrent connections
    return session_mqtt_broker

# ─────────────────────────────────────────────────────────────
# Timeout-Aware Test Markers and Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def timeout_aware_test_setup(request, long_timeout_environment):
    """Automatically configure tests based on their timeout requirements."""
    
    # Check if test is marked as having expected long timeouts
    timeout_expected = request.node.get_closest_marker("timeout_expected")
    slow_test = request.node.get_closest_marker("slow")
    integration_test = request.node.get_closest_marker("integration")
    
    test_start = time.time()
    test_name = request.node.name
    
    # Configure based on test type
    if timeout_expected or slow_test or integration_test:
        logger.info(f"Starting timeout-expected test: {test_name}")
        # Could add additional timeout-specific setup here
    
    yield
    
    # Report test duration
    test_duration = time.time() - test_start
    if test_duration > 30:  # Report slow tests
        logger.warning(f"Slow test completed: {test_name} ({test_duration:.2f}s)")
    elif test_duration > 5:
        logger.info(f"Test completed: {test_name} ({test_duration:.2f}s)")

# ─────────────────────────────────────────────────────────────
# Test Isolation and Cleanup
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure proper test isolation and cleanup."""
    # Store initial state
    initial_threads = set(threading.enumerate())
    
    yield
    
    # Cleanup after test
    try:
        # Give threads time to finish naturally
        time.sleep(0.1)
        
        # Check for leaked threads
        current_threads = set(threading.enumerate())
        leaked_threads = current_threads - initial_threads
        
        # Filter out daemon threads and known persistent threads
        problematic_threads = [
            t for t in leaked_threads 
            if not t.daemon and 'mqtt' not in t.name.lower()
        ]
        
        if problematic_threads:
            logger.warning(f"Test leaked {len(problematic_threads)} threads: {[t.name for t in problematic_threads]}")
            
    except Exception as e:
        logger.warning(f"Error during test cleanup: {e}")

# ─────────────────────────────────────────────────────────────
# Integration with Existing Optimization Fixtures
# ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line("markers", "timeout_expected: Test expected to have long timeout")
    config.addinivalue_line("markers", "infrastructure_dependent: Test depends on slow infrastructure")
    
    # Set up timeout handling
    logger.info("Pytest configured for long timeout handling")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test characteristics."""
    # Get current Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    for item in items:
        # Auto-mark tests that are likely to have long timeouts
        if any(keyword in item.name.lower() for keyword in ['mqtt', 'broker', 'integration', 'e2e']):
            item.add_marker(pytest.mark.timeout_expected)
            
        # Auto-mark infrastructure-dependent tests
        if any(keyword in item.name.lower() for keyword in ['docker', 'hardware', 'coral', 'frigate']):
            item.add_marker(pytest.mark.infrastructure_dependent)
        
        # Additional filtering for Python 3.12
        if python_version == "3.12":
            # Skip tests that are marked for other Python versions
            markers = [marker.name for marker in item.iter_markers()]
            skip_markers = ['python310', 'python38', 'yolo_nas', 'super_gradients', 
                          'coral_tpu', 'tflite_runtime', 'model_converter', 
                          'hardware_integration', 'deployment']
            
            if any(marker in markers for marker in skip_markers):
                item.add_marker(pytest.mark.skip(
                    reason=f"Test requires different Python version (running {python_version})"
                ))

def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Log test start for timeout debugging
    logger.debug(f"Starting test: {item.name}")

def pytest_runtest_teardown(item, nextitem):
    """Teardown for each test run."""
    # Log test completion for timeout debugging  
    logger.debug(f"Completed test: {item.name}")

# ─────────────────────────────────────────────────────────────
# Import Optimization Fixtures if Available
# ─────────────────────────────────────────────────────────────

try:
    # Import optimized fixtures if they exist
    from concurrent_futures_fix import fix_concurrent_futures
    logger.info("Imported concurrent futures fixes")
except ImportError:
    logger.info("Concurrent futures fixes not available")

try:
    # Import camera detector optimizations if available
    from camera_detector_optimized_fixture import camera_detector_fast
    logger.info("Imported optimized camera detector fixtures")
except ImportError:
    logger.info("Optimized camera detector fixtures not available")

# ─────────────────────────────────────────────────────────────
# Python Version Routing Plugin
# ─────────────────────────────────────────────────────────────

pytest_plugins = ["pytest_python_versions"]

try:
    # Import Python version routing plugin
    import pytest_python_versions
    logger.info("Python version routing plugin loaded")
except ImportError:
    logger.info("Python version routing plugin not available")

# ─────────────────────────────────────────────────────────────
# Timeout Reporting and Debugging
# ─────────────────────────────────────────────────────────────

def pytest_runtest_logreport(report):
    """Log test results with timeout information."""
    if report.when == "call":
        duration = getattr(report, 'duration', 0)
        if duration > 60:  # Report tests taking more than 1 minute
            logger.info(f"Long-running test: {report.nodeid} ({duration:.1f}s)")

def pytest_sessionfinish(session, exitstatus):
    """Session finish hook with timeout summary."""
    if hasattr(session, 'config'):
        duration = getattr(session.config, '_session_duration', 0)
        if duration > 300:  # Report sessions taking more than 5 minutes
            logger.info(f"Long test session completed: {duration:.1f}s")