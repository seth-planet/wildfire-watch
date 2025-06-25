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

# Configure logging for test infrastructure with proper cleanup handling
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] conftest: %(message)s',
    datefmt='%H:%M:%S',
    force=True  # Force reconfiguration to avoid conflicts
)
logger = logging.getLogger(__name__)

# Add a null handler as fallback to prevent I/O errors during cleanup
null_handler = logging.NullHandler()
logger.addHandler(null_handler)

# Import test isolation fixes and enhanced fixtures
try:
    from test_isolation_fixtures import (
        thread_monitor, state_manager, mqtt_broker, mqtt_client_factory,
        unique_id, mock_external_deps, fire_consensus_clean, camera_detector_clean,
        isolate_tests, cleanup_telemetry
    )
    logger.info("Test isolation fixtures loaded successfully")
except ImportError as e:
    logger.warning(f"Test isolation fixtures not available: {e}")
    
# Import enhanced test isolation with garbage collection
try:
    from enhanced_test_isolation import (
        comprehensive_test_isolation, cleanup_services,
        EnhancedThreadManager, ResourceMonitor
    )
    logger.info("Enhanced test isolation loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced test isolation not available: {e}")
    
try:
    from enhanced_mqtt_broker import TestMQTTBroker
    logger.info("Enhanced MQTT broker loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced MQTT broker not available: {e}")

# ─────────────────────────────────────────────────────────────
# Session Management and Performance Monitoring
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def session_performance_monitor():
    """Monitor and report test session performance."""
    session_start = time.time()
    safe_log("=== Test Session Started ===")
    
    # Track infrastructure setup times
    setup_times = {}
    
    yield setup_times
    
    session_duration = time.time() - session_start
    safe_log(f"=== Test Session Complete: {session_duration:.2f}s ===")
    
    # Report infrastructure timing if we have data
    if setup_times:
        safe_log("Infrastructure Setup Times:")
        for name, duration in setup_times.items():
            safe_log(f"  {name}: {duration:.3f}s")

@pytest.fixture(scope="session", autouse=True)
def handle_session_timeouts():
    """Handle session-level timeouts gracefully."""
    # Set up signal handlers for graceful shutdown
    original_sigterm = signal.signal(signal.SIGTERM, lambda s, f: safe_log("Received SIGTERM during test session"))
    original_sigint = signal.signal(signal.SIGINT, lambda s, f: safe_log("Received SIGINT during test session"))
    
    yield
    
    # Restore original handlers
    signal.signal(signal.SIGTERM, original_sigterm)
    signal.signal(signal.SIGINT, original_sigint)

def safe_log(message, level=logging.INFO):
    """Safely log messages, checking handler state and catching I/O errors during teardown.
    
    This enhanced version checks if logging has been shut down or if handlers
    are closed before attempting to log, preventing errors during teardown.
    """
    # Check if logging has been shut down globally
    if hasattr(logging, '_shutdown') and logging._shutdown:
        return
        
    # Check if the logger has handlers and if they're still operational
    if hasattr(logger, 'handlers'):
        for handler in logger.handlers:
            # Check if handler has a stream that might be closed
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                if handler.stream.closed:
                    return
                    
    try:
        logger.log(level, message)
    except (ValueError, OSError):
        # Ignore logging errors during teardown
        pass

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
    safe_log("Applied long timeout environment configuration")
    
    yield timeout_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
    safe_log("Restored original environment configuration")

# ─────────────────────────────────────────────────────────────
# Enhanced MQTT Test Infrastructure
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def session_mqtt_broker(session_performance_monitor):
    """Session-scoped MQTT broker for all tests - ensures real broker is always available."""
    setup_start = time.time()
    logger.info("Setting up session MQTT broker...")
    
    # Import here to avoid import errors
    sys.path.insert(0, os.path.dirname(__file__))
    from mqtt_test_broker import MQTTTestBroker
    
    # Create real broker instance
    broker = MQTTTestBroker()
    broker_started = False
    
    try:
        # Start the broker
        broker.start()
        
        # Wait for broker to be ready with retries
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            logger.info(f"Waiting for MQTT broker to be ready (attempt {attempt + 1}/{max_retries})...")
            
            if broker.wait_for_ready(timeout=30):
                broker_started = True
                break
            
            if attempt < max_retries - 1:
                logger.warning(f"MQTT broker not ready, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        if not broker_started:
            # Fail the test session if we can't get a real broker
            pytest.fail("MQTT broker failed to start after all retries - cannot run tests without real MQTT")
        
        setup_time = time.time() - setup_start
        session_performance_monitor['mqtt_broker_setup'] = setup_time
        safe_log(f"Session MQTT broker ready in {setup_time:.3f}s on port {broker.port}")
        
        # Verify broker is actually running
        if not broker.is_running():
            pytest.fail("MQTT broker reported as not running after successful start")
        
        yield broker
        
    except Exception as e:
        safe_log(f"Failed to start session MQTT broker: {e}", logging.ERROR)
        # Fail the test session - we need real MQTT for integration tests
        pytest.fail(f"Cannot run tests without MQTT broker: {e}")
        
    finally:
        try:
            if broker_started and hasattr(broker, 'stop'):
                safe_log("Stopping session MQTT broker...")
                # Use a thread with timeout to prevent hanging
                stop_thread = threading.Thread(target=broker.stop)
                stop_thread.start()
                stop_thread.join(timeout=5.0)  # 5 second timeout
                
                if stop_thread.is_alive():
                    safe_log("MQTT broker stop timed out after 5 seconds", logging.WARNING)
                else:
                    safe_log("Session MQTT broker stopped")
        except Exception as e:
            safe_log(f"Error stopping MQTT broker: {e}", logging.WARNING)

@pytest.fixture
def test_mqtt_broker(session_mqtt_broker):
    """Per-test MQTT broker fixture that reuses session broker."""
    # Just return the session broker - it's designed to handle multiple concurrent connections
    return session_mqtt_broker

# ─────────────────────────────────────────────────────────────
# Topic Isolation Fixtures for Test Independence
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def unique_topic_prefix():
    """
    Function-scoped fixture to generate a unique topic prefix for each test.
    This is the key to test isolation when using a shared broker.
    Example: 'test/abc123/fire/detection' -> 'test/def456/fire/detection'
    """
    import uuid
    return f"test/{uuid.uuid4().hex[:8]}"

@pytest.fixture
def mqtt_topic_factory(unique_topic_prefix):
    """
    A factory fixture that creates full, unique topic strings.
    This makes tests cleaner as they don't need to manually construct topics.
    
    Usage:
        def test_something(mqtt_topic_factory):
            control_topic = mqtt_topic_factory("control")
            # control_topic is now "test/some_unique_id/control"
    """
    def _topic_factory(base_topic: str) -> str:
        return f"{unique_topic_prefix}/{base_topic}"
    
    return _topic_factory

# ─────────────────────────────────────────────────────────────
# MQTT Client Management Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mqtt_client(session_mqtt_broker):
    """
    Function-scoped fixture providing a connected paho-mqtt client.
    Uses threading.Event for robust connect/disconnect logic, ensuring
    each test gets a clean, verified connection and that cleanup is graceful.
    """
    import paho.mqtt.client as mqtt
    import uuid
    
    client_id = f"test_client_{uuid.uuid4().hex[:8]}"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)

    connected_event = threading.Event()
    disconnected_event = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.debug(f"Client {client_id} connected successfully.")
            connected_event.set()
        else:
            logger.error(f"Client {client_id} failed to connect with reason code: {rc}")
            # The wait timeout below will handle this failure.

    def on_disconnect(client, userdata, rc, properties=None, reason_code=None):
        logger.debug(f"Client {client_id} disconnected with reason code: {rc}.")
        disconnected_event.set()

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        # Start the network loop. This is essential for callbacks to be processed.
        client.loop_start()

        # The connect call is non-blocking.
        client.connect(session_mqtt_broker.host, session_mqtt_broker.port, 60)

        # Wait for the on_connect callback to be fired.
        # A 10-second timeout is generous but prevents tests from hanging.
        if not connected_event.wait(timeout=10):
            # If the event isn't set, the connection failed. Stop the loop.
            client.loop_stop()
            raise ConnectionError(f"MQTT client {client_id} failed to connect within the timeout period.")

        yield client

    finally:
        # Graceful disconnect
        if client.is_connected():
            client.disconnect()
            # Wait for the on_disconnect callback to confirm disconnection.
            if not disconnected_event.wait(timeout=5):
                logger.warning(f"Client {client_id} did not disconnect gracefully within timeout.")
        
        # Always ensure the loop is stopped.
        client.loop_stop()

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
        # Stop any telemetry timers
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cam_telemetry"))
            import telemetry
            if hasattr(telemetry, 'shutdown_telemetry'):
                telemetry.shutdown_telemetry()
        except ImportError:
            pass
        
        # Give threads time to finish naturally
        time.sleep(0.1)
        
        # Check for leaked threads
        current_threads = set(threading.enumerate())
        leaked_threads = current_threads - initial_threads
        
        # Filter out daemon threads and known persistent threads
        problematic_threads = [
            t for t in leaked_threads 
            if not t.daemon and 'mqtt' not in t.name.lower() and not t.name.startswith('Thread-')
        ]
        
        if problematic_threads:
            logger.warning(f"Test leaked {len(problematic_threads)} threads: {[t.name for t in problematic_threads]}")
            
        # Force cleanup of any remaining timers
        for thread in list(threading.enumerate()):
            if thread != threading.current_thread() and hasattr(thread, 'cancel'):
                try:
                    thread.cancel()
                except:
                    pass
                    
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
                          'coral_tpu', 'tflite_runtime', 
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

# Note: pytest_python_versions plugin is optional and not required
# pytest_plugins = ["pytest_python_versions"]  # Commented out - not installed

try:
    # Import Python version routing plugin
    import pytest_python_versions
    # Only register the plugin if it's available
    pytest_plugins = ["pytest_python_versions"]
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