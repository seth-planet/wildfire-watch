#!/usr/bin/env python3.12
"""
Enhanced Test Isolation Fixtures for Wildfire Watch
Provides comprehensive isolation between tests
"""
import os
import time
import threading
import weakref
import logging
import pytest
import uuid
from typing import Set, Optional, Dict, Any
from unittest.mock import Mock, patch
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Enhanced Thread Management
# ─────────────────────────────────────────────────────────────

class ThreadManager:
    """Enhanced thread management with automatic cleanup"""
    
    def __init__(self):
        self._initial_threads: Set[int] = set()
        self._test_threads: Set[threading.Thread] = weakref.WeakSet()
        self._lock = threading.Lock()
        
    def start(self):
        """Record initial thread state"""
        self._initial_threads = {t.ident for t in threading.enumerate()}
        logger.debug(f"Initial threads: {len(self._initial_threads)}")
        
    def register_thread(self, thread: threading.Thread):
        """Register a thread for tracking"""
        with self._lock:
            self._test_threads.add(thread)
            
    def cleanup(self, timeout: float = 5.0):
        """Clean up all test threads"""
        current_threads = threading.enumerate()
        test_threads = [t for t in current_threads 
                       if t.ident not in self._initial_threads and t.is_alive()]
        
        logger.debug(f"Cleaning up {len(test_threads)} test threads")
        
        # Stop all tracked threads
        for thread in list(self._test_threads):
            if thread.is_alive():
                if hasattr(thread, 'stop') and callable(thread.stop):
                    thread.stop()
                elif hasattr(thread, 'cancel') and callable(thread.cancel):
                    thread.cancel()
                elif hasattr(thread, '_stop_event'):
                    thread._stop_event.set()
        
        # Wait for threads to finish
        start_time = time.time()
        while test_threads and time.time() - start_time < timeout:
            test_threads = [t for t in test_threads if t.is_alive()]
            if test_threads:
                time.sleep(0.1)
        
        # Force cleanup remaining threads
        remaining_threads = [t for t in test_threads if t.is_alive()]
        if remaining_threads:
            logger.warning(f"Force terminating {len(remaining_threads)} stubborn threads")
            for thread in remaining_threads:
                try:
                    if hasattr(thread, '_delete'):
                        thread._delete()
                    # Force daemon status to allow process exit
                    thread.daemon = True
                except Exception as e:
                    logger.debug(f"Error force-cleaning thread {thread.name}: {e}")
                    
        final_test_threads = [t for t in threading.enumerate() 
                             if t.ident not in self._initial_threads and t.is_alive()]
        if final_test_threads:
            logger.error(f"Failed to cleanup {len(final_test_threads)} threads: "
                        f"{[t.name for t in final_test_threads]}")
                         
        return len(final_test_threads) == 0

# ─────────────────────────────────────────────────────────────
# State Management
# ─────────────────────────────────────────────────────────────

class StateManager:
    """Manages service state and cleanup"""
    
    def __init__(self):
        self._services: weakref.WeakSet = weakref.WeakSet()
        self._state_snapshots: Dict[str, Any] = {}
        
    def register_service(self, name: str, service):
        """Register a service for state management"""
        self._services.add(service)
        
        # Take initial state snapshot
        if hasattr(service, '__dict__'):
            self._state_snapshots[name] = {
                k: v.copy() if hasattr(v, 'copy') else v
                for k, v in service.__dict__.items()
                if not k.startswith('_') and isinstance(v, (dict, list, set))
            }
    
    def reset_service_state(self, name: str, service):
        """Reset service to initial state"""
        if name in self._state_snapshots:
            for attr, value in self._state_snapshots[name].items():
                if hasattr(service, attr):
                    current = getattr(service, attr)
                    if isinstance(current, dict):
                        current.clear()
                        if isinstance(value, dict):
                            current.update(value)
                    elif isinstance(current, list):
                        current.clear()
                        if isinstance(value, list):
                            current.extend(value)
                    elif isinstance(current, set):
                        current.clear()
                        if isinstance(value, set):
                            current.update(value)
    
    def cleanup_all(self):
        """Clean up all registered services"""
        for service in list(self._services):
            try:
                # Stop background tasks
                if hasattr(service, 'stop_background_tasks'):
                    service.stop_background_tasks()
                
                # Stop MQTT client
                if hasattr(service, "_mqtt_client"):
                    if hasattr(service._mqtt_client, 'loop_stop'):
                        service.mqtt_client.loop_stop()
                    if hasattr(service._mqtt_client, 'disconnect'):
                        service.mqtt_client.disconnect()
                
                # Call cleanup method
                if hasattr(service, 'cleanup'):
                    service.cleanup()
                elif hasattr(service, 'stop'):
                    service.stop()
                elif hasattr(service, "_shutdown"):
                    service.shutdown()
                    
                # Set shutdown flag
                if hasattr(service, '_shutdown'):
                    service._shutdown = True
                    
            except Exception as e:
                logger.error(f"Error cleaning up service: {e}")

# ─────────────────────────────────────────────────────────────
# MQTT Client Factory
# ─────────────────────────────────────────────────────────────

class MQTTClientFactory:
    """Factory for creating isolated MQTT clients"""
    
    def __init__(self, broker_params: dict):
        self.broker_params = broker_params
        self.clients: weakref.WeakSet = weakref.WeakSet()
        
    def create_client(self, client_id: Optional[str] = None) -> mqtt.Client:
        """Create a new MQTT client with automatic cleanup"""
        if not client_id:
            client_id = f"test_client_{uuid.uuid4().hex[:8]}"
            
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        client.connect(
            self.broker_params['host'],
            self.broker_params['port'],
            self.broker_params.get('keepalive', 60)
        )
        client.loop_start()
        
        # Track for cleanup
        self.clients.add(client)
        
        # Wait for connection
        start_time = time.time()
        while not client.is_connected() and time.time() - start_time < 5:
            time.sleep(0.1)
            
        return client
    
    def cleanup(self):
        """Clean up all created clients"""
        for client in list(self.clients):
            try:
                client.loop_stop()
                client.disconnect()
            except Exception as e:
                logger.warning(f"Error cleaning up MQTT client: {e}")

# ─────────────────────────────────────────────────────────────
# Core Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def thread_monitor():
    """Monitor and cleanup threads"""
    monitor = ThreadManager()
    monitor.start()
    yield monitor
    monitor.cleanup()

@pytest.fixture
def state_manager():
    """Manage service state"""
    manager = StateManager()
    yield manager
    manager.cleanup_all()

@pytest.fixture
def mqtt_broker():
    """Enhanced MQTT broker with session reuse"""
    from enhanced_mqtt_broker import TestMQTTBroker
    
    broker = TestMQTTBroker(session_scope=True)
    broker.start()
    
    # Reset state for this test
    broker.reset_state()
    
    yield broker
    
    # Don't stop session broker

@pytest.fixture
def mqtt_client_factory(mqtt_broker):
    """Factory for creating MQTT clients"""
    factory = MQTTClientFactory(mqtt_broker.get_connection_params())
    yield factory.create_client
    factory.cleanup()

@pytest.fixture
def unique_id():
    """Generate unique IDs for parallel-safe testing"""
    def _generate(prefix: str = "test"):
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    return _generate

# ─────────────────────────────────────────────────────────────
# Service Fixtures with Full Isolation
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_external_deps():
    """Mock all external dependencies"""
    with patch('socket.socket') as mock_socket, \
         patch('subprocess.Popen') as mock_popen, \
         patch('requests.get') as mock_requests, \
         patch('cv2.VideoCapture') as mock_cv2:
        
        # Configure mocks
        mock_sock_instance = Mock()
        mock_sock_instance.settimeout = Mock()
        mock_sock_instance.connect_ex = Mock(return_value=1)  # Port closed
        mock_socket.return_value.__enter__ = Mock(return_value=mock_sock_instance)
        mock_socket.return_value.__exit__ = Mock(return_value=None)
        
        yield {
            'socket': mock_socket,
            'popen': mock_popen,
            'requests': mock_requests,
            'cv2': mock_cv2
        }

@pytest.fixture
def fire_consensus_clean(mqtt_broker, monkeypatch, state_manager, thread_monitor):
    """Create clean FireConsensus instance with full isolation"""
    # Fresh import
    import sys
    if 'fire_consensus.consensus' in sys.modules:
        del sys.modules['fire_consensus.consensus']
    if 'fire_consensus' in sys.modules:
        del sys.modules['fire_consensus']
    
    from fire_consensus.consensus import FireConsensus
    
    # Get connection parameters
    conn_params = mqtt_broker.get_connection_params()
    
    # Set clean environment
    test_env = {
        "CONSENSUS_THRESHOLD": "2",
        "CAMERA_WINDOW": "10",
        "INCREASE_COUNT": "3",
        "DETECTION_COOLDOWN": "0.5",
        "MIN_CONFIDENCE": "0.7",
        "TELEMETRY_INTERVAL": "3600",
        "CLEANUP_INTERVAL": "3600",
        "MQTT_BROKER": conn_params['host'],
        "MQTT_PORT": str(conn_params['port']),
        "MQTT_KEEPALIVE": "60",
        "MQTT_TLS": "false"
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Create service
    service = FireConsensus()
    state_manager.register_service('fire_consensus', service)
    
    # Wait for connection
    start_time = time.time()
    while not service._mqtt_connected and time.time() - start_time < 10:
        time.sleep(0.1)
    
    assert service._mqtt_connected, "Service must connect to MQTT broker"
    
    yield service
    
    # Cleanup
    try:
        service._shutdown = True
        
        # Cancel timers
        if hasattr(service, '_health_timer') and service._health_timer:
            service._health_timer.cancel()
        if hasattr(service, '_cleanup_timer') and service._cleanup_timer:
            service._cleanup_timer.cancel()
        
        # Stop MQTT
        if hasattr(service, "_mqtt_client"):
            service.mqtt_client.loop_stop()
            service.mqtt_client.disconnect()
        
        # Clear state
        service.cameras.clear()
        service.detections.clear()
        
        time.sleep(0.2)
        
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")

@pytest.fixture
def camera_detector_clean(mqtt_broker, monkeypatch, state_manager, thread_monitor, mock_external_deps):
    """Create clean CameraDetector instance with full isolation"""
    # Fresh import
    import sys
    if 'camera_detector.detect' in sys.modules:
        del sys.modules['camera_detector.detect']
    if 'camera_detector' in sys.modules:
        del sys.modules['camera_detector']
    
    from camera_detector.detect import CameraDetector
    
    # Get connection parameters
    conn_params = mqtt_broker.get_connection_params()
    
    # Set clean environment
    test_env = {
        "MQTT_BROKER": conn_params['host'],
        "MQTT_PORT": str(conn_params['port']),
        "DISCOVERY_INTERVAL": "3600",
        "HEALTH_CHECK_INTERVAL": "3600",
        "ONVIF_TIMEOUT": "1",
        "RTSP_TIMEOUT": "1",
        "CAMERA_CREDENTIALS": ""
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    # Create detector
    detector = CameraDetector()
    state_manager.register_service('camera_detector', detector)
    
    # Stop background tasks immediately
    detector.stop_background_tasks()
    
    # Clear any initial state
    detector.cameras.clear()
    detector.mac_tracker.mac_to_ip.clear()
    
    # Wait for connection
    start_time = time.time()
    while not detector._mqtt_connected and time.time() - start_time < 10:
        time.sleep(0.1)
    
    assert detector._mqtt_connected, "Detector must connect to MQTT broker"
    
    yield detector
    
    # Cleanup
    try:
        detector.stop_background_tasks()
        
        if hasattr(detector, "_mqtt_client"):
            detector.mqtt_client.loop_stop()
            detector.mqtt_client.disconnect()
        
        if hasattr(detector, 'executor'):
            detector._executor.shutdown(wait=False)
        
        # Clear state
        detector.cameras.clear()
        detector.mac_tracker.mac_to_ip.clear()
        
        time.sleep(0.2)
        
    except Exception as e:
        logger.error(f"Error during detector cleanup: {e}")

# ─────────────────────────────────────────────────────────────
# Auto-use Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolate_tests(request):
    """Automatic test isolation"""
    # Record test start
    test_name = request.node.name
    logger.debug(f"Starting isolated test: {test_name}")
    
    yield
    
    # Clean up after test
    import gc
    gc.collect()
    
    # Clear module-level state
    import sys
    for module in list(sys.modules.keys()):
        if module.startswith(('fire_consensus', 'camera_detector', 'cam_telemetry')):
            if hasattr(sys.modules[module], '_instances'):
                sys.modules[module]._instances.clear()
            if hasattr(sys.modules[module], '_state'):
                sys.modules[module]._state.clear()
    
    logger.debug(f"Completed isolated test: {test_name}")

@pytest.fixture(autouse=True) 
def cleanup_telemetry():
    """Ensure telemetry is cleaned up after each test"""
    yield
    
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cam_telemetry"))
        import telemetry
        if hasattr(telemetry, 'shutdown_telemetry'):
            telemetry.shutdown_telemetry()
    except:
        pass