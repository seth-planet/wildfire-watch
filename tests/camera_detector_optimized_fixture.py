#!/usr/bin/env python3.12
"""
Drop-in replacement for camera_detector fixture with optimized MQTT handling.

This can be imported into conftest.py or test_detect.py to replace the slow fixture.
"""

import pytest
import time
import threading
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# Import from the main codebase
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_detector.detect import CameraDetector, Camera


# Global session-scoped MQTT connections
_mqtt_connections = {}
_mqtt_lock = threading.Lock()


def get_shared_mqtt_connection(broker_params):
    """Get or create a shared MQTT connection for the test session"""
    key = f"{broker_params['host']}:{broker_params['port']}"
    
    with _mqtt_lock:
        if key not in _mqtt_connections:
            # Create new connection
            import paho.mqtt.client as mqtt
            
            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"test-shared-{os.getpid()}",
                clean_session=True
            )
            
            connected_event = threading.Event()
            
            def on_connect(client, userdata, flags, rc, properties=None):
                if rc == 0:
                    connected_event.set()
            
            client.on_connect = on_connect
            
            # Connect
            client.connect(
                broker_params['host'],
                broker_params['port'],
                keepalive=60
            )
            client.loop_start()
            
            # Wait for connection
            if not connected_event.wait(timeout=5):
                raise RuntimeError("Failed to connect to MQTT broker")
            
            _mqtt_connections[key] = {
                'client': client,
                'connected': True,
                'ref_count': 0
            }
        
        # Increment reference count
        _mqtt_connections[key]['ref_count'] += 1
        return _mqtt_connections[key]['client']


def release_shared_mqtt_connection(broker_params):
    """Release a shared MQTT connection"""
    key = f"{broker_params['host']}:{broker_params['port']}"
    
    with _mqtt_lock:
        if key in _mqtt_connections:
            _mqtt_connections[key]['ref_count'] -= 1
            
            # Don't actually disconnect - let session cleanup handle it
            # This keeps the connection alive for other tests


@pytest.fixture
def camera_detector_fast(test_mqtt_broker, network_mocks, mock_onvif, config, monkeypatch):
    """
    Optimized camera detector fixture that reuses MQTT connections.
    
    This is a drop-in replacement for the original camera_detector fixture
    but with ~16x faster setup time.
    """
    # Get connection parameters from the test broker
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Configure MQTT for testing
    monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
    monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
    monkeypatch.setenv("MQTT_TLS", "false")
    
    # Get shared MQTT client
    shared_client = get_shared_mqtt_connection(conn_params)
    
    # Monkey-patch CameraDetector to use our shared client
    original_setup_mqtt = CameraDetector._setup_mqtt
    original_connect = CameraDetector._mqtt_connect_with_retry
    original_start_tasks = CameraDetector._start_background_tasks
    
    def mock_setup_mqtt(self):
        """Use shared MQTT client instead of creating new one"""
        self.mqtt_client = shared_client
        self.mqtt_connected = True
        
        # Still need to set callbacks for this instance
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        # Set LWT for this specific detector
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'camera_detector',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        self.mqtt_client.will_set(lwt_topic, lwt_payload, qos=1, retain=True)
    
    def mock_connect(self):
        """Skip connection - already connected"""
        pass
    
    def mock_start_tasks(self):
        """Don't start background tasks"""
        pass
    
    # Apply patches
    CameraDetector._setup_mqtt = mock_setup_mqtt
    CameraDetector._mqtt_connect_with_retry = mock_connect
    CameraDetector._start_background_tasks = mock_start_tasks
    
    try:
        # Create detector - this will be MUCH faster now
        detector = CameraDetector()
        
        # Stop any background tasks that might have started
        detector._running = False
        
        # Clear any cameras discovered during init
        with detector.lock:
            detector.cameras.clear()
        
        # Add helper methods for controlled task execution
        def run_discovery_once():
            """Run one discovery cycle manually"""
            detector._run_full_discovery()
        
        def run_health_check_once():
            """Run one health check cycle manually"""
            current_time = time.time()
            with detector.lock:
                for mac, camera in list(detector.cameras.items()):
                    if current_time - camera.last_seen > detector.config.OFFLINE_THRESHOLD:
                        if camera.online:
                            camera.online = False
                            camera.stream_active = False
                            detector._publish_camera_status(camera, "offline")
        
        def enable_background_tasks():
            """Enable background tasks for testing scenarios that need them"""
            detector._running = True
            original_start_tasks(detector)
        
        def disable_background_tasks():
            """Disable background tasks for controlled testing"""
            detector._running = False
        
        # Attach helper methods
        detector.test_run_discovery_once = run_discovery_once
        detector.test_run_health_check_once = run_health_check_once
        detector.test_enable_background_tasks = enable_background_tasks
        detector.test_disable_background_tasks = disable_background_tasks
        
        yield detector
        
    finally:
        # Restore original methods
        CameraDetector._setup_mqtt = original_setup_mqtt
        CameraDetector._mqtt_connect_with_retry = original_connect
        CameraDetector._start_background_tasks = original_start_tasks
        
        # Cleanup detector state
        try:
            detector._running = False
            
            # Clear cameras
            with detector.lock:
                detector.cameras.clear()
            
            # Don't disconnect the shared client!
            # Just clear this detector's state
            detector.mqtt_client = None
            
        except Exception as e:
            print(f"Detector cleanup error: {e}")
        
        # Release our reference to the shared connection
        release_shared_mqtt_connection(conn_params)


# Session cleanup
def pytest_sessionfinish(session, exitstatus):
    """Clean up all shared MQTT connections at end of session"""
    with _mqtt_lock:
        for key, conn_info in _mqtt_connections.items():
            try:
                conn_info['client'].loop_stop()
                conn_info['client'].disconnect()
            except:
                pass
        _mqtt_connections.clear()


# Alternative: Modify the existing fixture in-place
def optimize_camera_detector_fixture():
    """
    Call this function to monkey-patch the existing fixture for better performance.
    
    Usage in conftest.py:
        from camera_detector_optimized_fixture import optimize_camera_detector_fixture
        optimize_camera_detector_fixture()
    """
    import tests.test_detect
    
    # Store original fixture
    original_fixture = tests.test_detect.camera_detector
    
    # Replace with optimized version
    tests.test_detect.camera_detector = camera_detector_fast
    
    return original_fixture  # In case someone needs it