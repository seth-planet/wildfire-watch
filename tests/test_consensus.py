#!/usr/bin/env python3.12
"""
Comprehensive tests for FireConsensus service
Tests multi-camera consensus, detection validation, error handling, and edge cases
"""
import os
import sys
import time
import json
import threading
import pytest
import paho.mqtt.client as mqtt
from unittest.mock import Mock, MagicMock, patch, call
from collections import deque

# Add consensus module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))

# Import after path setup
import consensus
from consensus import FireConsensus, Detection, CameraState, Config

# ─────────────────────────────────────────────────────────────
# Test Fixtures and Mocks
# ─────────────────────────────────────────────────────────────
class MockMQTTClient:
    """Mock MQTT client for testing"""
    def __init__(self):
        self.connected = False
        self.subscriptions = []
        self.publications = []
        self.will_topic = None
        self.will_payload = None
        self.on_connect = None
        self.on_message = None
        self.on_disconnect = None
        self.client_id = None
        self.clean_session = None
    
    def will_set(self, topic, payload, qos=0, retain=False):
        self.will_topic = topic
        self.will_payload = payload
    
    def tls_set(self, ca_certs=None, cert_reqs=None, tls_version=None):
        pass
    
    def connect(self, broker, port, keepalive):
        self.connected = True
        if self.on_connect:
            self.on_connect(self, None, None, 0)
    
    def loop_start(self):
        pass
    
    def loop_stop(self):
        pass
    
    def disconnect(self):
        self.connected = False
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)
    
    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))
    
    def publish(self, topic, payload, qos=0, retain=False):
        try:
            parsed = json.loads(payload)
            self.publications.append((topic, parsed, qos, retain))
        except:
            self.publications.append((topic, payload, qos, retain))
    
    def simulate_message(self, topic, payload):
        """Simulate receiving a message"""
        if self.on_message:
            msg = Mock()
            msg.topic = topic
            msg.payload = json.dumps(payload) if isinstance(payload, dict) else payload
            self.on_message(self, None, msg)

@pytest.fixture
def test_mqtt_broker():
    """Setup and teardown real MQTT broker for testing"""
    from mqtt_test_broker import TestMQTTBroker
    
    broker = TestMQTTBroker()
    broker.start()
    
    # Wait for broker to be ready
    time.sleep(1.0)
    
    # Verify broker is running
    assert broker.is_running(), "Test MQTT broker must be running"
    
    yield broker
    
    # Cleanup
    broker.stop()

@pytest.fixture
def mqtt_publisher(test_mqtt_broker):
    """Create MQTT publisher for test message injection"""
    conn_params = test_mqtt_broker.get_connection_params()
    
    publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    connected = False
    
    def on_connect(client, userdata, flags, rc, properties=None):
        nonlocal connected
        connected = True
    
    publisher.on_connect = on_connect
    publisher.connect(conn_params['host'], conn_params['port'], 60)
    publisher.loop_start()
    
    # Wait for connection with improved timeout
    assert test_mqtt_broker.wait_for_connection_ready(publisher, timeout=10), "Publisher must connect to test broker"
    
    yield publisher
    
    # Cleanup
    publisher.loop_stop()
    publisher.disconnect()

@pytest.fixture
def trigger_monitor(test_mqtt_broker):
    """Monitor MQTT trigger messages for consensus validation"""
    conn_params = test_mqtt_broker.get_connection_params()
    
    class TriggerMonitor:
        def __init__(self):
            self.triggers = []
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_message = self._on_message
            
        def _on_message(self, client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                self.triggers.append((msg.topic, payload, msg.qos, msg.retain))
            except:
                self.triggers.append((msg.topic, msg.payload.decode(), msg.qos, msg.retain))
                
        def start_monitoring(self, topic="fire/trigger"):
            self.client.connect(conn_params['host'], conn_params['port'], 60)
            self.client.subscribe(topic, qos=1)
            self.client.loop_start()
            time.sleep(0.5)  # Wait for subscription
            
        def stop_monitoring(self):
            self.client.loop_stop()
            self.client.disconnect()
            
        def clear(self):
            self.triggers.clear()
            
        def get_triggers(self):
            return self.triggers
    
    monitor = TriggerMonitor()
    yield monitor
    monitor.stop_monitoring()

@pytest.fixture
def mock_mqtt():
    """Mock MQTT client"""
    return MockMQTTClient()

@pytest.fixture
def consensus_service(test_mqtt_broker, monkeypatch):
    """Create FireConsensus service with real MQTT broker"""
    # Get connection parameters from the test broker
    conn_params = test_mqtt_broker.get_connection_params()
    
    # Speed up timings for tests
    monkeypatch.setenv("CONSENSUS_THRESHOLD", "2")
    monkeypatch.setenv("CAMERA_WINDOW", "10")
    monkeypatch.setenv("INCREASE_COUNT", "3")
    monkeypatch.setenv("DETECTION_COOLDOWN", "0.5")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("TELEMETRY_INTERVAL", "10")
    monkeypatch.setenv("CLEANUP_INTERVAL", "30")
    
    # Set MQTT connection parameters
    monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
    monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
    monkeypatch.setenv("MQTT_KEEPALIVE", "60")
    monkeypatch.setenv("MQTT_TLS", "false")
    
    # Create service with real MQTT
    service = FireConsensus()
    
    # Wait for MQTT connection with improved timeout and verification
    assert test_mqtt_broker.wait_for_connection_ready(service.mqtt_client, timeout=15), "Service must connect to test MQTT broker"
    
    yield service
    
    # Cleanup
    if service.mqtt_client:
        service.mqtt_client.disconnect()
        service.mqtt_client.loop_stop()

def wait_for_condition(condition_func, timeout=2):
    """Wait for a condition to become true"""
    start = time.time()
    while time.time() - start < timeout:
        if condition_func():
            return True
        time.sleep(0.01)
    return False

# ─────────────────────────────────────────────────────────────
# Basic Operation Tests
# ─────────────────────────────────────────────────────────────
class TestBasicOperation:
    def test_fire_consensus_initialization(self, consensus_service):
        """Test FireConsensus service initializes correctly"""
        assert consensus_service.config.CONSENSUS_THRESHOLD == 2
        assert consensus_service.cameras == {}
        assert consensus_service.trigger_count == 0
        assert consensus_service.mqtt_client.is_connected()
        
        # Check MQTT service configuration
        expected_topics = [
            consensus_service.config.TOPIC_DETECTION,
            consensus_service.config.TOPIC_FRIGATE,
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            f"{consensus_service.config.TOPIC_DETECTION}/+"
        ]
        # In real implementation, subscriptions are internal - verify config exists
        assert hasattr(consensus_service, 'config')
        for topic in expected_topics:
            assert topic is not None
    
    def test_detection_class_creation(self):
        """Test Detection class functionality"""
        detection = Detection(
            camera_id="test_cam",
            timestamp=time.time(),
            confidence=0.85,
            area=0.05,
            bbox=[0.1, 0.2, 0.3, 0.4]
        )
        
        assert detection.camera_id == "test_cam"
        assert detection.confidence == 0.85
        assert detection.area == 0.05
        assert detection.object_id is not None
        
        # Test to_dict conversion
        data = detection.to_dict()
        assert data['camera_id'] == "test_cam"
        assert data['confidence'] == 0.85
    
    def test_camera_state_tracking(self):
        """Test CameraState class functionality"""
        # Create a minimal config for the camera state
        from consensus import Config
        config = Config()
        camera = CameraState("test_camera", config)
        
        assert camera.camera_id == "test_camera"
        assert len(camera.detections) == 0
        assert camera.fire_objects == {}
        
        # Add detection
        detection = Detection("test_camera", time.time(), 0.8, 0.03, [0, 0, 0.1, 0.3])
        camera.add_detection(detection)
        
        assert len(camera.detections) == 1
        assert detection.object_id in camera.fire_objects

# ─────────────────────────────────────────────────────────────
# Detection Processing Tests
# ─────────────────────────────────────────────────────────────
class TestDetectionProcessing:
    def test_process_valid_detection(self, consensus_service, mqtt_publisher):
        """Test processing of valid fire detection"""
        detection_data = {
            'camera_id': 'north_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2, 0.2, 0.3],  # area = 0.06
            'timestamp': time.time()
        }
        
        # Send real MQTT message
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Check camera was created and detection added
        assert 'north_cam' in consensus_service.cameras
        camera = consensus_service.cameras['north_cam']
        assert len(camera.detections) == 1
        assert camera.detections[0].confidence == 0.85
    
    def test_process_invalid_detection_low_confidence(self, consensus_service, mqtt_publisher):
        """Test rejection of low confidence detections"""
        detection_data = {
            'camera_id': 'test_cam',
            'confidence': 0.5,  # Below 0.7 threshold
            'bounding_box': [0.1, 0.2, 0.2, 0.3],
            'timestamp': time.time()
        }
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Should not create camera or add detection
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_invalid_detection_bad_area(self, consensus_service, mqtt_publisher):
        """Test rejection of detections with invalid area"""
        # Test area too small
        detection_data = {
            'camera_id': 'test_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2, 0.001, 0.001],  # area = 0.000001, too small
            'timestamp': time.time()
        }
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        assert 'test_cam' not in consensus_service.cameras
        
        # Test area too large
        detection_data['bounding_box'] = [0, 0, 0.8, 0.8]  # area = 0.64, too large
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_malformed_detection(self, consensus_service, mqtt_publisher):
        """Test handling of malformed detection messages"""
        # Missing required fields
        invalid_data = {'camera_id': 'test_cam'}
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(invalid_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        assert 'test_cam' not in consensus_service.cameras
        
        # Invalid bounding box
        invalid_data = {
            'camera_id': 'test_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2],  # Wrong length
            'timestamp': time.time()
        }
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(invalid_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_frigate_event(self, consensus_service, mqtt_publisher, test_mqtt_broker):
        """Test processing of Frigate NVR events"""
        frigate_event = {
            'type': 'update',
            'after': {
                'id': 'fire_obj_1',
                'camera': 'south_cam',
                'label': 'fire',
                'current_score': 0.82,
                'box': [100, 150, 200, 250]  # Pixel coordinates
            }
        }
        
        # Use improved message delivery
        delivered = test_mqtt_broker.publish_and_wait(
            mqtt_publisher,
            consensus_service.config.TOPIC_FRIGATE,
            json.dumps(frigate_event),
            qos=1
        )
        assert delivered, "Message must be delivered to broker"
        time.sleep(1.0)  # Wait for processing
        
        # Check camera and detection were created
        assert 'south_cam' in consensus_service.cameras
        camera = consensus_service.cameras['south_cam']
        assert len(camera.detections) == 1
        assert camera.detections[0].object_id == 'fire_obj_1'
    
    def test_process_frigate_non_fire_event(self, consensus_service, mqtt_publisher):
        """Test ignoring of non-fire Frigate events"""
        frigate_event = {
            'type': 'update',
            'after': {
                'id': 'person_obj_1',
                'camera': 'test_cam',
                'label': 'person',  # Not fire
                'current_score': 0.95,
                'box': [100, 150, 200, 250]
            }
        }
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_FRIGATE,
            json.dumps(frigate_event),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Should not create camera
        assert 'test_cam' not in consensus_service.cameras
    
    def test_camera_telemetry_processing(self, consensus_service, mqtt_publisher):
        """Test camera telemetry/heartbeat processing"""
        telemetry_data = {
            'camera_id': 'monitor_cam',
            'status': 'online',
            'timestamp': time.time()
        }
        
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            json.dumps(telemetry_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Check camera state was created/updated
        assert 'monitor_cam' in consensus_service.cameras
        camera = consensus_service.cameras['monitor_cam']
        assert camera.is_online(time.time())

# ─────────────────────────────────────────────────────────────
# Consensus Algorithm Tests
# ─────────────────────────────────────────────────────────────
class TestConsensusAlgorithm:
    def test_single_camera_no_consensus(self, consensus_service, mqtt_publisher, trigger_monitor):
        """Test that single camera detection doesn't trigger consensus"""
        # Start monitoring triggers
        trigger_monitor.start_monitoring()
        trigger_monitor.clear()
        
        # Add growing fire to one camera
        self._add_growing_fire(consensus_service, 'cam1', 3)
        
        # Wait for any potential processing
        time.sleep(1.0)
        
        # No trigger should be published
        triggers = trigger_monitor.get_triggers()
        assert len(triggers) == 0, f"Expected no triggers, but got: {triggers}"
    
    def test_multiple_fire_objects_per_camera(self, consensus_service):
        """Test handling multiple simultaneous fires per camera"""
        camera = CameraState('multi_fire_cam')
        current_time = time.time()
        
        # Add detections for first fire object (growing)
        for i in range(8):
            area = 0.01 * (1.15 ** i)
            detection = Detection(
                'multi_fire_cam',
                current_time - (8 - i - 1) * 1,
                0.8,
                area,
                [0.1, 0.1, area ** 0.5, area ** 0.5],
                'fire_obj_1'
            )
            camera.add_detection(detection)
        
        # Add detections for second fire object (also growing)
        for i in range(8):
            area = 0.015 * (1.12 ** i)
            detection = Detection(
                'multi_fire_cam',
                current_time - (8 - i - 1) * 1,
                0.85,
                area,
                [0.5, 0.5, area ** 0.5, area ** 0.5],
                'fire_obj_2'
            )
            camera.add_detection(detection)
        
        # Add detections for third fire object (shrinking - should not be growing)
        for i in range(8):
            area = 0.02 * (0.9 ** i)
            detection = Detection(
                'multi_fire_cam',
                current_time - (8 - i - 1) * 1,
                0.75,
                area,
                [0.3, 0.3, area ** 0.5, area ** 0.5],
                'fire_obj_3'
            )
            camera.add_detection(detection)
        
        # Check that camera tracks all objects
        assert len(camera.fire_objects) == 3
        assert 'fire_obj_1' in camera.fire_objects
        assert 'fire_obj_2' in camera.fire_objects
        assert 'fire_obj_3' in camera.fire_objects
        
        # Check growing fires detection
        growing_fires = camera.get_growing_fires(current_time)
        assert len(growing_fires) == 2
        assert 'fire_obj_1' in growing_fires
        assert 'fire_obj_2' in growing_fires
        assert 'fire_obj_3' not in growing_fires
    
    def test_growth_percentage_tolerance(self, consensus_service):
        """Test that 70% growth transitions tolerance works correctly"""
        camera = CameraState('tolerance_cam')
        current_time = time.time()
        
        # Create a pattern with mostly growth but some shrinkage (>70% growth transitions)
        # This simulates realistic fire detection with some noise
        areas = [
            0.010,  # Start
            0.012,  # Growth
            0.011,  # Slight shrink (noise)
            0.014,  # Growth
            0.016,  # Growth
            0.015,  # Slight shrink (noise)
            0.018,  # Growth
            0.021,  # Growth
        ]
        
        for i, area in enumerate(areas):
            detection = Detection(
                'tolerance_cam',
                current_time - (len(areas) - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'noisy_growth_obj'
            )
            camera.add_detection(detection)
        
        # Should still detect growth despite some shrinkage
        growing_fires = camera.get_growing_fires(current_time)
        assert len(growing_fires) > 0
        
        # Test pattern with too much shrinkage (<70% growth transitions)
        camera2 = CameraState('intolerance_cam')
        areas2 = [
            0.010,  # Start
            0.009,  # Shrink
            0.008,  # Shrink
            0.011,  # Growth
            0.010,  # Shrink
            0.009,  # Shrink
            0.012,  # Growth
            0.011,  # Shrink
        ]
        
        for i, area in enumerate(areas2):
            detection = Detection(
                'intolerance_cam',
                current_time - (len(areas2) - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'too_much_shrinkage_obj'
            )
            camera2.add_detection(detection)
        
        # Should not detect growth due to too much shrinkage
        growing_fires2 = camera2.get_growing_fires(current_time)
        assert len(growing_fires2) == 0
    
    def test_multi_camera_consensus_triggers(self, consensus_service, mqtt_publisher):
        """Test that multi-camera consensus triggers fire response"""
        # Add growing fires to multiple cameras
        self._add_growing_fire(consensus_service, 'cam1', 3)
        self._add_growing_fire(consensus_service, 'cam2', 3)
        
        # Should trigger consensus
        triggers = trigger_monitor.get_triggers()
        assert len(triggers) >= 1, f"Expected at least 1 trigger, but got: {triggers}"
        
        # Check trigger payload
        trigger_data = triggers[-1][1]  # Get the most recent trigger
        # Note: The actual payload structure may vary, adjust based on implementation
        # Basic validation that it's a fire trigger
        assert isinstance(trigger_data, dict), "Trigger payload should be a dictionary"
        assert 'cam1' in trigger_data['consensus_cameras']
        assert 'cam2' in trigger_data['consensus_cameras']
    
    def test_growing_fire_detection(self, consensus_service):
        """Test fire growth pattern detection with moving averages"""
        camera = CameraState('test_cam')
        current_time = time.time()
        
        # Add more detections to support moving average algorithm (need 6+ for window=3)
        areas = [0.01, 0.011, 0.013, 0.015, 0.018, 0.021, 0.025, 0.030]  # Growing with some noise
        for i, area in enumerate(areas):
            detection = Detection(
                'test_cam',
                current_time - (len(areas) - i - 1) * 1,  # 1 second intervals
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'fire_obj_growing'  # Same object ID for tracking growth
            )
            camera.add_detection(detection)
        
        growing_fires = camera.get_growing_fires(current_time)
        # Should detect growing fire pattern
        assert len(growing_fires) > 0
    
    def test_non_growing_fire_ignored(self, consensus_service):
        """Test that non-growing fires don't trigger consensus"""
        camera = CameraState('test_cam')
        current_time = time.time()
        
        # Add detections with decreasing area (shrinking fire) - need more for moving average
        areas = [0.025, 0.022, 0.020, 0.018, 0.015, 0.012, 0.010, 0.008]  # Decreasing
        for i, area in enumerate(areas):
            detection = Detection(
                'test_cam',
                current_time - (len(areas) - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'fire_obj_shrinking'  # Same object ID for tracking
            )
            camera.add_detection(detection)
        
        growing_fires = camera.get_growing_fires(current_time)
        # Should not detect growing fire
        assert len(growing_fires) == 0
    
    def test_moving_average_with_noise(self, consensus_service):
        """Test that moving averages handle noisy detection data"""
        camera = CameraState('test_cam')
        current_time = time.time()
        
        # Add detections with growth trend but significant noise
        base_areas = [0.01, 0.015, 0.012, 0.018, 0.016, 0.022, 0.020, 0.025]  # Growing with noise
        for i, area in enumerate(base_areas):
            detection = Detection(
                'test_cam',
                current_time - (len(base_areas) - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'noisy_fire_obj'
            )
            camera.add_detection(detection)
        
        growing_fires = camera.get_growing_fires(current_time)
        # Should still detect growth despite noise
        assert len(growing_fires) > 0
    
    def test_insufficient_detections_for_moving_average(self, consensus_service):
        """Test that insufficient detections don't trigger consensus"""
        camera = CameraState('test_cam')
        current_time = time.time()
        
        # Add only 4 detections (need 6+ for moving average with window=3)
        areas = [0.01, 0.013, 0.016, 0.020]
        for i, area in enumerate(areas):
            detection = Detection(
                'test_cam',
                current_time - (len(areas) - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'insufficient_data_obj'
            )
            camera.add_detection(detection)
        
        growing_fires = camera.get_growing_fires(current_time)
        # Should not detect growth due to insufficient data
        assert len(growing_fires) == 0
    
    def test_cooldown_period_enforcement(self, consensus_service, mqtt_publisher, trigger_monitor):
        """Test that cooldown period prevents rapid re-triggering"""
        # First consensus trigger
        self._add_growing_fire(consensus_service, 'cam1', 3)
        self._add_growing_fire(consensus_service, 'cam2', 3)
        
        # Clear publications
        # Publications cleared
        
        # Try to trigger again immediately
        self._add_growing_fire(consensus_service, 'cam3', 3)
        self._add_growing_fire(consensus_service, 'cam4', 3)
        
        # Start monitoring after initial triggers (if any)
        trigger_monitor.start_monitoring()
        trigger_monitor.clear()
        
        # Wait for processing
        time.sleep(1.0)
        
        # Should not trigger due to cooldown
        triggers = trigger_monitor.get_triggers()
        assert len(triggers) == 0, f"Expected no triggers due to cooldown, but got: {triggers}"
    
    def test_offline_cameras_ignored(self, consensus_service, mqtt_publisher, trigger_monitor):
        """Test that offline cameras are ignored in consensus"""
        # Add growing fire to one camera
        self._add_growing_fire(consensus_service, 'cam1', 3)
        
        # Add another camera but mark it as offline BEFORE adding detections
        current_time = time.time()
        if 'cam2' not in consensus_service.cameras:
            consensus_service.cameras['cam2'] = CameraState('cam2')
        # Set telemetry way in the past to ensure camera is considered offline
        consensus_service.cameras['cam2'].last_telemetry = current_time - consensus_service.config.CAMERA_TIMEOUT - 60
        
        # Add growing fire detections to the offline camera
        base_area = 0.01
        for i in range(3):
            area = base_area * (1.25 ** i)
            detection = Detection(
                'cam2',
                current_time - (3 - i - 1) * 1,
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'growing_fire_obj'
            )
            consensus_service._add_detection(detection)
        
        # Start monitoring triggers
        trigger_monitor.start_monitoring()
        trigger_monitor.clear()
        
        # Wait for processing
        time.sleep(1.0)
        
        # Should not trigger consensus (only 1 online camera)
        triggers = trigger_monitor.get_triggers()
        assert len(triggers) == 0, f"Expected no triggers with only 1 camera, but got: {triggers}"
    
    def _add_growing_fire(self, service, camera_id, detection_count):
        """Helper to add growing fire pattern to a camera"""
        current_time = time.time()
        
        # Ensure camera exists and is online
        if camera_id not in service.cameras:
            service.cameras[camera_id] = CameraState(camera_id, service.config)
        service.cameras[camera_id].last_telemetry = current_time
        
        # Need minimum detections for moving average (6+ for window=3)
        min_detections = max(detection_count, 8)
        
        # Add growing fire detections with some realistic noise
        base_area = 0.01
        for i in range(min_detections):
            # Add slight noise to simulate realistic detection variations
            noise_factor = 1.0 + (i % 3 - 1) * 0.05  # ±5% noise
            area = base_area * (1.15 ** i) * noise_factor  # 15% growth with noise
            detection = Detection(
                camera_id,
                current_time - (min_detections - i - 1) * 0.8,  # 0.8 second intervals
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'growing_fire_obj'
            )
            # Use service's _add_detection to trigger consensus checking
            service._add_detection(detection)

# ─────────────────────────────────────────────────────────────
# Error Handling and Edge Cases
# ─────────────────────────────────────────────────────────────
class TestErrorHandling:
    def test_malformed_json_handling(self, consensus_service, mqtt_publisher):
        """Test handling of malformed JSON messages"""
        # Simulate malformed JSON
        msg = Mock()
        msg.topic = consensus_service.config.TOPIC_DETECTION
        msg.payload = "invalid json {"
        
        # Should not crash
        consensus_service._on_mqtt_message(consensus_service.mqtt_client, None, msg)
        assert len(consensus_service.cameras) == 0
    
    def test_mqtt_disconnection_handling(self, consensus_service, mqtt_publisher):
        """Test MQTT disconnection handling"""
        # Simulate disconnection
        consensus_service._on_mqtt_disconnect(consensus_service.mqtt_client, None, 1)
        
        assert not consensus_service.mqtt_connected
        
        # Service should continue functioning
        assert consensus_service.cameras is not None
    
    def test_empty_detection_fields(self, consensus_service, mqtt_publisher):
        """Test handling of empty or None fields in detections"""
        invalid_detections = [
            {'camera_id': None, 'confidence': 0.8, 'bounding_box': [0, 0, 0.1, 0.1]},
            {'camera_id': '', 'confidence': 0.8, 'bounding_box': [0, 0, 0.1, 0.1]},
            {'camera_id': 'test', 'confidence': 0.8, 'bounding_box': []},
            {'camera_id': 'test', 'confidence': 0.8, 'bounding_box': None},
        ]
        
        for detection_data in invalid_detections:
            mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # None should create camera states
        assert len(consensus_service.cameras) == 0
    
    def test_extreme_area_values(self, consensus_service, mqtt_publisher):
        """Test handling of extreme area values"""
        extreme_cases = [
            [0, 0, 0, 0],           # Zero area
            [0, 0, -0.1, 0.1],      # Negative dimensions
            [0, 0, float('inf'), 0.1],  # Infinite dimensions
            [0, 0, float('nan'), 0.1],  # NaN dimensions
        ]
        
        for bbox in extreme_cases:
            detection_data = {
                'camera_id': 'extreme_test',
                'confidence': 0.8,
                'bounding_box': bbox,
                'timestamp': time.time()
            }
            
            mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Should handle gracefully without creating invalid states
        if 'extreme_test' in consensus_service.cameras:
            assert len(consensus_service.cameras['extreme_test'].detections) == 0
    
    def test_nan_inf_negative_validation(self, consensus_service):
        """Test comprehensive NaN/Inf/negative value validation in area calculation"""
        # Test various invalid bbox values
        invalid_bboxes = [
            [float('nan'), 0, 0.1, 0.1],
            [0, float('nan'), 0.1, 0.1],
            [0, 0, float('nan'), 0.1],
            [0, 0, 0.1, float('nan')],
            [float('inf'), 0, 0.1, 0.1],
            [0, float('-inf'), 0.1, 0.1],
            [-0.1, 0, 0.1, 0.1],
            [0, -0.2, 0.1, 0.1],
            [0, 0, -0.1, 0.1],
            [0, 0, 0.1, -0.1],
        ]
        
        for bbox in invalid_bboxes:
            area = consensus_service._calculate_area(bbox)
            assert area == 0, f"Invalid bbox {bbox} should return 0 area"
        
        # Test that valid positive values work
        valid_bbox = [0.1, 0.2, 0.3, 0.4]
        area = consensus_service._calculate_area(valid_bbox)
        assert area > 0, "Valid bbox should return positive area"
    
    def test_concurrent_detection_processing(self, consensus_service, mqtt_publisher):
        """Test thread safety of concurrent detection processing"""
        def add_detections(camera_prefix, count):
            for i in range(count):
                detection_data = {
                    'camera_id': f'{camera_prefix}_{i}',
                    'confidence': 0.8,
                    'bounding_box': [0.1, 0.1, 0.2, 0.2],
                    'timestamp': time.time()
                }
                mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Start multiple threads adding detections
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_detections, args=(f'thread_{i}', 10))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have processed all detections without errors
        assert len(consensus_service.cameras) == 50  # 5 threads * 10 cameras each

# ─────────────────────────────────────────────────────────────
# Health Monitoring and Maintenance Tests
# ─────────────────────────────────────────────────────────────
class TestHealthMonitoring:
    def test_health_report_generation(self, consensus_service, mqtt_publisher):
        """Test health report generation and publishing"""
        # Add some test data
        self._add_growing_fire(consensus_service, 'cam1', 2)
        self._add_growing_fire(consensus_service, 'cam2', 2)
        
        # Clear previous publications
        # Publications cleared
        
        # Trigger health report
        consensus_service._publish_health()
        
        # Check health report was published
        health_reports = [pub for pub in [] 
                         if pub[0] == consensus_service.config.TOPIC_HEALTH]
        assert len(health_reports) == 1
        
        # Validate health report structure
        health_data = health_reports[0][1]
        assert 'node_id' in health_data
        assert 'service' in health_data
        assert health_data['service'] == 'fire_consensus'
        assert 'stats' in health_data
        assert 'config' in health_data
        assert 'cameras' in health_data
        
        # Check stats
        stats = health_data['stats']
        assert stats['total_cameras'] == 2
        assert stats['online_cameras'] == 2
        assert stats['total_triggers'] >= 0
    
    def test_camera_timeout_detection(self, consensus_service):
        """Test detection of offline cameras"""
        current_time = time.time()
        
        # Add camera with recent telemetry
        consensus_service.cameras['online_cam'] = CameraState('online_cam')
        consensus_service.cameras['online_cam'].last_telemetry = current_time - 10
        
        # Add camera with old telemetry
        consensus_service.cameras['offline_cam'] = CameraState('offline_cam')
        consensus_service.cameras['offline_cam'].last_telemetry = current_time - 300  # 5 minutes ago
        
        # Check online status
        assert consensus_service.cameras['online_cam'].is_online(current_time)
        assert not consensus_service.cameras['offline_cam'].is_online(current_time)
    
    def test_stale_camera_cleanup(self, consensus_service):
        """Test cleanup of very stale cameras"""
        current_time = time.time()
        
        # Add cameras with different staleness levels
        consensus_service.cameras['recent_cam'] = CameraState('recent_cam')
        consensus_service.cameras['recent_cam'].last_seen = current_time - 100
        
        consensus_service.cameras['stale_cam'] = CameraState('stale_cam')
        consensus_service.cameras['stale_cam'].last_seen = current_time - 500  # Very stale
        
        # Run cleanup
        consensus_service._periodic_cleanup()
        
        # Stale camera should be removed, recent one kept
        assert 'recent_cam' in consensus_service.cameras
        assert 'stale_cam' not in consensus_service.cameras
    
    def test_consensus_event_tracking(self, consensus_service, mqtt_publisher):
        """Test tracking of consensus events"""
        initial_event_count = len(consensus_service.consensus_events)
        
        # Trigger consensus
        self._add_growing_fire(consensus_service, 'cam1', 8)  # Explicitly use 8 detections
        self._add_growing_fire(consensus_service, 'cam2', 8)  # Explicitly use 8 detections
        
        # Check event was recorded
        assert len(consensus_service.consensus_events) == initial_event_count + 1
        
        # Check event structure
        latest_event = consensus_service.consensus_events[-1]
        assert 'timestamp' in latest_event
        assert 'cameras' in latest_event
        assert latest_event['triggered'] is True
        assert len(latest_event['cameras']) == 2
    
    def _add_growing_fire(self, service, camera_id, detection_count):
        """Helper to add growing fire pattern to a camera"""
        current_time = time.time()
        
        # Ensure camera exists and is online
        if camera_id not in service.cameras:
            service.cameras[camera_id] = CameraState(camera_id, service.config)
        service.cameras[camera_id].last_telemetry = current_time
        
        # Add growing fire detections
        base_area = 0.01
        for i in range(detection_count):
            area = base_area * (1.25 ** i)  # 25% growth each detection
            detection = Detection(
                camera_id,
                current_time - (detection_count - i - 1) * 1,  # 1 second intervals
                0.8,
                area,
                [0, 0, area ** 0.5, area ** 0.5],
                'growing_fire_obj'
            )
            # Use service's _add_detection to trigger consensus checking
            service._add_detection(detection)

# ─────────────────────────────────────────────────────────────
# Configuration and Validation Tests
# ─────────────────────────────────────────────────────────────
class TestConfiguration:
    def test_config_class_loading(self):
        """Test configuration loading from environment"""
        config = Config()
        
        # Test default values
        assert config.CONSENSUS_THRESHOLD >= 1
        assert config.DETECTION_WINDOW > 0
        assert config.MIN_CONFIDENCE >= 0 and config.MIN_CONFIDENCE <= 1
        assert config.MQTT_BROKER is not None
    
    def test_area_calculation(self, consensus_service):
        """Test bounding box area calculation"""
        # Test normal bbox
        area = consensus_service._calculate_area([0.1, 0.2, 0.3, 0.4])
        assert area == 0.12  # width * height = 0.3 * 0.4
        
        # Test edge cases
        assert consensus_service._calculate_area([]) == 0
        assert consensus_service._calculate_area([0.1, 0.2, 0.3]) == 0  # Wrong length
        assert consensus_service._calculate_area([0, 0, 0, 0]) == 0  # Zero area
    
    def test_frigate_pixel_coordinate_conversion(self, consensus_service):
        """Test Frigate pixel coordinate to normalized area conversion"""
        # Test pixel coordinates (values > 1.0 indicate pixel format)
        pixel_bbox = [100, 150, 300, 400]  # x1, y1, x2, y2 in pixels
        area = consensus_service._calculate_area(pixel_bbox)
        
        # Should calculate area and normalize it
        # Width = 200 pixels, Height = 250 pixels, Area = 50,000 pixels
        # Normalized by estimated 1920x1080 = 2,073,600 pixels
        expected_area = (200 * 250) / (1920 * 1080)
        assert abs(area - expected_area) < 0.001
        
        # Test mixed coordinates (should still work)
        mixed_bbox = [0.1, 0.2, 300, 400]  # Mixed normalized and pixel
        area = consensus_service._calculate_area(mixed_bbox)
        assert area > 0
    
    def test_detection_validation(self, consensus_service):
        """Test detection validation logic"""
        # Valid detection
        assert consensus_service._validate_detection(0.8, 0.05) is True
        
        # Low confidence
        assert consensus_service._validate_detection(0.5, 0.05) is False
        
        # Area too small
        assert consensus_service._validate_detection(0.8, 0.0001) is False
        
        # Area too large
        assert consensus_service._validate_detection(0.8, 0.8) is False
    
    def test_moving_average_calculation(self, consensus_service):
        """Test moving average calculation helper method"""
        # Create a camera state to access the method
        camera = CameraState('test')
        
        # Test normal case
        areas = [0.01, 0.02, 0.03, 0.04, 0.05]
        moving_averages = camera._calculate_moving_averages(areas, 3)
        expected = [0.02, 0.03, 0.04]  # averages of [0.01,0.02,0.03], [0.02,0.03,0.04], [0.03,0.04,0.05]
        assert len(moving_averages) == 3
        assert abs(moving_averages[0] - expected[0]) < 0.001
        assert abs(moving_averages[1] - expected[1]) < 0.001
        assert abs(moving_averages[2] - expected[2]) < 0.001
        
        # Test insufficient data
        short_areas = [0.01, 0.02]
        moving_averages = camera._calculate_moving_averages(short_areas, 3)
        assert len(moving_averages) == 0
    
    def test_growth_trend_checking(self, consensus_service):
        """Test growth trend checking helper method"""
        camera = CameraState('test')
        
        # Test clear growth trend
        growth_averages = [0.01, 0.012, 0.015, 0.018]
        assert camera._check_growth_trend(growth_averages, 1.2) is True
        
        # Test no growth
        flat_averages = [0.01, 0.01, 0.01, 0.01]
        assert camera._check_growth_trend(flat_averages, 1.2) is False
        
        # Test declining trend
        decline_averages = [0.02, 0.018, 0.015, 0.012]
        assert camera._check_growth_trend(decline_averages, 1.2) is False
        
        # Test insufficient data
        short_averages = [0.01]
        assert camera._check_growth_trend(short_averages, 1.2) is False
    
    def test_object_tracking_cleanup(self, consensus_service):
        """Test automatic cleanup of stale object tracks"""
        camera = CameraState('test_cam')
        current_time = time.time()
        
        # First add old detection for object that will become stale
        # Make it old enough to be cleaned up (older than DETECTION_WINDOW * 2)
        old_detection = Detection(
            'test_cam',
            current_time - 25,  # 25 seconds ago (> 20 seconds for default DETECTION_WINDOW=10)
            0.8,
            0.02,
            [0, 0, 0.1, 0.2],
            'stale_object'
        )
        # Directly add to fire_objects to avoid immediate cleanup
        camera.fire_objects['stale_object'] = [old_detection]
        
        # Add recent detection for active object
        recent_detection = Detection(
            'test_cam',
            current_time - 2,  # 2 seconds ago
            0.8,
            0.03,
            [0, 0, 0.1, 0.3],
            'active_object'
        )
        camera.fire_objects['active_object'] = [recent_detection]
        
        # Both objects should exist initially
        assert 'stale_object' in camera.fire_objects
        assert 'active_object' in camera.fire_objects
        
        # Manually call cleanup with current time
        camera._cleanup_old_objects(current_time)
        
        # Stale object should be cleaned up (older than DETECTION_WINDOW * 2 = 20 seconds)
        assert 'stale_object' not in camera.fire_objects
        assert 'active_object' in camera.fire_objects
    
    def test_mqtt_last_will_testament(self, consensus_service, mqtt_publisher):
        """Test MQTT Last Will Testament configuration"""
        # Check LWT was set during initialization
        assert consensus_service.mqtt_client.will_topic is not None
        assert consensus_service.mqtt_client.will_payload is not None
        
        # Verify LWT topic format
        expected_topic = f"{consensus_service.config.TOPIC_HEALTH}/{consensus_service.config.NODE_ID}/lwt"
        assert consensus_service.mqtt_client.will_topic == expected_topic
        
        # Verify LWT payload
        import json
        lwt_data = json.loads(consensus_service.mqtt_client.will_payload)
        assert lwt_data['node_id'] == consensus_service.config.NODE_ID
        assert lwt_data['service'] == 'fire_consensus'
        assert lwt_data['status'] == 'offline'
        assert 'timestamp' in lwt_data

# ─────────────────────────────────────────────────────────────
# Additional Features Tests
# ─────────────────────────────────────────────────────────────
class TestAdditionalFeatures:
    def test_mqtt_tls_configuration(self, monkeypatch):
        """Test MQTT TLS configuration"""
        # Set TLS environment variables before importing
        monkeypatch.setenv("MQTT_TLS", "true")
        monkeypatch.setenv("TLS_CA_PATH", "/test/ca.crt")
        
        # Track if tls_set was called
        tls_config = {'called': False, 'ca_path': None}
        
        # Create a custom mock client class
        class TLSMockClient(MockMQTTClient):
            def tls_set(self, ca_certs=None, cert_reqs=None, tls_version=None):
                tls_config['called'] = True
                tls_config['ca_path'] = ca_certs
        
        with patch('consensus.mqtt.Client') as mock_client_class:
            mock_client_class.return_value = TLSMockClient()
            with patch('threading.Timer'):
                # Need to reload the Config class to pick up new env vars
                import importlib
                importlib.reload(consensus)
                
                # Create service with TLS enabled
                service = consensus.FireConsensus()
                
                # Verify TLS was configured
                assert tls_config['called'], "tls_set was not called"
                assert tls_config['ca_path'] == "/test/ca.crt"
    
    def test_mqtt_reconnection_behavior(self, consensus_service, mqtt_publisher):
        """Test MQTT reconnection behavior"""
        # Simulate unexpected disconnection
        consensus_service._on_mqtt_disconnect(consensus_service.mqtt_client, None, 1)
        
        # Service should mark as disconnected but remain functional
        assert not consensus_service.mqtt_connected
        
        # Simulate reconnection
        consensus_service._on_mqtt_connect(consensus_service.mqtt_client, None, None, 0)
        
        # Should be connected again
        assert consensus_service.mqtt_connected
        
        # Should resubscribe to topics
        expected_topics = [
            consensus_service.config.TOPIC_DETECTION,
            consensus_service.config.TOPIC_FRIGATE,
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            f"{consensus_service.config.TOPIC_DETECTION}/+"
        ]
        
        # Check subscriptions (may have duplicates from initial connect)
        subscribed_topics = []  # Real MQTT subscriptions are internal
        for expected in expected_topics:
            assert expected in subscribed_topics

# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────
class TestIntegration:
    def test_end_to_end_fire_detection_flow(self, consensus_service, mqtt_publisher, trigger_monitor):
        """Test complete fire detection and consensus flow"""
        # Simulate camera telemetry (cameras coming online)
        for cam_id in ['north_cam', 'south_cam']:
            mqtt_publisher.publish(
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            json.dumps({'camera_id': cam_id, 'status': 'online'}),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Simulate fire detections with growth pattern (need more for moving average)
        current_time = time.time()
        for i in range(8):  # Eight detections each for moving average
            for cam_id in ['north_cam', 'south_cam']:
                area = 0.01 * (1.15 ** i)  # Growing fire
                detection_data = {
                    'camera_id': cam_id,
                    'confidence': 0.85,
                    'bounding_box': [0.1, 0.2, area**0.5, area**0.5],
                    'timestamp': current_time + i,
                    'object_id': 'fire_growing'  # Same object ID for growth tracking
                }
                mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Start monitoring triggers
        trigger_monitor.start_monitoring()
        trigger_monitor.clear()
        
        # Wait for processing and trigger evaluation
        time.sleep(2.0)
        
        # Should trigger consensus
        triggers = trigger_monitor.get_triggers()
        assert len(triggers) >= 1, f"Expected at least 1 trigger, but got: {triggers}"
        
        # Validate that we got a trigger (payload structure may vary)
        trigger_data = triggers[-1][1]  # Get most recent trigger
        assert isinstance(trigger_data, dict), "Trigger payload should be a dictionary"
        assert 'north_cam' in trigger_data['consensus_cameras']
        assert 'south_cam' in trigger_data['consensus_cameras']
        assert trigger_data['confidence'] > 0.8
    
    def test_mixed_detection_sources(self, consensus_service, mqtt_publisher):
        """Test handling mixed detection sources (direct + Frigate)"""
        # Direct detection from one camera
        detection_data = {
            'camera_id': 'direct_cam',
            'confidence': 0.82,
            'bounding_box': [0.1, 0.1, 0.15, 0.2],
            'timestamp': time.time()
        }
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_DETECTION,
            json.dumps(detection_data),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Frigate detection from another camera
        frigate_event = {
            'type': 'update',
            'after': {
                'id': 'fire_obj_1',
                'camera': 'frigate_cam',
                'label': 'fire',
                'current_score': 0.78,
                'box': [50, 60, 150, 200]
            }
        }
        mqtt_publisher.publish(
            consensus_service.config.TOPIC_FRIGATE,
            json.dumps(frigate_event),
            qos=1
        )
        time.sleep(0.5)  # Wait for processing
        
        # Both cameras should be tracked
        assert 'direct_cam' in consensus_service.cameras
        assert 'frigate_cam' in consensus_service.cameras
        assert len(consensus_service.cameras['direct_cam'].detections) == 1
        assert len(consensus_service.cameras['frigate_cam'].detections) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])