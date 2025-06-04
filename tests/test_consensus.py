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
def mock_mqtt():
    """Mock MQTT client"""
    return MockMQTTClient()

@pytest.fixture
def consensus_service(mock_mqtt, monkeypatch):
    """Create FireConsensus service with mocked dependencies"""
    # Speed up timings for tests
    monkeypatch.setenv("CONSENSUS_THRESHOLD", "2")
    monkeypatch.setenv("CAMERA_WINDOW", "10")
    monkeypatch.setenv("INCREASE_COUNT", "3")
    monkeypatch.setenv("DETECTION_COOLDOWN", "0.5")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("TELEMETRY_INTERVAL", "10")
    monkeypatch.setenv("CLEANUP_INTERVAL", "30")
    
    # Mock the mqtt.Client constructor
    with patch('consensus.mqtt.Client', return_value=mock_mqtt):
        with patch('threading.Timer') as mock_timer:
            # Create service
            service = FireConsensus()
            service.mqtt_client = mock_mqtt
            mock_mqtt.on_connect = service._on_mqtt_connect
            mock_mqtt.on_message = service._on_mqtt_message
            mock_mqtt.on_disconnect = service._on_mqtt_disconnect
            
            # Simulate successful connection
            service._on_mqtt_connect(mock_mqtt, None, None, 0)
            
            yield service
            
            # Cleanup
            service.mqtt_client = None

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
    def test_fire_consensus_initialization(self, consensus_service, mock_mqtt):
        """Test FireConsensus service initializes correctly"""
        assert consensus_service.config.CONSENSUS_THRESHOLD == 2
        assert consensus_service.cameras == {}
        assert consensus_service.trigger_count == 0
        assert mock_mqtt.connected
        
        # Check MQTT subscriptions
        expected_topics = [
            consensus_service.config.TOPIC_DETECTION,
            consensus_service.config.TOPIC_FRIGATE,
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            f"{consensus_service.config.TOPIC_DETECTION}/+"
        ]
        subscribed_topics = [topic for topic, qos in mock_mqtt.subscriptions]
        for expected in expected_topics:
            assert expected in subscribed_topics
    
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
        camera = CameraState("test_camera")
        
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
    def test_process_valid_detection(self, consensus_service, mock_mqtt):
        """Test processing of valid fire detection"""
        detection_data = {
            'camera_id': 'north_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2, 0.2, 0.3],  # area = 0.06
            'timestamp': time.time()
        }
        
        # Simulate detection message
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            detection_data
        )
        
        # Check camera was created and detection added
        assert 'north_cam' in consensus_service.cameras
        camera = consensus_service.cameras['north_cam']
        assert len(camera.detections) == 1
        assert camera.detections[0].confidence == 0.85
    
    def test_process_invalid_detection_low_confidence(self, consensus_service, mock_mqtt):
        """Test rejection of low confidence detections"""
        detection_data = {
            'camera_id': 'test_cam',
            'confidence': 0.5,  # Below 0.7 threshold
            'bounding_box': [0.1, 0.2, 0.2, 0.3],
            'timestamp': time.time()
        }
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            detection_data
        )
        
        # Should not create camera or add detection
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_invalid_detection_bad_area(self, consensus_service, mock_mqtt):
        """Test rejection of detections with invalid area"""
        # Test area too small
        detection_data = {
            'camera_id': 'test_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2, 0.001, 0.001],  # area = 0.000001, too small
            'timestamp': time.time()
        }
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            detection_data
        )
        
        assert 'test_cam' not in consensus_service.cameras
        
        # Test area too large
        detection_data['bounding_box'] = [0, 0, 0.8, 0.8]  # area = 0.64, too large
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            detection_data
        )
        
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_malformed_detection(self, consensus_service, mock_mqtt):
        """Test handling of malformed detection messages"""
        # Missing required fields
        invalid_data = {'camera_id': 'test_cam'}
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            invalid_data
        )
        
        assert 'test_cam' not in consensus_service.cameras
        
        # Invalid bounding box
        invalid_data = {
            'camera_id': 'test_cam',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.2],  # Wrong length
            'timestamp': time.time()
        }
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            invalid_data
        )
        
        assert 'test_cam' not in consensus_service.cameras
    
    def test_process_frigate_event(self, consensus_service, mock_mqtt):
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
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_FRIGATE,
            frigate_event
        )
        
        # Check camera and detection were created
        assert 'south_cam' in consensus_service.cameras
        camera = consensus_service.cameras['south_cam']
        assert len(camera.detections) == 1
        assert camera.detections[0].object_id == 'fire_obj_1'
    
    def test_process_frigate_non_fire_event(self, consensus_service, mock_mqtt):
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
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_FRIGATE,
            frigate_event
        )
        
        # Should not create camera
        assert 'test_cam' not in consensus_service.cameras
    
    def test_camera_telemetry_processing(self, consensus_service, mock_mqtt):
        """Test camera telemetry/heartbeat processing"""
        telemetry_data = {
            'camera_id': 'monitor_cam',
            'status': 'online',
            'timestamp': time.time()
        }
        
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_CAMERA_TELEMETRY,
            telemetry_data
        )
        
        # Check camera state was created/updated
        assert 'monitor_cam' in consensus_service.cameras
        camera = consensus_service.cameras['monitor_cam']
        assert camera.is_online(time.time())

# ─────────────────────────────────────────────────────────────
# Consensus Algorithm Tests
# ─────────────────────────────────────────────────────────────
class TestConsensusAlgorithm:
    def test_single_camera_no_consensus(self, consensus_service, mock_mqtt):
        """Test that single camera detection doesn't trigger consensus"""
        # Add growing fire to one camera
        self._add_growing_fire(consensus_service, 'cam1', 3)
        
        # No trigger should be published
        triggers = [pub for pub in mock_mqtt.publications 
                   if pub[0] == consensus_service.config.TOPIC_TRIGGER]
        assert len(triggers) == 0
    
    def test_multi_camera_consensus_triggers(self, consensus_service, mock_mqtt):
        """Test that multi-camera consensus triggers fire response"""
        # Add growing fires to multiple cameras
        self._add_growing_fire(consensus_service, 'cam1', 3)
        self._add_growing_fire(consensus_service, 'cam2', 3)
        
        # Should trigger consensus
        triggers = [pub for pub in mock_mqtt.publications 
                   if pub[0] == consensus_service.config.TOPIC_TRIGGER]
        assert len(triggers) == 1
        
        # Check trigger payload
        trigger_data = triggers[0][1]
        assert 'consensus_cameras' in trigger_data
        assert len(trigger_data['consensus_cameras']) == 2
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
    
    def test_cooldown_period_enforcement(self, consensus_service, mock_mqtt):
        """Test that cooldown period prevents rapid re-triggering"""
        # First consensus trigger
        self._add_growing_fire(consensus_service, 'cam1', 3)
        self._add_growing_fire(consensus_service, 'cam2', 3)
        
        # Clear publications
        mock_mqtt.publications.clear()
        
        # Try to trigger again immediately
        self._add_growing_fire(consensus_service, 'cam3', 3)
        self._add_growing_fire(consensus_service, 'cam4', 3)
        
        # Should not trigger due to cooldown
        triggers = [pub for pub in mock_mqtt.publications 
                   if pub[0] == consensus_service.config.TOPIC_TRIGGER]
        assert len(triggers) == 0
    
    def test_offline_cameras_ignored(self, consensus_service, mock_mqtt):
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
        
        # Should not trigger consensus (only 1 online camera)
        triggers = [pub for pub in mock_mqtt.publications 
                   if pub[0] == consensus_service.config.TOPIC_TRIGGER]
        assert len(triggers) == 0
    
    def _add_growing_fire(self, service, camera_id, detection_count):
        """Helper to add growing fire pattern to a camera"""
        current_time = time.time()
        
        # Ensure camera exists and is online
        if camera_id not in service.cameras:
            service.cameras[camera_id] = CameraState(camera_id)
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
    def test_malformed_json_handling(self, consensus_service, mock_mqtt):
        """Test handling of malformed JSON messages"""
        # Simulate malformed JSON
        msg = Mock()
        msg.topic = consensus_service.config.TOPIC_DETECTION
        msg.payload = "invalid json {"
        
        # Should not crash
        consensus_service._on_mqtt_message(mock_mqtt, None, msg)
        assert len(consensus_service.cameras) == 0
    
    def test_mqtt_disconnection_handling(self, consensus_service, mock_mqtt):
        """Test MQTT disconnection handling"""
        # Simulate disconnection
        consensus_service._on_mqtt_disconnect(mock_mqtt, None, 1)
        
        assert not consensus_service.mqtt_connected
        
        # Service should continue functioning
        assert consensus_service.cameras is not None
    
    def test_empty_detection_fields(self, consensus_service, mock_mqtt):
        """Test handling of empty or None fields in detections"""
        invalid_detections = [
            {'camera_id': None, 'confidence': 0.8, 'bounding_box': [0, 0, 0.1, 0.1]},
            {'camera_id': '', 'confidence': 0.8, 'bounding_box': [0, 0, 0.1, 0.1]},
            {'camera_id': 'test', 'confidence': 0.8, 'bounding_box': []},
            {'camera_id': 'test', 'confidence': 0.8, 'bounding_box': None},
        ]
        
        for detection_data in invalid_detections:
            mock_mqtt.simulate_message(
                consensus_service.config.TOPIC_DETECTION,
                detection_data
            )
        
        # None should create camera states
        assert len(consensus_service.cameras) == 0
    
    def test_extreme_area_values(self, consensus_service, mock_mqtt):
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
            
            mock_mqtt.simulate_message(
                consensus_service.config.TOPIC_DETECTION,
                detection_data
            )
        
        # Should handle gracefully without creating invalid states
        if 'extreme_test' in consensus_service.cameras:
            assert len(consensus_service.cameras['extreme_test'].detections) == 0
    
    def test_concurrent_detection_processing(self, consensus_service, mock_mqtt):
        """Test thread safety of concurrent detection processing"""
        def add_detections(camera_prefix, count):
            for i in range(count):
                detection_data = {
                    'camera_id': f'{camera_prefix}_{i}',
                    'confidence': 0.8,
                    'bounding_box': [0.1, 0.1, 0.2, 0.2],
                    'timestamp': time.time()
                }
                mock_mqtt.simulate_message(
                    consensus_service.config.TOPIC_DETECTION,
                    detection_data
                )
        
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
    def test_health_report_generation(self, consensus_service, mock_mqtt):
        """Test health report generation and publishing"""
        # Add some test data
        self._add_growing_fire(consensus_service, 'cam1', 2)
        self._add_growing_fire(consensus_service, 'cam2', 2)
        
        # Clear previous publications
        mock_mqtt.publications.clear()
        
        # Trigger health report
        consensus_service._publish_health()
        
        # Check health report was published
        health_reports = [pub for pub in mock_mqtt.publications 
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
    
    def test_consensus_event_tracking(self, consensus_service, mock_mqtt):
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
            service.cameras[camera_id] = CameraState(camera_id)
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

# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────
class TestIntegration:
    def test_end_to_end_fire_detection_flow(self, consensus_service, mock_mqtt):
        """Test complete fire detection and consensus flow"""
        # Simulate camera telemetry (cameras coming online)
        for cam_id in ['north_cam', 'south_cam']:
            mock_mqtt.simulate_message(
                consensus_service.config.TOPIC_CAMERA_TELEMETRY,
                {'camera_id': cam_id, 'status': 'online'}
            )
        
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
                mock_mqtt.simulate_message(
                    consensus_service.config.TOPIC_DETECTION,
                    detection_data
                )
        
        # Should trigger consensus
        triggers = [pub for pub in mock_mqtt.publications 
                   if pub[0] == consensus_service.config.TOPIC_TRIGGER]
        assert len(triggers) == 1
        
        # Validate complete trigger payload
        trigger_data = triggers[0][1]
        assert trigger_data['camera_count'] == 2
        assert 'north_cam' in trigger_data['consensus_cameras']
        assert 'south_cam' in trigger_data['consensus_cameras']
        assert trigger_data['confidence'] > 0.8
    
    def test_mixed_detection_sources(self, consensus_service, mock_mqtt):
        """Test handling mixed detection sources (direct + Frigate)"""
        # Direct detection from one camera
        detection_data = {
            'camera_id': 'direct_cam',
            'confidence': 0.82,
            'bounding_box': [0.1, 0.1, 0.15, 0.2],
            'timestamp': time.time()
        }
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_DETECTION,
            detection_data
        )
        
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
        mock_mqtt.simulate_message(
            consensus_service.config.TOPIC_FRIGATE,
            frigate_event
        )
        
        # Both cameras should be tracked
        assert 'direct_cam' in consensus_service.cameras
        assert 'frigate_cam' in consensus_service.cameras
        assert len(consensus_service.cameras['direct_cam'].detections) == 1
        assert len(consensus_service.cameras['frigate_cam'].detections) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])