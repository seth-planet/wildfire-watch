#!/usr/bin/env python3.12
"""
Enhanced tests for FireConsensus service using real MQTT broker
Tests multi-camera consensus, growing fire detection, and edge cases
Properly handles cleanup and isolation between tests
"""
import os
import sys
import time
import json
import pytest
import threading
import logging
from typing import List, Dict, Any
from unittest import mock
import paho.mqtt.client as mqtt

# Add consensus module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))

# Import after path setup
from consensus import FireConsensus, Detection, CameraState, FireConsensusConfig

logger = logging.getLogger(__name__)


def create_growing_fire_detections(camera_id, object_id, base_time, count=8, initial_size=0.03, growth_rate=0.005):
    """Helper to create a series of growing fire detections with proper normalized format"""
    detections = []
    for i in range(count):
        width = initial_size + i * growth_rate  # Growing normalized width
        height = initial_size + i * (growth_rate * 0.8)  # Growing normalized height
        detections.append({
            'camera_id': camera_id,
            'object': 'fire',
            'object_id': object_id,
            'confidence': 0.8 + i * 0.01,
            'bounding_box': [0.1, 0.1, width, height],  # [x, y, width, height] normalized format
            'timestamp': base_time + i * 0.5
        })
    return detections


@pytest.fixture
def consensus_with_env(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Create consensus service with proper environment isolation"""
    # Get unique topic prefix
    full_topic = mqtt_topic_factory("dummy")
    prefix = full_topic.rsplit('/', 1)[0]
    
    # Set environment variables using monkeypatch for proper cleanup
    monkeypatch.setenv('MQTT_TOPIC_PREFIX', prefix)
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')
    monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
    monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'false')
    monkeypatch.setenv('COOLDOWN_PERIOD', '60')
    
    # Create consensus service
    consensus = FireConsensus()
    
    # Wait for MQTT connection
    start_time = time.time()
    while time.time() - start_time < 10:
        if hasattr(consensus, 'mqtt_connected') and consensus.mqtt_connected:
            time.sleep(0.5)  # Give extra time for subscriptions
            break
        time.sleep(0.1)
    
    yield consensus, prefix
    
    # Cleanup
    try:
        consensus.cleanup()
        time.sleep(0.5)  # Allow cleanup to complete
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")


@pytest.fixture
def single_camera_consensus(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Consensus service configured for single camera triggering"""
    # Get unique topic prefix
    full_topic = mqtt_topic_factory("dummy")
    prefix = full_topic.rsplit('/', 1)[0]
    
    # Set environment variables
    monkeypatch.setenv('MQTT_TOPIC_PREFIX', prefix)
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    monkeypatch.setenv('CONSENSUS_THRESHOLD', '1')
    monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'true')
    monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
    monkeypatch.setenv('COOLDOWN_PERIOD', '5')
    
    consensus = FireConsensus()
    
    # Wait for connection
    start_time = time.time()
    while time.time() - start_time < 10:
        if hasattr(consensus, 'mqtt_connected') and consensus.mqtt_connected:
            time.sleep(0.5)
            break
        time.sleep(0.1)
    
    yield consensus, prefix
    
    # Cleanup
    try:
        consensus.cleanup()
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestConsensusTrigger:
    """Test consensus trigger logic with real MQTT"""
    
    def test_single_camera_multiple_detections_should_trigger(self, single_camera_consensus, mqtt_client):
        """Test that single camera with growing fire triggers consensus"""
        consensus, prefix = single_camera_consensus
        
        # Create topics with the same prefix
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track received messages
        received_triggers = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Send camera telemetry first
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        time.sleep(0.2)
        
        # Send growing fire detections
        base_time = time.time()
        detections = create_growing_fire_detections('cam1', 'fire1', base_time)
        
        for detection in detections:
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.1)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "No trigger received within timeout"
        
        # Verify trigger
        assert len(received_triggers) > 0
        trigger = received_triggers[0]
        assert trigger['consensus_cameras'] == ['cam1']
        assert trigger['camera_count'] == 1
    
    def test_multi_camera_consensus_trigger(self, consensus_with_env, mqtt_client):
        """Test that multiple cameras detecting fire triggers consensus"""
        consensus, prefix = consensus_with_env
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track received messages
        received_triggers = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Register cameras
        for cam_id in ['cam1', 'cam2', 'cam3']:
            telemetry = {
                'camera_id': cam_id,
                'status': 'online',
                'timestamp': time.time()
            }
            mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        
        time.sleep(0.5)
        
        # Send detections from multiple cameras
        base_time = time.time()
        for cam_id in ['cam1', 'cam2']:
            detections = create_growing_fire_detections(cam_id, f'{cam_id}_fire', base_time)
            for detection in detections:  # Send all 8 detections
                mqtt_client.publish(detection_topic, json.dumps(detection))
                time.sleep(0.05)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "No trigger received within timeout"
        
        # Verify trigger
        assert len(received_triggers) > 0
        trigger = received_triggers[0]
        assert set(trigger['consensus_cameras']) == {'cam1', 'cam2'}
        assert trigger['camera_count'] == 2


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestOnlineOfflineCameras:
    """Test handling of online/offline camera states"""
    
    def test_offline_cameras_excluded_from_consensus(self, consensus_with_env, mqtt_client):
        """Test that offline cameras are excluded from consensus calculation"""
        consensus, prefix = consensus_with_env
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track messages
        received_triggers = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Register 3 cameras - cam3 will be offline
        current_time = time.time()
        for cam_id in ['cam1', 'cam2']:
            telemetry = {
                'camera_id': cam_id,
                'status': 'online',
                'timestamp': current_time
            }
            mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        
        # cam3 is offline (old timestamp)
        telemetry = {
            'camera_id': 'cam3',
            'status': 'online',
            'timestamp': current_time - 3600  # 1 hour old
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        
        time.sleep(0.5)
        
        # Send detections from all 3 cameras
        base_time = time.time()
        for cam_id in ['cam1', 'cam2', 'cam3']:
            detections = create_growing_fire_detections(cam_id, f'{cam_id}_fire', base_time)
            for detection in detections[:8]:
                mqtt_client.publish(detection_topic, json.dumps(detection))
                time.sleep(0.05)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "No trigger received within timeout"
        
        # Verify trigger only includes online cameras
        trigger = received_triggers[0]
        assert set(trigger['consensus_cameras']) == {'cam1', 'cam2'}
        assert 'cam3' not in trigger['consensus_cameras']


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestCooldownPeriod:
    """Test cooldown period between triggers"""
    
    def test_cooldown_prevents_rapid_triggers(self, single_camera_consensus, mqtt_client):
        """Test cooldown period prevents rapid re-triggering"""
        consensus, prefix = single_camera_consensus
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track triggers
        received_triggers = []
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append({
                    'data': json.loads(msg.payload.decode()),
                    'time': time.time()
                })
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Register camera
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        time.sleep(0.2)
        
        # First burst - should trigger
        base_time = time.time()
        detections = create_growing_fire_detections('cam1', 'fire1', base_time)
        
        for detection in detections[:8]:  # Send all 8 to ensure trigger
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.05)
        
        # Wait for first trigger
        time.sleep(2)
        
        triggers_before = len(received_triggers)
        assert triggers_before > 0, "First detection burst should trigger"
        
        # Second burst immediately after - should NOT trigger due to cooldown
        base_time2 = time.time()
        detections2 = create_growing_fire_detections('cam1', 'fire2', base_time2)
        
        for detection in detections2[:8]:  # Send all 8 for consistency
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.05)
        
        # Wait to see if it triggers
        time.sleep(2)
        
        # Should still have same number of triggers
        triggers_after = len(received_triggers)
        assert triggers_after == triggers_before, "Cooldown should prevent second trigger"
        
        # Wait for cooldown to expire (5 seconds total in test)
        time.sleep(4)
        
        # Third burst after cooldown - should trigger
        base_time3 = time.time()
        detections3 = create_growing_fire_detections('cam1', 'fire3', base_time3)
        
        for detection in detections3[:8]:  # Send all 8 to ensure trigger
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.05)
        
        time.sleep(2)
        
        # Should now have more triggers
        final_triggers = len(received_triggers)
        assert final_triggers > triggers_before, "Should trigger after cooldown expires"


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_invalid_detection_format_ignored(self, consensus_with_env, mqtt_client):
        """Test that invalid detection messages are ignored gracefully"""
        consensus, prefix = consensus_with_env
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        
        # Send various invalid messages
        invalid_messages = [
            "not json",
            json.dumps({}),  # Missing required fields
            json.dumps({'camera_id': 'cam1'}),  # Missing confidence
            json.dumps({'camera_id': 'cam1', 'confidence': 'not_a_number'}),  # Invalid type
            json.dumps({'camera_id': 'cam1', 'confidence': 0.8, 'bounding_box': 'not_a_list'}),
        ]
        
        for msg in invalid_messages:
            mqtt_client.publish(detection_topic, msg)
            time.sleep(0.1)
        
        # Send valid detection to ensure service is still running
        valid_detection = {
            'camera_id': 'cam1',
            'confidence': 0.85,
            'object': 'fire',
            'object_id': 'fire1',
            'bounding_box': [0.1, 0.1, 0.2, 0.2],
            'timestamp': time.time()
        }
        
        # Need to send enough valid detections to potentially trigger
        for i in range(6):
            detection = valid_detection.copy()
            detection['timestamp'] = time.time()
            detection['bounding_box'] = [0.1, 0.1, 0.2 + i*0.01, 0.2 + i*0.01]
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.1)
        
        time.sleep(2)
        
        # Service should handle malformed messages gracefully
        # Main test is that invalid messages don't crash the service
        assert 'cam1' in consensus.cameras
    
    def test_low_confidence_detections_ignored(self, single_camera_consensus, mqtt_client):
        """Test that low confidence detections are filtered out"""
        consensus, prefix = single_camera_consensus
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track triggers
        received_triggers = []
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Register camera
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry))
        time.sleep(0.2)
        
        # Send low confidence detections
        base_time = time.time()
        for i in range(6):
            detection = {
                'camera_id': 'cam1',
                'confidence': 0.5,  # Below threshold of 0.7
                'object': 'fire',
                'object_id': 'fire1',
                'bounding_box': [0.1, 0.1, 0.2 + i*0.01, 0.2 + i*0.01],
                'timestamp': base_time + i * 0.5
            }
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.1)
        
        time.sleep(2)
        
        # Should not trigger
        assert len(received_triggers) == 0, "Low confidence detections should not trigger"
        
        # Now send high confidence detections
        for i in range(6):
            detection = {
                'camera_id': 'cam1',
                'confidence': 0.85,  # Above threshold
                'object': 'fire',
                'object_id': 'fire2',
                'bounding_box': [0.1, 0.1, 0.2 + i*0.01, 0.2 + i*0.01],
                'timestamp': time.time()
            }
            mqtt_client.publish(detection_topic, json.dumps(detection))
            time.sleep(0.1)
        
        time.sleep(2)
        
        # Should trigger now
        assert len(received_triggers) > 0, "High confidence detections should trigger"


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestZoneBasedActivation:
    """Test zone-based activation feature with real MQTT"""
    
    def test_zone_mapping_in_trigger_payload(self, test_mqtt_broker, monkeypatch):
        """Test that zone information is included in trigger payload"""
        # Set up environment for zone activation
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('ZONE_ACTIVATION', 'true')
        monkeypatch.setenv('ZONE_MAPPING', '{"cam1": "zone_a", "cam2": "zone_b"}')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        
        # Import after environment setup
        import importlib
        if 'consensus' in sys.modules:
            del sys.modules['consensus']
        from consensus import FireConsensus
        
        # Create MQTT subscriber to capture trigger messages
        received_messages = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if 'fire/trigger' in msg.topic:
                received_messages.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_zone_subscriber")
        subscriber.on_message = on_message
        subscriber.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        subscriber.subscribe("fire/trigger")
        subscriber.loop_start()
        
        # Give subscriber time to connect
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus()
        
        # Wait for consensus to connect
        time.sleep(1.0)
        
        # Simulate fire detections from multiple cameras
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_zone_publisher")
        publisher.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        publisher.loop_start()
        
        # Send camera telemetry first
        for camera_id in ['cam1', 'cam2']:
            telemetry = {
                'camera_id': camera_id,
                'status': 'online',
                'timestamp': time.time()
            }
            publisher.publish('system/camera_telemetry', json.dumps(telemetry))
        time.sleep(0.5)
        
        # Send growing fire detections from cameras in different zones
        base_time = time.time()
        for camera_id in ['cam1', 'cam2']:
            detections = create_growing_fire_detections(camera_id, f'{camera_id}_fire1', base_time)
            for detection in detections:
                publisher.publish('fire/detection', json.dumps(detection))
                time.sleep(0.1)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "Trigger not received within timeout"
        
        # Verify payload includes zone information
        assert len(received_messages) > 0
        payload = received_messages[0]
        
        assert 'zones' in payload
        assert 'zone_activation' in payload
        assert payload['zone_activation'] is True
        assert set(payload['zones']) == {'zone_a', 'zone_b'}
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()
    
    def test_zone_activation_disabled(self, test_mqtt_broker, monkeypatch):
        """Test trigger payload when zone activation is disabled"""
        # Set up environment without zone activation
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('ZONE_ACTIVATION', 'false')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '1')
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'true')
        
        # Import after environment setup
        import importlib
        if 'consensus' in sys.modules:
            del sys.modules['consensus']
        from consensus import FireConsensus
        
        # Create MQTT subscriber
        received_messages = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if 'fire/trigger' in msg.topic:
                received_messages.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_nozone_subscriber")
        subscriber.on_message = on_message
        subscriber.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        subscriber.subscribe("fire/trigger")
        subscriber.loop_start()
        
        # Give subscriber time to connect
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus()
        
        # Wait for consensus to connect
        time.sleep(1.0)
        
        # Send detection to trigger
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_nozone_publisher")
        publisher.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        publisher.loop_start()
        
        # Send camera telemetry first
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        publisher.publish('system/camera_telemetry', json.dumps(telemetry))
        time.sleep(0.5)
        
        # Send growing fire detections
        base_time = time.time()
        detections = create_growing_fire_detections('cam1', 'fire1', base_time)
        
        for detection in detections:
            publisher.publish('fire/detection', json.dumps(detection))
            time.sleep(0.1)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "Trigger not received within timeout"
        
        # Verify payload
        assert len(received_messages) > 0
        payload = received_messages[0]
        
        assert payload['zones'] is None
        assert payload['zone_activation'] is False
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestSingleCameraMode:
    """Test single camera trigger mode with real MQTT"""
    
    def test_single_camera_immediate_trigger(self, test_mqtt_broker, monkeypatch):
        """Test single camera triggers immediately when enabled"""
        # Enable single camera mode
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'true')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')  # Still requires 2 normally
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
        
        # Import after environment setup
        import importlib
        if 'consensus' in sys.modules:
            del sys.modules['consensus']
        from consensus import FireConsensus
        
        # Set up MQTT monitoring
        received_triggers = []
        trigger_event = threading.Event()
        
        def on_message(client, userdata, msg):
            if 'fire/trigger' in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_single_subscriber")
        subscriber.on_message = on_message
        subscriber.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        subscriber.subscribe("fire/trigger")
        subscriber.loop_start()
        
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus()
        time.sleep(1.0)
        
        # Send detection from single camera
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_single_publisher")
        publisher.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        publisher.loop_start()
        
        # Send camera telemetry so consensus knows camera is online
        telemetry_msg = {
            'camera_id': 'single_cam',
            'timestamp': time.time(),
            'status': 'online'
        }
        publisher.publish('system/camera_telemetry', json.dumps(telemetry_msg))
        time.sleep(0.5)
        
        # Send growing fire detections - ensure enough for minimum window
        base_time = time.time()
        detections = create_growing_fire_detections('single_cam', 'fire1', base_time, count=12, initial_size=0.02, growth_rate=0.01)
        
        for detection in detections:
            publisher.publish('fire/detection', json.dumps(detection))
            time.sleep(0.05)  # Shorter interval to ensure all within window
        
        # Should trigger with just one camera
        assert trigger_event.wait(timeout=5), "Single camera should trigger immediately"
        
        assert len(received_triggers) > 0
        trigger = received_triggers[0]
        assert trigger['camera_count'] == 1
        assert trigger['consensus_cameras'] == ['single_cam']
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()
    
    def test_single_camera_mode_disabled(self, test_mqtt_broker, monkeypatch):
        """Test single camera does not trigger when mode is disabled"""
        # Disable single camera mode
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'false')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        
        # Import after environment setup
        import importlib
        if 'consensus' in sys.modules:
            del sys.modules['consensus']
        from consensus import FireConsensus
        
        # Set up MQTT monitoring
        received_triggers = []
        
        def on_message(client, userdata, msg):
            if 'fire/trigger' in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
        
        subscriber = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_multi_subscriber")
        subscriber.on_message = on_message
        subscriber.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        subscriber.subscribe("fire/trigger")
        subscriber.loop_start()
        
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus()
        time.sleep(1.0)
        
        # Send detection from single camera
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_multi_publisher")
        publisher.connect(test_mqtt_broker.host, test_mqtt_broker.port)
        publisher.loop_start()
        
        detection = {
            'camera_id': 'single_cam',
            'confidence': 0.9,
            'area': 0.03,
            'timestamp': time.time(),
            'bbox': [100, 100, 250, 250],
            'object_type': 'fire'
        }
        
        # Send detections
        for i in range(8):
            detection['area'] = 0.03 + (i * 0.005)
            detection['timestamp'] = time.time()
            publisher.publish('frigate/single_cam/fire', json.dumps(detection))
            time.sleep(0.1)
        
        # Wait to ensure no trigger
        time.sleep(3)
        
        # Should not trigger with just one camera
        assert len(received_triggers) == 0, "Should not trigger with single camera when disabled"
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])