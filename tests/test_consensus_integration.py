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

# Note: We'll import FireConsensus dynamically in fixtures after env vars are set
# This prevents ConnectionRefusedError from using default localhost:1883

logger = logging.getLogger(__name__)


def create_growing_fire_detections(camera_id, object_id, base_time, count=8, initial_size=0.03, growth_rate=0.005):
    """Helper to create a series of growing fire detections with proper [x1,y1,x2,y2] format"""
    detections = []
    for i in range(count):
        x1, y1 = 0.1, 0.1  # Top-left corner
        width = initial_size + i * growth_rate  # Growing normalized width
        height = initial_size + i * (growth_rate * 0.8)  # Growing normalized height
        x2 = x1 + width  # Bottom-right x
        y2 = y1 + height  # Bottom-right y
        detections.append({
            'camera_id': camera_id,
            'object': 'fire',
            'object_id': object_id,
            'confidence': 0.8 + i * 0.01,
            'bbox': [x1, y1, x2, y2],  # [x1, y1, x2, y2] format for consensus
            'timestamp': base_time + i * 0.5
        })
    return detections


@pytest.fixture
def consensus_with_env(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Create consensus service with proper environment isolation"""
    # Get unique topic prefix with additional isolation
    full_topic = mqtt_topic_factory("dummy")
    prefix = full_topic.rsplit('/', 1)[0]
    
    # Add process ID and timestamp for complete isolation
    import os
    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000) % 100000}"
    prefix = f"{prefix}_{unique_suffix}"
    
    # Set environment variables using monkeypatch for proper cleanup
    monkeypatch.setenv('TOPIC_PREFIX', prefix)
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')
    monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
    monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'false')
    monkeypatch.setenv('COOLDOWN_PERIOD', '60')
    
    # Ensure broker is running before creating consensus
    assert test_mqtt_broker.is_running(), "MQTT broker not running"
    logger.info(f"MQTT broker running on {test_mqtt_broker.host}:{test_mqtt_broker.port}")
    
    # Import FireConsensus after environment variables are set
    # This ensures the configuration picks up the test broker settings
    from fire_consensus.consensus import FireConsensus
    
    # Create consensus service with auto_connect=False to prevent immediate connection
    consensus = FireConsensus(auto_connect=False)
    
    # Now initialize the MQTT connection after environment is set up
    consensus._initialize_mqtt_connection()
    
    # Wait for MQTT connection using the proper method
    connected = consensus.wait_for_connection(timeout=10.0)
    
    if not connected:
        logger.warning(f"Consensus service may not be connected (prefix: {prefix})")
    
    yield consensus, prefix
    
    # Cleanup with proper shutdown
    try:
        if hasattr(consensus, 'cleanup'):
            consensus.cleanup()
        elif hasattr(consensus, 'shutdown'):
            consensus.shutdown()
        time.sleep(0.5)  # Allow cleanup to complete
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")


@pytest.fixture
def single_camera_consensus(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Consensus service configured for single camera triggering"""
    # Get unique topic prefix with test isolation
    full_topic = mqtt_topic_factory("dummy")
    prefix = full_topic.rsplit('/', 1)[0]
    
    # Add process ID and timestamp for complete isolation
    import os
    unique_suffix = f"{os.getpid()}_{int(time.time() * 1000) % 100000}"
    prefix = f"{prefix}_{unique_suffix}"
    
    # Set environment variables
    monkeypatch.setenv('TOPIC_PREFIX', prefix)
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    monkeypatch.setenv('CONSENSUS_THRESHOLD', '1')
    monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'true')
    monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
    monkeypatch.setenv('COOLDOWN_PERIOD', '5')
    
    # Ensure broker is running before creating consensus
    assert test_mqtt_broker.is_running(), "MQTT broker not running"
    logger.info(f"MQTT broker running on {test_mqtt_broker.host}:{test_mqtt_broker.port}")
    
    # Import FireConsensus after environment variables are set
    from fire_consensus.consensus import FireConsensus
    
    # Create consensus with unique service name to avoid conflicts
    consensus = FireConsensus(auto_connect=False)
    
    # Now initialize the MQTT connection after environment is set up
    consensus._initialize_mqtt_connection()
    
    # Wait for MQTT connection using the proper method
    connected = consensus.wait_for_connection(timeout=10.0)
    connection_verified = connected
    
    if not connection_verified:
        logger.warning(f"Consensus service not connected to MQTT (prefix: {prefix})")
    else:
        logger.info(f"Consensus service connected successfully (prefix: {prefix})")
        # Give a bit of time for subscriptions to complete
        time.sleep(0.5)
    
    yield consensus, prefix
    
    # Cleanup with proper shutdown
    try:
        if hasattr(consensus, 'cleanup'):
            consensus.cleanup()
        elif hasattr(consensus, 'shutdown'):
            consensus.shutdown()
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Error during consensus cleanup: {e}")


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
        assert len(trigger['fire_locations']) > 0
    
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
        assert len(trigger['fire_locations']) > 0


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


class TestCooldownPeriod:
    """Test cooldown period between triggers"""
    
    def test_cooldown_prevents_rapid_triggers(self, single_camera_consensus, mqtt_client):
        """Test cooldown period prevents rapid re-triggering"""
        consensus, prefix = single_camera_consensus
        
        # Create topics
        detection_topic = f"{prefix}/fire/detection"
        trigger_topic = f"{prefix}/fire/trigger"
        telemetry_topic = f"{prefix}/system/camera_telemetry"
        
        # Track triggers with timestamps and events
        received_triggers = []
        trigger_lock = threading.Lock()
        first_trigger_event = threading.Event()
        subscription_ready = threading.Event()
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                with trigger_lock:
                    received_triggers.append({
                        'data': json.loads(msg.payload.decode()),
                        'time': time.time()
                    })
                    if len(received_triggers) == 1:
                        first_trigger_event.set()
        
        def on_subscribe(client, userdata, mid, granted_qos, properties=None):
            subscription_ready.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.on_subscribe = on_subscribe
        result, mid = mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Wait for subscription confirmation
        assert subscription_ready.wait(timeout=5), "Failed to subscribe to trigger topic"
        
        # Also subscribe to all topics for debugging
        mqtt_client.subscribe(f"{prefix}/#")
        
        # Register camera with QoS 1 for reliability
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry), qos=1)
        
        # Give consensus more time to process camera registration
        time.sleep(2.0)
        
        # First burst - should trigger
        base_time = time.time()
        detections = create_growing_fire_detections('cam1', 'fire1', base_time, count=12, initial_size=0.04, growth_rate=0.01)
        
        # Send detections with QoS 1 for reliability
        for i, detection in enumerate(detections):
            mqtt_client.publish(detection_topic, json.dumps(detection), qos=1)
            time.sleep(0.05)  # Faster to stay within detection window
            if i % 3 == 0:
                time.sleep(0.1)  # Occasional longer delay
        
        # Wait for first trigger with event
        assert first_trigger_event.wait(timeout=10), "First detection burst should trigger within 10 seconds"
        
        # Verify we got the trigger
        with trigger_lock:
            triggers_before = len(received_triggers)
        assert triggers_before > 0, f"First detection burst should trigger, but got {triggers_before} triggers"
        
        # Note the time of first trigger
        with trigger_lock:
            first_trigger_time = received_triggers[0]['time']
        
        # Wait a bit to ensure we're in cooldown period
        time.sleep(2)
        
        # Second burst during cooldown - should NOT trigger
        base_time2 = time.time()
        detections2 = create_growing_fire_detections('cam1', 'fire2', base_time2, count=12, initial_size=0.04, growth_rate=0.01)
        
        for detection in detections2:
            mqtt_client.publish(detection_topic, json.dumps(detection), qos=1)
            time.sleep(0.1)
        
        # Wait to see if it triggers (it shouldn't)
        time.sleep(3)
        
        # Should still have same number of triggers
        with trigger_lock:
            triggers_after = len(received_triggers)
        assert triggers_after == triggers_before, f"Cooldown should prevent second trigger (before: {triggers_before}, after: {triggers_after})"
        
        # Calculate how long to wait for cooldown to expire
        # Cooldown is 5 seconds, we've already waited about 5-6 seconds
        elapsed_since_trigger = time.time() - first_trigger_time
        remaining_cooldown = max(0, 6 - elapsed_since_trigger)  # Add 1s buffer for safety
        if remaining_cooldown > 0:
            logger.info(f"Waiting {remaining_cooldown:.1f}s for cooldown to expire...")
            time.sleep(remaining_cooldown)
        
        # Third burst after cooldown - should trigger
        base_time3 = time.time()
        detections3 = create_growing_fire_detections('cam1', 'fire3', base_time3, count=12, initial_size=0.04, growth_rate=0.01)
        
        # Send with QoS 1 for reliability
        for i, detection in enumerate(detections3):
            mqtt_client.publish(detection_topic, json.dumps(detection), qos=1)
            time.sleep(0.05)  # Faster to stay within detection window
            if i % 3 == 0:
                time.sleep(0.1)  # Occasional longer delay
        
        # Wait for new trigger with event-based approach
        new_trigger_event = threading.Event()
        
        def check_for_new_trigger():
            start_wait = time.time()
            while time.time() - start_wait < 10:
                with trigger_lock:
                    if len(received_triggers) > triggers_before:
                        new_trigger_event.set()
                        return
                time.sleep(0.1)
        
        # Run check in thread to avoid blocking
        check_thread = threading.Thread(target=check_for_new_trigger)
        check_thread.start()
        check_thread.join(timeout=11)
        
        assert new_trigger_event.is_set(), "Should trigger after cooldown expires"
        
        # Verify we got a new trigger
        with trigger_lock:
            final_triggers = len(received_triggers)
        assert final_triggers > triggers_before, f"Should have more triggers after cooldown (before: {triggers_before}, final: {final_triggers})"


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
            'bbox': [0.1, 0.1, 0.2, 0.2],  # [x1, y1, x2, y2] format
            'timestamp': time.time()
        }
        
        # Need to send enough valid detections to potentially trigger
        for i in range(6):
            detection = valid_detection.copy()
            detection['timestamp'] = time.time()
            detection['bbox'] = [0.1, 0.1, 0.2 + i*0.01, 0.2 + i*0.01]
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
        
        # Track triggers with event
        received_triggers = []
        trigger_event = threading.Event()
        subscription_ready = threading.Event()
        
        def on_message(client, userdata, msg):
            if trigger_topic in msg.topic:
                received_triggers.append(json.loads(msg.payload.decode()))
                trigger_event.set()
        
        def on_subscribe(client, userdata, mid, granted_qos, properties=None):
            subscription_ready.set()
        
        mqtt_client.on_message = on_message
        mqtt_client.on_subscribe = on_subscribe
        result, mid = mqtt_client.subscribe(f"{trigger_topic}/#")
        
        # Wait for subscription to be confirmed
        assert subscription_ready.wait(timeout=5), "Failed to subscribe to trigger topic"
        
        # Register camera and wait for consensus to process it
        telemetry = {
            'camera_id': 'cam1',
            'status': 'online',
            'timestamp': time.time()
        }
        mqtt_client.publish(telemetry_topic, json.dumps(telemetry), qos=1)
        
        # Give consensus time to process camera registration
        time.sleep(1.0)
        
        # Send low confidence detections - ensure consensus service is ready
        base_time = time.time()
        low_conf_detections = []
        for i in range(10):  # Send more detections
            detection = {
                'camera_id': 'cam1',
                'confidence': 0.5,  # Below threshold of 0.7
                'object': 'fire',
                'object_id': 'fire1',
                'bbox': [0.1, 0.1, 0.2 + i*0.01, 0.2 + i*0.01],  # [x1, y1, x2, y2] format
                'timestamp': base_time + i * 0.1  # Closer together for detection window
            }
            low_conf_detections.append(detection)
            mqtt_client.publish(detection_topic, json.dumps(detection), qos=1)
            time.sleep(0.1)
        
        # Wait for processing with polling
        start_wait = time.time()
        while time.time() - start_wait < 5:
            if len(received_triggers) > 0:
                break  # Unexpected trigger
            time.sleep(0.1)
        
        # Should not trigger
        assert len(received_triggers) == 0, f"Low confidence detections should not trigger, but got {len(received_triggers)} triggers"
        
        # Clear event for high confidence test
        trigger_event.clear()
        
        # Ensure enough time has passed to avoid detection window overlap
        time.sleep(2)
        
        # Now send high confidence detections with fresh timestamps
        new_base_time = time.time()
        high_conf_detections = []
        for i in range(12):  # Send more detections to ensure trigger
            detection = {
                'camera_id': 'cam1',
                'confidence': 0.85,  # Above threshold
                'object': 'fire',
                'object_id': 'fire2',
                'bbox': [0.1, 0.1, 0.25 + i*0.02, 0.25 + i*0.02],  # [x1, y1, x2, y2] format
                'timestamp': new_base_time + i * 0.2  # Closer together
            }
            high_conf_detections.append(detection)
            mqtt_client.publish(detection_topic, json.dumps(detection), qos=1)
            time.sleep(0.1)
        
        # Wait for trigger with timeout
        assert trigger_event.wait(timeout=15), f"High confidence detections should trigger within 15 seconds. Sent {len(high_conf_detections)} detections"
        assert len(received_triggers) > 0, "Should have received at least one trigger"


class TestZoneBasedActivation:
    """Test zone-based activation feature with real MQTT"""
    
    def test_zone_mapping_in_trigger_payload(self, test_mqtt_broker, mqtt_topic_factory, monkeypatch):
        """Test that zone information is included in trigger payload"""
        # Get unique topic prefix
        full_topic = mqtt_topic_factory("dummy")
        prefix = full_topic.rsplit('/', 1)[0]
        
        # Set up environment for zone activation
        monkeypatch.setenv('TOPIC_PREFIX', prefix)  # Critical for refactored services
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('ZONE_ACTIVATION', 'true')
        monkeypatch.setenv('ZONE_MAPPING', '{"cam1": ["zone_a"], "cam2": ["zone_b"]}')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '2')
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        
        # Import after environment setup
        import importlib
        if 'fire_consensus.consensus' in sys.modules:
            del sys.modules['fire_consensus.consensus']
        from fire_consensus.consensus import FireConsensus
        
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
        subscriber.subscribe(f"{prefix}/fire/trigger")  # Use prefix for topic
        subscriber.loop_start()
        
        # Give subscriber time to connect
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus(auto_connect=False)
        
        # Initialize connection after environment is set
        consensus._initialize_mqtt_connection()
        
        # Wait for consensus to connect
        connected = consensus.wait_for_connection(timeout=5.0)
        if not connected:
            raise RuntimeError("Failed to connect consensus to MQTT broker")
        
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
            publisher.publish(f'{prefix}/system/camera_telemetry', json.dumps(telemetry))
        time.sleep(0.5)
        
        # Send growing fire detections from cameras in different zones
        base_time = time.time()
        for camera_id in ['cam1', 'cam2']:
            detections = create_growing_fire_detections(camera_id, f'{camera_id}_fire1', base_time)
            for detection in detections:
                publisher.publish(f'{prefix}/fire/detection', json.dumps(detection))
                time.sleep(0.1)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "Trigger not received within timeout"
        
        # Verify payload includes zone information
        assert len(received_messages) > 0
        payload = received_messages[0]
        
        assert 'zones' in payload
        assert set(payload['zones']) == {'zone_a', 'zone_b'}
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()
    
    def test_zone_activation_disabled(self, test_mqtt_broker, mqtt_topic_factory, monkeypatch):
        """Test trigger payload when zone activation is disabled"""
        # Get unique topic prefix
        full_topic = mqtt_topic_factory("dummy")
        prefix = full_topic.rsplit('/', 1)[0]
        
        # Set up environment without zone activation
        monkeypatch.setenv('TOPIC_PREFIX', prefix)  # Critical for refactored services
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
        monkeypatch.setenv('ZONE_ACTIVATION', 'false')
        monkeypatch.setenv('CONSENSUS_THRESHOLD', '1')
        monkeypatch.setenv('MIN_CONFIDENCE', '0.7')
        monkeypatch.setenv('SINGLE_CAMERA_TRIGGER', 'true')
        
        # Import after environment setup
        import importlib
        if 'fire_consensus.consensus' in sys.modules:
            del sys.modules['fire_consensus.consensus']
        from fire_consensus.consensus import FireConsensus
        
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
        subscriber.subscribe(f"{prefix}/fire/trigger")  # Use prefix for topic
        subscriber.loop_start()
        
        # Give subscriber time to connect
        time.sleep(0.5)
        
        # Create consensus instance
        consensus = FireConsensus(auto_connect=False)
        
        # Initialize connection after environment is set
        consensus._initialize_mqtt_connection()
        
        # Wait for consensus to connect
        connected = consensus.wait_for_connection(timeout=5.0)
        if not connected:
            raise RuntimeError("Failed to connect consensus to MQTT broker")
        
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
        publisher.publish(f'{prefix}/system/camera_telemetry', json.dumps(telemetry))
        time.sleep(0.5)
        
        # Send growing fire detections
        base_time = time.time()
        detections = create_growing_fire_detections('cam1', 'fire1', base_time)
        
        for detection in detections:
            publisher.publish(f'{prefix}/fire/detection', json.dumps(detection))
            time.sleep(0.1)
        
        # Wait for trigger
        assert trigger_event.wait(timeout=5), "Trigger not received within timeout"
        
        # Verify payload
        assert len(received_messages) > 0
        payload = received_messages[0]
        
        # When zone activation is disabled, zones key should not be present
        assert 'zones' not in payload or payload.get('zones') is None
        
        # Cleanup
        publisher.disconnect()
        subscriber.disconnect()
        consensus.cleanup()


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
        if 'fire_consensus.consensus' in sys.modules:
            del sys.modules['fire_consensus.consensus']
        from fire_consensus.consensus import FireConsensus
        
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
        consensus = FireConsensus(auto_connect=False)
        consensus._initialize_mqtt_connection()
        
        # Wait for connection
        connected = consensus.wait_for_connection(timeout=5.0)
        if not connected:
            raise RuntimeError("Failed to connect consensus to MQTT broker")
        
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
        assert trigger['consensus_cameras'] == ['single_cam']
        assert len(trigger['fire_locations']) > 0
        
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
        if 'fire_consensus.consensus' in sys.modules:
            del sys.modules['fire_consensus.consensus']
        from fire_consensus.consensus import FireConsensus
        
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
        consensus = FireConsensus(auto_connect=False)
        consensus._initialize_mqtt_connection()
        
        # Wait for connection
        connected = consensus.wait_for_connection(timeout=5.0)
        if not connected:
            raise RuntimeError("Failed to connect consensus to MQTT broker")
        
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