#!/usr/bin/env python3.12
"""
Real Integration Tests - Tests actual services with real MQTT broker
Converted from internal mocking to real implementation testing
"""

import os
import sys
import time
import json
import pytest
import threading
from unittest.mock import patch
from pathlib import Path
import paho.mqtt.client as mqtt

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'fire_consensus'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpio_trigger'))
sys.path.insert(0, str(Path(__file__).parent))

# Import test MQTT broker infrastructure
from mqtt_test_broker import MQTTTestBroker

# Import consensus which doesn't depend on environment setup
from consensus import FireConsensus, Detection, CameraState
# Note: PumpController must be imported AFTER environment setup in tests

@pytest.fixture
def mqtt_monitor(test_mqtt_broker):
    """Setup MQTT message monitoring for testing real MQTT communication"""
    # Storage for captured messages
    captured_messages = []
    
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            # Subscribe to all topics for monitoring
            client.subscribe("#", 0)
    
    def on_message(client, userdata, msg):
        captured_messages.append({
            'topic': msg.topic,
            'payload': msg.payload.decode() if msg.payload else '',
            'qos': msg.qos,
            'retain': msg.retain
        })
    
    # Create monitoring client
    conn_params = test_mqtt_broker.get_connection_params()
    monitor_client = mqtt.Client()
    monitor_client.on_connect = on_connect
    monitor_client.on_message = on_message
    
    # Connect monitor client
    monitor_client.connect(conn_params['host'], conn_params['port'], 60)
    monitor_client.loop_start()
    
    # Wait for connection
    time.sleep(0.5)
    
    class MessageCapture:
        @property
        def messages(self):
            return captured_messages
        
        def get_messages_for_topic(self, topic):
            return [msg for msg in captured_messages if msg['topic'] == topic]
        
        def get_latest_message(self, topic):
            matching = self.get_messages_for_topic(topic)
            return matching[-1] if matching else None
        
        def clear(self):
            captured_messages.clear()
    
    capture = MessageCapture()
    
    yield capture
    
    # Cleanup monitor client
    monitor_client.loop_stop()
    monitor_client.disconnect()

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Ensure proper cleanup after each test"""
    import gc
    import threading
    
    # Store initial state
    initial_threads = threading.active_count()
    
    yield
    
    # Force cleanup
    gc.collect()
    
    # Clean up GPIO state if using simulated GPIO
    try:
        from trigger import GPIO
        if hasattr(GPIO, '_lock'):
            with GPIO._lock:
                if hasattr(GPIO, '_state'):
                    GPIO._state.clear()
        if hasattr(GPIO, 'cleanup'):
            GPIO.cleanup()
    except Exception:
        pass
    
    # Clear any module state
    for module in ['trigger', 'consensus']:
        if module in sys.modules:
            mod = sys.modules[module]
            # Reset any global variables
            if hasattr(mod, 'controller'):
                try:
                    if mod.controller:
                        mod.controller._shutdown = True
                        mod.controller.cleanup()
                    mod.controller = None
                except:
                    pass
    
    # Wait for thread cleanup
    timeout = time.time() + 2.0
    while time.time() < timeout:
        if threading.active_count() <= initial_threads:
            break
        time.sleep(0.1)

class TestRealIntegration:
    """Real integration tests using actual services with real MQTT broker"""
    
    def test_fire_detection_to_consensus_trigger(self, test_mqtt_broker, mqtt_monitor, monkeypatch):
        """Test fire detection flows through consensus to trigger with real MQTT"""
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Configure environment for consensus service
        monkeypatch.setenv("CONSENSUS_THRESHOLD", "2")
        monkeypatch.setenv("CAMERA_WINDOW", "10")
        monkeypatch.setenv("DETECTION_COOLDOWN", "0.5")
        monkeypatch.setenv("MIN_CONFIDENCE", "0.7")
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        
        # Mock threading.Timer to prevent background tasks
        with patch('consensus.threading.Timer'):
            # Create real FireConsensus service
            consensus = FireConsensus()
            
            # Wait for MQTT connection
            time.sleep(1.0)
            
            # Clear any startup messages
            mqtt_monitor.clear()
            
            # Create publisher client for sending detection messages
            publisher = mqtt.Client()
            publisher.connect(conn_params['host'], conn_params['port'], 60)
            publisher.loop_start()
            
            time.sleep(0.5)
            
            # Simulate fire detections from multiple cameras
            base_time = time.time()
            
            # Camera 1 - growing fire
            for i in range(8):
                size = 50 + i * 5
                detection_data = {
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'object_id': 'fire1',
                    'confidence': 0.8 + i * 0.01,
                    'bounding_box': [100, 100, 100 + size, 100 + size],
                    'timestamp': base_time + i * 0.5
                }
                
                # Publish real MQTT message
                publisher.publish(
                    "fire/detection", 
                    json.dumps(detection_data), 
                    qos=1
                )
                time.sleep(0.1)
            
            # Camera 2 - growing fire
            for i in range(8):
                size = 60 + i * 4
                detection_data = {
                    'camera_id': 'cam2',
                    'object': 'fire', 
                    'object_id': 'fire2',
                    'confidence': 0.75 + i * 0.01,
                    'bounding_box': [200, 200, 200 + size, 200 + size],
                    'timestamp': base_time + i * 0.5
                }
                
                # Publish real MQTT message
                publisher.publish(
                    "fire/detection", 
                    json.dumps(detection_data), 
                    qos=1
                )
                time.sleep(0.1)
            
            # Wait for processing
            time.sleep(2.0)
            
            # Check for consensus trigger messages in real MQTT traffic
            trigger_messages = mqtt_monitor.get_messages_for_topic("fire/trigger")
            assert len(trigger_messages) > 0, "Fire consensus should trigger"
            
            # Verify trigger payload format
            trigger_payload = json.loads(trigger_messages[0]['payload'])
            assert 'consensus_cameras' in trigger_payload
            assert 'confidence' in trigger_payload
            assert 'timestamp' in trigger_payload
            assert len(trigger_payload['consensus_cameras']) >= 2
            
            # Cleanup
            publisher.loop_stop()
            publisher.disconnect()
            consensus.cleanup()
    
    def test_trigger_receives_consensus_and_activates_pump(self, test_mqtt_broker, mqtt_monitor, monkeypatch):
        """Test GPIO trigger receives consensus and activates pump with real MQTT"""
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Configure environment for trigger service
        monkeypatch.setenv("GPIO_SIMULATION", "true")
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        monkeypatch.setenv("MAX_ENGINE_RUNTIME", "30")
        monkeypatch.setenv("TELEMETRY_INTERVAL", "3600")  # Long interval to prevent health timer issues
        monkeypatch.setenv("HARDWARE_VALIDATION_ENABLED", "false")
        monkeypatch.setenv("RESERVOIR_FLOAT_PIN", "")  # Disable reservoir monitoring
        monkeypatch.setenv("EMERGENCY_BUTTON_PIN", "")  # Disable emergency button
        
        # Force reload of trigger module to pick up new environment variables
        if 'trigger' in sys.modules:
            del sys.modules['trigger']
        
        # Import trigger module AFTER environment setup
        from trigger import PumpController
        
        # Mock threading.Timer to prevent background tasks
        with patch('trigger.threading.Timer'):
            # Create real PumpController
            trigger = PumpController()
            trigger._test_mode = True  # Set test mode for shorter sleeps
            
            # Wait for MQTT connection
            time.sleep(1.0)
            
            # Clear any startup messages
            mqtt_monitor.clear()
            
            # Import GPIO after PumpController to get the simulated GPIO
            from trigger import GPIO
            
            # Create publisher client for sending trigger message
            publisher = mqtt.Client()
            publisher.connect(conn_params['host'], conn_params['port'], 60)
            publisher.loop_start()
            
            time.sleep(0.5)
            
            # Send real consensus trigger message
            consensus_data = {
                'node_id': 'test-node',
                'timestamp': time.time(),
                'trigger_number': 1,
                'consensus_cameras': ['cam1', 'cam2'],
                'camera_count': 2,
                'confidence': 0.85
            }
            
            # Publish real MQTT message
            publisher.publish(
                "fire/trigger", 
                json.dumps(consensus_data), 
                qos=2
            )
            
            # Wait for processing
            time.sleep(1.0)
            
            # Verify pump state changes
            assert trigger._state.name in ['PRIMING', 'STARTING', 'RUNNING'], f"Pump should be activated, but state is {trigger._state.name}"
            
            # Verify main valve was opened
            assert GPIO.input(trigger.cfg['MAIN_VALVE_PIN']), "Main valve should be opened"
            
            # Check for telemetry messages in real MQTT traffic
            # The trigger service publishes to system/trigger_telemetry, not gpio/status
            telemetry_messages = mqtt_monitor.get_messages_for_topic("system/trigger_telemetry")
            assert len(telemetry_messages) > 0, "Telemetry messages should be published"
            
            # Cleanup
            publisher.loop_stop()
            publisher.disconnect()
            trigger.cleanup()
    
    def test_detection_validation_logic(self, test_mqtt_broker, monkeypatch):
        """Test fire detection validation logic with real service"""
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Configure environment for consensus service
        monkeypatch.setenv("CONSENSUS_THRESHOLD", "2")
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        
        # Mock threading.Timer to prevent background tasks
        with patch('consensus.threading.Timer'):
            # Create real FireConsensus service
            consensus = FireConsensus()
            
            # Wait for MQTT connection
            time.sleep(1.0)
            
            # Test area calculation
            bbox_pixel = [100, 100, 200, 200]  # 100x100 pixels
            bbox_normalized = [0.1, 0.1, 0.05, 0.05]  # 5% width/height
            
            # Pixel format area calculation
            area_pixel = consensus._calculate_area(bbox_pixel)
            assert area_pixel > 0, "Should calculate area for pixel coordinates"
            
            # Normalized format area calculation  
            area_norm = consensus._calculate_area(bbox_normalized)
            assert area_norm > 0, "Should calculate area for normalized coordinates"
            # Use approximate comparison for floating point
            assert abs(area_norm - 0.0025) < 0.0001, f"Expected ~0.0025, got {area_norm}"
            
            # Test validation
            valid_detection = consensus._validate_detection(0.8, 0.01)
            assert valid_detection, "High confidence, reasonable size should be valid"
            
            invalid_confidence = consensus._validate_detection(0.5, 0.01)
            assert not invalid_confidence, "Low confidence should be invalid"
            
            invalid_size = consensus._validate_detection(0.8, 0.0001)
            assert not invalid_size, "Too small area should be invalid"
            
            # Cleanup
            consensus.cleanup()
    
    def test_growing_fire_detection_algorithm(self, test_mqtt_broker, monkeypatch):
        """Test the growing fire detection algorithm with real implementation"""
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Configure environment to use test broker
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        
        # Import and use real consensus module classes
        from consensus import CameraState, Detection, Config
        
        # Create config with test settings
        config = Config()
        
        camera = CameraState("test_cam", config)
        current_time = time.time()
        
        # Create detections with growing area
        detections = []
        for i in range(8):
            size = 50 + i * 10  # Growing from 50 to 120
            area = size * size / (1920 * 1080)  # Normalize
            
            detection = Detection(
                camera_id="test_cam",
                timestamp=current_time + i * 0.5,
                confidence=0.8,
                area=area,
                bbox=[100, 100, 100 + size, 100 + size],
                object_id="fire1"
            )
            detections.append(detection)
            camera.add_detection(detection)
        
        # Test growing fire detection
        growing_fires = camera.get_growing_fires(current_time + 10)
        assert len(growing_fires) > 0, "Should detect growing fire"
        assert "fire1" in growing_fires, "Should detect the specific fire object"
        
        # Test with shrinking fire
        camera2 = CameraState("test_cam2", config)
        for i in range(8):
            size = 120 - i * 10  # Shrinking from 120 to 50
            area = size * size / (1920 * 1080)
            
            detection = Detection(
                camera_id="test_cam2", 
                timestamp=current_time + i * 0.5,
                confidence=0.8,
                area=area,
                bbox=[100, 100, 100 + size, 100 + size],
                object_id="fire2"
            )
            camera2.add_detection(detection)
        
        shrinking_fires = camera2.get_growing_fires(current_time + 10)
        assert len(shrinking_fires) == 0, "Should not detect shrinking fire"

class TestRealTelemetryReporting:
    """Test telemetry reporting functionality with real MQTT"""
    
    def test_telemetry_reporting(self, test_mqtt_broker, mqtt_monitor, monkeypatch):
        """Test that telemetry is published correctly with real MQTT"""
        # Get connection parameters from the test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Configure environment for trigger service
        monkeypatch.setenv("GPIO_SIMULATION", "true")
        monkeypatch.setenv("MQTT_BROKER", conn_params['host'])
        monkeypatch.setenv("MQTT_PORT", str(conn_params['port']))
        monkeypatch.setenv("MQTT_TLS", "false")
        monkeypatch.setenv("TELEMETRY_INTERVAL", "1")  # Short interval for testing
        monkeypatch.setenv("HARDWARE_VALIDATION_ENABLED", "false")
        monkeypatch.setenv("RESERVOIR_FLOAT_PIN", "")  # Disable reservoir monitoring
        monkeypatch.setenv("EMERGENCY_BUTTON_PIN", "")  # Disable emergency button
        
        # Force reload of trigger module to pick up new environment variables
        if 'trigger' in sys.modules:
            del sys.modules['trigger']
        
        # Import trigger module AFTER environment setup
        from trigger import PumpController
        
        # Mock threading.Timer to prevent background tasks
        with patch('trigger.threading.Timer'):
            # Create real PumpController
            trigger = PumpController()
            trigger._test_mode = True
            
            # Wait for MQTT connection
            time.sleep(1.0)
            
            # Clear any startup messages
            mqtt_monitor.clear()
            
            # Manually trigger health report
            trigger._publish_health()
            
            # Wait for message to be published
            time.sleep(0.5)
            
            # Check that telemetry was published to real MQTT
            telemetry_messages = [msg for msg in mqtt_monitor.messages 
                                if 'telemetry' in msg['topic']]
            assert len(telemetry_messages) > 0, "Telemetry should be published"
            
            # Verify telemetry content
            assert any(
                'timestamp' in json.loads(msg['payload'])
                for msg in telemetry_messages
            ), "Telemetry should contain timestamp"
            
            # Clean up
            trigger.cleanup()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])