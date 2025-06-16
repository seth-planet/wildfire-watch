#!/usr/bin/env python3.12
"""
Simplified Integration Tests - Tests actual code with mock MQTT
"""

import os
import sys
import time
import json
import pytest
import threading
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'fire_consensus'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'gpio_trigger'))

# Import after path setup
from consensus import FireConsensus, Detection, CameraState
from trigger import PumpController


class TestSimplifiedIntegration:
    """Simplified integration tests that focus on code functionality"""
    
    @patch('consensus.mqtt.Client')
    def test_fire_detection_to_consensus_trigger(self, mock_mqtt_class):
        """Test fire detection flows through consensus to trigger"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Set up environment for consensus
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '2',
            'MQTT_BROKER': 'localhost',
            'MQTT_TLS': 'false'
        }):
            consensus = FireConsensus()
            
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
                
                msg = Mock()
                msg.payload = json.dumps(detection_data).encode()
                msg.topic = "fire/detection"
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
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
                
                msg = Mock()
                msg.payload = json.dumps(detection_data).encode()
                msg.topic = "fire/detection"
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            # Should trigger consensus (2 cameras, growing fire)
            trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                           if 'fire/trigger' in str(call)]
            assert len(trigger_calls) > 0, "Fire consensus should trigger"
            
            # Verify trigger payload format
            trigger_payload = json.loads(trigger_calls[0].args[1])
            assert 'consensus_cameras' in trigger_payload
            assert 'confidence' in trigger_payload
            assert 'timestamp' in trigger_payload
            assert len(trigger_payload['consensus_cameras']) >= 2
    
    @patch('trigger.mqtt.Client')
    def test_trigger_receives_consensus_and_activates_pump(self, mock_mqtt_class):
        """Test GPIO trigger receives consensus and activates pump"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Mock the MQTT connection to prevent blocking
        mock_mqtt.connect = MagicMock()
        mock_mqtt.loop_start = MagicMock()
        mock_mqtt.loop_stop = MagicMock()
        mock_mqtt.disconnect = MagicMock()
        mock_mqtt.subscribe = MagicMock()
        mock_mqtt.publish = MagicMock()
        
        # Set up environment for trigger
        with patch.dict(os.environ, {
            'GPIO_SIMULATION': 'true',
            'MQTT_BROKER': 'localhost',
            'MQTT_TLS': 'false',
            'MAX_ENGINE_RUNTIME': '30',
            'TELEMETRY_INTERVAL': '3600'  # Long interval to prevent health timer issues
        }):
            # Patch the _mqtt_connect_with_retry to use test mode
            with patch.object(PumpController, '_mqtt_connect_with_retry') as mock_connect:
                mock_connect.return_value = None  # Prevent actual connection
                
                # Patch the monitoring thread starts to prevent background threads
                with patch.object(PumpController, '_start_monitoring_tasks'):
                    # Create the controller
                    trigger = PumpController()
                    trigger._test_mode = True  # Set test mode for shorter sleeps
                    
                    # Manually call on_connect to simulate MQTT connection
                    trigger._on_connect(mock_mqtt, None, None, 0)
                    
                    # Import GPIO after PumpController to get the mock
                    from trigger import GPIO
                    
                    # Simulate consensus trigger message
                    consensus_data = {
                        'node_id': 'test-node',
                        'timestamp': time.time(),
                        'trigger_number': 1,
                        'consensus_cameras': ['cam1', 'cam2'],
                        'camera_count': 2,
                        'confidence': 0.85
                    }
                    
                    msg = Mock()
                    msg.payload = json.dumps(consensus_data).encode()
                    msg.topic = "fire/trigger"
                    
                    # Call the message handler
                    trigger._on_message(mock_mqtt, None, msg)
                    
                    # Give the system a moment to process
                    time.sleep(0.1)
                    
                    # Verify pump state changes
                    assert trigger._state.name in ['PRIMING', 'STARTING', 'RUNNING'], f"Pump should be activated, but state is {trigger._state.name}"
                    
                    # Verify main valve was opened
                    assert GPIO.input(trigger.cfg['MAIN_VALVE_PIN']), "Main valve should be opened"
                    
                    # Clean up
                    trigger.cleanup()
    
    @patch('consensus.mqtt.Client')
    def test_detection_validation_logic(self, mock_mqtt_class):
        """Test fire detection validation logic"""
        # Mock MQTT to prevent connection attempts
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Test area calculation
        bbox_pixel = [100, 100, 200, 200]  # 100x100 pixels
        bbox_normalized = [0.1, 0.1, 0.05, 0.05]  # 5% width/height
        
        from consensus import FireConsensus
        
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '2',
            'MQTT_BROKER': 'localhost',
            'MQTT_TLS': 'false'
        }):
            consensus = FireConsensus()
            
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
    
    def test_growing_fire_detection_algorithm(self):
        """Test the growing fire detection algorithm specifically"""
        from consensus import CameraState, Detection
        
        camera = CameraState("test_cam")
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
        camera2 = CameraState("test_cam2")
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
    
    @patch('consensus.mqtt.Client')
    @pytest.mark.skip(reason="Config class reads env vars at import time, making this test unreliable with patch.dict")
    def test_consensus_cooldown_period(self, mock_mqtt_class):
        """Test consensus cooldown prevents rapid triggers"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '1',
            'SINGLE_CAMERA_TRIGGER': 'true',  # Enable single camera trigger
            'COOLDOWN_PERIOD': '10',
            'MQTT_BROKER': 'localhost',
            'MQTT_TLS': 'false'
        }):
            consensus = FireConsensus()
            
            # First trigger - create a growing fire pattern
            base_time = time.time()
            for i in range(8):
                # Increase size significantly to ensure growth is detected
                size = 50 + i * 10  # Growing from 50 to 120
                detection_data = {
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'object_id': 'fire1',
                    'confidence': 0.8,
                    'bounding_box': [100, 100, 100 + size, 100 + size],
                    'timestamp': base_time + i * 0.5
                }
                
                msg = Mock()
                msg.payload = json.dumps(detection_data).encode()
                msg.topic = "fire/detection"
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            initial_triggers = len([call for call in mock_mqtt.publish.call_args_list 
                                  if 'fire/trigger' in str(call)])
            
            # Reset mock
            mock_mqtt.publish.reset_mock()
            
            # Second trigger immediately (should be blocked by cooldown)
            for i in range(8):
                # Same growing pattern but different fire
                size = 50 + i * 10  # Growing from 50 to 120
                detection_data = {
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'object_id': 'fire2',
                    'confidence': 0.8,
                    'bounding_box': [100, 100, 100 + size, 100 + size],
                    'timestamp': base_time + 5 + i * 0.5
                }
                
                msg = Mock()
                msg.payload = json.dumps(detection_data).encode()
                msg.topic = "fire/detection"
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            cooldown_triggers = len([call for call in mock_mqtt.publish.call_args_list 
                                   if 'fire/trigger' in str(call)])
            
            assert initial_triggers > 0, "First trigger should work"
            assert cooldown_triggers == 0, "Second trigger should be blocked by cooldown"


class TestTelemetryReporting:
    """Test telemetry reporting functionality"""
    
    @patch('trigger.mqtt.Client')
    def test_telemetry_reporting(self, mock_mqtt_class):
        """Test that telemetry is published correctly"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Mock the MQTT connection to prevent blocking
        mock_mqtt.connect = MagicMock()
        mock_mqtt.loop_start = MagicMock()
        mock_mqtt.loop_stop = MagicMock()
        mock_mqtt.disconnect = MagicMock()
        mock_mqtt.subscribe = MagicMock()
        mock_mqtt.publish = MagicMock()
        
        with patch.dict(os.environ, {
            'GPIO_SIMULATION': 'true',
            'MQTT_BROKER': 'localhost',
            'MQTT_TLS': 'false',
            'TELEMETRY_INTERVAL': '1'  # Short interval for testing
        }):
            # Patch the _mqtt_connect_with_retry to use test mode
            with patch.object(PumpController, '_mqtt_connect_with_retry') as mock_connect:
                mock_connect.return_value = None
                
                # Patch the monitoring thread starts
                with patch.object(PumpController, '_start_monitoring_tasks'):
                    # Create the controller
                    trigger = PumpController()
                    trigger._test_mode = True
                    
                    # Manually call on_connect
                    trigger._on_connect(mock_mqtt, None, None, 0)
                    
                    # Manually trigger health report
                    trigger._publish_health()
                    
                    # Check that telemetry was published
                    telemetry_calls = [call for call in mock_mqtt.publish.call_args_list
                                     if 'telemetry' in str(call)]
                    assert len(telemetry_calls) > 0, "Telemetry should be published"
                    
                    # Verify telemetry content - at least check we got telemetry
                    # Multiple events may be published (mqtt_connected, health_report, etc)
                    assert any(
                        'timestamp' in json.loads(call.args[1])
                        for call in telemetry_calls
                    ), "Telemetry should contain timestamp"
                    
                    # Clean up
                    trigger.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])