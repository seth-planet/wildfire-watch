#!/usr/bin/env python3.12
"""
Enhanced tests for FireConsensus service
Tests multi-camera consensus, growing fire detection, and edge cases
"""
import os
import sys
import time
import json
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from collections import deque

# Add consensus module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))

# Import after path setup
from consensus import FireConsensus, Detection, CameraState, Config


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


class TestGrowingFireDetection:
    """Test growing fire detection logic"""
    
    @patch('consensus.mqtt.Client')
    def test_growing_fire_triggers_consensus(self, mock_mqtt_class):
        """Test that only growing fires trigger consensus"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Create consensus with 2 camera threshold
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '2',
            'CONSENSUS_WINDOW': '10'
        }):
            consensus = FireConsensus()
            
            # Simulate fire detections with increasing size
            # Need at least 6 detections for moving average (window=3, min=window*2)
            base_time = time.time()
            detections = []
            
            # Camera 1 - growing fire
            for i in range(8):
                size = 50 + i * 5  # Growing from 50x50 to 85x85
                detections.append({
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'object_id': 'fire1',
                    'confidence': 0.8 + i * 0.01,
                    'bounding_box': [100, 100, 100 + size, 100 + size],
                    'timestamp': base_time + i * 0.5
                })
            
            # Camera 2 - another growing fire
            for i in range(8):
                size = 60 + i * 4  # Growing from 60x60 to 88x88
                detections.append({
                    'camera_id': 'cam2', 
                    'object': 'fire',
                    'object_id': 'fire2',
                    'confidence': 0.75 + i * 0.01,
                    'bounding_box': [200, 200, 200 + size, 200 + size],
                    'timestamp': base_time + i * 0.5
                })
            
            # Process detections
            for detection in detections:
                payload = json.dumps(detection)
                msg = Mock()
                msg.payload = payload.encode()
                msg.topic = "fire/detection"  # Use the correct topic
                
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            # Should trigger consensus (2 cameras, growing fire)
            trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                           if 'fire/trigger' in str(call)]
            assert len(trigger_calls) > 0
    
    @patch('consensus.mqtt.Client')
    def test_shrinking_fire_no_consensus(self, mock_mqtt_class):
        """Test that shrinking fires don't trigger consensus"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '2',
            'CONSENSUS_WINDOW': '10'
        }):
            consensus = FireConsensus()
            
            # Simulate fire detections with decreasing size
            detections = [
                {
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'confidence': 0.8,
                    'bounding_box': [100, 100, 200, 200],  # 100x100 = 10000
                    'timestamp': time.time()
                },
                {
                    'camera_id': 'cam1',
                    'object': 'fire',
                    'confidence': 0.85,
                    'bounding_box': [100, 100, 150, 150],  # 50x50 = 2500 (shrinking)
                    'timestamp': time.time() + 1
                },
                {
                    'camera_id': 'cam2',
                    'object': 'fire',
                    'confidence': 0.82,
                    'bounding_box': [200, 200, 250, 250],  # 50x50 = 2500
                    'timestamp': time.time() + 2
                }
            ]
            
            # Process detections
            for detection in detections:
                payload = json.dumps(detection)
                msg = Mock()
                msg.payload = payload.encode()
                msg.topic = "fire/detection"  # Use the correct topic
                
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            # Should NOT trigger consensus (shrinking fire)
            trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                           if 'fire/trigger' in str(call)]
            assert len(trigger_calls) == 0


class TestFireSizeCalculation:
    """Test fire size calculation from bounding boxes"""
    
    def test_bbox_area_calculation(self):
        """Test bounding box area calculation"""
        # bbox format: [x1, y1, x2, y2]
        bbox1 = [100, 100, 200, 200]  # 100x100 = 10000
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        assert area1 == 10000
        
        bbox2 = [0, 0, 50, 100]  # 50x100 = 5000
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        assert area2 == 5000


class TestConsensusWithOfflineCameras:
    """Test consensus behavior with offline cameras"""
    
    @patch('consensus.mqtt.Client')
    def test_consensus_with_offline_cameras(self, mock_mqtt_class):
        """Test consensus adjusts threshold for offline cameras"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '3',
            'CONSENSUS_WINDOW': '10'
        }):
            consensus = FireConsensus()
            
            # Register 4 cameras, 1 offline
            cameras = ['cam1', 'cam2', 'cam3', 'cam4']
            for cam_id in cameras:
                consensus.cameras[cam_id] = CameraState(cam_id)
            
            # Mark cam4 as offline
            consensus.cameras['cam4'].online = False
            consensus.cameras['cam4'].last_seen = time.time() - 3600
            
            # Fire detections from 2 online cameras (should be enough with adjusted threshold)
            base_time = time.time()
            detections = []
            
            # Growing fires from cam1 and cam2
            detections.extend(create_growing_fire_detections('cam1', 'fire1', base_time))
            detections.extend(create_growing_fire_detections('cam2', 'fire2', base_time))
            
            # Process detections
            for detection in detections:
                payload = json.dumps(detection)
                msg = Mock()
                msg.payload = payload.encode()
                msg.topic = "fire/detection"  # Use the correct topic
                
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            # Should trigger consensus (2/3 online cameras)
            trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                           if 'fire/trigger' in str(call)]
            assert len(trigger_calls) > 0


class TestCooldownPeriod:
    """Test cooldown period between triggers"""
    
    @patch('consensus.mqtt.Client')
    def test_cooldown_prevents_rapid_triggers(self, mock_mqtt_class):
        """Test cooldown period prevents rapid re-triggering"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Prevent background tasks from interfering
        with patch.object(FireConsensus, '_start_background_tasks'), \
             patch.object(FireConsensus, '_mqtt_connect_with_retry'):
            consensus = FireConsensus()
        
        # Directly set config values to ensure they are used
        consensus.config.CONSENSUS_THRESHOLD = 1
        consensus.config.COOLDOWN_PERIOD = 30
        consensus.config.SINGLE_CAMERA_TRIGGER = True
        consensus.config.MOVING_AVERAGE_WINDOW = 3
        # Set MQTT client
        consensus.mqtt_client = mock_mqtt
        
        # First set of detections - ensure enough for moving averages
        base_time = time.time()
        detections = []
        
        # Create 8 growing detections from same object with clear growth pattern
        for i in range(8):
            size_width = 0.03 + i * 0.008  # Growing normalized width
            size_height = 0.03 + i * 0.006  # Growing normalized height
            detections.append({
                'camera_id': 'cam1',
                'object': 'fire',
                'object_id': 'fire1',  # Same object ID for tracking
                'confidence': 0.8 + i * 0.01,
                'bounding_box': [0.1, 0.1, size_width, size_height],  # normalized format
                'timestamp': base_time + i * 0.5
            })
        
        # Process all detections
        for detection in detections:
            msg = Mock()
            msg.payload = json.dumps(detection).encode()
            msg.topic = "fire/detection"
            consensus._on_mqtt_message(mock_mqtt, None, msg)
        
        # Give some time for processing
        time.sleep(0.1)
        
        # Should trigger
        trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                       if 'fire/trigger' in str(call)]
        assert len(trigger_calls) >= 1, f"Expected at least 1 trigger, got {len(trigger_calls)}"
        
        # Second detection immediately after - should not trigger due to cooldown
        mock_mqtt.publish.reset_mock()
        
        # More detections with new fire (different object)
        new_time = time.time() + 5
        new_detections = []
        for i in range(8):
            size_width = 0.035 + i * 0.010  # Different growing pattern
            size_height = 0.035 + i * 0.007
            new_detections.append({
                'camera_id': 'cam1',
                'object': 'fire',
                'object_id': 'fire2',  # Different object ID
                'confidence': 0.75 + i * 0.01,
                'bounding_box': [0.2, 0.2, size_width, size_height],  # normalized format
                'timestamp': new_time + i * 0.5
            })
        
        for detection in new_detections:
            msg = Mock()
            msg.payload = json.dumps(detection).encode()
            msg.topic = "fire/detection"
            consensus._on_mqtt_message(mock_mqtt, None, msg)
        
        # Give time for processing
        time.sleep(0.1)
        
        # Should NOT trigger (in cooldown)
        trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                       if 'fire/trigger' in str(call)]
        assert len(trigger_calls) == 0, f"Expected no triggers due to cooldown, got {len(trigger_calls)}"


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @patch('consensus.mqtt.Client')
    def test_single_camera_mode(self, mock_mqtt_class):
        """Test single camera can trigger if configured"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Prevent background tasks from interfering
        with patch.object(FireConsensus, '_start_background_tasks'), \
             patch.object(FireConsensus, '_mqtt_connect_with_retry'):
            consensus = FireConsensus()
        
        # Directly set config values to ensure they are used
        consensus.config.CONSENSUS_THRESHOLD = 1
        consensus.config.SINGLE_CAMERA_TRIGGER = True
        consensus.config.MOVING_AVERAGE_WINDOW = 3
        consensus.config.COOLDOWN_PERIOD = 0  # No cooldown for this test
        # Set MQTT client
        consensus.mqtt_client = mock_mqtt
        
        # Create growing fire detections from single camera
        base_time = time.time()
        detections = []
        
        # Create 8 growing detections with clear growth pattern
        for i in range(8):
            size_width = 0.025 + i * 0.010  # Growing normalized width
            size_height = 0.025 + i * 0.008  # Growing normalized height
            detections.append({
                'camera_id': 'cam1',
                'object': 'fire',
                'object_id': 'fire1',  # Same object ID for tracking
                'confidence': 0.8 + i * 0.01,
                'bounding_box': [0.1, 0.1, size_width, size_height],  # normalized format
                'timestamp': base_time + i * 0.5
            })
        
        # Process all detections
        for detection in detections:
            msg = Mock()
            msg.payload = json.dumps(detection).encode()
            msg.topic = "fire/detection"
            consensus._on_mqtt_message(mock_mqtt, None, msg)
        
        # Give time for processing
        time.sleep(0.1)
        
        # Should trigger with single camera
        trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                       if 'fire/trigger' in str(call)]
        assert len(trigger_calls) > 0, f"Expected trigger from single camera, got {len(trigger_calls)}"
    
    @patch('consensus.mqtt.Client')
    def test_malformed_detection_handling(self, mock_mqtt_class):
        """Test handling of malformed detection messages"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        consensus = FireConsensus()
        
        # Malformed JSON
        msg = Mock()
        msg.payload = b"not valid json"
        msg.topic = "frigate/cam1/fire"
        
        # Should not crash
        consensus._on_mqtt_message(mock_mqtt, None, msg)
        
        # Missing required fields
        msg.payload = json.dumps({'camera_id': 'cam1'}).encode()
        consensus._on_mqtt_message(mock_mqtt, None, msg)
        
        # No exceptions should be raised


class TestTLSSupport:
    """Test TLS configuration"""
    
    def test_tls_enabled_configuration(self):
        """Test MQTT client configures TLS when enabled"""
        # Mock the mqtt.Client class
        with patch('consensus.mqtt.Client') as mock_mqtt_class:
            mock_mqtt = MagicMock()
            mock_mqtt_class.return_value = mock_mqtt
            
            # Prevent background tasks from interfering
            with patch.object(FireConsensus, '_start_background_tasks'), \
                 patch.object(FireConsensus, '_mqtt_connect_with_retry'):
                
                # Mock environment with TLS enabled
                with patch.dict(os.environ, {'MQTT_TLS': 'true', 'TLS_CA_PATH': '/path/to/ca.crt'}):
                    # Create consensus with TLS config
                    consensus = FireConsensus()
                    # Set TLS config manually to test
                    consensus.config.MQTT_TLS = True
                    consensus.config.TLS_CA_PATH = '/path/to/ca.crt'
                    
                    # Call setup to verify TLS configuration
                    consensus._setup_mqtt()
                    
                    # Verify TLS was configured
                    assert mock_mqtt.tls_set.called
                    call_args = mock_mqtt.tls_set.call_args
                    assert call_args.kwargs['cert_reqs'] is not None
                    assert call_args.kwargs['tls_version'] is not None


class TestHealthMonitoring:
    """Test health monitoring and reporting"""
    
    @patch('consensus.mqtt.Client')
    def test_health_status_publishing(self, mock_mqtt_class):
        """Test periodic health status publishing"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        consensus = FireConsensus()
        
        # Manually trigger health report
        consensus._publish_health()
        
        # Check health status was published
        health_calls = [call for call in mock_mqtt.publish.call_args_list 
                       if 'system/consensus_telemetry' in str(call)]
        assert len(health_calls) > 0
        
        # Verify health payload
        health_call = health_calls[0]
        payload = json.loads(health_call.args[1])
        assert 'status' in payload
        assert 'cameras' in payload
        assert 'timestamp' in payload


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @patch('consensus.mqtt.Client')
    def test_full_fire_detection_flow(self, mock_mqtt_class):
        """Test complete flow from detection to trigger"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '2',
            'MIN_CONFIDENCE': '0.7'
        }):
            consensus = FireConsensus()
            
            # Simulate complete fire detection scenario
            # 1. Camera discovery
            discovery_msg = Mock()
            discovery_msg.payload = json.dumps({
                'camera_id': 'cam1',
                'name': 'North Camera',
                'online': True
            }).encode()
            discovery_msg.topic = 'cameras/discovered/cam1'
            consensus._on_mqtt_message(mock_mqtt, None, discovery_msg)
            
            # 2. Fire detections from multiple cameras
            base_time = time.time()
            detections = []
            
            # Growing fires from cam1 and cam2
            detections.extend(create_growing_fire_detections('cam1', 'fire1', base_time))
            detections.extend(create_growing_fire_detections('cam2', 'fire2', base_time))
            
            for detection in detections:
                msg = Mock()
                msg.payload = json.dumps(detection).encode()
                msg.topic = "fire/detection"  # Use the correct topic
                consensus._on_mqtt_message(mock_mqtt, None, msg)
            
            # 3. Verify trigger
            trigger_calls = [call for call in mock_mqtt.publish.call_args_list 
                           if 'fire/trigger' in str(call)]
            assert len(trigger_calls) > 0
            
            # 4. Verify trigger payload
            trigger_payload = json.loads(trigger_calls[0].args[1])
            assert 'consensus_cameras' in trigger_payload
            assert len(trigger_payload['consensus_cameras']) >= 2
            assert trigger_payload['confidence'] > 0.7
            assert 'timestamp' in trigger_payload
            assert 'trigger_number' in trigger_payload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])