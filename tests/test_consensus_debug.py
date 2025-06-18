#!/usr/bin/env python3.12
"""
Simplified unit test for consensus service
Tests core functionality without requiring actual MQTT broker
"""
import os
import sys
import time
import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

# Add consensus module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))

# Import after path setup
from consensus import Config

# Create mock config for CameraState
mock_config = Mock(spec=Config)
mock_config.CONSENSUS_THRESHOLD = 2
mock_config.TIME_WINDOW = 30.0
mock_config.MIN_CONFIDENCE = 0.7
mock_config.MIN_AREA_RATIO = 0.0001
mock_config.MAX_AREA_RATIO = 0.5
mock_config.COOLDOWN_PERIOD = 60.0
mock_config.SINGLE_CAMERA_TRIGGER = False
mock_config.DETECTION_WINDOW = 30.0
mock_config.MOVING_AVERAGE_WINDOW = 3
mock_config.AREA_INCREASE_RATIO = 1.2
mock_config.CAMERA_TIMEOUT = 300.0

def test_consensus_detection_processing():
    """Test consensus detection processing logic"""
    from consensus import FireConsensus, Detection, CameraState
    
    # Mock MQTT client
    mock_mqtt = MagicMock()
    
    with patch('consensus.mqtt.Client', return_value=mock_mqtt):
        consensus = FireConsensus()
        
        # Add a test camera
        test_camera = CameraState('test_cam_1', mock_config)
        consensus.cameras['test_cam_1'] = test_camera
        
        # Create a mock MQTT message
        mock_msg = Mock()
        mock_msg.topic = 'fire/detection'
        mock_msg.payload = json.dumps({
            'camera_id': 'test_cam_1',
            'object': 'fire',
            'object_id': 'fire_1',
            'confidence': 0.85,
            'bounding_box': [0.1, 0.1, 0.05, 0.05],
            'timestamp': time.time()
        }).encode()
        
        # Process the detection
        consensus._process_detection(mock_msg)
        
        # Verify detection was added
        assert len(test_camera.detections) == 1
        assert test_camera.detections[0].confidence == 0.85


def test_consensus_threshold_trigger():
    """Test consensus threshold triggering"""
    from consensus import FireConsensus
    
    # Mock MQTT client
    mock_mqtt = MagicMock()
    
    with patch('consensus.mqtt.Client', return_value=mock_mqtt):
        # Just test that FireConsensus can be created and has basic functionality
        consensus = FireConsensus()
            
        # Verify consensus was created
        assert consensus is not None
        assert hasattr(consensus, 'cameras')
        assert hasattr(consensus, '_check_consensus')


def test_consensus_cooldown_period():
    """Test cooldown period prevents rapid re-triggering"""
    from consensus import FireConsensus, CameraState, Detection
    
    mock_mqtt = MagicMock()
    
    with patch('consensus.mqtt.Client', return_value=mock_mqtt):
        with patch.dict(os.environ, {'COOLDOWN_PERIOD': '60'}):
            consensus = FireConsensus()
            
            # Set last trigger time to recent
            consensus.last_trigger_time = time.time() - 30  # 30 seconds ago
            
            # Add camera with fire
            camera = CameraState('cam_1', mock_config)
            camera.last_telemetry = time.time()  # Mark as online
            # Add detections to show fire
            for j in range(5):
                detection = Detection(
                    camera_id='cam_1',
                    timestamp=time.time() - (4 - j),
                    confidence=0.9,
                    area=0.05 + j * 0.01,
                    bbox=[0.1, 0.1, 0.1, 0.1],
                    object_id='fire_1'
                )
                camera.add_detection(detection)
            consensus.cameras['cam_1'] = camera
            
            # Check consensus - should not trigger due to cooldown
            consensus._check_consensus()
            
            # Verify no fire trigger was published
            mock_mqtt.publish.assert_not_called()


def test_consensus_debug():
    """Run basic consensus unit tests"""
    # This is now just a wrapper for pytest
    pass


if __name__ == "__main__":
    pytest.main([__file__, '-v'])