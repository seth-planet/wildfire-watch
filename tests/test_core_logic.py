#!/usr/bin/env python3.12
"""
Core Logic Tests - Tests functionality with real MQTT connections
"""

import os
import sys
import time
import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'fire_consensus'))
sys.path.insert(0, str(Path(__file__).parent))

# Import test helpers
from helpers import mqtt_test_environment, MqttMessageListener

# Import consensus classes
from consensus import Detection, CameraState, FireConsensusConfig, FireConsensus


# Create mock config for CameraState
mock_config = Mock(spec=FireConsensusConfig)
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

class TestCoreLogic:
    """Test core logic with real MQTT connections for integration testing"""
    
    def test_detection_area_calculation(self, test_mqtt_broker, monkeypatch):
        """Test detection area calculation logic"""
        # Set up MQTT environment
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            # Create real consensus instance with real MQTT
            consensus = FireConsensus()
            
            # Test pixel format (values > 1)
            bbox_pixel = [100, 100, 200, 200]  # 100x100 pixels
            area_pixel = consensus._calculate_area(bbox_pixel)
            expected_pixel = (100 * 100) / (1920 * 1080)  # Normalized
            assert abs(area_pixel - expected_pixel) < 0.001, f"Expected {expected_pixel}, got {area_pixel}"
            
            # Test normalized format (values < 1)
            bbox_norm = [0.1, 0.1, 0.05, 0.05]  # 5% width/height
            area_norm = consensus._calculate_area(bbox_norm)
            assert abs(area_norm - 0.0025) < 0.000001, f"Expected 0.0025, got {area_norm}"
            
            # Test invalid bbox
            bbox_invalid = [0, 0, 0, 0]
            area_invalid = consensus._calculate_area(bbox_invalid)
            assert area_invalid == 0, "Invalid bbox should return 0 area"
            
            # Cleanup
            consensus.cleanup()
    
    def test_detection_validation(self, test_mqtt_broker, monkeypatch):
        """Test detection validation logic"""
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            # Create real consensus instance
            consensus = FireConsensus()
            
            # Valid detection
            valid = consensus._validate_detection(0.8, 0.01)
            assert valid, "High confidence, reasonable size should be valid"
            
            # Invalid confidence
            invalid_conf = consensus._validate_detection(0.5, 0.01)
            assert not invalid_conf, "Low confidence should be invalid"
            
            # Invalid size (too small)
            invalid_small = consensus._validate_detection(0.8, 0.0001)
            assert not invalid_small, "Too small area should be invalid"
            
            # Invalid size (too large)
            invalid_large = consensus._validate_detection(0.8, 0.6)
            assert not invalid_large, "Too large area should be invalid"
            
            # Cleanup
            consensus.cleanup()
    
    def test_camera_state_detection_tracking(self):
        """Test camera state detection tracking"""
        camera = CameraState("test_cam", mock_config)
        current_time = time.time()
        
        # Add detections
        detection1 = Detection(
            camera_id="test_cam",
            timestamp=current_time,
            confidence=0.8,
            area=0.01,
            bbox=[100, 100, 150, 150],
            object_id="fire1"
        )
        
        detection2 = Detection(
            camera_id="test_cam", 
            timestamp=current_time + 1,
            confidence=0.85,
            area=0.015,
            bbox=[100, 100, 160, 160],
            object_id="fire1"
        )
        
        camera.add_detection(detection1)
        camera.add_detection(detection2)
        
        # Verify tracking
        assert len(camera.detections) == 2
        assert "fire1" in camera.fire_objects
        assert len(camera.fire_objects["fire1"]) == 2
        assert camera.last_seen == current_time + 1
    
    def test_growing_fire_detection_algorithm(self):
        """Test growing fire detection algorithm"""
        camera = CameraState("test_cam", mock_config)
        current_time = time.time()
        
        # Create detections with growing area
        for i in range(8):
            size = 50 + i * 10  # Growing from 50 to 120
            area = (size * size) / (1920 * 1080)  # Normalize
            
            detection = Detection(
                camera_id="test_cam",
                timestamp=current_time + i * 0.5,
                confidence=0.8,
                area=area,
                bbox=[100, 100, 100 + size, 100 + size],
                object_id="fire1"
            )
            camera.add_detection(detection)
        
        # Test growing fire detection
        growing_fires = camera.get_growing_fires(current_time + 10)
        assert len(growing_fires) > 0, "Should detect growing fire"
        assert "fire1" in growing_fires, "Should detect the specific fire object"
    
    def test_shrinking_fire_not_detected(self):
        """Test shrinking fires are not detected as growing"""
        camera = CameraState("test_cam", mock_config)
        current_time = time.time()
        
        # Create detections with shrinking area
        for i in range(8):
            size = 120 - i * 10  # Shrinking from 120 to 50
            area = (size * size) / (1920 * 1080)
            
            detection = Detection(
                camera_id="test_cam",
                timestamp=current_time + i * 0.5,
                confidence=0.8,
                area=area,
                bbox=[100, 100, 100 + size, 100 + size],
                object_id="fire2"
            )
            camera.add_detection(detection)
        
        shrinking_fires = camera.get_growing_fires(current_time + 10)
        assert len(shrinking_fires) == 0, "Should not detect shrinking fire"
    
    def test_moving_average_calculation(self):
        """Test moving average calculation"""
        camera = CameraState("test_cam", mock_config)
        
        # Test moving average calculation
        areas = [1.0, 2.0, 3.0, 4.0, 5.0]
        window_size = 3
        
        moving_averages = camera._calculate_moving_averages(areas, window_size)
        
        # Expected: [2.0, 3.0, 4.0] (averages of [1,2,3], [2,3,4], [3,4,5])
        expected = [2.0, 3.0, 4.0]
        assert moving_averages == expected, f"Expected {expected}, got {moving_averages}"
    
    def test_growth_trend_detection(self):
        """Test growth trend detection logic"""
        camera = CameraState("test_cam", mock_config)
        
        # Test clear growth trend
        growing_averages = [1.0, 1.2, 1.5, 1.8, 2.0]
        is_growing = camera._check_growth_trend(growing_averages, 1.2)
        assert is_growing, "Should detect growth trend"
        
        # Test no growth
        flat_averages = [1.0, 1.0, 1.0, 1.0, 1.0]
        is_flat = camera._check_growth_trend(flat_averages, 1.2)
        assert not is_flat, "Should not detect growth in flat trend"
        
        # Test shrinking trend
        shrinking_averages = [2.0, 1.8, 1.5, 1.2, 1.0]
        is_shrinking = camera._check_growth_trend(shrinking_averages, 1.2)
        assert not is_shrinking, "Should not detect growth in shrinking trend"
    
    def test_detection_object_cleanup(self):
        """Test old detection object cleanup"""
        camera = CameraState("test_cam", mock_config)
        current_time = time.time()
        
        # Add old detection
        old_detection = Detection(
            camera_id="test_cam",
            timestamp=current_time - 100,  # Very old
            confidence=0.8,
            area=0.01,
            bbox=[100, 100, 150, 150],
            object_id="old_fire"
        )
        camera.add_detection(old_detection)
        
        # Add new detection
        new_detection = Detection(
            camera_id="test_cam",
            timestamp=current_time,
            confidence=0.8,
            area=0.01,
            bbox=[100, 100, 150, 150], 
            object_id="new_fire"
        )
        camera.add_detection(new_detection)
        
        # Should have cleaned up old object
        assert "old_fire" not in camera.fire_objects, "Old fire object should be cleaned up"
        assert "new_fire" in camera.fire_objects, "New fire object should remain"
    
    def test_config_environment_variables(self):
        """Test configuration reads from environment variables"""
        with patch.dict(os.environ, {
            'CONSENSUS_THRESHOLD': '3',
            'MIN_CONFIDENCE': '0.8',
            'MQTT_TLS': 'true'
        }):
            # Re-import to get fresh config
            import importlib
            import consensus
            importlib.reload(consensus)
            
            config = consensus.FireConsensusConfig()
            assert config.CONSENSUS_THRESHOLD == 3
            assert config.MIN_CONFIDENCE == 0.8
            assert config.MQTT_TLS is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])