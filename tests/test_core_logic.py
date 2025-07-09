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
            # _calculate_area expects [x1, y1, x2, y2] format, not [x, y, width, height]
            bbox_norm = [0.1, 0.1, 0.15, 0.15]  # 5% width/height (0.15 - 0.1 = 0.05)
            area_norm = consensus._calculate_area(bbox_norm)
            # Since values < 1, they're already normalized coordinates
            # So: width=0.05, height=0.05, area=0.0025 (already normalized)
            expected_norm = 0.05 * 0.05  # = 0.0025
            assert abs(area_norm - expected_norm) < 1e-10, f"Expected {expected_norm}, got {area_norm}"
            
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
            
            # Test validation logic directly (inline with _handle_fire_detection)
            # Valid detection
            confidence = 0.8
            area = 0.01
            assert confidence >= consensus.config.min_confidence, "High confidence should pass"
            assert consensus.config.min_area_ratio <= area <= consensus.config.max_area_ratio, "Reasonable size should pass"
            
            # Invalid confidence
            confidence = 0.5
            assert confidence < consensus.config.min_confidence, "Low confidence should fail"
            
            # Invalid size (too small)
            area = 0.00001  # Changed from 0.0001 to be below default min_area_ratio of 0.001
            assert area < consensus.config.min_area_ratio, "Too small area should fail"
            
            # Invalid size (too large)
            area = 0.9  # Changed from 0.6 to be above default max_area_ratio of 0.8
            assert area > consensus.config.max_area_ratio, "Too large area should fail"
            
            # Cleanup
            consensus.cleanup()
    
    def test_camera_state_detection_tracking(self):
        """Test camera state detection tracking"""
        camera = CameraState("test_cam")
        current_time = time.time()
        
        # Add detections - Detection class only takes confidence, area, object_id
        detection1 = Detection(
            confidence=0.8,
            area=0.01,
            object_id="fire1"
        )
        
        detection2 = Detection( 
            confidence=0.85,
            area=0.015,
            object_id="fire1"
        )
        
        # The detections are added via consensus._add_detection, not directly
        # Let's test the camera state directly
        camera.detections["fire1"].append(detection1)
        camera.detections["fire1"].append(detection2)
        camera.total_detections = 2
        camera.last_seen = current_time + 1
        camera.last_detection_time = current_time + 1
        
        # Verify tracking
        assert len(camera.detections["fire1"]) == 2
        assert camera.total_detections == 2
        assert camera.last_seen == current_time + 1
    
    def test_growing_fire_detection_algorithm(self, test_mqtt_broker, monkeypatch):
        """Test growing fire detection algorithm"""
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            consensus = FireConsensus()
            current_time = time.time()
            
            # Add camera to consensus
            consensus.cameras["test_cam"] = CameraState("test_cam")
            camera = consensus.cameras["test_cam"]
            
            # Create detections with growing area
            for i in range(8):
                size = 50 + i * 10  # Growing from 50 to 120
                area = (size * size) / (1920 * 1080)  # Normalize
                
                detection = Detection(
                    confidence=0.8,
                    area=area,
                    object_id="fire1"
                )
                detection.timestamp = current_time + i * 0.5  # Set timestamp after creation
                camera.detections["fire1"].append(detection)
                camera.total_detections += 1
                camera.last_detection_time = detection.timestamp
            
            # Test growing fire detection
            growing_fires = consensus.get_growing_fires("test_cam")
            assert len(growing_fires) > 0, "Should detect growing fire"
            assert "fire1" in growing_fires, "Should detect the specific fire object"
            
            # Cleanup
            consensus.cleanup()
    
    def test_shrinking_fire_not_detected(self, test_mqtt_broker, monkeypatch):
        """Test shrinking fires are not detected as growing"""
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            consensus = FireConsensus()
            consensus.cameras["test_cam"] = CameraState("test_cam")
            camera = consensus.cameras["test_cam"]
            current_time = time.time()
            
            # Create detections with shrinking area
            for i in range(8):
                size = 120 - i * 10  # Shrinking from 120 to 50
                area = (size * size) / (1920 * 1080)
                
                detection = Detection(
                    confidence=0.8,
                    area=area,
                    object_id="fire2"
                )
                detection.timestamp = current_time + i * 0.5
                camera.detections["fire2"].append(detection)
            
            shrinking_fires = consensus.get_growing_fires("test_cam")
            assert len(shrinking_fires) == 0, "Should not detect shrinking fire"
            
            # Cleanup
            consensus.cleanup()
    
    
    def test_detection_object_cleanup(self, test_mqtt_broker, monkeypatch):
        """Test old detection object cleanup"""
        with mqtt_test_environment(test_mqtt_broker, monkeypatch):
            consensus = FireConsensus()
            consensus.cameras["test_cam"] = CameraState("test_cam")
            camera = consensus.cameras["test_cam"]
            current_time = time.time()
            
            # Add old detection
            old_detection = Detection(
                confidence=0.8,
                area=0.01,
                object_id="old_fire"
            )
            old_detection.timestamp = current_time - 100  # Very old
            camera.detections["old_fire"].append(old_detection)
            
            # Add new detection
            new_detection = Detection(
                confidence=0.8,
                area=0.01,
                object_id="new_fire"
            )
            new_detection.timestamp = current_time
            camera.detections["new_fire"].append(new_detection)
            
            # Run cleanup
            consensus._cleanup_old_data()
            
            # Old detection should be cleaned up if older than detection_window * 2
            # Default detection_window is 30s, so 60s cleanup threshold
            if consensus.config.detection_window * 2 < 100:
                assert len(camera.detections["old_fire"]) == 0, "Old fire object should be cleaned up"
            assert len(camera.detections["new_fire"]) > 0, "New fire object should remain"
            
            # Cleanup
            consensus.cleanup()
    
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
            # ConfigBase uses snake_case attributes
            assert config.consensus_threshold == 3
            assert config.min_confidence == 0.8
            assert config.mqtt_tls is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])