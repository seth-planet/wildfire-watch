#!/usr/bin/env python3.12
"""
Quick validation of test fixes.
"""
import pytest
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_concurrent_futures_fix():
    """Test that concurrent futures fix doesn't cause recursion."""
    from tests.concurrent_futures_fix import SafeExecutor
    
    # Create SafeExecutor without recursion
    executor = SafeExecutor(max_workers=2)
    
    # Submit a simple task
    future = executor.submit(lambda: 42)
    result = future.result()
    
    assert result == 42
    
    # Cleanup
    executor.shutdown()
    print("✓ Concurrent futures fix works without recursion")


def test_camera_state_creation():
    """Test that CameraState can be created with config."""
    from tests.camera_state_fixtures import create_test_camera_state
    
    # Create camera state with test config
    camera_state = create_test_camera_state("test-cam-001")
    
    assert camera_state.camera_id == "test-cam-001"
    assert camera_state.config.CONSENSUS_THRESHOLD == 2
    assert camera_state.config.MIN_CONFIDENCE == 0.7
    
    print("✓ CameraState creation with config works")


def test_mqtt_broker_attributes():
    """Test that TestMQTTBroker has required attributes."""
    from tests.mqtt_test_broker import MQTTTestBroker
    
    # Create broker
    broker = MQTTTestBroker()
    
    # Check required attributes
    assert hasattr(broker, 'host')
    assert hasattr(broker, 'port')
    assert broker.host == 'localhost'
    assert isinstance(broker.port, int)
    
    print("✓ TestMQTTBroker has required attributes")


def test_calculate_area_signature():
    """Test that _calculate_area uses camera_resolution parameter."""
    from fire_consensus.consensus import FireConsensus, FireConsensusConfig
    from unittest.mock import Mock, patch
    
    # Mock MQTT setup_mqtt method and background tasks
    with patch('fire_consensus.consensus.MQTTService.setup_mqtt'):
        with patch.object(FireConsensus, '_start_background_tasks'):
            # Create consensus (it creates its own config)
            consensus = FireConsensus()
    
    # Test with camera_resolution parameter
    bbox = [100, 100, 200, 200]
    area = consensus._calculate_area(bbox, camera_resolution=(1920, 1080))
    
    assert isinstance(area, float)
    assert 0 <= area <= 1
    
    print("✓ _calculate_area accepts camera_resolution parameter")


if __name__ == "__main__":
    # Run tests
    test_concurrent_futures_fix()
    test_camera_state_creation()
    test_mqtt_broker_attributes()
    test_calculate_area_signature()
    
    print("\n✅ All fixes validated successfully!")