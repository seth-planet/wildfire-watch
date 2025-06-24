#!/usr/bin/env python3.12
"""
Fixtures for CameraState testing.

Provides proper CameraState instances with required config parameter.
"""
import pytest
from unittest.mock import Mock
from fire_consensus.consensus import CameraState, Config


@pytest.fixture
def mock_config():
    """Create a mock Config object with default values for testing."""
    config = Mock(spec=Config)
    config.CONSENSUS_THRESHOLD = 2
    config.TIME_WINDOW = 30.0
    config.MIN_CONFIDENCE = 0.7
    config.MIN_AREA_RATIO = 0.0001
    config.MAX_AREA_RATIO = 0.5
    config.COOLDOWN_PERIOD = 60.0
    config.SINGLE_CAMERA_TRIGGER = False
    config.DETECTION_WINDOW = 30.0
    config.MOVING_AVERAGE_WINDOW = 3
    config.AREA_INCREASE_RATIO = 1.2
    config.LOG_LEVEL = "INFO"
    return config


@pytest.fixture
def camera_state_factory(mock_config):
    """Factory fixture to create CameraState instances."""
    def _create_camera_state(camera_id="camera-001", config=None):
        if config is None:
            config = mock_config
        return CameraState(camera_id=camera_id, config=config)
    return _create_camera_state


@pytest.fixture
def camera_state(camera_state_factory):
    """Create a default CameraState instance for testing."""
    return camera_state_factory()


def create_test_camera_state(camera_id="test-camera", **config_overrides):
    """
    Helper function to create CameraState with custom config.
    
    Args:
        camera_id: ID for the camera
        **config_overrides: Config attributes to override
        
    Returns:
        CameraState instance
    """
    config = Mock(spec=Config)
    
    # Set defaults
    defaults = {
        'CONSENSUS_THRESHOLD': 2,
        'TIME_WINDOW': 30.0,
        'MIN_CONFIDENCE': 0.7,
        'MIN_AREA_RATIO': 0.0001,
        'MAX_AREA_RATIO': 0.5,
        'COOLDOWN_PERIOD': 60.0,
        'SINGLE_CAMERA_TRIGGER': False,
        'DETECTION_WINDOW': 30.0,
        'MOVING_AVERAGE_WINDOW': 3,
        'AREA_INCREASE_RATIO': 1.2,
        'LOG_LEVEL': "INFO"
    }
    
    # Apply overrides
    for key, value in defaults.items():
        setattr(config, key, config_overrides.get(key, value))
    
    return CameraState(camera_id=camera_id, config=config)