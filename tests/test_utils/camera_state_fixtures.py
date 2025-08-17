#!/usr/bin/env python3.12
"""
Fixtures for CameraState testing.

Provides proper CameraState instances with required config parameter.
"""
import pytest
import os
from fire_consensus.consensus import CameraState, FireConsensusConfig


@pytest.fixture
def real_config():
    """Create a real FireConsensusConfig object with test values."""
    # Set test environment variables
    test_env = {
        'CONSENSUS_THRESHOLD': '2',
        'TIME_WINDOW': '30.0',
        'MIN_CONFIDENCE': '0.7',
        'MIN_AREA_RATIO': '0.0001',
        'MAX_AREA_RATIO': '0.5',
        'COOLDOWN_PERIOD': '60.0',
        'DETECTION_WINDOW': '30.0',
        'MOVING_AVERAGE_WINDOW': '3',
        'AREA_INCREASE_RATIO': '1.2',
        'LOG_LEVEL': 'INFO'
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Create real config instance
        config = FireConsensusConfig()
        yield config
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


# Keep the old name for compatibility but use real config
@pytest.fixture
def mock_config(real_config):
    """Legacy name - returns real config for compatibility."""
    return real_config


@pytest.fixture
def camera_state_factory(mock_config):
    """Factory fixture to create CameraState instances."""
    def _create_camera_state(camera_id="camera-001", config=None):
        # CameraState doesn't take a config parameter
        camera_state = CameraState(camera_id=camera_id)
        # Attach config as an attribute for tests that need it
        if config is None:
            config = mock_config
        camera_state.config = config
        return camera_state
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
        CameraState instance with config attached
    """
    # Set defaults
    defaults = {
        'CONSENSUS_THRESHOLD': '2',
        'TIME_WINDOW': '30.0',
        'MIN_CONFIDENCE': '0.7',
        'MIN_AREA_RATIO': '0.0001',
        'MAX_AREA_RATIO': '0.5',
        'COOLDOWN_PERIOD': '60.0',
        'DETECTION_WINDOW': '30.0',
        'MOVING_AVERAGE_WINDOW': '3',
        'AREA_INCREASE_RATIO': '1.2',
        'LOG_LEVEL': 'INFO'
    }
    
    # Apply overrides and convert to strings for env vars
    test_env = {}
    for key, default_value in defaults.items():
        value = config_overrides.get(key, default_value)
        test_env[key] = str(value).lower() if isinstance(value, bool) else str(value)
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Create real config instance
        config = FireConsensusConfig()
        
        # Create CameraState and attach config
        camera_state = CameraState(camera_id=camera_id)
        camera_state.config = config
        return camera_state
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value