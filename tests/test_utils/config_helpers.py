#!/usr/bin/env python3.12
"""
Configuration Helper Utilities for Testing

This module provides utilities for managing configuration in tests,
particularly for handling the issue of CONFIG dictionaries that are
loaded at module import time.
"""

import importlib
import sys
from typing import Optional, Dict, Any


def reload_service_config(service_name: str) -> None:
    """
    Force reload a service module to pick up new environment variables.
    
    This is necessary because services like trigger.py and consensus.py
    load their CONFIG dictionaries at module import time. In tests, we
    need to reload the module after setting environment variables.
    
    Args:
        service_name: Name of the service ('trigger' or 'consensus')
    
    Example:
        # Set environment variables
        monkeypatch.setenv('MQTT_BROKER', 'localhost')
        monkeypatch.setenv('MQTT_PORT', '1883')
        
        # Reload the module to pick up new env vars
        reload_service_config('trigger')
        
        # Now import and use the service
        from gpio_trigger.trigger import PumpController
    """
    module_map = {
        'trigger': 'gpio_trigger.trigger',
        'consensus': 'fire_consensus.consensus',
        'camera_detector': 'camera_detector.detect'
    }
    
    module_name = module_map.get(service_name)
    if not module_name:
        raise ValueError(f"Unknown service: {service_name}. Valid options: {list(module_map.keys())}")
    
    # Remove the module from sys.modules if it exists
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Also remove any parent modules to ensure clean reload
    parent_module = module_name.rsplit('.', 1)[0]
    if parent_module in sys.modules:
        # Store reference to parent
        parent_ref = sys.modules[parent_module]
        
        # Remove child attribute if it exists
        child_name = module_name.split('.')[-1]
        if hasattr(parent_ref, child_name):
            delattr(parent_ref, child_name)
    
    # Force reimport will now read fresh environment variables
    importlib.import_module(module_name)


def get_clean_controller(test_env: Dict[str, str], broker_params: Dict[str, Any], 
                        topic_prefix: str) -> 'PumpController':
    """
    Create a PumpController instance with clean configuration.
    
    This handles the complete setup process:
    1. Sets environment variables
    2. Reloads the trigger module
    3. Updates CONFIG dictionary
    4. Creates controller instance
    
    Args:
        test_env: Environment variables to set
        broker_params: MQTT broker connection parameters
        topic_prefix: Topic namespace prefix
        
    Returns:
        Configured PumpController instance
    """
    import os
    
    # Set all environment variables first
    for key, value in test_env.items():
        os.environ[key] = str(value)
    
    # Set broker parameters
    os.environ['MQTT_BROKER'] = broker_params['host']
    os.environ['MQTT_PORT'] = str(broker_params['port'])
    os.environ['MQTT_TLS'] = 'false'
    os.environ['TOPIC_PREFIX'] = topic_prefix
    
    # Reload the module to pick up environment variables
    reload_service_config('trigger')
    
    # Import the helper function
    from tests.test_utils.helpers import create_pump_controller_with_config
    
    # Create and return controller using the new helper
    return create_pump_controller_with_config(
        test_env=test_env,
        conn_params=broker_params,
        topic_prefix=topic_prefix,
        auto_connect=True  # Connect immediately since this helper is meant to provide ready-to-use instances
    )


def get_clean_consensus(test_env: Dict[str, str], broker_params: Dict[str, Any], 
                       topic_prefix: str) -> 'FireConsensus':
    """
    Create a FireConsensus instance with clean configuration.
    
    Similar to get_clean_controller but for FireConsensus service.
    
    Args:
        test_env: Environment variables to set
        broker_params: MQTT broker connection parameters
        topic_prefix: Topic namespace prefix
        
    Returns:
        Configured FireConsensus instance
    """
    import os
    
    # Set all environment variables first
    for key, value in test_env.items():
        os.environ[key] = str(value)
    
    # Set broker parameters
    os.environ['MQTT_BROKER'] = broker_params['host']
    os.environ['MQTT_PORT'] = str(broker_params['port'])
    os.environ['MQTT_TLS'] = 'false'
    os.environ['TOPIC_PREFIX'] = topic_prefix
    
    # Reload the module to pick up environment variables
    reload_service_config('consensus')
    
    # Import and create consensus instance
    from fire_consensus.consensus import FireConsensus
    
    # Create and return consensus
    return FireConsensus()