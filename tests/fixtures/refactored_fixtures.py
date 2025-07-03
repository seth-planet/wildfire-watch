#!/usr/bin/env python3.12
"""Shared test fixtures for Wildfire Watch refactored services.

These fixtures provide common test infrastructure that works with
the new base classes (MQTTService, HealthReporter, ThreadSafeService).
"""

import os
import time
import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional


@pytest.fixture
def mock_mqtt_config():
    """Create a mock config object that works with MQTTService."""
    config = MagicMock()
    config.mqtt_broker = 'localhost'
    config.mqtt_port = 1883
    config.mqtt_tls = False
    config.mqtt_username = ''
    config.mqtt_password = ''
    config.topic_prefix = 'test'
    
    # Add dict-style access for compatibility
    config.__getitem__ = lambda self, key: getattr(self, key.lower(), None)
    config.get = lambda self, key, default=None: getattr(self, key.lower(), default)
    
    return config


@pytest.fixture
def mock_service_config(mock_mqtt_config):
    """Extend mock MQTT config with common service settings."""
    config = mock_mqtt_config
    config.health_interval = 60
    config.log_level = 'INFO'
    config.service_id = 'test_service'
    return config


@pytest.fixture
def mock_mqtt_client():
    """Create a mock MQTT client for testing."""
    client = MagicMock()
    client.is_connected = MagicMock(return_value=True)
    client.connect = MagicMock(return_value=(0, None))
    client.disconnect = MagicMock()
    client.loop_start = MagicMock()
    client.loop_stop = MagicMock()
    client.publish = MagicMock(return_value=(0, 1))
    client.subscribe = MagicMock(return_value=(0, 1))
    
    # Track published messages
    client.published_messages = []
    
    def track_publish(topic, payload, **kwargs):
        client.published_messages.append({
            'topic': topic,
            'payload': payload,
            'kwargs': kwargs
        })
        return (0, 1)
    
    client.publish.side_effect = track_publish
    
    return client


@pytest.fixture
def refactored_service_factory(mock_service_config, mock_mqtt_client):
    """Factory for creating refactored services with mocked dependencies."""
    def create_service(service_class, config_overrides=None):
        # Apply config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(mock_service_config, key, value)
        
        # Create service
        service = service_class()
        
        # Replace MQTT client with mock
        if hasattr(service, '_mqtt_client'):
            service._mqtt_client = mock_mqtt_client
            service._mqtt_connected = True
        
        return service, mock_mqtt_client
    
    return create_service


@pytest.fixture
def wait_for_condition():
    """Helper to wait for a condition with timeout."""
    def wait(condition_fn, timeout=5, interval=0.1):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_fn():
                return True
            time.sleep(interval)
        return False
    
    return wait


@pytest.fixture
def capture_mqtt_messages(test_mqtt_broker, mqtt_topic_factory):
    """Capture MQTT messages published to specific topics."""
    messages = []
    
    def on_message(client, userdata, msg):
        messages.append({
            'topic': msg.topic,
            'payload': msg.payload.decode(),
            'timestamp': time.time()
        })
    
    # Create subscriber client
    import paho.mqtt.client as mqtt
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_capture")
    client.on_message = on_message
    
    conn_params = test_mqtt_broker.get_connection_params()
    client.connect(conn_params['host'], conn_params['port'])
    
    def capture(topics):
        """Start capturing messages on specified topics."""
        for topic in topics:
            client.subscribe(mqtt_topic_factory(topic))
        client.loop_start()
        
        # Return messages list for inspection
        return messages
    
    yield capture
    
    # Cleanup
    client.loop_stop()
    client.disconnect()
