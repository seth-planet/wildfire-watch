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
def real_mqtt_service_factory(test_mqtt_broker, mqtt_topic_factory):
    """Factory for creating services with real MQTT connections."""
    import paho.mqtt.client as mqtt
    
    def create_mqtt_service(service_name="test_service", config_overrides=None):
        """Create a service with real MQTT broker connection."""
        # Get broker connection params
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Create config dict with real broker info
        config = {
            'mqtt_broker': conn_params['host'],
            'mqtt_port': conn_params['port'],
            'mqtt_tls': False,
            'mqtt_username': '',
            'mqtt_password': '',
            'topic_prefix': mqtt_topic_factory(''),  # Use unique prefix
            'health_interval': 60,
            'log_level': 'INFO',
            'service_id': service_name
        }
        
        # Apply overrides
        if config_overrides:
            config.update(config_overrides)
        
        # Create real MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{service_name}_client")
        client.connect(config['mqtt_broker'], config['mqtt_port'])
        client.loop_start()
        
        # Return config and client for service initialization
        return config, client
    
    return create_mqtt_service


@pytest.fixture
def refactored_service_factory(real_mqtt_service_factory):
    """Factory for creating refactored services with real MQTT connections."""
    created_clients = []
    
    def create_service(service_class, config_overrides=None):
        # Get real MQTT config and client
        config, mqtt_client = real_mqtt_service_factory(
            service_name=service_class.__name__,
            config_overrides=config_overrides
        )
        
        # Track client for cleanup
        created_clients.append(mqtt_client)
        
        # Create service with real config
        # Note: Service initialization depends on the specific service class
        # This is a template that may need adjustment per service
        service = service_class(config)
        
        return service, mqtt_client
    
    yield create_service
    
    # Cleanup all created clients
    for client in created_clients:
        try:
            client.loop_stop()
            client.disconnect()
        except:
            pass


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
