#!/usr/bin/env python3.12
"""
MQTT Migration Helper Utilities for Wildfire Watch Tests

This module provides utilities and decorators to help migrate existing tests
to the optimized MQTT infrastructure while maintaining backward compatibility.
"""

import functools
import pytest
import time
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock
import paho.mqtt.client as mqtt

# Import test infrastructure
from enhanced_mqtt_broker import TestMQTTBroker
from tests.mqtt_test_broker import MQTTTestBroker as LegacyMQTTTestBroker


class MQTTMigrationAdapter:
    """
    Adapter to make old test patterns work with new infrastructure
    
    This allows gradual migration without breaking existing tests.
    """
    
    def __init__(self, test_mqtt_broker, mqtt_topic_factory):
        self.broker = test_mqtt_broker
        self.topic_factory = mqtt_topic_factory
        self._clients = []
        self._legacy_topics = {}
        
    def get_broker_port(self) -> int:
        """Get broker port for legacy compatibility"""
        return self.broker.port
        
    def get_broker_host(self) -> str:
        """Get broker host for legacy compatibility"""
        return self.broker.host
        
    def create_client(self, client_id: Optional[str] = None) -> mqtt.Client:
        """Create a client with automatic cleanup"""
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        client.connect(self.broker.host, self.broker.port, 60)
        client.loop_start()
        self._clients.append(client)
        return client
        
    def map_legacy_topic(self, legacy_topic: str) -> str:
        """Map legacy topic to isolated topic"""
        if legacy_topic not in self._legacy_topics:
            self._legacy_topics[legacy_topic] = self.topic_factory(legacy_topic)
        return self._legacy_topics[legacy_topic]
        
    def cleanup(self):
        """Clean up all created clients"""
        for client in self._clients:
            try:
                client.loop_stop()
                client.disconnect()
            except:
                pass
        self._clients.clear()


def mqtt_migration_fixture(func: Callable) -> Callable:
    """
    Decorator to add MQTT migration support to test methods
    
    This decorator:
    1. Injects mqtt_adapter as first argument
    2. Maps legacy broker attributes
    3. Handles cleanup automatically
    """
    @functools.wraps(func)
    def wrapper(self, test_mqtt_broker, mqtt_topic_factory, *args, **kwargs):
        # Create adapter
        adapter = MQTTMigrationAdapter(test_mqtt_broker, mqtt_topic_factory)
        
        # Map legacy attributes if test class expects them
        if hasattr(self, '__class__'):
            # Set broker attribute for backward compatibility
            self.broker = adapter.broker
            self.broker_port = adapter.get_broker_port()
            self.broker_host = adapter.get_broker_host()
            
        try:
            # Call test with adapter
            return func(self, adapter, *args, **kwargs)
        finally:
            # Cleanup
            adapter.cleanup()
            
    return wrapper


class LegacyBrokerAdapter:
    """
    Adapter to make new broker work like old MQTTTestBroker
    """
    
    def __init__(self, test_mqtt_broker):
        self._broker = test_mqtt_broker
        self.port = test_mqtt_broker.port
        self.host = test_mqtt_broker.host
        
    def start(self):
        """Legacy start method - broker already started"""
        pass
        
    def stop(self):
        """Legacy stop method - handled by fixture"""
        pass
        
    def is_running(self) -> bool:
        """Check if broker is running"""
        return self._broker.is_running()
        
    def wait_for_connection(self, timeout: float = 5) -> bool:
        """Wait for broker to be ready"""
        return self._broker.wait_for_ready(timeout)


def migrate_class_broker(test_class):
    """
    Class decorator to migrate class-level broker setup
    
    Usage:
        @migrate_class_broker
        class TestMyFeature:
            # Old class-level broker setup is replaced
            pass
    """
    # Remove old class methods if they exist
    if hasattr(test_class, 'setUpClass'):
        delattr(test_class, 'setUpClass')
    if hasattr(test_class, 'tearDownClass'):
        delattr(test_class, 'tearDownClass')
        
    # Add fixture to all test methods
    for attr_name in dir(test_class):
        attr = getattr(test_class, attr_name)
        if callable(attr) and attr_name.startswith('test_'):
            # Wrap test method
            wrapped = pytest.mark.usefixtures('test_mqtt_broker', 'mqtt_topic_factory')(attr)
            setattr(test_class, attr_name, wrapped)
            
    return test_class


class MQTTTestHelper:
    """
    Helper class with common MQTT test patterns
    """
    
    @staticmethod
    def wait_for_messages(client: mqtt.Client, topic: str, count: int = 1, 
                         timeout: float = 5.0) -> List[Any]:
        """Wait for specific number of messages on a topic"""
        messages = []
        event = threading.Event()
        
        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                messages.append(payload)
                if len(messages) >= count:
                    event.set()
            except:
                pass
                
        client.on_message = on_message
        client.subscribe(topic)
        
        # Wait for messages
        event.wait(timeout)
        
        return messages
        
    @staticmethod
    def publish_and_wait(client: mqtt.Client, topic: str, payload: Any, 
                        qos: int = 1, timeout: float = 2.0) -> bool:
        """Publish message and wait for confirmation"""
        published = threading.Event()
        
        def on_publish(client, userdata, mid):
            published.set()
            
        client.on_publish = on_publish
        
        # Publish
        payload_str = json.dumps(payload) if not isinstance(payload, str) else payload
        info = client.publish(topic, payload_str, qos=qos)
        
        # Wait for publish confirmation
        return published.wait(timeout)
        
    @staticmethod
    def create_mock_client(broker_host: str = 'localhost', 
                          broker_port: int = 1883) -> Mock:
        """Create a mock MQTT client for unit testing"""
        mock_client = Mock(spec=mqtt.Client)
        mock_client.is_connected.return_value = True
        mock_client.publish.return_value = (0, 1)  # Success
        mock_client.subscribe.return_value = (0, 1)  # Success
        return mock_client


# Migration patterns for common test scenarios
class MigrationPatterns:
    """
    Common migration patterns with examples
    """
    
    @staticmethod
    def class_to_function_pattern():
        """Example: Migrating class-based to function-based tests"""
        # Before:
        # class TestFeature:
        #     @classmethod
        #     def setUpClass(cls):
        #         cls.broker = MQTTTestBroker()
        #         cls.broker.start()
        
        # After:
        def test_feature(test_mqtt_broker, mqtt_client):
            # Broker and client are ready to use
            pass
            
    @staticmethod
    def topic_isolation_pattern():
        """Example: Migrating hardcoded topics"""
        # Before:
        # TOPIC = "sensors/temperature"
        # client.publish(TOPIC, data)
        
        # After:
        def test_with_isolated_topic(mqtt_client, mqtt_topic_factory):
            topic = mqtt_topic_factory("sensors/temperature")
            mqtt_client.publish(topic, data)
            
    @staticmethod
    def multi_client_pattern():
        """Example: Migrating multi-client tests"""
        # Before:
        # client1 = mqtt.Client()
        # client2 = mqtt.Client()
        # client1.connect(host, port)
        # client2.connect(host, port)
        
        # After:
        def test_multi_client(mqtt_client_factory):
            client1 = mqtt_client_factory("client1")
            client2 = mqtt_client_factory("client2")
            # Both clients are connected and managed


# Backward compatibility shims
def create_legacy_broker() -> LegacyBrokerAdapter:
    """
    Create a broker that looks like the old MQTTTestBroker
    
    For tests that absolutely need the old interface.
    """
    # Create new broker
    new_broker = TestMQTTBroker(session_scope=False)
    new_broker.start()
    
    # Wrap in adapter
    return LegacyBrokerAdapter(new_broker)


# Test migration validator
class MigrationValidator:
    """
    Validates that tests have been properly migrated
    """
    
    @staticmethod
    def check_test_file(filepath: str) -> Dict[str, List[str]]:
        """Check a test file for migration issues"""
        issues = {
            'class_brokers': [],
            'hardcoded_topics': [],
            'manual_clients': [],
            'missing_fixtures': []
        }
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
        for i, line in enumerate(lines):
            # Check for class-level broker setup
            if 'MQTTTestBroker()' in line and 'class' in lines[max(0, i-10):i]:
                issues['class_brokers'].append(f"Line {i+1}: {line.strip()}")
                
            # Check for hardcoded topics
            if 'client.publish(' in line or 'client.subscribe(' in line:
                if '"' in line and '/' in line:
                    # Likely a hardcoded topic
                    issues['hardcoded_topics'].append(f"Line {i+1}: {line.strip()}")
                    
            # Check for manual client creation
            if 'mqtt.Client()' in line:
                issues['manual_clients'].append(f"Line {i+1}: {line.strip()}")
                
        return issues


if __name__ == "__main__":
    # Example validation
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        issues = MigrationValidator.check_test_file(filepath)
        
        print(f"Migration issues in {filepath}:")
        for issue_type, occurrences in issues.items():
            if occurrences:
                print(f"\n{issue_type}:")
                for occurrence in occurrences:
                    print(f"  {occurrence}")