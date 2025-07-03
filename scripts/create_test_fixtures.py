#!/usr/bin/env python3
"""Create shared test fixtures for refactored code.

This script creates common test fixtures that can be used across
all test files to reduce duplication and ensure consistency.
"""

import os
from pathlib import Path

# Common test fixture template
FIXTURES_TEMPLATE = '''#!/usr/bin/env python3.12
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
'''

# Test helpers template
HELPERS_TEMPLATE = '''#!/usr/bin/env python3.12
"""Test helpers for working with refactored services."""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class MockCamera:
    """Mock camera for testing."""
    ip: str
    mac: str
    name: str = "Test Camera"
    manufacturer: str = "TestCam"
    model: str = "TC-1000"
    online: bool = True
    error_count: int = 0
    last_seen: float = field(default_factory=time.time)
    rtsp_urls: Dict[str, str] = field(default_factory=dict)
    discovery_method: str = "mock"
    
    def __post_init__(self):
        if not self.rtsp_urls:
            self.rtsp_urls = {
                'main': f'rtsp://{self.ip}:554/stream1',
                'sub': f'rtsp://{self.ip}:554/stream2'
            }


@dataclass  
class MockDetection:
    """Mock fire/smoke detection for testing."""
    camera_id: str
    confidence: float
    object_type: str = "fire"
    timestamp: float = field(default_factory=time.time)
    bounding_box: List[int] = field(default_factory=lambda: [100, 100, 50, 50])
    object_id: Optional[str] = None
    
    def to_mqtt_payload(self) -> str:
        """Convert to MQTT JSON payload."""
        return json.dumps({
            'camera_id': self.camera_id,
            'confidence': self.confidence,
            'object_type': self.object_type,
            'timestamp': self.timestamp,
            'bounding_box': self.bounding_box,
            'object_id': self.object_id or f"{self.object_type}_{int(self.timestamp)}"
        })


class ServiceTestHelper:
    """Helper for testing refactored services."""
    
    @staticmethod
    def wait_for_mqtt_connection(service, timeout=10):
        """Wait for service MQTT connection."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if hasattr(service, '_mqtt_connected') and service._mqtt_connected:
                return True
            time.sleep(0.1)
        return False
    
    @staticmethod
    def get_published_messages(mock_mqtt_client, topic_filter=None):
        """Get messages published by the service."""
        messages = mock_mqtt_client.published_messages
        if topic_filter:
            messages = [m for m in messages if topic_filter in m['topic']]
        return messages
    
    @staticmethod
    def trigger_health_report(service):
        """Trigger a health report from the service."""
        if hasattr(service, 'health_reporter'):
            service.health_reporter.report_health()
            return True
        return False
    
    @staticmethod
    def simulate_shutdown(service):
        """Simulate service shutdown."""
        if hasattr(service, 'shutdown'):
            service.shutdown()
        elif hasattr(service, '_shutdown'):
            service._shutdown = True
        
        # Stop any background tasks
        for attr in ['discovery_task', 'health_check_task', 'mac_tracking_task']:
            if hasattr(service, attr):
                task = getattr(service, attr)
                if hasattr(task, 'stop'):
                    task.stop()
'''


def main():
    """Create shared test fixtures."""
    # Get project root
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Create fixtures directory
    fixtures_dir = tests_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    
    # Create __init__.py
    (fixtures_dir / "__init__.py").write_text("")
    
    # Create shared fixtures file
    fixtures_file = fixtures_dir / "refactored_fixtures.py"
    fixtures_file.write_text(FIXTURES_TEMPLATE)
    print(f"✓ Created {fixtures_file}")
    
    # Create test helpers file
    helpers_file = fixtures_dir / "test_helpers.py"
    helpers_file.write_text(HELPERS_TEMPLATE)
    print(f"✓ Created {helpers_file}")
    
    # Create example usage file
    example_content = '''#!/usr/bin/env python3.12
"""Example of using shared fixtures with refactored services."""

import pytest
from tests.fixtures.refactored_fixtures import *
from tests.fixtures.test_helpers import ServiceTestHelper, MockCamera, MockDetection


def test_example_with_fixtures(mock_service_config, mock_mqtt_client):
    """Example test using shared fixtures."""
    # Your test code here
    assert mock_service_config.mqtt_broker == 'localhost'
    assert mock_mqtt_client.is_connected()


def test_example_with_helper(refactored_service_factory):
    """Example using the service factory."""
    # This would create a real service instance with mocked MQTT
    # service, mqtt_client = refactored_service_factory(YourServiceClass)
    pass
'''
    
    example_file = fixtures_dir / "example_usage.py"
    example_file.write_text(example_content)
    print(f"✓ Created {example_file}")
    
    print("\n✅ Shared test fixtures created successfully!")
    print("\nTo use in your tests:")
    print("  from tests.fixtures.refactored_fixtures import *")
    print("  from tests.fixtures.test_helpers import ServiceTestHelper, MockCamera")


if __name__ == "__main__":
    main()