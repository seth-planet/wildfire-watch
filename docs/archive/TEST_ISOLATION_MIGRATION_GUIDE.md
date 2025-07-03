# Test Isolation Migration Guide

This guide explains how to migrate existing tests to use the new isolation fixtures.

## Quick Start

1. **Update conftest.py** to import the new fixtures:
```python
# At the top of tests/conftest.py
from test_isolation_fixtures import *
from enhanced_mqtt_broker import TestMQTTBroker
```

2. **Replace test fixtures** in your test files:

### Old Pattern → New Pattern

#### MQTT Broker
```python
# OLD
@pytest.fixture
def test_mqtt_broker():
    from mqtt_test_broker import TestMQTTBroker
    broker = TestMQTTBroker()
    broker.start()
    yield broker
    broker.stop()

# NEW - Just use the provided fixture
def test_something(mqtt_broker):  # Automatically cleaned up
    conn_params = mqtt_broker.get_connection_params()
```

#### Service Fixtures
```python
# OLD
@pytest.fixture
def consensus_service(test_mqtt_broker, monkeypatch):
    service = FireConsensus()
    yield service
    # Manual cleanup...

# NEW
def test_consensus(fire_consensus_clean):  # Automatic cleanup
    service = fire_consensus_clean
    # Service is already connected and ready
```

## Detailed Migration Steps

### 1. Update test_consensus.py

```python
# OLD
import pytest
from consensus import FireConsensus

@pytest.fixture
def consensus_service(test_mqtt_broker, monkeypatch):
    # ... complex setup ...
    service = FireConsensus()
    yield service
    # ... manual cleanup ...

# NEW
import pytest

def test_fire_consensus_initialization(fire_consensus_clean):
    """Test FireConsensus service initializes correctly"""
    consensus = fire_consensus_clean
    assert consensus.config.CONSENSUS_THRESHOLD == 2
    assert consensus.cameras == {}
    assert consensus.mqtt_client.is_connected()
```

### 2. Update test_detect.py

```python
# OLD
@pytest.fixture
def camera_detector(test_mqtt_broker, network_mocks):
    detector = CameraDetector()
    yield detector
    # Complex cleanup

# NEW  
def test_camera_discovery(camera_detector_clean, mock_external_deps):
    detector = camera_detector_clean
    # Detector is ready with mocked external deps
```

### 3. Use Thread Monitoring

```python
def test_with_threading(thread_monitor, fire_consensus_clean):
    """Test that creates threads"""
    consensus = fire_consensus_clean
    
    # Your test code here
    # Any threads/timers created will be automatically cleaned up
```

### 4. Use State Management

```python
def test_state_isolation(state_manager, camera_detector_clean):
    """Test with explicit state management"""
    detector = camera_detector_clean
    state_manager.register_service('my_detector', detector)
    
    # Modify detector state
    detector.cameras['test'] = Camera(ip="192.168.1.1", mac="AA:BB:CC:DD:EE:FF")
    
    # State will be automatically reset after test
```

## Common Patterns

### 1. MQTT Message Testing
```python
def test_mqtt_messaging(mqtt_broker, mqtt_client_factory):
    """Test MQTT message flow"""
    # Create test clients
    publisher = mqtt_client_factory("publisher")
    subscriber = mqtt_client_factory("subscriber")
    
    # Set up subscription
    messages = []
    def on_message(client, userdata, msg):
        messages.append(msg)
    
    subscriber.on_message = on_message
    subscriber.subscribe("test/topic")
    
    # Publish message
    publisher.publish("test/topic", "test payload")
    
    # Wait for message
    time.sleep(0.5)
    assert len(messages) == 1
    # Clients automatically cleaned up
```

### 2. Service Integration Testing
```python
def test_service_integration(fire_consensus_clean, camera_detector_clean):
    """Test interaction between services"""
    consensus = fire_consensus_clean
    detector = camera_detector_clean
    
    # Services share the same MQTT broker automatically
    # Test their interaction...
```

### 3. Parallel-Safe Testing
```python
def test_parallel_safe(unique_id, mqtt_broker):
    """Test that's safe for parallel execution"""
    # Use unique IDs for topics/client IDs
    topic = unique_id("test/topic")
    client_id = unique_id("test_client")
    
    # Test will not conflict with parallel instances
```

## Troubleshooting

### Issue: Tests still failing in bulk

1. **Check imports**: Ensure `test_isolation_fixtures` is imported in conftest.py
2. **Check fixture usage**: Replace old fixtures with new ones
3. **Check for global state**: Look for module-level variables that persist
4. **Check thread cleanup**: Use thread_monitor fixture for tests creating threads

### Issue: MQTT connection errors

1. **Use provided broker**: Don't create your own broker instances
2. **Check client cleanup**: Use mqtt_client_factory for automatic cleanup
3. **Verify broker is running**: The session broker should stay up for all tests

### Issue: State leaking between tests

1. **Use clean fixtures**: fire_consensus_clean, camera_detector_clean
2. **Register services**: Use state_manager for custom services
3. **Clear collections**: Ensure lists/dicts are cleared in cleanup

## Validation

After migration, run the validation script:

```bash
python tests/validate_test_isolation.py
```

This will verify:
- Individual tests still pass
- Tests pass in groups
- All tests pass together
- Parallel execution works

## Best Practices

1. **Use session fixtures** for expensive resources (MQTT broker)
2. **Use function fixtures** for service instances
3. **Always use context managers** for resource acquisition
4. **Prefer composition** over inheritance in fixtures
5. **Make tests independent** - each test should work in isolation
6. **Use markers** to categorize tests (@pytest.mark.mqtt, @pytest.mark.slow)

## Example: Complete Test File

```python
import pytest
import time
import json

class TestFireConsensusWithIsolation:
    """Fire consensus tests with proper isolation"""
    
    def test_initialization(self, fire_consensus_clean):
        """Test service initialization"""
        consensus = fire_consensus_clean
        assert consensus.mqtt_connected
        assert len(consensus.cameras) == 0
    
    def test_detection_processing(self, fire_consensus_clean, mqtt_client_factory):
        """Test detection processing with MQTT"""
        consensus = fire_consensus_clean
        publisher = mqtt_client_factory("test_publisher")
        
        # Publish detection
        detection = {
            'camera_id': 'cam1',
            'confidence': 0.9,
            'timestamp': time.time()
        }
        publisher.publish("detection/fire", json.dumps(detection))
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify processing
        assert 'cam1' in consensus.cameras
    
    @pytest.mark.slow
    def test_consensus_with_timeout(self, fire_consensus_clean, thread_monitor):
        """Test consensus with timeouts"""
        consensus = fire_consensus_clean
        # Test code...
        # Threads automatically cleaned up
```

## Summary

The new isolation system provides:
- Automatic resource cleanup
- Thread management
- State isolation
- MQTT connection pooling
- Parallel execution support

By following this guide, tests will be more reliable and maintainable.