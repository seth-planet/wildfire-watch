# MQTT Fixture Optimization Guide

## Problem Statement
- Current `camera_detector` fixture takes ~16 seconds per test
- 51 tests × 16 seconds = 13.6 minutes just for MQTT setup
- Concurrent futures cleanup errors during test teardown
- Each test creates/destroys full MQTT connection

## Solution Overview

### 1. Session-Scoped MQTT Connection Pool
```python
@pytest.fixture(scope="session")
def shared_mqtt_pool(test_mqtt_broker):
    """Share MQTT connection across all tests in session"""
    pool = SharedMQTTPool(test_mqtt_broker.get_connection_params())
    yield pool
    pool.cleanup()
```

### 2. Fast Camera Detector Fixture
```python
@pytest.fixture
def fast_camera_detector(shared_mqtt_pool, ...):
    """<1s setup using shared MQTT connection"""
    mqtt_client, mqtt_connected = shared_mqtt_pool.get_client()
    detector = MockCameraDetector(mqtt_client=mqtt_client, mqtt_connected=mqtt_connected)
    yield detector
    detector.reset_state()  # Clear state, keep connection
```

### 3. MockCameraDetector Class
- Inherits from CameraDetector but skips heavy initialization
- No background threads started
- Reuses MQTT client from pool
- Provides `reset_state()` method for test isolation

## Migration Steps

### Step 1: Update conftest.py
Add the shared MQTT pool fixture to `tests/conftest.py`:

```python
from test_detect_optimized import shared_mqtt_pool, SharedMQTTPool
```

### Step 2: Update Individual Tests
Replace `camera_detector` with `fast_camera_detector`:

```python
# Before
def test_something(camera_detector):
    detector = camera_detector
    ...

# After
def test_something(fast_camera_detector):
    detector = fast_camera_detector
    ...
```

### Step 3: Handle Background Tasks
Tests that rely on background tasks need explicit calls:

```python
# Before (background tasks run automatically)
time.sleep(1)  # Wait for discovery

# After (explicit task execution)
detector.test_run_discovery_once()  # Run discovery manually
```

### Step 4: Fix Executor Cleanup
The MockCameraDetector doesn't use concurrent.futures, avoiding shutdown errors.

## Performance Improvements

### Before
- Setup time: ~16 seconds per test
- Total time for 51 tests: ~13.6 minutes
- Executor cleanup errors

### After
- Setup time: <1 second per test
- Total time for 51 tests: ~1 minute
- No executor cleanup errors
- 13x-16x speedup

## Test Patterns

### Pattern 1: Simple State Tests
```python
def test_initialization(fast_camera_detector):
    """Tests that just check state"""
    assert len(fast_camera_detector.cameras) == 0
    assert fast_camera_detector.mqtt_connected == True
```

### Pattern 2: Discovery Tests
```python
def test_discovery(fast_camera_detector):
    """Tests that need discovery to run"""
    # Mock network responses
    with patch('detect.discover_onvif_cameras') as mock_discover:
        mock_discover.return_value = [...]
        
        # Run discovery explicitly
        fast_camera_detector.test_run_discovery_once()
        
        # Check results
        assert len(fast_camera_detector.cameras) == 1
```

### Pattern 3: MQTT Publishing Tests
```python
def test_mqtt_publish(fast_camera_detector):
    """Tests that verify MQTT messages"""
    camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF")
    
    # The shared MQTT client can be mocked if needed
    fast_camera_detector.mqtt_client.publish = MagicMock()
    
    # Trigger publish
    fast_camera_detector._publish_camera_discovery(camera)
    
    # Verify
    fast_camera_detector.mqtt_client.publish.assert_called_once()
```

## Implementation Files

1. **test_detect_optimized.py** - New optimized fixtures and classes
2. **mqtt_test_broker.py** - No changes needed
3. **test_detect.py** - Migrate to use new fixtures

## Rollout Plan

1. **Phase 1**: Add optimized fixtures alongside existing ones
2. **Phase 2**: Migrate high-frequency tests first
3. **Phase 3**: Migrate remaining tests
4. **Phase 4**: Remove old fixtures

## Troubleshooting

### Issue: Tests fail with "detector has no attribute X"
**Solution**: The MockCameraDetector may be missing some attributes. Add them to `__init__`.

### Issue: Background tasks don't run
**Solution**: Use `test_run_discovery_once()` or `test_run_health_check_once()` explicitly.

### Issue: MQTT messages not published
**Solution**: Check that `mqtt_connected` is True and client is not mocked.

### Issue: State leaks between tests
**Solution**: Ensure `reset_state()` is comprehensive. Add any new state variables.

## Benefits Summary

1. **13-16x faster test execution**
2. **No concurrent.futures cleanup errors**
3. **Shared MQTT connection reduces broker load**
4. **Explicit task control improves test reliability**
5. **Better test isolation with reset_state()**
6. **Easier debugging without background threads**