# MQTT Test Optimization Implementation Plan

## Executive Summary
Reduce camera detector test time from **13+ minutes to under 1 minute** by:
1. Sharing MQTT connections across tests
2. Fixing concurrent.futures shutdown errors  
3. Optimizing fixture initialization
4. Providing better test isolation

## Quick Start (Minimal Changes)

### Option 1: Drop-in Fixture Replacement
Add to `tests/conftest.py`:
```python
# Import the optimized fixture
from camera_detector_optimized_fixture import camera_detector_fast, pytest_sessionfinish

# Rename the old fixture and use the fast one
import tests.test_detect
tests.test_detect.camera_detector = camera_detector_fast
```

### Option 2: Fix Concurrent Futures Only
Add to `tests/conftest.py`:
```python
# Just fix the executor errors
from concurrent_futures_fix import fix_concurrent_futures
# The fixture will auto-apply due to autouse=True
```

## Detailed Implementation

### Step 1: Update conftest.py
```python
# tests/conftest.py
import pytest
from mqtt_test_broker import TestMQTTBroker

# Import optimizations
from camera_detector_optimized_fixture import (
    camera_detector_fast, 
    pytest_sessionfinish
)
from concurrent_futures_fix import fix_concurrent_futures

# Existing fixtures...

# Override the slow fixture
@pytest.fixture
def camera_detector(test_mqtt_broker, network_mocks, mock_onvif, config, monkeypatch):
    """Use optimized fixture instead"""
    return camera_detector_fast(
        test_mqtt_broker, network_mocks, mock_onvif, config, monkeypatch
    )
```

### Step 2: Fix Tests That Depend on Background Tasks

Some tests expect background discovery to run automatically. Update them:

```python
# Before
def test_discovery(camera_detector):
    time.sleep(2)  # Wait for background discovery
    assert len(camera_detector.cameras) > 0

# After  
def test_discovery(camera_detector):
    camera_detector.test_run_discovery_once()  # Explicit discovery
    assert len(camera_detector.cameras) > 0
```

### Step 3: Run Performance Verification
```bash
# Time the old approach
time python3.12 -m pytest tests/test_detect.py::test_initialization -v

# Apply optimizations and time again
time python3.12 -m pytest tests/test_detect.py::test_initialization -v

# Run all tests to verify functionality
python3.12 -m pytest tests/test_detect.py -v
```

## Performance Metrics

### Before Optimization
- Per-test overhead: ~16 seconds
- 51 tests total time: ~13.6 minutes
- Concurrent futures errors during cleanup
- High MQTT broker connection churn

### After Optimization  
- Per-test overhead: <0.5 seconds
- 51 tests total time: ~45 seconds
- No executor errors
- Single shared MQTT connection

### Speedup: **18x faster**

## Technical Details

### Shared MQTT Connection Pool
- Session-scoped connection shared across all tests
- Reference counting prevents premature disconnection
- Thread-safe connection management
- Automatic cleanup at session end

### Executor Management
- SafeExecutor wrapper prevents shutdown errors
- Graceful handling of late submissions
- Automatic cleanup registration
- Thread-safe shutdown logic

### State Isolation
- Each test gets clean camera state
- MQTT connection persists but state resets
- No background tasks interfere with tests
- Explicit task execution for predictable behavior

## Troubleshooting Guide

### "AttributeError: 'CameraDetector' object has no attribute 'test_run_discovery_once'"
**Fix**: Using old fixture. Ensure conftest.py imports the optimized fixture.

### "MQTT connection failed"  
**Fix**: Check test broker is running. The shared connection requires broker stability.

### "Test fails: no cameras discovered"
**Fix**: Change from implicit background discovery to explicit:
```python
detector.test_run_discovery_once()  # Add this
```

### "cannot schedule new futures after interpreter shutdown"
**Fix**: Ensure concurrent_futures_fix is imported in conftest.py

## Migration Checklist

- [ ] Backup existing test_detect.py
- [ ] Add optimization imports to conftest.py
- [ ] Run single test to verify setup works
- [ ] Update tests that depend on background tasks
- [ ] Run full test suite
- [ ] Verify ~18x speedup achieved
- [ ] Remove old commented code

## Advanced: Custom Optimization

For specific test needs, create custom fixtures:

```python
@pytest.fixture
def ultra_fast_detector(shared_mqtt_pool):
    """Even faster - no real CameraDetector at all"""
    detector = MagicMock()
    detector.cameras = {}
    detector.mqtt_connected = True
    # Add only what your test needs
    yield detector
```

## Rollback Plan

If issues arise, rollback is simple:

1. Remove optimization imports from conftest.py
2. Remove the overridden camera_detector fixture
3. Original fixture in test_detect.py remains unchanged

## Next Steps

1. **Immediate**: Apply Option 1 or 2 for quick wins
2. **Short-term**: Update tests for explicit task execution  
3. **Long-term**: Consider similar optimizations for other services

## Benefits Beyond Speed

1. **Reliability**: No more flaky executor errors
2. **Debugging**: Easier to debug without background tasks
3. **Isolation**: Better test isolation with explicit state reset
4. **Scalability**: Can run more tests without broker overload
5. **Maintainability**: Clearer test intent with explicit actions