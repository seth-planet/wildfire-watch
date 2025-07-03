# MQTT Race Condition Fix Summary

## Problem Identified

The error "Failed to publish health: 'NoneType' object has no attribute 'publish'" was occurring when running tests in parallel with pytest-xdist. This was caused by a race condition between:

1. Background health timers starting immediately in `__init__`
2. MQTT client initialization/connection happening asynchronously
3. Service cleanup not properly preventing timer execution

## Root Causes

1. **Early Timer Start**: Health timers started in `_start_background_tasks()` immediately after object creation, before MQTT connection was established
2. **No Safety Checks**: `_publish_health()` methods didn't check if `mqtt_client` was None before using it
3. **Incomplete Cleanup**: Services weren't setting `mqtt_client = None` after disconnect, and timers weren't daemon threads
4. **Test Isolation**: Parallel tests could leave services running that interfered with subsequent tests

## Fixes Applied

### 1. Added Safety Checks in `_publish_health()`
```python
if self.mqtt_client is None or not self.mqtt_connected:
    logger.warning("Cannot publish health: MQTT client not ready")
    return
```

### 2. Made Timers Daemon Threads
```python
self._health_timer.daemon = True  # Don't block process shutdown
```

### 3. Improved Cleanup Methods
```python
# Clear the reference to prevent further use
self.mqtt_client = None
```

### 4. Added Checks in Periodic Methods
```python
if not self.mqtt_connected or self.mqtt_client is None:
    logger.debug("Skipping health report - MQTT not ready")
    # Still reschedule if not shutting down
    if not self._shutdown:
        # Reschedule timer...
```

## Files Modified

1. `fire_consensus/consensus.py`
   - Added mqtt_client None checks in `_publish_health()`
   - Made timers daemon threads
   - Improved cleanup to set mqtt_client = None
   - Added MQTT ready checks in `_periodic_health_report()`

2. `camera_detector/detect.py`
   - Same fixes as consensus.py
   - Used `safe_log()` instead of `logger` for consistency

## Test Results

- ✅ No more "NoneType object has no attribute 'publish'" errors
- ✅ Tests pass with parallel execution (`-n 4`)
- ✅ Services properly clean up between tests
- ✅ Background timers don't block process shutdown

## Additional Benefits

1. **Better Error Messages**: Clear warnings when MQTT isn't ready
2. **Graceful Degradation**: Services continue to function even if MQTT is temporarily unavailable
3. **Cleaner Shutdown**: Daemon threads and proper cleanup prevent hanging processes
4. **Test Reliability**: Parallel test execution is now stable

## Remaining Work

- Fix `test_fire_trigger_via_mqtt` which has an unrelated AttributeError
- Complete updating remaining E2E test files with parallel isolation
- Monitor for any other edge cases in production use