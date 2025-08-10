# Test Fixes - Session 11

## Summary of Test Failures Fixed

This document summarizes the test fixes implemented in session 11 to address extensive test failures reported by the user.

## Root Causes Identified

### 1. Frigate Shared Memory Lock Conflicts
**Problem**: Multiple Frigate containers were mounting the host's `/dev/shm` directory directly, causing lock conflicts when tests ran in parallel.
```
s6-log: fatal: unable to lock /dev/shm/logs/*/lock: Resource busy
```

**Fix**: Changed from direct bind mount to container-specific tmpfs mount:
```python
# Old (causes conflicts):
'/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'}

# New (isolated per container):
'tmpfs': {
    '/dev/shm': 'size=1g,exec,dev,suid,noatime,mode=1777'
}
```

### 2. GPIO State Machine Test Failures
**Problem**: Test expected REFILLING state after shutdown but got IDLE due to unconfigured line pressure sensor triggering low pressure detection.

**Fix**: Added line pressure sensor configuration to test fixtures:
```python
# Configure line pressure sensor
line_pressure_pin="17",
line_pressure_active_low="true",  # LOW = low pressure, HIGH = normal

# Ensure normal pressure state
if gpio_test_setup and controller.config.line_pressure_pin:
    gpio_test_setup._state[controller.config.line_pressure_pin] = GPIO.HIGH
```

### 3. Environment Variable Pollution
**Problem**: Tests were modifying `os.environ` directly, causing cross-test pollution in parallel execution.

**Fix**: 
- For test functions: Use `monkeypatch.setenv()` instead of direct `os.environ` modification
- For class methods: Implemented proper cleanup with original value restoration

### 4. Container Lifecycle Management
**Analysis**: DockerContainerManager already has robust features:
- Reference counting to prevent premature cleanup
- Container age checking (30-second grace period)
- Retry logic with proper error handling
- Thread-safe operations

**Conclusion**: Existing implementation is sufficient; issues were due to other root causes.

### 5. Enhanced Debugging
**Added**: Comprehensive debug helper module (`test_utils/debug_helpers.py`) with:
- Event logging with timestamps and context
- Container lifecycle tracking
- MQTT message flow logging
- State transition tracking
- Environment variable logging
- Automatic debug log saving

## Files Modified

1. `/home/seth/wildfire-watch/tests/test_security_nvr_integration.py`
   - Replaced direct /dev/shm mount with tmpfs
   - Removed redundant shm_size parameter

2. `/home/seth/wildfire-watch/tests/test_gpio_state_machine_integrity.py`
   - Added line pressure sensor configuration to all fixtures
   - Set line pressure sensor to HIGH (normal pressure) state

3. `/home/seth/wildfire-watch/tests/test_e2e_hardware_integration.py`
   - Added monkeypatch parameter to test function
   - Replaced os.environ assignments with monkeypatch.setenv

4. `/home/seth/wildfire-watch/tests/test_integration_e2e_improved.py`
   - Implemented proper environment variable cleanup
   - Added restoration of original values in cleanup method
   - Added debug helper imports

5. `/home/seth/wildfire-watch/tests/test_utils/debug_helpers.py` (NEW)
   - Created comprehensive debugging utilities
   - Event logging and tracking
   - MQTT debug wrapper
   - Debug context manager

## Testing Recommendations

1. **Enable Debug Mode**: Set `TEST_DEBUG=true` environment variable for detailed logging
2. **Save Debug Logs**: Set `TEST_DEBUG_SAVE=true` to automatically save debug logs
3. **Increase Timeouts**: Already configured in pytest ini files:
   - pytest-python312.ini: 1 hour per test
   - pytest-python310.ini: 2 hours per test (for training)
   - pytest-python38.ini: 2 hours per test (for model conversion)

## Next Steps

1. Run tests with debug mode enabled to gather more information if failures persist
2. Monitor for any new lock conflicts or timing issues
3. Consider adding debug helpers to more test files as needed

## Key Insights

1. **Shared Memory Isolation**: Container tmpfs mounts are essential for parallel test execution
2. **Sensor Configuration**: All GPIO sensors must be properly configured in tests to avoid unexpected behavior
3. **Environment Isolation**: Use pytest's monkeypatch for all environment variable modifications
4. **Debugging Infrastructure**: Comprehensive logging is crucial for diagnosing parallel test failures