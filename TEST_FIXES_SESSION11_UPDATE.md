# Test Fixes Session 11 - Update

## Additional Issues Found and Fixed

### 1. Missing pytest marker definition
**Problem**: Tests were failing with "'frigate_slow' not found in `markers` configuration option"
**Fix**: Added `frigate_slow` marker to all pytest configuration files

### 2. Duplicate shared memory configuration
**Problem**: Found a second occurrence of the old shared memory configuration in the test file
**Location**: Lines 342-358 in test_security_nvr_integration.py
**Fix**: Updated second occurrence to use tmpfs mount instead of direct /dev/shm binding

## Summary of All Fixes Applied

1. **Frigate Shared Memory Issues**:
   - Changed from direct `/dev/shm` bind mount to container-specific tmpfs
   - Fixed in TWO places in test_security_nvr_integration.py (lines ~233 and ~346)
   - Removed `shm_size` parameter, using tmpfs mount configuration instead

2. **GPIO State Machine Test**:
   - Added line pressure sensor configuration to prevent unwanted IDLE state
   - Set line pressure sensor to HIGH (normal pressure) in all fixtures

3. **Environment Variable Isolation**:
   - Updated test_e2e_hardware_integration.py to use monkeypatch
   - Added proper cleanup in test_integration_e2e_improved.py

4. **pytest Configuration**:
   - Added missing `frigate_slow` marker to pytest.ini, pytest-python312.ini, pytest-python310.ini, and pytest-python38.ini

## Code Changes Summary

### test_security_nvr_integration.py
```python
# OLD (2 occurrences):
'/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'},
'shm_size': '128m',

# NEW:
'tmpfs': {
    '/dev/shm': 'size=1g,exec,dev,suid,noatime,mode=1777'
},
# shm_size removed - using tmpfs mount instead
```

### test_gpio_state_machine_integrity.py
```python
# Added to fixtures:
line_pressure_pin="17",
line_pressure_active_low="true",

# Added to setup:
if gpio_test_setup and controller.config.line_pressure_pin:
    gpio_test_setup._state[controller.config.line_pressure_pin] = GPIO.HIGH
```

### All pytest configuration files
```ini
frigate_slow: marks tests as slow Frigate tests that may take extended time
```

## Next Steps

The fixes have been applied. If tests are still failing:

1. Check for Python package issues:
   ```bash
   python3.12 -m pip check
   ```

2. Verify Docker is running:
   ```bash
   docker ps
   ```

3. Run tests with debug mode:
   ```bash
   TEST_DEBUG=true TEST_DEBUG_SAVE=true python3.12 -m pytest tests/test_gpio_state_machine_integrity.py -v
   ```

4. Check for stale processes:
   ```bash
   ps aux | grep pytest
   killall -9 pytest  # if needed
   ```