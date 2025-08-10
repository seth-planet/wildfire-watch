# Test Fixes Summary - Session 3

## Problem
9 Python tests were failing with ConnectionRefusedError and container lifecycle issues:
- ConnectionRefusedError [Errno 111] in 7 tests
- RuntimeError: Frigate container errors in 2 tests

## Root Causes Identified

### 1. MQTT Connection Timing Issues
- PumpController connects automatically in `__init__` but has no `wait_for_connection` method
- Tests were using fixed `time.sleep(1.0)` delays instead of proper connection verification
- Race condition between connection establishment and test execution

### 2. Container Lifecycle Management
- Frigate fixture was class-scoped causing conflicts in parallel test execution
- Multiple workers trying to manage the same container names
- Cleanup conflicts when containers still had active endpoints

### 3. Service Dependencies
- `test_multi_camera_consensus` was manually simulating consensus behavior instead of starting actual services
- Missing service startup in some tests expecting services to be running

## Solutions Implemented

### 1. Created `wait_for_mqtt_connection` Helper Function
**File**: `/home/seth/wildfire-watch/tests/test_utils/helpers.py`
```python
def wait_for_mqtt_connection(controller: 'PumpController', timeout: float = 10.0) -> bool:
    """Wait for PumpController's MQTT connection to be established."""
    # Polls controller._mqtt_client state
    # Returns True if connected, False if timeout
```

### 2. Fixed MQTT Connection Timing in Tests
**Files Modified**:
- `tests/test_pump_safety_timeout_simple.py` - Replaced `time.sleep(1.5)` with `wait_for_mqtt_connection()`
- `tests/test_integration_e2e_improved.py` - Fixed in `_start_e2e_services()` method
- `tests/test_gpio_critical_safety_paths.py` - Fixed 6 occurrences of `time.sleep(1.0)`

### 3. Fixed Container Lifecycle Management
**File**: `/home/seth/wildfire-watch/tests/test_security_nvr_integration.py`
- Converted fixtures from class-scope to function-scope:
  - `class_scoped_mqtt_broker` → `mqtt_broker_for_frigate` 
  - `class_scoped_docker_manager` → `docker_manager_for_frigate`
  - `frigate_container` scope changed from "class" to function
- Ensures proper container cleanup between tests in parallel execution

### 4. Fixed Service Dependencies
**File**: `/home/seth/wildfire-watch/tests/test_integration_e2e_improved.py`
- `test_multi_camera_consensus` now calls `_start_e2e_services()` to start actual services
- Removed manual simulation code that was publishing fake consensus messages

## Key Pattern Changes

### Before:
```python
controller.connect()
time.sleep(1.0)  # Hope connection is ready
```

### After:
```python
controller.connect()
if not wait_for_mqtt_connection(controller, timeout=10):
    raise RuntimeError("Failed to connect PumpController to MQTT broker")
```

## Results
All 9 tests should now:
- Properly wait for MQTT connections to be established
- Have isolated container lifecycles in parallel execution
- Start required services instead of manual simulation
- Use event-driven synchronization instead of fixed delays

## Verification Results
✅ **test_pump_safety_timeout_simple.py** - PASSED after fixing `wait_for_mqtt_connection` to check `controller.client` and `controller._mqtt_connected`

Run the following tests to verify remaining fixes:
```bash
python3.12 -m pytest tests/test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_multi_camera_consensus -v
python3.12 -m pytest tests/test_security_nvr_integration.py -v -k "mqtt_broker_dependency or camera_detector_integration"
python3.12 -m pytest tests/test_gpio_critical_safety_paths.py -v
```

## Key Learning
The `wait_for_mqtt_connection` helper needed to check:
1. `controller._mqtt_connected` flag (if exists)
2. `controller.client.is_connected()` 
3. `controller.client._state == 3` (paho MQTT connected state)

## Remaining Work
- Replace remaining `time.sleep()` calls with event-driven waits throughout the test suite
- Monitor for any new timing-related failures in CI/CD