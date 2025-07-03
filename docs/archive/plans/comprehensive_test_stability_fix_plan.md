# Comprehensive Test Stability Fix Plan

## Root Causes Identified

1. **File Descriptor Exhaustion**
   - Multiple test files creating temporary files without proper cleanup
   - Model converter tests creating large numbers of calibration images
   - Socket connections not being properly closed in some tests
   - ThreadPoolExecutor instances not being properly shutdown

2. **MQTT Broker Stability**
   - Session-scoped MQTT broker failing midway through test run
   - mqtt_monitor fixture asserting broker is running without recovery mechanism
   - Tests failing in cascade when broker becomes unavailable

3. **Resource Leaks**
   - Temporary directories not being cleaned up
   - File handles left open in model converter tests
   - Thread pools not being properly terminated

## Fixes Required

### 1. Fix MQTT Monitor Fixture Resilience
The mqtt_monitor fixture should handle broker unavailability gracefully:
- Check if broker is available before connecting
- Return a mock monitor if broker is unavailable
- Log warning instead of asserting

### 2. Add Resource Cleanup to Model Converter Tests
- Ensure all temporary directories are cleaned up
- Close all file handles explicitly
- Limit number of calibration images created during tests

### 3. Fix ThreadPoolExecutor Cleanup
- Ensure all ThreadPoolExecutor instances use context managers
- Add explicit shutdown calls in finally blocks
- Reduce maximum worker counts

### 4. Add Global Resource Management
- Create a pytest plugin to monitor file descriptor usage
- Add warnings when approaching limits
- Force garbage collection between test modules

### 5. Fix Session MQTT Broker Resilience
- Add health checks to session broker
- Implement broker restart capability
- Add connection pooling to prevent exhaustion

## Implementation Order

1. **Immediate Fix**: Make mqtt_monitor resilient to broker failures
2. **Quick Win**: Add tempfile cleanup to model converter tests  
3. **Medium Term**: Fix ThreadPoolExecutor usage patterns
4. **Long Term**: Implement comprehensive resource monitoring

## Expected Results

- Tests should not fail due to "Too many open files"
- MQTT-dependent tests should gracefully skip if broker unavailable
- Test suite should complete without resource exhaustion
- Clear logging of resource issues for debugging