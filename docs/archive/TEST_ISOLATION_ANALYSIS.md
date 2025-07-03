# Test Isolation Analysis for Wildfire Watch

## Problem Summary

Tests are failing when run in bulk but passing individually, indicating:
1. **Resource exhaustion** - MQTT connections, threads, file handles
2. **State leakage** - Tests modifying shared state
3. **Race conditions** - Concurrent test execution issues
4. **Fixture teardown** - Incomplete cleanup between tests

## Root Causes Identified

### 1. MQTT Connection Management
- Tests create new MQTT connections without proper cleanup
- SimpleMQTTBroker fallback has message delivery issues (noted in mqtt_test_broker.py:136)
- Connection pool exhaustion from multiple test clients
- Broker restart/cleanup issues between tests

### 2. Thread Lifecycle Issues
- Background threads not properly terminated
- Timer threads from telemetry and consensus services
- Threading.Timer objects accumulating (cam_telemetry/telemetry.py uses timers)
- No thread join() with timeout in many places

### 3. State Persistence
- Global state in service instances
- Shared detector instances between tests
- MAC tracker state not reset
- Camera state carrying over

### 4. Resource Cleanup
- ProcessPoolExecutor not properly shutdown
- File handles left open
- Socket connections not closed
- OpenCV resources not released

## Symptoms Observed

1. **test_consensus.py**: EEEEEEEEE pattern suggests setup/fixture errors
2. **test_detect.py**: Multiple errors indicate resource issues
3. **Timeout on bulk runs**: Tests hang waiting for resources
4. **Individual tests pass**: No resource contention when run alone

## Systematic Fix Approach

### Phase 1: Immediate Fixes (High Impact)

1. **Fix MQTT Test Infrastructure**
   - Use session-scoped MQTT broker
   - Implement connection pooling
   - Add proper broker cleanup with timeouts
   - Fix SimpleMQTTBroker message delivery

2. **Thread Management**
   - Add thread cleanup fixtures
   - Implement thread timeout handling
   - Cancel all timers in teardown
   - Use daemon threads where appropriate

3. **State Isolation**
   - Reset global state in fixtures
   - Use fresh instances per test
   - Clear singleton patterns
   - Reset MAC tracker between tests

### Phase 2: Test Optimization

1. **Fixture Optimization**
   - Session-scoped expensive resources
   - Lazy initialization
   - Resource pooling
   - Parallel-safe fixtures

2. **Mock Strategy**
   - Mock external dependencies only
   - Use real components where possible
   - Avoid mocking internal modules
   - Process isolation for RTSP validation

### Phase 3: Best Practices Implementation

1. **Test Markers**
   ```python
   @pytest.mark.mqtt  # Tests requiring MQTT broker
   @pytest.mark.slow  # Long-running tests
   @pytest.mark.integration  # Integration tests
   ```

2. **Fixture Scoping**
   - Session: MQTT broker, network setup
   - Module: Service configurations
   - Function: Service instances, state

3. **Cleanup Patterns**
   ```python
   @pytest.fixture
   def service():
       svc = Service()
       yield svc
       svc.cleanup()  # Explicit cleanup
       # Force thread termination
       # Close connections
       # Clear state
   ```

## Recommended Implementation Order

1. **Fix conftest.py** (already has session broker but needs improvements)
2. **Update mqtt_test_broker.py** to handle concurrent clients better
3. **Add thread cleanup to all service fixtures**
4. **Implement state reset helpers**
5. **Add test grouping and markers**
6. **Optimize fixture scoping**

## Specific File Changes Needed

### 1. tests/conftest.py
- Enhance session_mqtt_broker with better cleanup
- Add thread monitoring and cleanup
- Implement state reset helpers

### 2. tests/mqtt_test_broker.py
- Fix SimpleMQTTBroker message delivery
- Add connection limit handling
- Improve broker cleanup

### 3. tests/test_consensus.py
- Use session broker fixture
- Add proper service cleanup
- Reset consensus state between tests

### 4. tests/test_detect.py
- Fix camera_detector fixture cleanup
- Add ProcessPoolExecutor shutdown
- Clear camera state properly

### 5. Service Files
- Add cleanup() methods to all services
- Implement thread cancellation
- Add state reset capabilities

## Testing the Fixes

1. Run tests individually to establish baseline
2. Run tests in small groups to identify conflicts
3. Use pytest-xdist for parallel execution testing
4. Monitor resource usage during test runs
5. Verify no thread/connection leaks

## Success Criteria

- All tests pass when run with `pytest tests/`
- No resource leaks after test completion
- Consistent pass/fail behavior
- Tests complete in reasonable time (<5 min)
- No hanging tests or timeouts