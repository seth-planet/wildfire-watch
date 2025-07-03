# Comprehensive Test Fix Plan

## Overview
This plan addresses all failing tests identified during individual test runs. The goal is to fix the underlying code issues (not just the tests) and ensure no internal functionality is mocked.

## Test Status Summary

### Passing Tests ✅
- `test_consensus.py` - 42/42 tests passing
- `test_detect.py` - 51/51 tests passing  
- `test_telemetry.py` - 9/9 tests passing
- `test_mqtt_broker.py` - 27/27 tests passing
- `test_frigate_integration.py` - 16/16 tests passing

### Failing Tests ❌

#### 1. `test_trigger.py` - 3 failures
- `TestConcurrency::test_concurrent_triggers`
- `TestErrorHandling::test_gpio_failure_handling`
- `TestREADMECompliance::test_state_consistency_under_failures`

#### 2. `test_integration_e2e.py` - 1 failure
- `TestE2EIntegration::test_camera_discovery_to_frigate`

#### 3. `test_model_converter_e2e.py` - 1 failure
- `ModelConverterE2ETests::test_tflite_conversion`

#### 4. `test_integration_docker.py` - 1 failure
- `test_docker_integration`

#### 5. `test_simplified_integration.py` - 2 failures
- `TestRealIntegration::test_trigger_receives_consensus_and_activates_pump`
- `TestRealTelemetryReporting::test_telemetry_reporting`

### Skipped Tests ⏭️
- `test_security_nvr_integration.py` - 19/22 tests skipped (requires running Frigate)

## Phase 1: GPIO Trigger State Machine Fixes - ✅ COMPLETE

### Issue Analysis
The three failing trigger tests all relate to state machine consistency and error recovery:
1. Concurrent triggers causing race conditions
2. GPIO failures not transitioning to ERROR state properly
3. State consistency issues during failures

### Root Cause
The `trigger.py` state machine had issues with:
- Timer cleanup during state transitions
- ERROR state transition logic incomplete
- Recovery mechanism not working correctly
- Generic exceptions in timer callbacks don't trigger ERROR state
- SafeGPIO not catching generic Exception
- Recursive error state entry

### Fixes Applied
1. ✅ Added error state transition for critical timer failures (start_engine, emergency_stop, ignition_off)
2. ✅ Updated SafeGPIO to catch generic Exception and convert to HardwareError
3. ✅ Added error state transition for critical pins (MAIN_VALVE, IGN_ON, IGN_START) in _set_pin
4. ✅ Prevented recursive error state entry in _enter_error_state
5. ✅ Made TimerDict iterable for test compatibility

### Progress Notes
- All three trigger tests now passing
- ERROR state properly entered on GPIO failures
- No more infinite recursion
- Thread safety maintained

## Phase 2: MQTT Connection Issues - ✅ COMPLETE

### Issue Analysis
Multiple tests failing due to MQTT connection problems:
- `test_integration_e2e.py` - No events received (client doesn't receive own messages)
- `test_simplified_integration.py` - DNS resolution failure for 'mqtt_broker'

### Root Cause
1. test_simplified_integration.py: PumpController CONFIG dict reads environment at module import time
2. test_integration_e2e.py: Test was subscribing to wrong topic and expecting to receive own messages

### Fixes Applied
1. ✅ Fixed test_simplified_integration.py by deleting and reimporting trigger module after env setup
2. ✅ Fixed test_integration_e2e.py by using separate subscriber client and correct topics
3. ✅ Fixed topic assertion in test_simplified_integration.py (system/trigger_telemetry not gpio/status)

### Progress Notes
- All MQTT connection issues resolved
- Tests now properly use real MQTT broker
- No internal mocking of MQTT functionality

## Phase 3: Model Converter Network Issues - ✅ COMPLETE

### Issue Analysis
TFLite conversion test failing due to incorrect filename expectations

### Root Cause
Test was looking for `yolov8n_320x320.tflite` but converter creates:
- `yolov8n_cpu.tflite` (FP16 optimized)
- `yolov8n_quant.tflite` (INT8 quantized)
- `yolov8n_dynamic.tflite` (Dynamic range quantized)
- `yolov8n_edgetpu.tflite` (Edge TPU compiled)

### Fixes Applied
1. ✅ Updated test to check for all possible TFLite filename variants
2. ✅ Test now passes with 61.7s runtime (model conversion is compute-intensive)

### Progress Notes
- Conversion completed successfully
- All TFLite variants created
- Edge TPU compilation successful

## Phase 4: Docker Integration Test - ✅ COMPLETE

### Issue Analysis
Multiple issues with Docker container networking and configuration

### Root Cause
1. Docker compose v2 vs v1 command differences
2. Container networking - host.docker.internal not resolving
3. Mosquitto default config not listening on container network
4. Test Dockerfile had syntax errors in embedded shell script

### Fixes Applied
1. ✅ Added fallback for both docker compose v2 and v1 commands
2. ✅ Created dedicated Docker network for test containers
3. ✅ Fixed container-to-container communication using container names
4. ✅ Added mosquitto config to listen on all interfaces
5. ✅ Created proper test entrypoint handling D-Bus/Avahi failures gracefully

### Progress Notes
- Docker integration test now passes in 37 seconds
- mDNS/Avahi warnings handled gracefully (expected in containers)
- Fire consensus successfully triggered
- All containers communicate properly

## Phase 5: Hardware Test Coverage - ⏳ PENDING

### Requirements
- Ensure Coral TPU tests work with actual hardware
- Ensure AMD GPU tests work with actual hardware
- Ensure TensorRT tests work with NVIDIA hardware
- Camera tests should use actual network cameras (username:password)

### Fix Plan
1. Create hardware detection tests
2. Add environment-based test skipping for missing hardware
3. Pass camera credentials via environment variables
4. Ensure Raspberry Pi 5 compatibility

## Execution Order

1. **Fix GPIO Trigger State Machine** (HIGH PRIORITY)
   - Critical safety component
   - Multiple test failures
   - Affects pump control

2. **Fix MQTT Connection Issues** (HIGH PRIORITY)
   - Affects multiple integration tests
   - Core communication layer

3. **Fix Model Converter Network** (MEDIUM PRIORITY)
   - Single test affected
   - Can add retry/caching

4. **Fix Docker Integration** (MEDIUM PRIORITY)
   - Container deployment testing
   - Command compatibility

5. **Add Hardware Test Coverage** (LOW PRIORITY)
   - Ensure all hardware paths tested
   - Add proper skip conditions

## Success Criteria

- All tests pass with `--timeout 1800`
- No internal mocking of wildfire-watch components
- Hardware tests execute on available hardware
- Tests compatible with `scripts/run_tests_by_python_version.sh --all`
- No timeouts, failures, or improperly skipped tests

## Notes

- Camera credentials: username:password (pass via env vars, not hardcoded)
- Hailo hardware not installed - skip those tests
- Must maintain Raspberry Pi 5 / Balena compatibility
- Use long timeouts (30 minutes) for infrastructure-heavy tests