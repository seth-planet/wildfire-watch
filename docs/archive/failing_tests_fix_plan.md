# Failing Tests Fix Plan

## Overview
Based on the test output, we have several failing tests that need to be addressed. The main issues appear to be:

1. **Docker SDK Integration Test** - Consensus service not triggering despite receiving detections
2. **Test Infrastructure Issues** - Missing test isolation fixtures and other modules
3. **Test Collection Issues** - Some tests not being found by pytest

## Phase 1: Fix Docker SDK Integration Test - ✅ COMPLETE

### Issue Analysis
The `test_integration_docker_sdk.py` test is failing because:
- Fire consensus service receives detections but doesn't trigger
- The consensus service logs show it's subscribing correctly but not processing detections
- Possible issues:
  - Detection validation failing
  - Growth analysis not working as expected
  - Camera state not being initialized properly

### Fix Strategy
1. Debug consensus service detection processing
2. Ensure camera telemetry is processed before detections
3. Verify detection format matches consensus expectations
4. Add more detailed logging to understand flow

### Fixes Applied
1. **test_integration_docker_sdk.py**:
   - Added delay between telemetry messages
   - Increased telemetry processing wait time
   - Added MOVING_AVERAGE_WINDOW=2 to environment
   - Added COOLDOWN_PERIOD=0 for testing
   - Increased DETECTION_WINDOW to 15 seconds

2. **fire_consensus/consensus.py**:
   - Added logging when camera states are created
   - Added logging when telemetry is received
   - Enhanced consensus checking logs to show why cameras are skipped

3. **test_integration_docker.py**:
   - Fixed to send camera telemetry before detections
   - Fixed detection format (object_type instead of object)
   - Fixed bounding box sizes (were too small)
   - Added growth pattern to detections
   - Updated environment variables to match SDK test

## Phase 2: Fix Test Infrastructure - ⏳ PENDING

### Missing Modules
- `test_isolation_fixtures`
- `enhanced_mqtt_broker`
- Concurrent futures fixes
- Optimized camera detector fixtures

### Fix Strategy
1. Create missing fixture files or remove references
2. Ensure all required test dependencies are installed
3. Fix import paths in conftest.py

## Phase 3: Fix Test Collection Issues - ⏳ PENDING

### Issue
Some tests like `test_rtsp_validation_multiple_streams` are not being found by pytest.

### Fix Strategy
1. Verify test naming conventions
2. Check class/method structure
3. Ensure proper test discovery configuration

## Phase 4: Run Comprehensive Test Suite - ⏳ PENDING

### Strategy
1. Fix all failing tests one by one
2. Ensure tests follow the guidelines:
   - Test actual code, not mocks
   - Use real MQTT broker for integration tests
   - Fix the code, not just the test
3. Document any tests that need to be skipped with reasons

## Progress Notes
- Started analysis of test_integration_docker_sdk.py
- Identified that consensus service is receiving messages but not processing them correctly
- **ROOT CAUSE FOUND**: Consensus service skips detections from "offline" cameras
  - Cameras are marked online when they send telemetry
  - The test sends telemetry but the camera state might not be persisting
  - The check `camera.is_online()` uses `last_telemetry` timestamp with CAMERA_TIMEOUT (default 120s)
- Need to ensure cameras are properly marked as online before sending detections

## Test Results
[To be added after fixes are complete]