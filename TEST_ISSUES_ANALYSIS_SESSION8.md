# Test Issues Analysis - Session 8

## Summary of Remaining Issues

After applying all the fixes, some tests are still failing. Here's a detailed analysis:

## 1. test_single_camera_immediate_trigger

**Status:** Still failing
**Root Cause:** Growth detection logic requirements not met

The test is failing because:
1. FireConsensus requires "growing fires" to trigger, not just detections
2. Growing fire detection requires:
   - At least `moving_average_window * 2` detections (4 with default config)
   - Recent average area >= early average area * area_increase_ratio
   
3. The test parameters might not create sufficient growth:
   - initial_size=0.02, growth_rate=0.01
   - Need to verify if this creates >= 1.1x growth (default area_increase_ratio)

**Potential Fix:** Increase growth_rate or adjust detection parameters

## 2. test_dry_run_protection_prevents_damage

**Status:** Still failing
**Root Cause:** Timing race condition with protection thread

The test expects dry run protection to trigger within 8 seconds, but:
1. MAX_DRY_RUN_TIME is set to 3.0 seconds
2. The dry run monitor thread runs every 2 seconds
3. There might be a race condition or the thread isn't detecting the condition

**Potential Fix:** Investigate the dry run monitor thread execution

## 3. test_pump_state_transitions_work  

**Status:** Still failing
**Root Cause:** Timing issue with state transitions

The test fails waiting for STARTING state even though logs show the transition happens.
This suggests the wait_for_state function might be checking too early or there's a synchronization issue.

**Potential Fix:** Add proper synchronization or increase timeouts

## Key Patterns Observed

### 1. Timing and Synchronization Issues
Many failures are due to timing assumptions:
- State transitions happening but assertions checking too early
- Thread execution timing not meeting test expectations
- Real-time vs simulated time mismatches

### 2. Growth Detection Requirements
The consensus logic has specific requirements for detecting "growing" fires that tests might not be meeting:
- Minimum number of detections
- Sufficient area growth ratio
- Proper detection window timing

### 3. MQTT Topic Namespacing
While we fixed the topic namespacing in many places, there might still be edge cases where topics don't match between publishers and subscribers.

## Recommendations

1. **Add Debug Logging:** Add more detailed logging in the consensus and trigger services to understand exactly what's happening during tests

2. **Review Timing Assumptions:** Many tests have hardcoded timeouts and delays that might not be sufficient under load

3. **Verify Growth Parameters:** Ensure test data creates sufficient growth to trigger consensus

4. **Consider Test Isolation:** Some failures might be due to test interference in parallel execution

## Tests That Should Be Passing

Based on the fixes applied:
- test_wait_for_healthy - Fixed to fail fast on container not found
- test_fire_consensus_with_simulated_detections - Fixed MQTT topics
- test_mqtt_broker_recovery - Fixed to use Python processes
- test_docker_sdk_integration - Fixed session scopes
- test_complete_pipeline_with_real_cameras - Added Docker image checks
- test_docker_integration - Added proper error handling

These should work correctly now, but the remaining failures need deeper investigation into the service logic and timing behavior.