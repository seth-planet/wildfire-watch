# Frigate 0.0 Seconds Timing Fix - Session 7

## Problem
Frigate tests were failing with "RuntimeError: Frigate failed to become ready after 0.0 seconds. Last error: None"

## Root Cause Analysis

### Issue Flow
1. `start_container()` called at line 196 in test_security_nvr_integration.py
2. If container exited during the 5-second `wait_timeout`, helpers.py:698 raised RuntimeError
3. This RuntimeError was NOT caught, bypassing all retry logic and timing code (lines 237-431)
4. Exception bubbled up to line 446 showing "0.0 seconds" because `start_time` was never set

### Why 0.0 Seconds?
- Timing initialization (`start_time = time.time()`) was on line 249, AFTER container creation
- When container failed during creation, code jumped directly to final error handling
- `actual_wait_time = time.time() - start_time` used uninitialized `start_time`

## Fixes Applied

### 1. Wrapped start_container in try-except (lines 207-257)
```python
# Initialize timing variables before container creation for accurate error reporting
import time
start_time = time.time()
frigate_ready = False
last_error = None
retry_count = 0
max_retries = 1
container = None

# Try to start the container with error handling
try:
    container = docker_manager_for_frigate.start_container(...)
    # ... container setup ...
except RuntimeError as e:
    # Container failed during initial startup
    last_error = f"Container startup failed: {str(e)}"
    print(f"[Worker: {docker_manager_for_frigate.worker_id}] {last_error}")
    # Don't re-raise here, let the retry logic below handle it
```

### 2. Added Retry Logic for Container Creation (lines 261-313)
- Check if container is None at start of retry loop
- Attempt to create container again on first retry
- Proper error handling and logging

### 3. Updated Final Error Handling (lines 498-510)
- Check if container exists before trying to get logs
- Provide meaningful error message when container never started

## Results
✅ Accurate timing in error messages (shows actual elapsed time, not 0.0s)
✅ Proper error handling for container startup failures
✅ Retry capability for transient startup issues
✅ Tests now pass successfully when container starts normally

## Key Improvements
1. **Error Resilience**: Container startup failures are caught and handled gracefully
2. **Accurate Timing**: Error messages show actual time spent, not misleading 0.0s
3. **Retry Logic**: Automatic retry on container startup failure
4. **Better Diagnostics**: Clear messages about what failed and when

## Files Modified
- `/home/seth/wildfire-watch/tests/test_security_nvr_integration.py`

The Frigate timing issue has been resolved!