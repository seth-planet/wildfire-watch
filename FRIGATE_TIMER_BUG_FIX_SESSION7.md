# Frigate Timer Bug Fix - Session 7

## Problem
All Frigate integration tests were failing with "RuntimeError: Frigate failed to become ready after 30 minutes", but the tests weren't actually waiting 30 minutes. They failed much faster than reported.

## Root Cause
A logic bug in the wait loop timer on line 249:

```python
while retry_count <= max_retries:
    wait_time = start_time  # BUG: This resets to old timestamp!
    backoff_delay = 1.0
    
    while time.time() - wait_time < 1800:  # 30 minute timeout
```

### Why This Failed
1. `start_time` was set much earlier (line 236) before container creation
2. When entering the retry loop, `wait_time = start_time` used that old timestamp
3. The condition `time.time() - wait_time` could already exceed 1800 seconds
4. The inner while loop exited immediately without waiting
5. Container health was never checked
6. Test failed with misleading "after 30 minutes" message

## Fix Applied

### 1. Timer Bug Fix (Line 249)
Changed from:
```python
wait_time = start_time  # Used old timestamp
```

To:
```python
wait_time = time.time()  # Fresh timer for each retry attempt
```

### 2. Accurate Error Message (Line 439-440)
Changed from:
```python
raise RuntimeError(f"Frigate failed to become ready after 30 minutes. Last error: {last_error}")
```

To:
```python
actual_wait_time = time.time() - start_time
raise RuntimeError(f"Frigate failed to become ready after {actual_wait_time:.1f} seconds. Last error: {last_error}")
```

## Test Results

**Before Fix:**
- Tests failed immediately but reported "after 30 minutes"
- No actual waiting occurred
- Container health never checked

**After Fix:**
- Test passed in 19.38 seconds ✅
- Frigate API ready after 5.0 seconds
- Proper timeout behavior restored

## Key Learnings

1. **Timer Logic**: Always use fresh timestamps for timeout calculations
2. **Error Messages**: Report actual elapsed time, not hardcoded values
3. **Variable Names**: Use clear names like `retry_start_time` instead of `wait_time`
4. **Testing**: Small logic bugs can cause confusing symptoms

## Verification

Run the test to verify the fix:
```bash
python3.12 -m pytest tests/test_security_nvr_integration.py::TestSecurityNVRIntegration::test_frigate_service_running -xvs
```

Expected output:
- Container starts successfully
- API becomes ready within seconds
- Test passes without timeout errors

## Related Issues
This was the actual root cause after previous attempts to fix with:
- Memory limits (already adequate at 4GB)
- Exit code detection (working but never reached due to timer bug)

The timer bug prevented all the other fixes from being effective.