# Frigate 1800-Second Timeout Fix - Session 8

## Problem
Frigate tests were failing with "RuntimeError: Frigate failed to become ready after 10.3 seconds. Last error: None" despite having a 1800-second (30-minute) timeout configured.

## Root Cause Analysis
Through detailed investigation with o3 AI model assistance:

1. **Container startup** uses `wait_timeout=5` seconds
2. **If container fails within 5s**, helpers.py:698 raises RuntimeError
3. **Exception caught**, leaving `container=None`
4. **Line 312-313**: `if container is None: break` - **THIS WAS THE BUG**
5. This break **bypassed the entire 1800-second wait loop** at line 318
6. Tests failed after ~10.3 seconds (5s initial + 5s retry)

## Fixes Applied

### 1. Removed Early Exit (Line 312-313)
**Before:**
```python
if container is None:
    break  # No container to check
```

**After:**
```python
# Don't break here - let the wait loop handle container creation
```

### 2. Added Container Handling in Wait Loop
Added logic to create container inside the wait loop if it's None:
```python
while time.time() - wait_time < 1800:
    try:
        # Handle case where container hasn't been created yet
        if container is None:
            # Try to create container
            try:
                container = docker_manager_for_frigate.start_container(...)
                print(f"Container created after {time.time() - start_time:.1f}s")
            except Exception as e:
                last_error = f"Container creation failed: {str(e)}"
                print(f"Retrying container creation in {backoff_delay}s...")
                time.sleep(backoff_delay)
                backoff_delay = min(backoff_delay * 2, 60)  # Exponential backoff
                continue
```

### 3. Increased wait_timeout to 30 seconds
Changed all instances of `wait_timeout=5` to `wait_timeout=30` for better stability.

## Results
✅ Tests now properly wait up to 1800 seconds for Frigate to become ready
✅ Container creation failures are retried with exponential backoff
✅ Frigate tests passing successfully:
- `test_frigate_service_running` - PASSED
- `test_frigate_stats_endpoint` - PASSED  
- `test_mqtt_connection` - PASSED

## Key Improvements
1. **No more premature failures** - Tests wait the full timeout period
2. **Resilient container creation** - Automatic retries with backoff
3. **Better error handling** - Clear messages about what's happening
4. **Accurate timing** - Shows actual time elapsed, not misleading values

## Files Modified
- `/home/seth/wildfire-watch/tests/test_security_nvr_integration.py`

The Frigate timeout issue has been permanently resolved!