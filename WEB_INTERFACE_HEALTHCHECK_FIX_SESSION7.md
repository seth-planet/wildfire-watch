# Web Interface Health Check Fix - Session 7

## Problem
Web interface tests were failing with "RuntimeError: Web interface container disappeared during health check" even after increasing memory limits to 2GB.

## Root Cause Analysis

### Issue Found
The `wait_for_healthy()` method in `tests/test_utils/helpers.py` had flawed logic:

1. When containers disappeared (docker.errors.NotFound), it would get exit status but not act on it
2. It would continue looping for 1800 seconds (30 minutes) regardless of failure type
3. This caused tests to wait 30 minutes before failing with the "disappeared" error

### Code Analysis
```python
# OLD CODE (lines 816-829):
except docker.errors.NotFound:
    # Get exit status...
    print(f"  Container exit info: {result.stdout.strip()}")
    # Don't return False - this might be a transient issue
    # Let the timeout handle it
    time.sleep(2)
    continue  # <-- PROBLEM: Always continues, even for permanent failures
```

### Pattern Inconsistency
- Line 784-786: When container exists with status='exited', it raises RuntimeError immediately
- Line 816-829: When container NotFound, it waits 30 minutes
- This inconsistency caused the reported failures

## Fix Applied

Modified `wait_for_healthy()` to parse exit status and return False immediately for permanent failures:

```python
# NEW CODE:
if result.returncode == 0 and result.stdout.strip():
    exit_info = result.stdout.strip()
    print(f"  Container exit info: {exit_info}")
    
    # Parse exit code and OOM status
    parts = exit_info.split()
    if len(parts) >= 2:
        exit_code = parts[0]
        oom_killed = parts[1]
        
        # Check for permanent failure conditions
        if exit_code != '0' or oom_killed == 'true':
            print(f"  Container failed permanently (exit_code={exit_code}, oom_killed={oom_killed})")
            return False
```

## How It Works

1. Docker inspect returns format: "ExitCode OOMKilled" (e.g., "42 false")
2. If ExitCode != 0 → Container failed, return False immediately
3. If OOMKilled == "true" → Out of memory, return False immediately  
4. Otherwise → Continue loop (might be transient issue)

## Benefits

1. **Fast Failure**: Tests fail immediately when containers crash instead of waiting 30 minutes
2. **Clear Diagnostics**: Logs show exit code and OOM status for debugging
3. **Consistent Behavior**: NotFound handling now matches exited status handling

## Next Steps

With this fix, the tests should:
1. Fail quickly when containers crash (within seconds, not 30 minutes)
2. Show clear error messages with exit codes
3. Allow debugging of the actual container failure cause

The memory limit of 2GB is still in effect, so if containers are still failing, we'll now see the actual exit codes to diagnose why.