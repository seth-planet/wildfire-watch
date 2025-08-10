# Frigate Test Timeout Fix - Session 7

## Problem
All Frigate integration tests were failing with "RuntimeError: Frigate failed to become ready after 30 minutes. Last error: None"

## Root Cause Analysis

### Issues Found
1. **Missing Memory Limits**: Frigate container configurations had no `mem_limit` set
2. **Inefficient Wait Loop**: Container failures weren't detected early, causing 30-minute waits
3. **No Exit Code Detection**: Unlike `wait_for_healthy`, no checking for permanent failures

### Code Analysis
- Lines 199-226: First container config missing `mem_limit`
- Lines 343-366: Retry container config also missing `mem_limit`
- Line 251: Always waits 1800 seconds (30 minutes) regardless of container state
- Lines 257-283: Detects container stopped but only breaks to retry, doesn't fail fast
- Lines 284-308: Handles NotFound but continues waiting

## Fixes Applied

### 1. Added Memory Limits
Modified both container configurations to include 4GB memory limit:

```python
# Line 201 and Line 345
'mem_limit': '4g',  # Frigate needs more memory for video processing
```

Frigate requires significant memory for:
- Video stream processing
- AI object detection
- Multiple camera handling
- Internal caching

### 2. Added Exit Code Detection
Added permanent failure detection in three places:

#### When Container Stops (Lines 262-275)
```python
# Check exit code to determine if this is a permanent failure
try:
    result = subprocess.run(['docker', 'inspect', container.id, 
                           '--format={{.State.ExitCode}} {{.State.OOMKilled}}'], 
                          capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            exit_code, oom_killed = parts[0], parts[1]
            if exit_code != '0' or oom_killed == 'true':
                print(f"Container failed permanently (exit_code={exit_code}, oom_killed={oom_killed})")
                raise RuntimeError(f"Frigate container failed with exit code {exit_code}, OOM: {oom_killed}")
except subprocess.SubprocessError:
    pass  # Continue with existing logic if inspect fails
```

#### When Container NotFound (Lines 288-301)
Similar exit code detection added for docker.errors.NotFound exception

#### When API Error 404 (Lines 313-326)
Same pattern applied for docker.errors.APIError with 404

## How It Works

1. **Memory Protection**: 4GB limit prevents OOM kills during video processing
2. **Fast Failure**: Exit code detection causes immediate failure instead of 30-minute wait
3. **Clear Diagnostics**: Shows exit code and OOM status for debugging
4. **Consistent Pattern**: Matches the successful `wait_for_healthy` fix

## Benefits

1. **Time Savings**: Tests fail in seconds instead of 30 minutes when containers crash
2. **Resource Protection**: Prevents memory exhaustion on test machines
3. **Better Debugging**: Clear indication of failure reason (exit code, OOM status)
4. **Reliability**: Containers less likely to fail with proper memory allocation

## Testing

To test these fixes:
```bash
# Run a single test first
python3.12 -m pytest tests/test_security_nvr_integration.py::TestSecurityNVRIntegration::test_frigate_service_running -xvs

# If successful, run all Frigate tests
python3.12 -m pytest tests/test_security_nvr_integration.py -xvs
```

## Expected Behavior

### Before Fix
- Container OOM killed → Wait 30 minutes → Timeout error
- No indication of actual failure reason

### After Fix  
- Container OOM killed → Immediate detection → Fast failure with clear error
- Shows: "Container failed permanently (exit_code=137, oom_killed=true)"

## Monitoring

Watch for:
1. Container memory usage during tests
2. Exit codes (137 = OOM kill, 1 = general error, 0 = clean exit)
3. API response times if container starts successfully

## Future Considerations

If 4GB is insufficient:
1. Increase to 6GB or 8GB
2. Optimize Frigate test configuration
3. Reduce number of test cameras
4. Disable unnecessary features in test mode

## Related Fixes
This follows the same pattern as the web interface health check fix, applying:
- Memory limit addition
- Exit code detection
- Fast failure on permanent errors