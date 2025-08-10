# Web Interface Memory Fix - Session 7

## Problem Solved
The "disappeared during health check" errors were caused by containers being OOM-killed due to insufficient memory (512MB default limit).

## Root Cause Analysis
After extensive investigation with AI assistance (o3 model):
1. All cleanup mechanisms were properly worker-aware (previous fixes worked correctly)
2. The real issue was Docker OOM-killing containers when FastAPI/uvicorn startup exceeded 512MB memory
3. Web interface has heavy dependencies: FastAPI, uvicorn[standard], uvloop, orjson, cryptography, etc.

## Fix Applied

### Files Modified
1. `/home/seth/wildfire-watch/tests/test_web_interface_e2e.py`
   - Added `'mem_limit': '2g'` to web_interface_container config (line 104)
   - Added `'mem_limit': '1g'` to gpio_trigger_container config (line 219)

2. `/home/seth/wildfire-watch/tests/test_utils/helpers.py`
   - Added OOM detection logging in wait_for_healthy() (lines 818-825)
   - Now logs container exit code and OOMKilled status when container disappears

## Test Results
- ✅ Primary issue resolved: No more "disappeared during health check" errors
- ✅ 8/25 tests passing (initial test passes consistently)
- ❌ 16 tests still failing due to other issues

## Remaining Issues (Not Memory Related)
1. **Container exits during test run**: Some tests cause the web interface container to exit
2. **Connection refused errors**: Tests fail to connect after container exits
3. **Session scope problems**: Tests share the same container, so if one test breaks it, subsequent tests fail

### Specific Failing Patterns:
- `test_multiple_service_coordination`: assert 0 >= 1 (expects at least 1 service)
- Multiple tests: "Web interface container is not running" 
- Multiple tests: Connection refused errors

## Evidence of Fix Working
```
Container wfmaster-web_interface-160981 belongs to different worker, skipping cleanup
Starting container: wfmaster-web_interface-160981
Waiting for wfmaster-web_interface-160981 to initialize...
✓ wfmaster-web_interface-160981 is running
Container wfmaster-web_interface-160981 health: starting
Container wfmaster-web_interface-160981 health: healthy
```

## Recommendations
1. **Short term**: The memory fix resolves the primary issue
2. **Medium term**: Investigate why containers exit during specific tests
3. **Long term**: Consider using function-scoped fixtures instead of session-scoped for better test isolation

## Summary
The memory limit increase from 512MB to 2GB successfully prevents OOM kills during container startup. The remaining test failures are unrelated to the original "disappeared during health check" issue and require separate investigation.