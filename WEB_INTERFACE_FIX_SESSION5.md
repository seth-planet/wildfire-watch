# Web Interface Container Fix - Session 5

## Problem
Web interface tests were failing with "disappeared during health check" errors, even after previous fixes that added timestamps to container names and updated wait_for_healthy logic.

## Root Cause
The ProcessCleaner.cleanup_docker_containers() method was forcefully removing ALL containers with names starting with "wf-" using `docker rm -f`, regardless of which worker created them. This caused containers to disappear while other workers were still using them.

## Solution Implemented

### 1. Updated Container Naming Pattern
**File: /home/seth/wildfire-watch/tests/test_utils/helpers.py**
- Changed container naming to put worker ID before "wf" prefix
- From: `wf-{worker_id}-{service}-{timestamp}`
- To: `wf{worker_id}-{service}-{timestamp}` 
- Example: `wfgw11-web_interface-133531`

This prevents containers from matching the generic "wf-" filter used by cleanup.

### 2. Made ProcessCleaner Worker-Aware
**File: /home/seth/wildfire-watch/tests/test_utils/process_cleanup.py**
- Modified cleanup_docker_containers() to only remove containers for the current worker
- Gets worker ID from PYTEST_XDIST_WORKER environment variable
- If no worker ID found, skips Docker cleanup entirely to prevent cross-worker interference
- Filter changed from `name=wf-` to `name=wf{worker_id}-`

### 3. Timeout Configuration
**Current timeout settings (already sufficient):**
- pytest.ini: 3600s (1 hour) per test
- pytest-python310.ini: 7200s (2 hours) per test
- pytest-python38.ini: 7200s (2 hours) per test
- All session timeouts: 14400s (4 hours)

These values already exceed the requested 1800s (30 minutes).

## Expected Results
1. Each worker creates containers with unique prefixes (wfgw0-, wfgw1-, etc.)
2. Cleanup processes only remove containers belonging to their own worker
3. No more "disappeared during health check" errors
4. Parallel test execution works reliably

## Testing the Fix
Run parallel tests with:
```bash
python3.12 -m pytest tests/test_web_simple.py tests/test_web_interface_e2e.py -n 4
```

Monitor container names to verify they follow the new pattern and that no cross-worker cleanup occurs.