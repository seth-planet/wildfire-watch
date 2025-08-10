# Web Interface Parallel Test Fix - Session 8

## Problem
Web interface tests were being skipped in parallel execution with errors:
- Containers marked as "belongs to different worker"
- 409 Conflict errors when accessing container logs
- Tests passing individually but failing in parallel

## Root Cause
Container prefix mismatch in DockerContainerManager:
- **Container prefix**: `self.container_prefix = f"wf-{worker_id}"` (e.g., `wf-gw1`)
- **Container names**: `f"wf{self.worker_id}-{service}-{timestamp}"` (e.g., `wfgw1-web_interface-123456`)
- Cleanup check couldn't match `wf-gw1` prefix with `wfgw1` container names

## Fix Applied
Updated `/home/seth/wildfire-watch/tests/test_utils/helpers.py` line 869:

**From:**
```python
return f"wf{self.worker_id}-{service}-{timestamp}"
```

**To:**
```python
return f"{self.container_prefix}-{service}-{timestamp}"
```

This ensures container names use the same prefix format as the cleanup checks.

## Results
✅ All 8 web interface tests pass in parallel execution
✅ No "belongs to different worker" messages
✅ No 409 Conflict errors
✅ Tests complete in ~80 seconds with 4 workers

## Testing
```bash
# Individual test (passed)
python3.12 -m pytest tests/test_web_interface_e2e.py::TestWebInterfaceE2E::test_dashboard_displays_real_service_health -xvs

# All tests in parallel (all 8 passed)
python3.12 -m pytest tests/test_web_interface_e2e.py -xvs -n 4
```

## Key Improvement
Container naming is now consistent throughout the codebase, ensuring proper worker isolation in parallel test execution.