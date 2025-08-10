# Container Naming Best Practices Fix

## Issue
My initial fix for `test_complete_pipeline_with_real_cameras` used container names like:
- `f"e2e-mqtt-{worker_id}"`
- `f"e2e-camera-detector-{worker_id}"`

This didn't follow the established best practices in the codebase.

## Best Practice Discovery
After examining the codebase, I found that `DockerContainerManager` follows this pattern:
```python
self.container_prefix = f"wf-{worker_id}"
def get_container_name(self, service: str) -> str:
    return f"{self.container_prefix}-{service}"
```

This creates container names like: `wf-master-mqtt`, `wf-gw0-camera-detector`, etc.

## Corrected Implementation
Updated all container names to follow the `wf-{worker_id}-{service}` pattern:

### Before (incorrect):
```python
name=f"e2e-mqtt-{worker_id}"
name=f"e2e-camera-detector-{worker_id}"
```

### After (correct):
```python
container_prefix = f"wf-{worker_id}"
name=f"{container_prefix}-e2e-mqtt"
name=f"{container_prefix}-e2e-camera-detector"
```

## Changes Made
1. Updated container name cleanup list to use `wf-` prefix
2. Updated all container creation calls to use `container_prefix`
3. Added `worker_id` and `container_prefix` to the fixture's yield dictionary
4. Updated test method to extract these values from the setup

## Why This Matters
- **Consistency**: All tests use the same naming convention
- **Debugging**: Easy to identify which worker created which container
- **Cleanup**: Container management utilities expect this pattern
- **Parallel Testing**: Prevents conflicts between test workers

## Container Naming Pattern
All containers in the test suite should follow:
```
wf-{worker_id}-{service_name}
```

Where:
- `wf` = wildfire (project prefix)
- `worker_id` = pytest-xdist worker ID (master, gw0, gw1, etc.)
- `service_name` = descriptive service identifier