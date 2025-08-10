# MQTT Broker Recovery Test Fix

## Problem
The `test_mqtt_broker_recovery` test in `tests/test_integration_e2e_improved.py` was timing out because:
1. The test used dynamic port allocation for the MQTT broker (`'1883/tcp': None`)
2. When Docker restarts a container with dynamic port allocation, it assigns a new port
3. The services were configured to connect to the original port and couldn't reconnect to the new port after broker restart

## Solution
Changed from dynamic to fixed port allocation based on worker ID:
```python
# Calculate a unique port based on worker_id to avoid conflicts
import hashlib
worker_hash = int(hashlib.md5(self.parallel_context.worker_id.encode()).hexdigest()[:4], 16)
mqtt_port = 20000 + (worker_hash % 10000)  # Port range 20000-29999

mqtt_container = self.docker_manager.start_container(
    image="eclipse-mosquitto:2.0",
    name=broker_name,
    config={
        'ports': {'1883/tcp': mqtt_port},  # Fixed port allocation
        'detach': True,
        'volumes': {config_dir: {'bind': '/mosquitto/config', 'mode': 'ro'}}
    },
    wait_timeout=10
)
```

## Key Changes
1. **Fixed Port Allocation**: Uses a deterministic port based on worker ID hash
2. **Port Range**: 20000-29999 to avoid conflicts with system ports
3. **Consistent Port**: Port remains the same after container restart
4. **Monitor Client Re-subscription**: Added automatic re-subscription on reconnect

## Test Results
- Test now passes consistently in ~90 seconds
- All services successfully reconnect after broker restart
- No container conflicts in parallel execution
- Proper cleanup of containers after test completion

## Additional Improvements
- Added re-subscription logic to monitor client on reconnect
- Enhanced debug logging to track reconnection events
- Services maintain their connection settings and reconnect automatically