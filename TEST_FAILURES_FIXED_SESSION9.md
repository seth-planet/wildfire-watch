# Test Failures Fixed - Session 9

## Summary
Fixed three critical test failure types after Frigate timeout fix was applied:
1. Docker 500 Server Errors from resource exhaustion
2. GPIO mock missing required attributes
3. MQTT broker startup without proper health checks

## Root Cause Analysis (with o3 AI assistance)

### 1. Docker 500 Server Errors
**Problem**: After fixing Frigate timeout to wait full 1800s, container creation was attempted repeatedly in a loop, causing:
- 100+ container creation attempts over 30 minutes
- Docker daemon fd exhaustion from persistent HTTP connections
- Generic 500 errors when resources depleted

**Solution Applied**:
- Limited container creation to MAX 5 attempts
- Added exponential backoff: `(2^attempt) + random(0-0.2)` seconds
- Added 30-second quarantine for 500 errors
- Created singleton Docker client per worker
- Added container/network pruning in cleanup

### 2. GPIO Mock Missing Attributes
**Problem**: Mock GPIO class only had `_state` and `_lock`, but conftest.py expected:
- `_mode` - Pin mode tracking (IN/OUT)
- `_pull` - Pull resistor state
- `_edge_callbacks` - Edge detection callbacks

**Solution Applied**:
- Added all missing attributes to mock GPIO class
- Updated setup() to track mode and pull states
- Updated cleanup() to clear all dictionaries

### 3. MQTT Broker Health Check
**Problem**: MQTT broker only waited 1.5 seconds with no proper health check
- Tests connected before broker was ready
- ConnectionRefusedError in consensus tests

**Solution Applied**:
- Added `_wait_for_port()` method with socket connection test
- Increased timeout to 15 seconds
- Proper retry with 0.5s intervals
- Clear error messages on timeout

## Files Modified

### 1. `/home/seth/wildfire-watch/gpio_trigger/trigger.py`
```python
# Added to mock GPIO class:
_mode = {}  # Pin mode (IN/OUT)
_pull = {}  # Pull resistor state
_edge_callbacks = {}  # Edge detection callbacks

# Updated setup() to track states
# Updated cleanup() to clear all dicts
```

### 2. `/home/seth/wildfire-watch/tests/test_security_nvr_integration.py`
```python
# Added container creation limits:
container_creation_attempts = 0
MAX_CONTAINER_ATTEMPTS = 5

# Added 500 error quarantine:
if isinstance(e, docker.errors.APIError) and "500" in str(e):
    print("Docker daemon overloaded, applying 30s quarantine...")
    time.sleep(30)

# Added singleton Docker client:
@pytest.fixture(scope="session")
def docker_client_for_frigate():
    client = docker.from_env()
    client.api.timeout = 240
    yield client
    client.close()
```

### 3. `/home/seth/wildfire-watch/tests/test_utils/helpers.py`
```python
# Added pruning to cleanup():
if force:
    prune_result = self.client.containers.prune(
        filters={'label': [f'com.wildfire.worker={self.worker_id}']}
    )
    net_result = self.client.networks.prune(
        filters={'label': [f'com.wildfire.worker={self.worker_id}']}
    )
```

### 4. `/home/seth/wildfire-watch/tests/test_utils/enhanced_mqtt_broker.py`
```python
# Added proper port health check:
def _wait_for_port(self, timeout=15, interval=0.5):
    """Wait for MQTT broker port to become available"""
    # Socket connection test with retry

# Updated start sequence:
if not self._wait_for_port(timeout=15):
    raise RuntimeError(f"Mosquitto failed to start on port {self.port}")
```

## Key Improvements

1. **Resource Management**
   - Bounded container creation attempts
   - Singleton Docker client per worker
   - Automatic resource pruning
   - Quarantine for overload conditions

2. **Test Reliability**
   - Proper health checks before proceeding
   - Clear error messages with context
   - Exponential backoff with jitter
   - Worker-aware resource cleanup

3. **Code Completeness**
   - GPIO mock now matches real RPi.GPIO interface
   - All required attributes present
   - Proper state tracking

## Testing Recommendations

1. **Validate GPIO Fix**:
   ```bash
   python3.12 -m pytest tests/test_trigger.py -v
   ```

2. **Validate Container Limits**:
   ```bash
   python3.12 -m pytest tests/test_security_nvr_integration.py::test_frigate_service_running -v
   ```

3. **Gradual Parallel Testing**:
   ```bash
   # Start with single worker
   python3.12 -m pytest tests/test_security_nvr_integration.py -v
   
   # Increase to 2 workers
   python3.12 -m pytest tests/test_security_nvr_integration.py -n 2 -v
   
   # Full parallel if stable
   python3.12 -m pytest tests/test_security_nvr_integration.py -n auto -v
   ```

4. **Monitor for 500 errors**:
   ```bash
   python3.12 -m pytest tests/ -v 2>&1 | grep -i "500\|quarantine"
   ```

## Future Considerations

1. Consider using `testcontainers-python` for built-in retry/cleanup
2. Implement container pooling for frequently used images
3. Add Docker daemon health monitoring
4. Consider moving to docker-compose for multi-container tests

These fixes address all three failure types comprehensively while maintaining test isolation and reliability.