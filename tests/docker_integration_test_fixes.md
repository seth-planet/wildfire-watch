# Docker Integration Test Fixes

## Summary of Issues Fixed

### 1. Import Path Errors
**Issue**: Tests were importing from `tests.test_utils.helpers` instead of `test_utils.helpers`

**Fixed Files**:
- `tests/test_integration_docker.py`
- `tests/test_integration_docker_sdk.py`
- `tests/test_integration_e2e_improved.py`

**Fix**: Changed all imports from:
```python
from tests.test_utils.helpers import ParallelTestContext, DockerContainerManager
```
to:
```python
from test_utils.helpers import ParallelTestContext, DockerContainerManager
```

### 2. MQTT Broker Conflicts
**Issue**: `test_integration_docker.py` was creating its own MQTT container instead of using the test broker fixture

**Fix**: 
- Added `test_mqtt_broker` parameter to test function
- Modified `DockerIntegrationTest` to accept and use the test broker
- Updated `start_mqtt_container()` to use test broker instead of creating new container
- Updated all MQTT connections to use test broker host and port

### 3. Container Network Mode
**Issue**: Containers couldn't connect to test MQTT broker on localhost

**Fix**: Changed from custom network to host network mode:
```python
# Before
network=self.network.name,

# After
network_mode='host',  # Use host network to access test broker
```

### 4. Worker-Specific Container Names
**Issue**: Hardcoded container names caused conflicts in parallel test execution

**Fix in `test_integration_docker_sdk.py`**:
```python
# Before
container_name = "fire-consensus-test-sdk"

# After
container_name = self.docker_manager.get_container_name("fire-consensus-sdk")
```

## Test Results

All three tests now pass individually:
- ✅ `test_integration_docker.py::test_docker_integration` - PASSED
- ✅ `test_integration_docker_sdk.py::test_docker_sdk_integration` - PASSED  
- ✅ `test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_service_startup_order` - PASSED

## Best Practices Applied

1. **Use test fixtures**: Always use `test_mqtt_broker` fixture instead of creating separate MQTT containers
2. **Worker isolation**: Use `DockerContainerManager.get_container_name()` for unique container names
3. **Network configuration**: Use host network mode when containers need to access test services on localhost
4. **Import paths**: Use relative imports from `test_utils` not `tests.test_utils`
5. **MQTT configuration**: Get connection parameters from `test_mqtt_broker.get_connection_params()`

## Running the Tests

Individual tests:
```bash
python3.12 -m pytest tests/test_integration_docker.py::test_docker_integration -xvs
python3.12 -m pytest tests/test_integration_docker_sdk.py::test_docker_sdk_integration -xvs
python3.12 -m pytest tests/test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_service_startup_order -xvs
```

All integration tests:
```bash
python3.12 -m pytest tests/test_integration_*.py -xvs
```

## Notes

- Tests may timeout when run together due to resource constraints
- Consider using pytest-xdist for proper parallel execution with worker isolation
- Ensure Docker images are built before running tests (`./scripts/build_test_images.sh`)