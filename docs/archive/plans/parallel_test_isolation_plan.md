# Parallel Test Isolation Plan

## Overview
Fix integration test parallelization issues by implementing proper isolation through unique topic namespaces and dynamic port allocation, allowing tests to run concurrently without conflicts.

## Root Causes
1. Fixed container names causing Docker conflicts
2. Fixed MQTT port (1883) causing broker conflicts  
3. Services publishing to same topics causing message interference
4. No worker-based resource isolation

## Solution Approach
Instead of forcing sequential execution, implement proper isolation:
- **Topic Isolation**: Each test worker uses unique topic namespace
- **Port Isolation**: Dynamic port allocation based on worker ID
- **Container Isolation**: Unique container names per worker
- **Network Isolation**: Separate Docker networks per worker

## Phases

### Phase 1: Enhanced MQTT Broker Fixture - ⏳ PENDING
**Goal**: Create parallel-safe MQTT broker with dynamic port allocation

**Tasks**:
1. Update `enhanced_mqtt_broker.py` to support dynamic port allocation
2. Implement port allocation based on worker ID (base_port + worker_number)
3. Add port conflict detection and retry logic
4. Ensure broker cleanup doesn't affect other workers

**Key Changes**:
```python
class TestMQTTBroker:
    def __init__(self, worker_id='master', base_port=11883):
        self.worker_id = worker_id
        self.port = self._allocate_port(base_port)
```

### Phase 2: Topic Namespace Isolation - ⏳ PENDING  
**Goal**: Implement unique topic namespaces per test worker

**Tasks**:
1. Create `TopicNamespace` utility class
2. Update all MQTT publish/subscribe calls to use namespaced topics
3. Implement topic translation for service communication
4. Add topic namespace to service environment variables

**Key Pattern**:
```python
# Instead of: client.publish("frigate/camera_0/fire", data)
# Use: client.publish(namespace.topic("frigate/camera_0/fire"), data)
```

### Phase 3: Container Name Isolation - ⏳ PENDING
**Goal**: Use worker-based container naming to prevent conflicts

**Tasks**:
1. Update `DockerContainerManager` to use worker ID in names
2. Implement container name generation utility
3. Update all integration tests to use unique container names
4. Add container cleanup by worker ID

**Naming Pattern**:
```
mqtt-broker-{worker_id}
camera-detector-{worker_id}
fire-consensus-{worker_id}
```

### Phase 4: Service Isolation Utilities - ⏳ PENDING
**Goal**: Create comprehensive utilities for parallel-safe integration testing

**Tasks**:
1. Create `ParallelTestContext` class combining all isolation features
2. Implement service configuration with isolated resources
3. Add inter-service communication translation
4. Create pytest fixtures for easy usage

### Phase 5: Update Integration Tests - ⏳ PENDING
**Goal**: Apply isolation to all integration test files

**Test Files to Update**:
- `test_integration_e2e_improved.py`
- `test_e2e_hardware_docker.py`  
- `test_hardware_integration.py`
- `test_e2e_hardware_integration.py`
- `test_mqtt_optimized_example.py`

**Updates per file**:
1. Use `parallel_test_context` fixture
2. Replace hardcoded ports with dynamic allocation
3. Use topic namespaces for all MQTT operations
4. Update container names to include worker ID

### Phase 6: Validation - ⏳ PENDING
**Goal**: Verify parallel execution works correctly

**Tests**:
1. Run with `-n 12` to verify high parallelization
2. Check no "Failed to publish health" errors
3. Verify no container name conflicts
4. Ensure all tests pass consistently
5. Monitor resource usage and cleanup

## Implementation Details

### Topic Namespace Design
```python
class TopicNamespace:
    def __init__(self, worker_id):
        self.prefix = f"test/{worker_id}"
    
    def topic(self, original_topic):
        return f"{self.prefix}/{original_topic}"
    
    def strip(self, namespaced_topic):
        return namespaced_topic.replace(f"{self.prefix}/", "")
```

### Port Allocation Strategy
- Base port: 20000 (high range to avoid conflicts)
- Worker allocation: base + (worker_number * 100)
- Port range per worker: 100 ports
- Example: gw0=20000, gw1=20100, gw2=20200

### Container Lifecycle
1. Create with worker-specific names
2. Use worker-specific networks
3. Cleanup only worker's containers on teardown
4. Health check with worker-isolated endpoints

## Success Criteria
- All integration tests pass with `-n 12`
- No "Failed to publish health" errors
- Test execution time reduced by >50%
- No resource conflicts between workers
- Clean teardown with no orphaned resources

## Timeline
- Phase 1-2: 30 minutes (Core isolation infrastructure)
- Phase 3-4: 30 minutes (Container and service utilities)
- Phase 5: 45 minutes (Update all test files)
- Phase 6: 15 minutes (Validation)
- Total: ~2 hours