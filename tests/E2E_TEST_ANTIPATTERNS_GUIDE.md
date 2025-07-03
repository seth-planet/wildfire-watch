# E2E Test Anti-Patterns Guide

## Overview
This guide identifies common anti-patterns in end-to-end (E2E) tests and provides correct approaches for testing the Wildfire Watch system.

## Anti-Pattern #1: Mocking Internal Services

### ❌ Bad: Mocking Internal Components
```python
# NEVER do this in E2E tests
@patch('fire_consensus.consensus.FireConsensus')
def test_fire_detection_e2e(mock_consensus):
    mock_consensus.return_value.check_consensus.return_value = True
    # This doesn't test the actual consensus logic!
```

### ✅ Good: Use Real Services with Test MQTT Broker
```python
def test_fire_detection_e2e(test_mqtt_broker, docker_container_manager):
    # Start real services
    consensus = docker_container_manager.start_container(
        "wildfire-watch/fire_consensus:latest",
        config={'environment': {'MQTT_BROKER': test_mqtt_broker.host}}
    )
    # Test actual service interaction via MQTT
```

## Anti-Pattern #2: Mocking MQTT Client

### ❌ Bad: Mocking MQTT Communication
```python
# NEVER mock MQTT in integration tests
@patch('paho.mqtt.client.Client')
def test_service_communication(mock_mqtt):
    # This doesn't test real message flow!
    pass
```

### ✅ Good: Use TestMQTTBroker
```python
def test_service_communication(test_mqtt_broker):
    # Use real MQTT broker from fixtures
    client = mqtt.Client()
    client.connect(test_mqtt_broker.host, test_mqtt_broker.port)
    # Test real MQTT message flow
```

## Anti-Pattern #3: Not Using Topic Namespacing

### ❌ Bad: Hardcoded Topics
```python
def test_fire_trigger():
    client.publish("fire/trigger", payload)  # Conflicts with parallel tests!
```

### ✅ Good: Use Topic Namespacing
```python
def test_fire_trigger(parallel_test_context):
    topic_prefix = parallel_test_context.get_topic_prefix()
    client.publish(f"{topic_prefix}/fire/trigger", payload)
```

## Anti-Pattern #4: Time-Based Waiting

### ❌ Bad: Fixed Sleep Times
```python
def test_consensus():
    publish_detection()
    time.sleep(5)  # Flaky and slow!
    assert consensus_reached
```

### ✅ Good: Event-Based Waiting
```python
def test_consensus():
    event = Event()
    def on_consensus(msg):
        event.set()
    
    client.on_message = on_consensus
    publish_detection()
    assert event.wait(timeout=10), "Consensus not reached"
```

## Anti-Pattern #5: Manual Container Management

### ❌ Bad: Direct Docker Client Usage
```python
def test_service():
    client = docker.from_env()
    container = client.containers.run(...)  # No cleanup!
```

### ✅ Good: Use DockerContainerManager
```python
def test_service(docker_container_manager):
    container = docker_container_manager.start_container(
        image="service:latest",
        name=docker_container_manager.get_container_name("service")
    )
    # Automatic cleanup on test completion
```

## Anti-Pattern #6: Testing with Production Config

### ❌ Bad: Using Production Settings
```python
def test_fire_suppression():
    # 5 minute cooldown makes tests slow!
    os.environ['COOLDOWN_PERIOD'] = '300'
```

### ✅ Good: Use Test-Specific Config
```python
def test_fire_suppression(parallel_test_context):
    env = parallel_test_context.get_service_env('fire_consensus')
    env['COOLDOWN_PERIOD'] = '5'  # Fast for tests
    env['MAX_ENGINE_RUNTIME'] = '10'  # Safety timeout
```

## Anti-Pattern #7: Not Testing Error Conditions

### ❌ Bad: Only Testing Happy Path
```python
def test_camera_discovery():
    # Only tests successful discovery
    publish_camera("192.168.1.100")
    assert camera_discovered
```

### ✅ Good: Test Error Scenarios
```python
def test_camera_discovery():
    # Test success
    publish_camera("192.168.1.100")
    assert camera_discovered
    
    # Test offline camera
    publish_camera_offline("192.168.1.101")
    assert camera_marked_offline
    
    # Test invalid RTSP
    publish_invalid_rtsp("192.168.1.102")
    assert camera_rejected
```

## Anti-Pattern #8: Incomplete Service Dependencies

### ❌ Bad: Starting Services in Wrong Order
```python
def test_system():
    start_consensus()  # Fails - MQTT not ready!
    start_mqtt()
```

### ✅ Good: Proper Service Orchestration
```python
def test_system(test_mqtt_broker):
    # MQTT already running from fixture
    
    # Start services with dependencies
    services = ['fire-consensus', 'gpio-trigger']
    for service in services:
        start_service_with_mqtt(service, test_mqtt_broker)
```

## Correct E2E Test Structure

```python
@pytest.mark.integration
@pytest.mark.timeout(300)
class TestFireSuppressionE2E:
    """Complete E2E test with proper patterns"""
    
    @pytest.fixture(autouse=True)
    def setup(self, parallel_test_context, test_mqtt_broker, docker_container_manager):
        self.context = parallel_test_context
        self.mqtt_broker = test_mqtt_broker
        self.docker = docker_container_manager
        
    def test_multi_camera_fire_suppression(self):
        # 1. Start real services
        consensus = self._start_service('fire-consensus')
        gpio = self._start_service('gpio-trigger')
        
        # 2. Create MQTT client with namespace
        client = self._create_namespaced_client()
        
        # 3. Setup event-based monitoring
        trigger_event = Event()
        client.message_callback_add("*/fire/trigger", 
                                   lambda c,u,m: trigger_event.set())
        
        # 4. Simulate fire detections from multiple cameras
        for camera_id in ['cam1', 'cam2']:
            self._publish_fire_detection(client, camera_id)
        
        # 5. Verify consensus and triggering
        assert trigger_event.wait(timeout=30), "Fire not triggered"
        
        # 6. Verify pump activation
        pump_status = self._get_pump_status(client)
        assert pump_status['state'] == 'active'
        
    def _start_service(self, name):
        """Start service with test configuration"""
        env = self.context.get_service_env(name)
        env.update({
            'LOG_LEVEL': 'DEBUG',
            'CONSENSUS_THRESHOLD': '2',
            'COOLDOWN_PERIOD': '5'
        })
        
        return self.docker.start_container(
            f"wildfire-watch/{name}:latest",
            self.docker.get_container_name(name),
            config={'environment': env, 'network_mode': 'host'}
        )
```

## Key Principles

1. **Test Real Services**: Always use actual service containers, not mocks
2. **Use Test Infrastructure**: Leverage TestMQTTBroker and DockerContainerManager
3. **Namespace Everything**: Use topic prefixes for parallel test isolation
4. **Event-Driven**: Use events/callbacks instead of sleep
5. **Fast Configuration**: Override timeouts for quick tests
6. **Test Failures**: Include negative test cases
7. **Proper Cleanup**: Use fixtures for automatic cleanup
8. **Clear Assertions**: Make failure messages descriptive

## Common E2E Test Scenarios

### 1. Fire Detection → Suppression
- Start consensus and GPIO services
- Publish detections from multiple cameras
- Verify consensus reached
- Verify pump activation
- Test cooldown period

### 2. Camera Discovery → Configuration
- Start camera detector service
- Simulate camera discovery (ONVIF/mDNS)
- Verify Frigate config generation
- Test camera offline handling

### 3. System Resilience
- Test MQTT disconnection/reconnection
- Test service crashes and restarts
- Test network partitions
- Verify message persistence

### 4. Safety Features
- Test emergency stop
- Test max runtime limits
- Test dry run protection
- Test manual override

## Running E2E Tests

```bash
# Run all E2E tests with proper Python version
./scripts/run_tests_by_python_version.sh --test tests/test_integration_e2e_improved.py

# Run with coverage
./scripts/run_tests_by_python_version.sh --coverage --test tests/

# Run specific E2E test
python3.12 -m pytest tests/test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_multi_camera_consensus -v
```

## Debugging E2E Tests

1. **Check Container Logs**:
   ```python
   logs = container.logs(tail=100).decode('utf-8')
   print(f"Service logs:\n{logs}")
   ```

2. **Monitor MQTT Traffic**:
   ```python
   client.on_log = lambda c,u,l,m: print(f"MQTT: {m}")
   ```

3. **Increase Timeouts**:
   ```python
   @pytest.mark.timeout(600)  # 10 minutes for debugging
   ```

4. **Use DEBUG Logging**:
   ```python
   env['LOG_LEVEL'] = 'DEBUG'
   ```

Remember: E2E tests should validate the **entire system** working together, not individual components in isolation!