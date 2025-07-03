# MQTT Namespace Issue Fix Guide

## Problem Summary

The E2E health monitoring test is failing because of a namespace mismatch between:
1. **Docker containers**: Publish to standard topics (e.g., `system/camera_detector_health`)
2. **Test client**: Subscribes to namespaced topics (e.g., `test/gw0/system/camera_detector_health`)

This results in the test client receiving NO messages even though the services are publishing successfully.

## Root Causes

### 1. Namespace Isolation Design
The `TopicNamespace` wrapper was designed for parallel test isolation but creates issues when:
- Services in Docker containers don't know about the test namespace
- The test client only listens to namespaced topics
- There's no bridge between namespaced and non-namespaced topics

### 2. Camera Detector Disconnections
The frequent MQTT disconnections in camera-detector suggest:
- Network issues between container and host
- MQTT broker rejecting connections (possibly due to client ID conflicts)
- Resource constraints or timeout issues

## Solutions

### Solution 1: Bypass Namespace for Docker Tests (Recommended)

Use the `BypassNamespaceClient` for Docker-based tests:

```python
from tests.mqtt_namespace_bypass import setup_health_monitoring_test, verify_service_health

def test_e2e_health_monitoring(mqtt_broker, docker_containers):
    # Use bypass client that subscribes to actual topics
    client = setup_health_monitoring_test(mqtt_broker.port)
    
    try:
        # Wait for services
        time.sleep(5)
        
        # Verify health
        health_status = verify_service_health(
            client, 
            ['camera_detector', 'fire_consensus', 'gpio_trigger'],
            timeout=30
        )
        
        assert all(health_status.values()), "Missing health messages"
    finally:
        client.loop_stop()
        client.disconnect()
```

### Solution 2: Namespace Bridge Client

Use the `NamespaceBridgeMQTTClient` that subscribes to both namespaced and non-namespaced topics:

```python
from tests.mqtt_namespace_fix import create_health_monitoring_client

def test_e2e_health_monitoring(mqtt_broker):
    # Create bridge client
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    namespace = f"test/{worker_id}" if worker_id != 'master' else None
    
    client, messages, event = create_health_monitoring_client(
        port=mqtt_broker.port,
        namespace=namespace
    )
    
    # Client will receive both namespaced and non-namespaced messages
```

### Solution 3: Pass Namespace to Docker Containers

Modify service code to support namespace environment variable:

```python
# In camera_detector/detect.py, fire_consensus/consensus.py, etc.
class ServiceBase:
    def __init__(self):
        self.namespace = os.environ.get('MQTT_TOPIC_NAMESPACE', '')
        
    def publish(self, topic, payload):
        if self.namespace:
            topic = f"{self.namespace}/{topic}"
        self.mqtt_client.publish(topic, payload)
```

Then pass namespace when starting containers:

```python
config = {
    'environment': {
        'MQTT_TOPIC_NAMESPACE': f'test/{worker_id}',
        # ... other env vars
    }
}
```

## Implementation Steps

### 1. Immediate Fix (No Code Changes to Services)

1. Replace the namespaced client in the test with `BypassNamespaceClient`:

```python
# Instead of:
client = create_namespaced_client(mqtt.Client(), worker_id)

# Use:
client = BypassNamespaceClient(f'test_{worker_id}')
```

2. Subscribe to actual topics the services publish to:

```python
client.subscribe('system/camera_detector_health')
client.subscribe('system/fire_consensus_health') 
client.subscribe('system/gpio_trigger_health')
```

### 2. Fix Camera Detector Disconnections

1. Add connection retry logic:

```python
class MQTTClientWithRetry:
    def __init__(self, client_id):
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_disconnect = self._on_disconnect
        self.connected = False
        
    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logging.warning(f"Disconnected with rc={rc}, will retry...")
            self._reconnect()
            
    def _reconnect(self):
        max_retries = 5
        for i in range(max_retries):
            try:
                time.sleep(2 ** i)  # Exponential backoff
                self.client.reconnect()
                logging.info("Reconnected successfully")
                break
            except Exception as e:
                logging.error(f"Reconnect attempt {i+1} failed: {e}")
```

2. Use unique client IDs to avoid conflicts:

```python
client_id = f"{service_name}_{container_id}_{int(time.time())}"
```

### 3. Debug MQTT Communication

Add detailed logging to understand message flow:

```python
def debug_mqtt_messages(client):
    """Log all MQTT traffic for debugging."""
    def on_log(client, userdata, level, buf):
        print(f"MQTT LOG [{level}]: {buf}")
        
    client.on_log = on_log
    client.enable_logger()
```

## Testing the Fix

1. Run a simple test to verify MQTT communication:

```python
def test_mqtt_basic_communication(mqtt_broker):
    # Publisher (simulating Docker container)
    pub = mqtt.Client('publisher')
    pub.connect('localhost', mqtt_broker.port)
    
    # Subscriber (test client) 
    sub = BypassNamespaceClient('subscriber')
    sub.connect('localhost', mqtt_broker.port)
    sub.loop_start()
    sub.subscribe('#')
    
    time.sleep(1)
    
    # Publish test message
    pub.publish('system/test_health', 'healthy')
    
    # Verify received
    assert sub.wait_for_messages(1, timeout=5)
    messages = sub.get_messages()
    assert any(m['topic'] == 'system/test_health' for m in messages)
```

2. Run the health monitoring test with debug output:

```python
def test_e2e_health_monitoring_debug(mqtt_broker, docker_containers):
    client = BypassNamespaceClient('debug_client')
    client.connect('localhost', mqtt_broker.port)
    client.loop_start()
    
    # Subscribe to everything
    client.subscribe('#')
    
    # Wait and collect all messages
    time.sleep(20)
    
    # Print all received messages
    messages = client.get_messages()
    print(f"\nReceived {len(messages)} messages:")
    for msg in messages:
        print(f"  {msg['topic']} = {msg['payload']}")
```

## Best Practices

1. **Use bypass client for Docker tests**: Since Docker containers can't easily be made namespace-aware
2. **Keep namespace for unit tests**: Where you control all MQTT clients
3. **Add health endpoints**: Consider adding HTTP health checks as backup
4. **Monitor broker logs**: Check MQTT broker logs for connection/auth issues
5. **Use unique client IDs**: Prevent conflicts in parallel tests

## Alternative: Disable Namespace for E2E Tests

If namespace isolation isn't critical for E2E tests, disable it:

```python
@pytest.fixture
def mqtt_client_e2e(mqtt_broker):
    """MQTT client for E2E tests without namespace isolation."""
    client = mqtt.Client(f'e2e_test_{int(time.time())}')
    client.connect('localhost', mqtt_broker.port)
    client.loop_start()
    yield client
    client.loop_stop()
    client.disconnect()
```

This avoids the complexity of namespace translation while still providing test isolation through unique client IDs and separate test brokers.