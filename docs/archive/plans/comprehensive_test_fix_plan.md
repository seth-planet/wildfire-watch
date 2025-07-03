# Comprehensive Test Fix Plan for Wildfire Watch

## Phase 1: Fix MQTT Infrastructure (Highest Priority)

### Issue: MQTT Broker Startup Failures
The `MQTTTestBroker` has several issues:
1. Hardcoded 2-second wait after starting mosquitto (line 71)
2. No proper health check or retry mechanism
3. Fallback to embedded brokers that may not support all MQTT features

### Fix 1.1: Improve MQTTTestBroker
```python
# In mqtt_test_broker.py, replace the _start_mosquitto method:

def _start_mosquitto(self):
    """Start mosquitto broker with proper health checking"""
    # Create temporary directories
    self.data_dir = tempfile.mkdtemp(prefix="mqtt_test_")
    
    # Create mosquitto config for testing
    config_content = f"""
port {self.port}
allow_anonymous true
persistence false
log_type error
log_dest stdout
"""
    
    self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
    with open(self.config_file, 'w') as f:
        f.write(config_content)
    
    # Start mosquitto broker
    self.process = subprocess.Popen([
        'mosquitto', '-c', self.config_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for broker to be ready with proper health check
    if not self.wait_for_ready(timeout=30):
        # Get output for debugging
        stdout, stderr = self.process.communicate()
        raise RuntimeError(f"Mosquitto failed to start within 30s. Stderr: {stderr.decode()}")
```

### Fix 1.2: Add Broker Diagnostics to conftest.py
```python
# In conftest.py session_mqtt_broker fixture, add:

def get_broker_diagnostics(broker):
    """Get diagnostic info from broker for debugging"""
    diagnostics = []
    
    # Check if process is running
    if hasattr(broker, 'process') and broker.process:
        diagnostics.append(f"Process PID: {broker.process.pid}")
        diagnostics.append(f"Process running: {broker.process.poll() is None}")
        
        # Try to get recent output
        try:
            stdout, stderr = broker.process.communicate(timeout=0.1)
            if stdout:
                diagnostics.append(f"Stdout: {stdout.decode()[:200]}")
            if stderr:
                diagnostics.append(f"Stderr: {stderr.decode()[:200]}")
        except:
            pass
    
    # Check port availability
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', broker.port))
            diagnostics.append(f"Port {broker.port} accessible: {result == 0}")
    except:
        diagnostics.append("Could not check port")
    
    return "\n".join(diagnostics)
```

## Phase 2: Fix Docker Infrastructure

### Issue: Docker Network Not Found
Tests are trying to use `wildfire_test_net` before it's created.

### Fix 2.1: Add Session-Scoped Docker Fixtures
```python
# Add to conftest.py:

@pytest.fixture(scope="session")
def docker_client():
    """Provides a Docker client, skipping tests if Docker unavailable"""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        logger.info("Docker client connected successfully")
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")

@pytest.fixture(scope="session")
def docker_test_network(docker_client):
    """Create a persistent test network for all Docker tests"""
    network_name = "wildfire_test_net"
    
    # Check if network exists
    try:
        network = docker_client.networks.get(network_name)
        logger.info(f"Using existing network: {network_name}")
    except docker.errors.NotFound:
        logger.info(f"Creating network: {network_name}")
        network = docker_client.networks.create(
            network_name,
            driver="bridge",
            options={"com.docker.network.bridge.name": "wildfire_test"}
        )
    
    yield network
    
    # Cleanup
    try:
        network.reload()
        if not network.containers:
            network.remove()
            logger.info(f"Removed network: {network_name}")
    except Exception as e:
        logger.warning(f"Could not remove network: {e}")
```

### Fix 2.2: Update Docker-Based Tests
All Docker tests should depend on `docker_test_network` fixture.

## Phase 3: Fix Hardware-Dependent Tests

### Issue: Tests Failing on Missing Hardware
Model converter and hardware tests fail when specific hardware isn't available.

### Fix 3.1: Add Hardware Detection
```python
# Add to conftest.py:

def has_coral_tpu():
    """Check if Coral TPU is available"""
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        return len(list_edge_tpus()) > 0
    except ImportError:
        return False

def has_tensorrt():
    """Check if TensorRT is available"""
    try:
        import tensorrt
        # Also check for GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        return result.returncode == 0
    except:
        return False

def has_camera_on_network():
    """Check if test camera is accessible"""
    import socket
    # Try to connect to common camera ports
    for port in [554, 8554]:  # RTSP ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                # Try common camera IPs
                for ip in ['192.168.1.64', '192.168.1.65']:
                    if s.connect_ex((ip, port)) == 0:
                        return True
        except:
            pass
    return False

# Add markers
def pytest_configure(config):
    config.addinivalue_line("markers", "requires_coral: Test requires Coral TPU")
    config.addinivalue_line("markers", "requires_tensorrt: Test requires TensorRT")
    config.addinivalue_line("markers", "requires_camera: Test requires network camera")
```

### Fix 3.2: Skip Tests Conditionally
```python
# In test files:
@pytest.mark.requires_coral
@pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
def test_coral_model_conversion():
    pass

@pytest.mark.requires_tensorrt  
@pytest.mark.skipif(not has_tensorrt(), reason="TensorRT not available")
def test_tensorrt_conversion():
    pass

@pytest.mark.requires_camera
@pytest.mark.skipif(not has_camera_on_network(), reason="No camera on network")
def test_real_camera_discovery():
    pass
```

## Phase 4: Remove Internal Mocking

### Issue: Tests Mocking Internal Modules
This violates the "no internal mocking" requirement.

### Fix 4.1: Find All Internal Mocks
```bash
# Run these commands to find mocks:
grep -r "mock.*wildfire" tests/
grep -r "mock.*camera_detector" tests/
grep -r "mock.*fire_consensus" tests/
grep -r "mock.*gpio_trigger" tests/
grep -r "patch.*detect" tests/
```

### Fix 4.2: Replace with Real Components
For each mock found:
1. Remove the mock/patch
2. Use the real component with test fixtures
3. Use `mqtt_topic_factory` for topic isolation
4. Use `test_mqtt_broker` for real MQTT communication

## Phase 5: Apply MQTT Optimization Migration

### Fix 5.1: Update Old Test Patterns
Replace class-level broker setup with fixtures:

```python
# OLD PATTERN:
class TestSomething:
    @classmethod
    def setUpClass(cls):
        cls.broker = MQTTTestBroker()
        cls.broker.start()

# NEW PATTERN:
class TestSomething:
    def test_something(self, test_mqtt_broker, mqtt_client, mqtt_topic_factory):
        # Use provided fixtures
```

### Fix 5.2: Use Topic Isolation
```python
# OLD:
client.publish("fire/detection", data)

# NEW:
topic = mqtt_topic_factory("fire/detection")
mqtt_client.publish(topic, data)
```

## Phase 6: Test Execution Strategy

### 6.1: Run Tests in Order
1. First fix MQTT broker (Phase 1)
2. Run basic MQTT tests to verify broker works
3. Fix Docker infrastructure (Phase 2)
4. Run Docker tests to verify
5. Fix hardware tests (Phase 3)
6. Remove internal mocking (Phase 4)
7. Apply MQTT optimizations (Phase 5)

### 6.2: Test Commands
```bash
# Test MQTT broker only
python3.12 -m pytest tests/test_mqtt_test_broker.py -v

# Test specific functionality
python3.12 -m pytest tests/test_consensus.py -v -k "not integration"

# Run all tests with proper timeout
./scripts/run_tests_by_python_version.sh --all --timeout 1800

# Run tests for specific Python version
./scripts/run_tests_by_python_version.sh --python312 --timeout 1800
```

## Phase 7: Missing Package Documentation

### Python 3.12 Requirements
- paho-mqtt>=2.0.0
- docker>=6.0.0
- pytest>=8.0.0
- pytest-timeout>=2.0.0
- pytest-xdist>=3.0.0

### Python 3.10 Requirements (YOLO-NAS)
- super-gradients>=3.0.0
- torch>=2.0.0

### Python 3.8 Requirements (Coral TPU)
- tflite-runtime==2.5.0.post1
- pycoral>=2.0.0

## Implementation Priority

1. **Immediate**: Fix MQTT broker startup (Phase 1)
2. **High**: Fix Docker networking (Phase 2)
3. **High**: Add hardware detection/skipping (Phase 3)
4. **Medium**: Remove internal mocking (Phase 4)
5. **Medium**: Apply MQTT optimizations (Phase 5)
6. **Low**: Document missing packages (Phase 7)

## Success Criteria

- All tests pass or skip gracefully when hardware unavailable
- No timeouts on properly configured systems
- No internal module mocking
- Real MQTT broker used for all tests
- Docker tests use real containers
- Hardware tests run on available hardware