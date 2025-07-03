# Wildfire Watch Comprehensive Test Plan

## Test Consolidation Plan

### Tests to Remove (Redundant)
1. **test_camera_discovery.py** - Merge with test_detect.py
2. **test_camera_detector_tls.py** - Merge TLS tests into main camera detector tests
3. **test_model_converter_comprehensive.py** - Merge with test_model_converter.py
4. **test_model_converter_e2e_enhanced.py** - Keep only test_model_converter_e2e.py
5. **test_services_tls_integration.py** - Merge with test_tls_integration.py
6. **test_docker_tls_integration.py** - Merge with test_tls_integration.py

### Consolidated Test Structure

#### 1. **test_camera_detector.py** (Merge of camera tests)
- Camera discovery (ONVIF, mDNS, RTSP scanning)
- MAC address tracking
- IP change handling
- Credential testing
- Frigate config generation
- TLS connection support
- Health monitoring

#### 2. **test_consensus.py** (Enhanced)
- Multi-camera consensus logic
- Growing fire detection
- Cooldown periods
- Single camera handling
- Edge cases (offline cameras)
- TLS support

#### 3. **test_trigger.py** (Enhanced)
- GPIO state machine
- Safety interlocks
- Refill logic
- Maximum runtime protection
- Sensor integration
- Simulation mode

#### 4. **test_model_converter.py** (Consolidated)
- Model conversion to all formats
- Quantization support
- Calibration data handling
- Multi-resolution support
- Deployment script generation

#### 5. **test_integration_e2e.py** (Enhanced)
- Full system workflow
- Multi-camera fire detection
- Consensus to trigger flow
- Service health monitoring
- Error recovery
- TLS communication

#### 6. **test_tls_integration.py** (Consolidated TLS)
- Certificate validation
- MQTT broker TLS
- Service TLS connections
- Docker TLS setup
- Certificate rotation

#### 7. **test_security_nvr_integration.py**
- Frigate configuration
- Camera integration
- Detection pipeline
- API authentication
- Recording management

#### 8. **test_mqtt_broker.py**
- MQTT communication
- Topic structure
- Retained messages
- Will messages
- TLS support

#### 9. **test_telemetry.py**
- Health monitoring
- Metrics collection
- Alert generation

#### 10. **test_hardware_integration.py**
- Physical hardware tests
- Accelerator detection
- GPIO functionality

### Tests to Add (Missing Coverage)

#### 11. **test_scripts.py** (NEW)
- Certificate generation
- Security configuration
- Multi-platform builds
- Startup coordination

#### 12. **test_deployment.py** (NEW)
- Docker Compose deployment
- Service dependencies
- Health checks
- Resource limits
- Network configuration

## README Specification Compliance

### Key Requirements from README:

1. **Multi-camera Consensus** ✓
   - test_consensus.py covers this
   - Need to add: growing fire detection tests

2. **Edge AI Acceleration** ✓
   - test_hardware_integration.py covers this
   - Need to add: auto-detection tests

3. **Automated Pump Control** ✓
   - test_trigger.py covers this
   - Need to add: refill multiplier tests

4. **24/7 Recording** ✓
   - test_security_nvr_integration.py covers this
   - Need to add: retention policy tests

5. **Secure by Default** ✓
   - test_tls_integration.py covers this
   - Need to add: certificate rotation tests

6. **Self-healing** ✓
   - test_camera_detector.py covers this
   - Need to add: network change detection tests

### Inconsistencies Found:

1. **Python Version**
   - Some files use python3, others python3.12
   - **Action**: Standardize to python3.12

2. **Default Credentials**
   - README mentions default Frigate credentials in logs
   - Tests don't verify this behavior
   - **Action**: Add test for credential generation

3. **Smart Discovery Mode**
   - Camera detector README mentions phases
   - Tests don't verify phase transitions
   - **Action**: Add discovery phase tests

4. **Consensus Growing Fire**
   - Fire consensus README mentions growing fire detection
   - Tests don't verify this behavior
   - **Action**: Add growing fire tests

5. **Refill Valve Behavior**
   - GPIO trigger README states refill opens immediately
   - Tests may not verify this specific behavior
   - **Action**: Add refill timing tests

## Host Environment Requirements

Based on test analysis, the following packages are required:
```bash
# Python 3.12
sudo apt install python3.12 python3.12-dev python3.12-venv

# System packages
sudo apt install docker.io docker-compose
sudo apt install mosquitto-clients  # For MQTT testing
sudo apt install openssl  # For certificate testing

# Python packages (in requirements.txt)
pytest>=8.0.0
pytest-asyncio
pytest-timeout
pytest-mock
paho-mqtt>=2.0.0
PyYAML
opencv-python
numpy
requests
cryptography
```

## Test Execution Order

1. Unit tests first (mocked dependencies)
2. Integration tests (real services)
3. Hardware tests (if hardware available)

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run without hardware
pytest tests/ -v --tb=short -m "not hardware"

# Run only critical tests
pytest tests/ -v --tb=short -m "critical"
```