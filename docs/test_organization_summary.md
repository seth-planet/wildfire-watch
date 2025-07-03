# Test Organization Summary

## Overview
The wildfire-watch test suite is organized by Python version requirements and test categories.

## Python Version Requirements

### Python 3.12 Tests (Default)
Most tests run on Python 3.12, including:
- Core functionality tests
- Unit tests
- Integration tests (non-Docker)
- MQTT communication tests
- Configuration tests
- Thread safety tests

### Python 3.10 Tests
Tests requiring super-gradients library:
- `test_yolo_nas_training.py`
- `test_yolo_nas_training_updated.py`
- `test_api_usage.py`
- `test_qat_functionality.py`

### Python 3.8 Tests
Tests requiring Coral TPU or TensorFlow Lite:
- `test_model_converter.py`
- `test_model_converter_e2e.py`
- `test_model_converter_hardware.py`
- `test_hardware_integration.py`
- `test_deployment.py`
- `test_int8_quantization.py`

## Test Categories

### Core Tests
- `test_consensus.py` - Multi-camera consensus logic
- `test_detect.py` - Camera detection functionality
- `test_trigger.py` - GPIO pump control
- `test_telemetry.py` - System health monitoring

### Integration Tests
- `test_integration_e2e.py` - End-to-end workflow
- `test_simplified_integration.py` - Simplified integration scenarios
- `test_frigate_integration.py` - Frigate NVR integration
- `test_security_nvr_integration.py` - Security NVR setup

### Configuration & Infrastructure
- `test_configuration_system.py` - Configuration validation
- `test_mqtt_broker.py` - MQTT broker configuration
- `test_thread_safety.py` - Thread safety verification
- `test_timeout_configuration.py` - Timeout handling

### Hardware & Model Tests
- `test_hardware_integration.py` - Hardware accelerator tests
- `test_model_converter*.py` - Model conversion tests
- `test_int8_quantization.py` - INT8 quantization tests

### Docker Tests
- `test_integration_docker.py` - Docker container tests
- `test_integration_docker_sdk.py` - Docker SDK tests
- `test_deployment.py` - Deployment scenarios

## Running Tests

### Automatic Python Version Selection
```bash
# Run all tests with appropriate Python versions
./scripts/run_tests_by_python_version.sh --all

# Run specific Python version tests
./scripts/run_tests_by_python_version.sh --python312
./scripts/run_tests_by_python_version.sh --python310
./scripts/run_tests_by_python_version.sh --python38

# Run specific test file
./scripts/run_tests_by_python_version.sh --test tests/test_consensus.py
```

### Manual Test Execution
```bash
# Python 3.12 tests (most tests)
python3.12 -m pytest -c pytest-python312.ini

# Python 3.10 tests (YOLO-NAS)
python3.10 -m pytest -c pytest-python310.ini

# Python 3.8 tests (Coral TPU)
python3.8 -m pytest -c pytest-python38.ini
```

## Test Fixtures

### Session-Scoped Fixtures
- `session_mqtt_broker` - Real MQTT broker for entire session
- `session_performance_monitor` - Performance tracking
- `long_timeout_environment` - Timeout-friendly environment

### Test-Scoped Fixtures
- `test_mqtt_broker` - Per-test MQTT broker access
- `cleanup_telemetry` - Automatic telemetry cleanup
- `thread_monitor` - Thread leak detection
- `state_manager` - Service state management

## Logging Configuration

Tests use safe logging to prevent I/O errors during teardown:
- `safe_log()` function in conftest.py catches logging errors
- Service cleanup methods handle logging errors gracefully
- NullHandler fallback prevents uncaught logging exceptions

## Timeout Configuration

All pytest configs have appropriate timeouts:
- **pytest-python312.ini**: 1 hour per test, 2 hour session
- **pytest-python310.ini**: 2 hours per test, 4 hour session
- **pytest-python38.ini**: 2 hours per test, 4 hour session

Extended timeouts accommodate:
- MQTT broker startup
- Model conversion operations
- Docker container builds
- Hardware initialization