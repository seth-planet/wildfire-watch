# Test Suite Requirements

## Overview
This document outlines the requirements and environment setup needed to run the complete wildfire-watch test suite.

## Python Version Requirements

### Python 3.12 (Primary)
- **Used for**: Most tests (597 tests)
- **Install**: `sudo apt install python3.12 python3.12-dev python3.12-venv`
- **Required for**: General functionality, integration tests, GPIO tests

### Python 3.10 (YOLO-NAS)
- **Used for**: YOLO-NAS training and super-gradients tests (43 tests)
- **Install**: `sudo apt install python3.10 python3.10-dev python3.10-venv`
- **Required for**: `super-gradients` compatibility

### Python 3.8 (Coral TPU)
- **Used for**: Coral TPU and TensorFlow Lite tests (34 tests)
- **Install**: `sudo apt install python3.8 python3.8-dev python3.8-venv`
- **Required for**: `tflite-runtime` compatibility

## Hardware Requirements

### Optional Hardware (Tests Skip if Unavailable)
1. **Coral TPU USB Accelerator**
   - Tests marked with `@pytest.mark.coral_tpu`
   - Requires Python 3.8 environment

2. **Hailo-8 AI Accelerator**
   - Tests marked with `@pytest.mark.hailo`
   - Requires Hailo SDK and Python 3.10

3. **NVIDIA GPU with TensorRT**
   - Tests marked with `@pytest.mark.tensorrt`
   - Requires CUDA and TensorRT installation

4. **Raspberry Pi GPIO**
   - Tests marked with `@pytest.mark.rpi_gpio`
   - GPIO simulation used on non-Pi systems

5. **IP Cameras**
   - Tests marked with `@pytest.mark.requires_camera`
   - Set `CAMERA_CREDENTIALS=username:password`

## Software Requirements

### Required Software
1. **Docker & Docker Compose**
   ```bash
   sudo apt install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

2. **MQTT Broker (for TLS tests)**
   - Mosquitto or similar MQTT broker
   - TLS support on port 8883

3. **Build Tools**
   ```bash
   sudo apt install build-essential cmake pkg-config
   sudo apt install libssl-dev libffi-dev
   ```

### Python Dependencies
Install for each Python version:
```bash
# Python 3.12
python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -r requirements-dev.txt

# Python 3.10 (for YOLO-NAS)
python3.10 -m pip install super-gradients torch torchvision

# Python 3.8 (for Coral TPU)
python3.8 -m pip install tflite-runtime
```

## Environment Variables

### Required for Full Test Suite
```bash
# Camera credentials (if testing with real cameras)
export CAMERA_CREDENTIALS="admin:S3thrule"

# Enable TLS tests (optional)
export MQTT_TLS=true

# Set test timeouts
export PYTEST_TIMEOUT=1800
```

### Optional Environment Variables
```bash
# Force GPIO simulation (auto-detected)
export GPIO_SIMULATION=true

# Debug output
export LOG_LEVEL=DEBUG

# Hardware detection
export FRIGATE_DETECTOR=cpu  # or coral, hailo, tensorrt
```

## Running the Test Suite

### Automated Multi-Python Test Runner (Recommended)
```bash
# Run all tests with automatic Python version selection
CAMERA_CREDENTIALS=admin:S3thrule ./scripts/run_tests_by_python_version.sh --all --timeout 1800

# Run specific Python version tests
./scripts/run_tests_by_python_version.sh --python312
./scripts/run_tests_by_python_version.sh --python310
./scripts/run_tests_by_python_version.sh --python38
```

### Manual Test Execution
```bash
# Python 3.12 tests
python3.12 -m pytest -c pytest-python312.ini

# Python 3.10 tests (YOLO-NAS)
python3.10 -m pytest -c pytest-python310.ini -m yolo_nas

# Python 3.8 tests (Coral TPU)
python3.8 -m pytest -c pytest-python38.ini -m coral_tpu

# Run tests serially (for debugging)
pytest -n 1 -v

# Run with specific markers
pytest -m "not slow and not hardware"
```

## Expected Test Results

### Tests That May Skip
- **Hardware tests**: Skip when hardware not available
- **TLS tests**: Skip when MQTT_TLS!=true or broker not configured
- **Camera tests**: Skip when no cameras accessible
- **GPU tests**: Skip when no CUDA/TensorRT available

### Known Slow Tests
Tests marked with `@pytest.mark.slow` or `@pytest.mark.very_slow`:
- Model training tests (>5 minutes)
- End-to-end integration tests
- Hardware initialization tests

### Parallel Execution
- Most tests run in parallel (pytest-xdist)
- GPIO tests may need serial execution: `pytest -n 1 -m gpio`
- Hardware tests use resource locks for exclusive access

## Troubleshooting

### Common Issues

1. **"I/O operation on closed file"**
   - Fixed by safe logging implementation
   - Should not occur with current codebase

2. **"Address already in use"**
   - Tests use dynamic port allocation
   - Check for stale Docker containers: `docker ps -a`

3. **"Module not found"**
   - Ensure correct Python version for test
   - Check virtual environment activation

4. **Timeout Errors**
   - Increase timeout: `--timeout 3600`
   - Run tests serially: `-n 1`

5. **Docker Permission Denied**
   - Add user to docker group: `sudo usermod -aG docker $USER`
   - Logout and login again

### Debug Commands
```bash
# Verbose output with no capture
pytest -xvs --capture=no tests/test_specific.py

# Show test setup/teardown
pytest --setup-show tests/test_specific.py

# List all markers
pytest --markers

# Dry run (collect only)
pytest --collect-only
```

## CI/CD Considerations

### GitHub Actions
- Use matrix builds for Python versions
- Cache pip dependencies
- Use Docker layer caching
- Run hardware tests on self-hosted runners

### Resource Limits
- Set memory limits for Docker containers
- Use timeout limits for all tests
- Clean up resources after test runs

## Summary

The wildfire-watch test suite is comprehensive and supports:
- ✅ Multiple Python versions (3.8, 3.10, 3.12)
- ✅ Optional hardware accelerators
- ✅ Parallel execution with proper isolation
- ✅ Integration with real cameras and services
- ✅ TLS/SSL security testing
- ✅ GPIO hardware testing with simulation fallback

Follow this guide to ensure all tests run successfully in your environment.