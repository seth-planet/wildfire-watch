# Remaining Test Failures Analysis

## Overview
After fixing the pytest configuration issues, most tests are now running. However, there are still some specific test failures that need investigation.

## Test Failures by Category

### 1. Camera Discovery Tests
- **test_camera_detector.py::TestCameraDiscovery::test_onvif_discovery_with_mqtt**
  - Likely issue: ONVIF discovery timeout or mock setup

### 2. INT8 Quantization Tests  
- **test_int8_quantization.py::TestINT8Quantization::test_quantization_config_generation**
  - Likely issue: Model conversion configuration or file paths

### 3. Docker Integration Tests
- **test_integration_docker.py::test_docker_integration**
- **test_integration_docker_sdk.py::test_docker_sdk_integration**
  - Likely issues: Container lifecycle, port conflicts, or cleanup

### 4. E2E Integration Tests
- **test_integration_e2e_improved.py::TestE2EPipelineWithRealCamerasImproved::test_complete_pipeline_with_real_cameras[insecure]**
- **test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_multi_camera_consensus**
- **test_integration_e2e_improved.py::TestE2EIntegrationImproved::test_pump_safety_timeout**
  - Likely issues: MQTT timing, camera credentials, or service coordination

### 5. Hardware Docker Tests
- **test_e2e_hardware_docker.py::TestE2EHardwareDocker::test_real_camera_integration**
- **test_e2e_hardware_docker.py::TestE2EHardwareDocker::test_complete_pipeline_auto_hardware**
- **test_e2e_hardware_docker.py::TestE2EHardwareDocker::test_tensorrt_gpu_pipeline**
- **test_e2e_hardware_docker.py::TestE2EHardwareDocker::test_multi_accelerator_failover**
- **test_e2e_hardware_docker.py::TestE2EHardwareDocker::test_performance_comparison**
  - Likely issues: Hardware not available, Docker device mapping, or timing

### 6. QAT Functionality Tests
- **test_qat_functionality.py::QATQuantizationTests::test_qat_calibration_data_usage**
  - Likely issue: Calibration data download or path configuration

### 7. Process Cleanup Tests
- **test_process_leak_fix.py::test_mosquitto_process_cleanup**
- **test_process_leak_fix.py::test_docker_container_cleanup**
  - Likely issues: Process tracking or cleanup timing

## Common Patterns

### 1. Logging Errors
Many tests show "I/O operation on closed file" errors, indicating logging cleanup issues during test teardown.

### 2. MQTT Broker Issues
Multiple stray mosquitto processes being force killed suggests broker lifecycle management problems.

### 3. Hardware Dependencies
Several tests fail due to missing hardware (TensorRT, Hailo, etc.) which is expected without actual hardware.

## Recommended Fixes

### 1. Add Hardware Skip Decorators
```python
@pytest.mark.skipif(not has_tensorrt(), reason="TensorRT not available")
def test_tensorrt_gpu_pipeline():
    pass
```

### 2. Improve Process Cleanup
- Add proper cleanup in fixtures
- Use context managers for resources
- Implement graceful shutdown sequences

### 3. Fix Logging Configuration
- Configure logging per test to avoid closed file errors
- Use separate log handlers for tests

### 4. Mock External Dependencies
- Mock camera discovery for non-hardware tests
- Mock model downloads for CI/CD environments

### 5. Increase Timeouts
- E2E tests may need longer timeouts
- Docker container startup can be slow

## Priority Fixes

1. **High Priority**: Fix test_fixes_validation.py failures (✅ DONE)
2. **Medium Priority**: Fix Docker integration tests
3. **Low Priority**: Fix hardware-dependent tests (can skip when hardware absent)

## Next Steps

1. Run individual failing tests with debug output
2. Check for resource conflicts (ports, files, processes)
3. Verify test environment setup (credentials, paths, permissions)
4. Consider splitting large E2E tests into smaller units