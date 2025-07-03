# Test Failures Analysis

## Summary
All 3 Python test suites failed:
- Python 3.12: FAILED
- Python 3.10: FAILED  
- Python 3.8: FAILED

## Key Issues Identified

### 1. MQTT Broker Connection Failures
Many tests are failing because they cannot connect to the MQTT broker:
- `Connection refused` errors
- `MQTT broker failed to become healthy`
- Tests expecting real MQTT broker but not finding it

### 2. Docker Integration Issues
- Container networking errors: `network wildfire_test_net not found`
- Containers exiting unexpectedly
- Docker SDK connection issues

### 3. Model Converter Tests
Multiple model converter tests failing:
- test_qat_model_conversion
- test_tensorrt_conversion
- test_tflite_conversion
- test_conversion_with_validation
- test_hailo_conversion_with_python310
- test_multi_size_conversion
- test_onnx_conversion

### 4. Integration Tests
E2E tests failing:
- test_complete_pipeline_with_real_cameras (both secure and insecure)
- test_integration_e2e_real_cameras_int8

### 5. Telemetry Tests
- test_real_mqtt_publish_qos_and_retain
- test_lwt_is_set

### 6. Consensus Tests
- test_camera_telemetry_processing
- test_end_to_end_fire_detection_flow

## Root Causes

1. **MQTT Infrastructure**: Tests expect real MQTT broker but session-scoped broker not starting properly
2. **Docker Network**: Tests creating isolated networks that conflict or don't exist
3. **Mocking Violations**: Some tests may still be mocking internal functionality
4. **Resource Cleanup**: Logging errors indicate file handles not being cleaned up properly
5. **Hardware Dependencies**: Model converter tests may need actual hardware available