# Security NVR Integration Tests

This directory contains integration tests for the Security NVR (Frigate) service, verifying that it operates correctly according to the documentation in `security_nvr/README.md`.

## Test Files

### 1. `test_security_nvr_integration.py`
Comprehensive integration tests that verify:
- Service health and accessibility
- Hardware detection functionality
- Camera discovery integration
- MQTT event publishing
- Storage configuration
- Model configuration
- API endpoints
- Performance metrics

### 2. `test_security_nvr_documentation.py`
Documentation verification tests that ensure:
- All documented files exist
- Environment variables match documentation
- Configuration files are properly formatted
- Hardware support table is accurate
- Troubleshooting guides are complete

### 3. `verify_security_nvr_deployment.sh`
Bash script for quick deployment verification:
- Docker container status
- Service connectivity
- Hardware detection
- MQTT integration
- Storage configuration
- Performance checks

## Running the Tests

### Prerequisites
```bash
# Ensure services are running
docker-compose up -d

# Install test dependencies
pip3.12 install pytest requests paho-mqtt pyyaml
```

### Run Python Integration Tests
```bash
# Run all integration tests
python3.12 -m pytest tests/test_security_nvr_integration.py -v

# Run without slow integration tests
python3.12 -m pytest tests/test_security_nvr_integration.py -v -k "not integration"

# Run documentation tests
python3.12 -m pytest tests/test_security_nvr_documentation.py -v
```

### Run Deployment Verification
```bash
# Quick verification of deployment
./tests/verify_security_nvr_deployment.sh
```

## Test Coverage

### Service Functionality
- ✅ Frigate service running and accessible
- ✅ API endpoints responding correctly
- ✅ Hardware acceleration detection
- ✅ USB storage management
- ✅ Model configuration for wildfire detection

### Integration Points
- ✅ MQTT broker connectivity
- ✅ Camera discovery integration
- ✅ Event publishing format
- ✅ Multi-node support

### Configuration
- ✅ Environment variables
- ✅ Model sizes (640x640 default, 320x320 fallback)
- ✅ Recording retention settings
- ✅ Power profiles

### Performance
- ✅ CPU usage monitoring
- ✅ Inference speed verification
- ✅ Memory usage tracking

## Expected Test Results

### Successful Deployment
```
=== Security NVR Deployment Verification ===

1. Checking Docker container status
-----------------------------------
Testing: Security NVR container running ... PASSED
Testing: Container healthy ... PASSED

2. Checking service connectivity
--------------------------------
Testing: Frigate API accessible ... PASSED
Testing: Frigate stats endpoint ... PASSED
Testing: Web UI accessible ... PASSED

...

============================================
Tests Passed: 20
Tests Failed: 0

✓ All tests passed! Security NVR is properly deployed.
```

### Common Test Failures

#### No Hardware Acceleration
```python
# Test will show high CPU usage
assert cpu_percent < 80, f"CPU usage too high: {cpu_percent}%"
```
**Solution**: Enable hardware acceleration in docker-compose.yml

#### No Cameras Configured
```
Cameras configured: 0
This is normal if camera_detector hasn't discovered cameras yet
```
**Solution**: Wait for camera_detector to discover cameras or manually configure

#### MQTT Connection Failed
```
Testing: Frigate MQTT availability ... FAILED
```
**Solution**: Ensure mqtt_broker service is running

## Mock Camera Testing

To test object detection without real cameras:

```python
# Publish fake camera discovery
mqtt_client.publish("cameras/discovered", json.dumps({
    "camera_id": "test_cam_001",
    "ip": "192.168.1.100",
    "rtsp_url": "rtsp://username:password@192.168.1.100:554/stream1",
    "manufacturer": "TestCam",
    "model": "TC-1000"
}))
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Security NVR Tests
  run: |
    docker-compose up -d
    sleep 30  # Wait for services
    pytest tests/test_security_nvr_integration.py
    ./tests/verify_security_nvr_deployment.sh
```

## Debugging Failed Tests

### View Logs
```bash
# Frigate logs
docker logs security_nvr -f

# Check specific errors
docker logs security_nvr 2>&1 | grep ERROR

# Hardware detection logs
docker exec security_nvr cat /tmp/hardware_detection.log
```

### Check Configuration
```bash
# View current Frigate config
curl http://localhost:5000/api/config | jq

# Check detector status
curl http://localhost:5000/api/stats | jq .detectors
```

### Test MQTT Messages
```bash
# Subscribe to all Frigate messages
mosquitto_sub -h localhost -t "frigate/#" -v

# Watch for fire detection events
mosquitto_sub -h localhost -t "frigate/+/fire" -v
```

## Contributing

When adding new features to Security NVR:
1. Update the documentation in `security_nvr/README.md`
2. Add corresponding tests in `test_security_nvr_integration.py`
3. Update `test_security_nvr_documentation.py` to verify docs
4. Run all tests to ensure nothing breaks