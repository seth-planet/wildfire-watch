# Camera Detector Test Fix Plan

## Current Issues:
The test file has extensive mocking of `mqtt.Client` which violates our integration testing philosophy. 

## Mocking Found:
```python
@patch('detect.mqtt.Client')  # Found throughout the file
@patch('detect.WSDiscovery')  # External dependency - OK to mock
@patch('detect.ONVIFCamera')  # External dependency - OK to mock  
@patch('detect.ProcessPoolExecutor')  # Internal Python - debatable
@patch('detect.netifaces.interfaces')  # External dependency - OK to mock
@patch('detect.srp')  # External scapy - OK to mock
@patch('os.geteuid')  # System call - OK to mock
```

## What to Keep Mocked (External Dependencies):
- WSDiscovery (ONVIF discovery)
- ONVIFCamera (ONVIF camera control)
- netifaces (network interface info)
- srp (scapy ARP scanning)
- os.geteuid (system permissions)
- cv2.VideoCapture (OpenCV video capture)

## What to Remove (Internal Mocking):
- mqtt.Client - Use real MQTT broker instead

## Approach:
1. Add fixtures for real MQTT broker (use test_mqtt_broker fixture)
2. Add topic isolation support using mqtt_topic_factory
3. Replace mock mqtt.Client with real MQTT connections
4. Ensure proper cleanup between tests
5. Add monkeypatch for environment variables

## Test Categories:
1. **Config Tests** - No MQTT needed, can stay as-is
2. **Camera Model Tests** - No MQTT needed, can stay as-is  
3. **MAC Tracker Tests** - No MQTT needed, keep external mocks
4. **Discovery Tests** - Need real MQTT broker
5. **TLS Tests** - Need real MQTT broker with TLS
6. **Smart Discovery Tests** - Need real MQTT broker

## Implementation Strategy:
- Create a base fixture that provides CameraDetector with real MQTT
- Mock only external dependencies (cameras, network interfaces)
- Use topic isolation for parallel test execution
- Ensure background tasks are properly managed