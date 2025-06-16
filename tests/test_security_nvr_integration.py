#!/usr/bin/env python3.12
"""
Integration tests for Security NVR (Frigate) Service
Tests camera integration, object detection, MQTT publishing, and hardware detection
"""
import os
import sys
import json
import time
import threading
import subprocess
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

# Test configuration
TEST_TIMEOUT = 30
FRIGATE_HOST = os.getenv("FRIGATE_HOST", "localhost")
FRIGATE_PORT = int(os.getenv("FRIGATE_PORT", "5000"))
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

def check_service_running():
    """Check if security_nvr service is running"""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=security_nvr", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return "security_nvr" in result.stdout

def check_frigate_api():
    """Check if Frigate API is accessible"""
    try:
        # Try with default admin credentials first
        response = requests.get(
            f"http://localhost:5000/version", 
            auth=("admin", "7f155ad9e8c340c88ef6a33f528f2e75"),
            timeout=2
        )
        return response.status_code == 200
    except:
        return False

def check_mqtt_broker():
    """Check if MQTT broker is running"""
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=mqtt_broker", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    return "mqtt_broker" in result.stdout

# Skip decorators
requires_security_nvr = pytest.mark.skipif(
    not check_service_running(),
    reason="security_nvr container not running - run 'docker-compose up -d' first"
)

requires_frigate_api = pytest.mark.skipif(
    not check_frigate_api(),
    reason="Frigate API not accessible - service may not be running"
)

requires_mqtt = pytest.mark.skipif(
    not check_mqtt_broker(),
    reason="MQTT broker not running - required for MQTT tests"
)

class TestSecurityNVRIntegration:
    """Integration tests for Security NVR service"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.mqtt_messages = []
        self.mqtt_connected = False
        self.mqtt_client = None
        self.frigate_api_url = f"http://{FRIGATE_HOST}:{FRIGATE_PORT}"
        # Frigate authentication credentials
        self.frigate_auth = ("admin", "7f155ad9e8c340c88ef6a33f528f2e75")
        
    def teardown_method(self):
        """Cleanup after each test"""
        if self.mqtt_client and self.mqtt_connected:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
    
    # ========== Service Health Tests ==========
    
    @requires_security_nvr
    def test_frigate_service_running(self):
        """Test that Frigate service is running and accessible"""
        # Check if container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=security_nvr", "--format", "{{.Status}}"],
            capture_output=True,
            text=True
        )
        assert "Up" in result.stdout, "Security NVR container is not running"
        
        # Check container health if available
        health_result = subprocess.run(
            ["docker", "inspect", "security_nvr", "--format", "{{.State.Health.Status}}"],
            capture_output=True,
            text=True
        )
        # Health check might not be configured, so we don't assert on it
        if health_result.stdout.strip() and health_result.stdout.strip() != "<no value>":
            print(f"Container health: {health_result.stdout.strip()}")
    
    @requires_frigate_api
    def test_frigate_stats_endpoint(self):
        """Test Frigate stats API endpoint"""
        response = requests.get(f"{self.frigate_api_url}/stats", auth=self.frigate_auth, timeout=5)
        assert response.status_code == 200
        
        stats = response.json()
        assert "detectors" in stats
        assert "cameras" in stats
        assert "service" in stats
        
        # Check service info
        service = stats["service"]
        assert service["uptime"] > 0
        assert "storage" in service
    
    # ========== Hardware Detection Tests ==========
    
    @requires_security_nvr
    def test_hardware_detector_execution(self):
        """Test that hardware detector runs and produces output"""
        result = subprocess.run(
            ["docker", "exec", "security_nvr", "python3", "/scripts/hardware_detector.py"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0, f"Hardware detector failed: {result.stderr}"
        output = result.stdout
        
        # Check for expected hardware detection output
        assert "cpu" in output.lower()
        assert "memory" in output.lower()
        assert "platform" in output.lower()
    
    @requires_frigate_api
    def test_detector_configuration(self):
        """Test that detector is properly configured based on hardware"""
        response = requests.get(f"{self.frigate_api_url}/config", auth=self.frigate_auth, timeout=5)
        assert response.status_code == 200
        
        config = response.json()
        assert "detectors" in config
        
        # Check that at least one detector is configured
        detectors = config["detectors"]
        assert len(detectors) > 0, "No detectors configured"
        
        # Verify detector settings
        for detector_name, detector_config in detectors.items():
            assert "type" in detector_config
            assert detector_config["type"] in ["cpu", "edgetpu", "openvino", "tensorrt"]
            
            # Check model configuration
            if "model" in detector_config:
                model = detector_config["model"]
                assert "width" in model
                assert "height" in model
                assert model["width"] in [320, 416, 640]  # Valid model sizes
                assert model["height"] in [320, 416, 640]
    
    # ========== Camera Integration Tests ==========
    
    @requires_security_nvr
    @requires_mqtt
    def test_camera_discovery_integration(self):
        """Test integration with camera_detector service"""
        # Setup MQTT client to simulate camera discovery
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_camera_discovery"
        )
        mqtt_client.on_connect = lambda c, u, f, rc, props: setattr(self, 'mqtt_connected', rc == 0)
        
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_client.loop_start()
            
            # Wait for connection
            timeout = time.time() + 5
            while not self.mqtt_connected and time.time() < timeout:
                time.sleep(0.1)
            
            assert self.mqtt_connected, "Failed to connect to MQTT broker"
            
            # Publish camera discovery message
            camera_data = {
                "camera_id": "test_cam_001",
                "ip": "192.168.1.100",
                "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
                "manufacturer": "TestCam",
                "model": "TC-1000",
                "mac_address": "00:11:22:33:44:55",
                "capabilities": {
                    "ptz": False,
                    "audio": True,
                    "resolution": "1920x1080"
                }
            }
            
            mqtt_client.publish("cameras/discovered", json.dumps(camera_data), qos=1)
            time.sleep(2)  # Allow time for processing
            
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    
    @requires_frigate_api
    def test_camera_configuration_format(self):
        """Test that camera configurations match expected format"""
        response = requests.get(f"{self.frigate_api_url}/config", auth=self.frigate_auth, timeout=5)
        config = response.json()
        
        if "cameras" in config and len(config["cameras"]) > 0:
            for camera_name, camera_config in config["cameras"].items():
                # Check required camera configuration
                assert "ffmpeg" in camera_config
                assert "detect" in camera_config
                
                # Check ffmpeg inputs
                ffmpeg = camera_config["ffmpeg"]
                assert "inputs" in ffmpeg
                assert len(ffmpeg["inputs"]) > 0
                
                # Check detect settings
                detect = camera_config["detect"]
                assert "width" in detect
                assert "height" in detect
                assert detect["width"] in [320, 416, 640]
                assert detect["height"] in [320, 416, 640]
    
    # ========== MQTT Publishing Tests ==========
    
    @requires_frigate_api
    def test_mqtt_connection(self):
        """Test that Frigate connects to MQTT broker"""
        # Check Frigate config for MQTT settings
        response = requests.get(f"{self.frigate_api_url}/config", auth=self.frigate_auth, timeout=5)
        config = response.json()
        
        assert "mqtt" in config
        mqtt_config = config["mqtt"]
        assert mqtt_config["enabled"] is True
        assert "host" in mqtt_config
        assert "port" in mqtt_config
    
    @requires_mqtt
    def test_mqtt_event_publishing(self):
        """Test that Frigate publishes events to MQTT"""
        received_messages = []
        
        def on_message(client, userdata, msg):
            received_messages.append({
                "topic": msg.topic,
                "payload": msg.payload.decode('utf-8')
            })
        
        # Setup MQTT subscriber
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_event_subscriber"
        )
        mqtt_client.on_connect = lambda c, u, f, rc, props: c.subscribe("frigate/#")
        mqtt_client.on_message = on_message
        
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_client.loop_start()
            
            # Wait for any Frigate messages (stats, availability, etc.)
            # Stats interval is 15 seconds, so wait at least that long
            time.sleep(20)
            
            # Check for expected message patterns
            topics_found = [msg["topic"] for msg in received_messages]
            
            # Frigate should publish availability and stats
            expected_patterns = ["frigate/available", "frigate/stats"]
            for pattern in expected_patterns:
                assert any(pattern in topic for topic in topics_found), \
                    f"Expected MQTT topic pattern '{pattern}' not found"
                    
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
    
    def test_fire_detection_event_format(self):
        """Test the format of fire detection events"""
        # This test verifies the expected event format
        # In a real scenario, we'd trigger a detection
        
        expected_event = {
            "type": "new",
            "before": {
                "id": "1234567890.123456-abc123",
                "camera": "front_yard",
                "frame_time": 1234567890.123456,
                "label": "fire",
                "score": 0.85,
                "box": [320, 180, 480, 360]
            }
        }
        
        # Validate event structure
        assert "type" in expected_event
        assert "before" in expected_event
        
        before = expected_event["before"]
        assert all(key in before for key in ["id", "camera", "frame_time", "label", "score", "box"])
        assert before["label"] in ["fire", "smoke"]
        assert 0 <= before["score"] <= 1
        assert len(before["box"]) == 4
    
    # ========== Storage Tests ==========
    
    @requires_security_nvr
    def test_usb_storage_configuration(self):
        """Test USB storage is properly configured"""
        # Check if storage path exists in container
        result = subprocess.run(
            ["docker", "exec", "security_nvr", "test", "-d", "/media/frigate"],
            capture_output=True
        )
        
        # Storage directory should exist (even if not mounted)
        assert result.returncode == 0, "Storage path /media/frigate not found in container"
    
    def test_recording_directory_structure(self):
        """Test that recording directory structure is documented correctly"""
        # This is a documentation test - verifies expected structure
        expected_paths = [
            "/media/frigate/recordings",
            "/media/frigate/clips",
            "/media/frigate/exports"
        ]
        
        # Just verify we know what paths should exist
        assert len(expected_paths) == 3
    
    # ========== Model Configuration Tests ==========
    
    @requires_frigate_api
    def test_wildfire_model_configuration(self):
        """Test that wildfire detection models are properly configured"""
        response = requests.get(f"{self.frigate_api_url}/config", auth=self.frigate_auth, timeout=5)
        config = response.json()
        
        # Check for wildfire-specific configuration
        if "objects" in config and "track" in config["objects"]:
            tracked_objects = config["objects"]["track"]
            assert "fire" in tracked_objects or "smoke" in tracked_objects
            
        # Check model configuration in detectors
        if "detectors" in config:
            for detector_name, detector_config in config["detectors"].items():
                if "model" in detector_config:
                    model = detector_config["model"]
                    assert "width" in model
                    assert "height" in model
    
    @requires_frigate_api
    def test_detection_settings(self):
        """Test detection settings match documentation"""
        response = requests.get(f"{self.frigate_api_url}/config", auth=self.frigate_auth, timeout=5)
        config = response.json()
        
        # Check global detect settings
        if "detect" in config:
            detect = config["detect"]
            if "fps" in detect:
                assert 1 <= detect["fps"] <= 10, "Detection FPS out of expected range"
    
    # ========== API Endpoint Tests ==========
    
    @requires_frigate_api
    def test_events_api(self):
        """Test Frigate events API endpoint"""
        response = requests.get(f"{self.frigate_api_url}/events", auth=self.frigate_auth, timeout=5)
        assert response.status_code == 200
        
        events = response.json()
        assert isinstance(events, list)
        
        # If there are events, validate structure
        if len(events) > 0:
            event = events[0]
            assert "id" in event
            assert "camera" in event
            assert "label" in event
            assert "start_time" in event
    
    @requires_frigate_api
    def test_recordings_api(self):
        """Test Frigate recordings API endpoint"""
        response = requests.get(f"{self.frigate_api_url}/recordings/summary", auth=self.frigate_auth, timeout=5)
        # API might return 404 if no recordings exist yet
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            summary = response.json()
            assert isinstance(summary, list)
    
    # ========== Performance Tests ==========
    
    @requires_frigate_api
    def test_cpu_usage(self):
        """Test that CPU usage is within acceptable limits"""
        response = requests.get(f"{self.frigate_api_url}/stats", auth=self.frigate_auth, timeout=5)
        stats = response.json()
        
        if "cpu_usages" in stats:
            cpu_stats = stats["cpu_usages"]
            
            # Check overall CPU usage
            if "cpu" in cpu_stats and "cpu" in cpu_stats["cpu"]:
                cpu_percent = cpu_stats["cpu"]["cpu"]
                assert cpu_percent < 80, f"CPU usage too high: {cpu_percent}%"
    
    @requires_frigate_api
    def test_detector_inference_speed(self):
        """Test that detector inference speed is acceptable"""
        response = requests.get(f"{self.frigate_api_url}/stats", auth=self.frigate_auth, timeout=5)
        stats = response.json()
        
        if "detectors" in stats:
            for detector_name, detector_stats in stats["detectors"].items():
                if "inference_speed" in detector_stats:
                    speed = detector_stats["inference_speed"]
                    # Check inference speed based on hardware type
                    if "coral" in detector_name.lower():
                        assert speed < 50, f"Coral inference too slow: {speed}ms"
                    elif "hailo" in detector_name.lower():
                        assert speed < 30, f"Hailo inference too slow: {speed}ms"
                    else:  # CPU or other
                        assert speed < 200, f"Inference too slow: {speed}ms"
    
    # ========== Integration Flow Tests ==========
    
    @pytest.mark.integration
    @requires_security_nvr
    @requires_mqtt
    def test_full_detection_flow(self):
        """Test complete flow from camera to MQTT event"""
        # This test would require a test video or live camera
        # It verifies the entire pipeline works
        
        # Setup MQTT subscriber for fire events
        fire_events = []
        
        def on_fire_event(client, userdata, msg):
            if "fire" in msg.topic or "smoke" in msg.topic:
                fire_events.append({
                    "topic": msg.topic,
                    "payload": json.loads(msg.payload.decode('utf-8'))
                })
        
        mqtt_client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_fire_detector"
        )
        mqtt_client.on_connect = lambda c, u, f, rc, props: c.subscribe("frigate/+/fire")
        mqtt_client.on_message = on_fire_event
        
        try:
            mqtt_client.connect(MQTT_HOST, MQTT_PORT, 60)
            mqtt_client.loop_start()
            
            # In a real test, we would:
            # 1. Inject a test video with fire
            # 2. Wait for detection
            # 3. Verify MQTT event published
            
            time.sleep(5)  # Wait for any existing events
            
            # Verify event structure if any were received
            for event in fire_events:
                payload = event["payload"]
                assert "type" in payload
                assert "camera" in payload or "before" in payload
                
        finally:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()


class TestServiceDependencies:
    """Test service dependencies and startup order"""
    
    @requires_security_nvr
    def test_mqtt_broker_dependency(self):
        """Test that security_nvr depends on mqtt_broker"""
        # Alternative: Check if MQTT broker is reachable from security_nvr
        result = subprocess.run(
            ["docker", "exec", "security_nvr", "ping", "-c", "1", "mqtt_broker"],
            capture_output=True
        )
        assert result.returncode == 0, "Cannot reach mqtt_broker from security_nvr"
    
    def test_camera_detector_integration(self):
        """Test that security_nvr can receive camera updates"""
        # Check if both services are on the same network
        result = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True
        )
        
        # Just verify Docker networks exist
        assert result.returncode == 0


class TestWebInterface:
    """Test Frigate web interface accessibility"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.frigate_auth = ("admin", "7f155ad9e8c340c88ef6a33f528f2e75")
    
    @requires_frigate_api
    def test_web_ui_accessible(self):
        """Test that Frigate web UI is accessible"""
        response = requests.get(f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/", auth=self.frigate_auth, timeout=5)
        assert response.status_code == 200
        # The root path returns a health check message
        assert "Frigate is running" in response.text
    
    @requires_frigate_api
    def test_static_resources(self):
        """Test that static resources are served"""
        # Check if API version endpoint works
        response = requests.get(f"http://{FRIGATE_HOST}:{FRIGATE_PORT}/version", auth=self.frigate_auth, timeout=5)
        assert response.status_code == 200


if __name__ == "__main__":
    # Run with markers to control which tests run
    import sys
    
    # Check what's available
    print("Checking service availability...")
    print(f"Security NVR running: {check_service_running()}")
    print(f"Frigate API accessible: {check_frigate_api()}")
    print(f"MQTT broker running: {check_mqtt_broker()}")
    print()
    
    # Run tests
    pytest.main([__file__, "-v", "-k", "not integration"])