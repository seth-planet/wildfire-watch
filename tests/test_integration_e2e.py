#!/usr/bin/env python3.12
#!/usr/bin/env python3
"""
End-to-end integration tests for Wildfire Watch
"""
import os
import sys
import time
import json
import pytest
import docker
import paho.mqtt.client as mqtt
from typing import Dict, List

class TestE2EIntegration:
    """Test complete system integration"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Get Docker client"""
        return docker.from_env()
    
    @pytest.fixture(scope="class")
    def mqtt_client(self):
        """Create MQTT test client"""
        client = mqtt.Client("test_client")
        client.connect("localhost", 1883, 60)
        client.loop_start()
        yield client
        client.loop_stop()
        client.disconnect()
    
    def test_service_startup_order(self, docker_client):
        """Test services start in correct order"""
        # Check mqtt_broker starts first
        broker = docker_client.containers.get("mqtt-broker")
        assert broker.status == "running"
        
        # Check dependent services
        services = ["camera-detector", "fire-consensus", "gpio-trigger"]
        for service in services:
            container = docker_client.containers.get(service)
            assert container.status == "running"
    
    def test_camera_discovery_to_frigate(self, mqtt_client):
        """Test camera discovery flow to Frigate config"""
        received_events = []
        
        def on_message(client, userdata, msg):
            received_events.append((msg.topic, json.loads(msg.payload)))
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe("camera/discovery/+")
        mqtt_client.subscribe("frigate/config/cameras")
        
        # Simulate camera discovery
        camera_data = {
            "camera": {
                "id": "test123",
                "ip": "192.168.1.100",
                "mac": "AA:BB:CC:DD:EE:FF",
                "online": True,
                "primary_rtsp_url": "rtsp://192.168.1.100/stream"
            }
        }
        
        mqtt_client.publish("camera/discovery/test123", json.dumps(camera_data))
        time.sleep(2)
        
        # Verify events received
        assert len(received_events) > 0
        assert any("frigate/config/cameras" in evt[0] for evt in received_events)
    
    def test_fire_detection_to_pump_activation(self, mqtt_client):
        """Test complete fire detection to pump activation flow"""
        pump_activated = False
        
        def on_message(client, userdata, msg):
            nonlocal pump_activated
            if msg.topic == "fire/trigger":
                pump_activated = True
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe("fire/trigger")
        
        # Simulate multiple camera detections
        for i in range(3):
            detection = {
                "camera_id": f"camera_{i}",
                "confidence": 0.85,
                "bounding_box": [0.1, 0.1, 0.05, 0.05],
                "timestamp": time.time()
            }
            mqtt_client.publish(f"fire/detection/camera_{i}", json.dumps(detection))
            time.sleep(0.5)
        
        # Wait for consensus
        time.sleep(5)
        
        assert pump_activated, "Pump should have been activated"
    
    def test_health_monitoring(self, mqtt_client):
        """Test all services report health"""
        health_reports = {}
        
        def on_message(client, userdata, msg):
            if "health" in msg.topic or "telemetry" in msg.topic:
                service = msg.topic.split("/")[1]
                health_reports[service] = json.loads(msg.payload)
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe("system/+/health")
        mqtt_client.subscribe("system/+_telemetry")
        
        # Wait for health reports
        time.sleep(65)  # Slightly longer than telemetry interval
        
        expected_services = ["camera_detector", "consensus", "trigger"]
        for service in expected_services:
            assert any(service in key for key in health_reports.keys()), \
                f"No health report from {service}"
    
    def test_error_recovery(self, docker_client, mqtt_client):
        """Test system recovers from service failures"""
        # Stop a service
        docker_client.containers.get("camera-detector").stop()
        time.sleep(5)
        
        # Service should restart
        detector = docker_client.containers.get("camera-detector")
        assert detector.status == "running"
        
        # Check it reconnects to MQTT
        connected = False
        
        def on_message(client, userdata, msg):
            nonlocal connected
            if "camera_detector" in msg.topic and "online" in str(msg.payload):
                connected = True
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe("system/camera_detector_health")
        
        time.sleep(10)
        assert connected, "Camera detector did not reconnect"
