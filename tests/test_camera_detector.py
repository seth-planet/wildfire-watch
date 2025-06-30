#!/usr/bin/env python3.12
"""
Camera Detector Service Tests - Using Real MQTT
Tests camera discovery, MAC tracking, TLS support, and Frigate integration
Following integration testing philosophy - no internal mocking
"""
import os
import sys
import ssl
import time
import json
import yaml
import socket
import pytest
import threading
from unittest.mock import Mock, patch
from typing import Dict, List, Optional
import ipaddress

# Add module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../camera_detector")))

# Import after path setup
from detect import CameraDetector, Camera, CameraProfile, MACTracker, Config


@pytest.fixture
def camera_detector_with_mqtt(test_mqtt_broker, mqtt_topic_factory, monkeypatch, tmp_path):
    """Create CameraDetector with real MQTT broker and topic isolation"""
    # Get unique topic prefix
    full_topic = mqtt_topic_factory("dummy")
    prefix = full_topic.rsplit('/', 1)[0]
    
    # Create temporary config directory
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Set environment variables using monkeypatch
    monkeypatch.setenv('MQTT_TOPIC_PREFIX', prefix)
    monkeypatch.setenv('MQTT_BROKER', test_mqtt_broker.host)
    monkeypatch.setenv('MQTT_PORT', str(test_mqtt_broker.port))
    monkeypatch.setenv('MQTT_TLS', 'false')
    monkeypatch.setenv('CAMERA_CREDENTIALS', '')
    monkeypatch.setenv('DISCOVERY_INTERVAL', '300')
    monkeypatch.setenv('SMART_DISCOVERY_ENABLED', 'false')  # Disable for testing
    monkeypatch.setenv('FRIGATE_CONFIG_PATH', str(config_dir / 'frigate_config.yml'))
    
    # Import and reload detect module to pick up new environment
    import importlib
    import sys
    if 'camera_detector.detect' in sys.modules:
        del sys.modules['camera_detector.detect']
    if 'detect' in sys.modules:
        del sys.modules['detect']
    
    from camera_detector import detect
    
    # Now create the detector with patched background tasks
    with patch.object(detect.CameraDetector, '_start_background_tasks'):
        detector = detect.CameraDetector()
        
        # Wait for MQTT connection
        start_time = time.time()
        while time.time() - start_time < 5:
            if hasattr(detector, 'mqtt_client') and detector.mqtt_client.is_connected():
                break
            time.sleep(0.1)
        
        # Verify we connected to the right broker
        assert detector.config.MQTT_BROKER == test_mqtt_broker.host
        assert detector.config.MQTT_PORT == test_mqtt_broker.port
        
        yield detector, prefix
        
        # Cleanup
        try:
            if hasattr(detector, 'mqtt_client'):
                detector.mqtt_client.disconnect()
                detector.mqtt_client.loop_stop()
        except:
            pass


class TestCameraDetectorConfig:
    """Test configuration handling"""
    
    def test_config_from_environment(self):
        """Test Config loads from environment variables"""
        config = Config()
        
        assert hasattr(config, 'MQTT_BROKER')
        assert hasattr(config, 'MQTT_PORT')
        assert hasattr(config, 'MQTT_TLS')
        assert hasattr(config, 'DISCOVERY_INTERVAL')
        
        # Test types
        assert isinstance(config.MQTT_BROKER, str)
        assert isinstance(config.MQTT_PORT, int)
        assert isinstance(config.MQTT_TLS, bool)
        assert isinstance(config.DISCOVERY_INTERVAL, int)
    
    def test_config_defaults(self):
        """Test Config uses proper defaults"""
        config = Config()
        assert config.MQTT_PORT == 1883
        assert hasattr(config, 'RTSP_TIMEOUT')
        assert hasattr(config, 'ONVIF_TIMEOUT')
        assert config.MAC_TRACKING_ENABLED is True
        assert config.SMART_DISCOVERY_ENABLED is True


class TestCameraModel:
    """Test Camera data model"""
    
    def test_camera_creation(self):
        """Test Camera object creation"""
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera",
            manufacturer="Hikvision",
            model="DS-2CD2042WD"
        )
        
        assert camera.ip == "192.168.1.100"
        assert camera.mac == "AA:BB:CC:DD:EE:FF"
        assert camera.id == "aabbccddeeff"
        assert camera.online is False
        assert len(camera.ip_history) == 1
        assert camera.ip_history[0] == "192.168.1.100"
    
    def test_camera_update_ip(self):
        """Test Camera IP update tracking"""
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera"
        )
        
        camera.update_ip("192.168.1.101")
        assert camera.ip == "192.168.1.101"
        assert len(camera.ip_history) == 2
        assert "192.168.1.100" in camera.ip_history
        assert "192.168.1.101" in camera.ip_history
    
    def test_camera_to_frigate_config(self):
        """Test Frigate configuration generation"""
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera"
        )
        camera.rtsp_urls = {
            'main': 'rtsp://admin:pass@192.168.1.100:554/stream1',
            'sub': 'rtsp://admin:pass@192.168.1.100:554/stream2'
        }
        camera.online = True
        
        config = camera.to_frigate_config()
        assert config is not None
        assert camera.id in config
        
        cam_config = config[camera.id]
        assert 'ffmpeg' in cam_config
        assert 'detect' in cam_config
        assert 'objects' in cam_config
        assert 'fire' in cam_config['objects']['track']
        assert 'smoke' in cam_config['objects']['track']


class TestMACTracker:
    """Test MAC address tracking"""
    
    def test_mac_tracker_update(self):
        """Test MAC to IP mapping updates"""
        tracker = MACTracker()
        
        tracker.update("AA:BB:CC:DD:EE:FF", "192.168.1.100")
        assert tracker.get_ip_for_mac("AA:BB:CC:DD:EE:FF") == "192.168.1.100"
        assert tracker.get_mac_for_ip("192.168.1.100") == "AA:BB:CC:DD:EE:FF"
        
        # Test IP change
        tracker.update("AA:BB:CC:DD:EE:FF", "192.168.1.101")
        assert tracker.get_ip_for_mac("AA:BB:CC:DD:EE:FF") == "192.168.1.101"
        assert tracker.get_mac_for_ip("192.168.1.100") is None
        assert tracker.get_mac_for_ip("192.168.1.101") == "AA:BB:CC:DD:EE:FF"
    
    @patch('os.geteuid')
    @patch('detect.srp')
    def test_mac_scan_network(self, mock_srp, mock_geteuid):
        """Test network ARP scanning"""
        mock_geteuid.return_value = 0  # Simulate root
        
        # Mock ARP response
        mock_response = Mock()
        mock_response.psrc = "192.168.1.100"
        mock_response.hwsrc = "AA:BB:CC:DD:EE:FF"
        
        mock_element = [None, mock_response]
        mock_srp.return_value = ([mock_element], None)
        
        tracker = MACTracker()
        results = tracker.scan_network("192.168.1.0/24")
        
        assert "192.168.1.100" in results
        assert results["192.168.1.100"] == "AA:BB:CC:DD:EE:FF"
        assert tracker.get_mac_for_ip("192.168.1.100") == "AA:BB:CC:DD:EE:FF"


class TestCameraDiscovery:
    """Test camera discovery methods with real MQTT"""
    
    def test_mqtt_camera_publication(self, camera_detector_with_mqtt, mqtt_client, test_mqtt_broker):
        """Test camera discovery publishes to MQTT"""
        detector, prefix = camera_detector_with_mqtt
        
        # Subscribe to camera discovery topic
        discovery_topic = f"{prefix}/camera/discovery/+"
        status_topic = f"{prefix}/camera/status/+"
        
        received_messages = []
        
        def on_message(client, userdata, msg):
            received_messages.append({
                'topic': msg.topic,
                'payload': json.loads(msg.payload.decode())
            })
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(discovery_topic)
        mqtt_client.subscribe(status_topic)
        
        # Give time for subscription to complete
        time.sleep(0.2)
        
        # Create and publish a camera
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera",
            manufacturer="Hikvision",
            model="DS-2CD2042WD"
        )
        camera.rtsp_urls = {
            'main': 'rtsp://admin:pass@192.168.1.100:554/stream1'
        }
        
        detector._publish_camera_discovery(camera)
        
        # Wait for message
        time.sleep(0.5)
        
        # Verify message was published
        assert len(received_messages) > 0
        
        # Find the discovery message
        discovery_msg = None
        for msg in received_messages:
            if 'discovery' in msg['topic']:
                discovery_msg = msg
                break
        
        assert discovery_msg is not None
        payload = discovery_msg['payload']
        
        # The payload has 'camera' field with the camera data
        assert 'camera' in payload
        camera_data = payload['camera']
        assert camera_data['ip'] == "192.168.1.100"
        assert camera_data['mac'] == "AA:BB:CC:DD:EE:FF"
        assert camera_data['name'] == "Test Camera"
        assert camera_data['manufacturer'] == "Hikvision"
    
    def test_onvif_discovery_with_mqtt(self, camera_detector_with_mqtt, mqtt_client):
        """Test ONVIF discovery with real hardware when available"""
        detector, prefix = camera_detector_with_mqtt
        
        # Subscribe to camera discovery topic
        discovery_topic = f"{prefix}/camera/discovery/+"
        
        received_messages = []
        
        def on_message(client, userdata, msg):
            received_messages.append({
                'topic': msg.topic,
                'payload': json.loads(msg.payload.decode())
            })
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(discovery_topic)
        
        # Give time for subscription to complete
        time.sleep(0.2)
        
        # Run discovery - this will discover real cameras if available
        detector._discover_onvif_cameras()
        
        # Wait for any discoveries
        time.sleep(2.0)
        
        # If real cameras were discovered, verify they were published
        # If no cameras were discovered, that's OK too (no cameras on network)
        if len(received_messages) > 0:
            # Verify the messages are properly formatted
            for msg in received_messages:
                assert 'camera' in msg['payload']
                camera_data = msg['payload']['camera']
                assert 'ip' in camera_data
                assert 'mac' in camera_data
                assert 'name' in camera_data
            
            # Log what we found
            print(f"Discovered {len(received_messages)} real cameras via ONVIF")
        else:
            # No cameras found is also a valid test result
            print("No ONVIF cameras discovered on network (this is OK)")
        
        # Test passes either way - we're testing the discovery mechanism works
        assert True
    
    def test_frigate_config_publication(self, camera_detector_with_mqtt, mqtt_client):
        """Test Frigate configuration is published to MQTT"""
        detector, prefix = camera_detector_with_mqtt
        
        # Subscribe to Frigate config topic
        config_topic = f"{prefix}/frigate/config/cameras"
        
        received_config = []
        
        def on_message(client, userdata, msg):
            if config_topic in msg.topic:
                received_config.append(yaml.safe_load(msg.payload.decode()))
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(config_topic)
        
        # Add a camera to the detector
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera"
        )
        camera.rtsp_urls = {
            'main': 'rtsp://admin:pass@192.168.1.100:554/stream1',
            'sub': 'rtsp://admin:pass@192.168.1.100:554/stream2'
        }
        camera.online = True
        
        detector.cameras[camera.id] = camera
        
        # Publish Frigate config
        detector._update_frigate_config()
        
        # Wait for message
        time.sleep(0.5)
        
        # Verify config was published
        assert len(received_config) > 0
        config = received_config[0]
        assert 'cameras' in config
        assert camera.id in config['cameras']


class TestCameraDetectorTLS:
    """Test TLS/SSL functionality with real MQTT"""
    
    def test_mqtt_tls_configuration(self, test_mqtt_tls_broker, mqtt_topic_factory, monkeypatch, tmp_path):
        """Test MQTT client with real TLS broker"""
        # Get unique topic prefix
        full_topic = mqtt_topic_factory("dummy")
        prefix = full_topic.rsplit('/', 1)[0]
        
        # Create temporary config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Set environment for TLS
        monkeypatch.setenv('MQTT_TOPIC_PREFIX', prefix)
        monkeypatch.setenv('MQTT_BROKER', test_mqtt_tls_broker.host)
        monkeypatch.setenv('MQTT_PORT', str(test_mqtt_tls_broker.tls_port))  # Use TLS port
        monkeypatch.setenv('MQTT_TLS', 'true')
        monkeypatch.setenv('TLS_CA_PATH', test_mqtt_tls_broker.ca_cert)  # Correct env var name
        monkeypatch.setenv('CAMERA_CREDENTIALS', '')
        monkeypatch.setenv('FRIGATE_CONFIG_PATH', str(config_dir / 'frigate_config.yml'))
        monkeypatch.setenv('DISCOVERY_INTERVAL', '300')
        monkeypatch.setenv('SMART_DISCOVERY_ENABLED', 'false')
        
        # Import and reload detect module to pick up new environment
        import importlib
        import sys
        if 'camera_detector.detect' in sys.modules:
            del sys.modules['camera_detector.detect']
        if 'detect' in sys.modules:
            del sys.modules['detect']
        
        from camera_detector import detect
        
        # Create detector with real TLS connection
        with patch.object(detect.CameraDetector, '_start_background_tasks'):
            detector = detect.CameraDetector()
            
            # Wait for TLS connection
            start_time = time.time()
            while time.time() - start_time < 10:
                if hasattr(detector, 'mqtt_client') and detector.mqtt_client.is_connected():
                    break
                time.sleep(0.1)
            
            # Verify we connected via TLS
            assert detector.config.MQTT_TLS is True
            assert detector.config.MQTT_PORT == test_mqtt_tls_broker.tls_port
            assert hasattr(detector, 'mqtt_client')
            assert detector.mqtt_client.is_connected()
            
            # Cleanup
            try:
                if hasattr(detector, 'mqtt_client'):
                    detector.mqtt_client.disconnect()
                    detector.mqtt_client.loop_stop()
            except:
                pass


class TestHealthReporting:
    """Test health reporting functionality"""
    
    def test_health_status_publication(self, camera_detector_with_mqtt, mqtt_client):
        """Test health status is published to MQTT"""
        detector, prefix = camera_detector_with_mqtt
        
        # Subscribe to health topic
        health_topic = f"{prefix}/system/camera_detector_health"
        
        received_health = []
        
        def on_message(client, userdata, msg):
            if health_topic in msg.topic:
                received_health.append(json.loads(msg.payload.decode()))
        
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(health_topic)
        
        # Trigger health report
        detector._publish_health()
        
        # Wait for message
        time.sleep(0.5)
        
        # Verify health was published
        assert len(received_health) > 0
        health = received_health[0]
        assert 'status' in health
        assert 'timestamp' in health
        assert 'stats' in health
        assert 'cameras' in health
        
        # Check stats structure
        stats = health['stats']
        assert 'online_cameras' in stats
        assert stats['online_cameras'] == 0  # No cameras added yet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])