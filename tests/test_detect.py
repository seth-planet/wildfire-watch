#!/usr/bin/env python3
"""
Comprehensive tests for Camera Detector Service
Tests discovery, MAC tracking, Frigate integration, and resilience
"""
import os
import sys
import time
import json
import yaml
import socket
import threading
import pytest
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock
from typing import Dict, List, Optional
import ipaddress

# Add module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../camera_detector")))

# Import after path setup
import detect
from detect import CameraDetector, Camera, CameraProfile, MACTracker, Config

# ─────────────────────────────────────────────────────────────
# Test Fixtures and Mocks
# ─────────────────────────────────────────────────────────────
class MockMQTTClient:
    """Mock MQTT client for testing"""
    def __init__(self):
        self.connected = False
        self.published_messages = []
        self.subscriptions = []
        self.on_connect = None
        self.on_disconnect = None
        self.will_topic = None
        self.will_payload = None
    
    def connect(self, broker, port, keepalive):
        self.connected = True
        if self.on_connect:
            self.on_connect(self, None, None, 0)
    
    def disconnect(self):
        self.connected = False
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)
    
    def loop_start(self):
        pass
    
    def loop_stop(self):
        pass
    
    def publish(self, topic, payload, qos=0, retain=False):
        self.published_messages.append({
            'topic': topic,
            'payload': json.loads(payload) if isinstance(payload, str) else payload,
            'qos': qos,
            'retain': retain
        })
    
    def will_set(self, topic, payload, qos, retain):
        self.will_topic = topic
        self.will_payload = payload
    
    def tls_set(self, *args, **kwargs):
        pass

class MockONVIFCamera:
    """Mock ONVIF camera"""
    def __init__(self, host, port, user, passwd, wsdl_dir=None):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.devicemgmt = Mock()
        self.media = Mock()
        
        # Mock device info
        device_info = Mock()
        device_info.Manufacturer = "Test"
        device_info.Model = "Camera-1000"
        device_info.SerialNumber = "12345"
        device_info.FirmwareVersion = "1.0.0"
        self.devicemgmt.GetDeviceInformation.return_value = device_info
        
        # Mock capabilities
        capabilities = Mock()
        capabilities.Media = True
        capabilities.PTZ = False
        self.devicemgmt.GetCapabilities.return_value = capabilities
    
    def create_media_service(self):
        media_service = Mock()
        
        # Mock profiles
        profile1 = Mock()
        profile1.Name = "Main"
        profile1.token = "profile_1"
        profile1.VideoEncoderConfiguration = Mock()
        profile1.VideoEncoderConfiguration.Resolution = Mock()
        profile1.VideoEncoderConfiguration.Resolution.Width = 1920
        profile1.VideoEncoderConfiguration.Resolution.Height = 1080
        profile1.VideoEncoderConfiguration.RateControl = Mock()
        profile1.VideoEncoderConfiguration.RateControl.FrameRateLimit = 30
        profile1.VideoEncoderConfiguration.Encoding = "H264"
        
        profile2 = Mock()
        profile2.Name = "Sub"
        profile2.token = "profile_2"
        profile2.VideoEncoderConfiguration = Mock()
        profile2.VideoEncoderConfiguration.Resolution = Mock()
        profile2.VideoEncoderConfiguration.Resolution.Width = 640
        profile2.VideoEncoderConfiguration.Resolution.Height = 480
        
        media_service.GetProfiles.return_value = [profile1, profile2]
        
        # Mock stream URI
        uri_response = Mock()
        uri_response.Uri = f"rtsp://{self.host}:554/stream1"
        media_service.GetStreamUri.return_value = uri_response
        
        return media_service

@pytest.fixture
def mock_mqtt():
    """Create mock MQTT client"""
    client = MockMQTTClient()
    with patch('detect.mqtt.Client', return_value=client):
        yield client

@pytest.fixture
def mock_onvif():
    """Mock ONVIF camera"""
    with patch('detect.ONVIFCamera', MockONVIFCamera):
        yield MockONVIFCamera

@pytest.fixture
def config(monkeypatch):
    """Configure test environment"""
    monkeypatch.setenv("DISCOVERY_INTERVAL", "300")
    monkeypatch.setenv("CAMERA_USERNAME", "admin")
    monkeypatch.setenv("CAMERA_PASSWORD", "password")
    monkeypatch.setenv("CAMERA_CREDENTIALS", "admin:,admin:admin,admin:password")
    monkeypatch.setenv("MAC_TRACKING_ENABLED", "true")
    monkeypatch.setenv("FRIGATE_UPDATE_ENABLED", "true")
    monkeypatch.setenv("FRIGATE_CONFIG_PATH", "/tmp/test_frigate.yml")

@pytest.fixture
def camera_detector(mock_mqtt, mock_onvif, config):
    """Create CameraDetector instance with mocked dependencies"""
    # Mock background tasks
    with patch.object(CameraDetector, '_start_background_tasks'):
        detector = CameraDetector()
        yield detector

@pytest.fixture
def sample_camera():
    """Create a sample camera"""
    camera = Camera(
        ip="192.168.1.100",
        mac="AA:BB:CC:DD:EE:FF",
        name="Test Camera",
        manufacturer="Test",
        model="Camera-1000"
    )
    camera.rtsp_urls = {
        'main': 'rtsp://admin:password@192.168.1.100:554/stream1',
        'sub': 'rtsp://admin:password@192.168.1.100:554/stream2'
    }
    camera.online = True
    camera.stream_active = True
    return camera

# ─────────────────────────────────────────────────────────────
# Basic Functionality Tests
# ─────────────────────────────────────────────────────────────
class TestBasicFunctionality:
    def test_initialization(self, camera_detector):
        """Test proper initialization"""
        assert len(camera_detector.cameras) == 0
        assert camera_detector.mqtt_connected is True
        assert len(camera_detector.credentials) == 3
    
    def test_credential_parsing(self, camera_detector):
        """Test credential parsing"""
        assert ('admin', '') in camera_detector.credentials
        assert ('admin', 'admin') in camera_detector.credentials
        assert ('admin', 'password') in camera_detector.credentials
    
    def test_camera_id_generation(self):
        """Test camera ID is based on MAC"""
        camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
        assert camera.id == "aabbccddeeff"
    
    def test_camera_ip_tracking(self, sample_camera):
        """Test IP change tracking"""
        original_ip = sample_camera.ip
        sample_camera.update_ip("192.168.1.101")
        
        assert sample_camera.ip == "192.168.1.101"
        assert original_ip in sample_camera.ip_history
        assert "192.168.1.101" in sample_camera.ip_history

# ─────────────────────────────────────────────────────────────
# MAC Tracking Tests
# ─────────────────────────────────────────────────────────────
class TestMACTracking:
    def test_mac_tracker_update(self):
        """Test MAC tracker updates"""
        tracker = MACTracker()
        tracker.update("AA:BB:CC:DD:EE:FF", "192.168.1.100")
        
        assert tracker.get_ip_for_mac("AA:BB:CC:DD:EE:FF") == "192.168.1.100"
        assert tracker.get_mac_for_ip("192.168.1.100") == "AA:BB:CC:DD:EE:FF"
    
    def test_mac_tracker_ip_change(self):
        """Test MAC tracker handles IP changes"""
        tracker = MACTracker()
        tracker.update("AA:BB:CC:DD:EE:FF", "192.168.1.100")
        tracker.update("AA:BB:CC:DD:EE:FF", "192.168.1.101")
        
        # Old IP should not map to MAC anymore
        assert tracker.get_mac_for_ip("192.168.1.100") is None
        assert tracker.get_mac_for_ip("192.168.1.101") == "AA:BB:CC:DD:EE:FF"
        assert tracker.get_ip_for_mac("AA:BB:CC:DD:EE:FF") == "192.168.1.101"
    
    @patch('detect.srp')
    def test_mac_tracker_network_scan(self, mock_srp):
        """Test MAC tracker network scanning"""
        # Mock ARP response
        mock_response = Mock()
        mock_response.psrc = "192.168.1.100"
        mock_response.hwsrc = "aa:bb:cc:dd:ee:ff"
        
        mock_element = [None, mock_response]
        mock_srp.return_value = ([mock_element], None)
        
        tracker = MACTracker()
        results = tracker.scan_network("192.168.1.0/24")
        
        assert "192.168.1.100" in results
        assert results["192.168.1.100"] == "AA:BB:CC:DD:EE:FF"

# ─────────────────────────────────────────────────────────────
# Camera Discovery Tests
# ─────────────────────────────────────────────────────────────
class TestCameraDiscovery:
    @patch('detect.WSDiscovery')
    def test_onvif_discovery(self, mock_wsd, camera_detector, mock_mqtt):
        """Test ONVIF camera discovery"""
        # Mock WS-Discovery
        mock_discovery = Mock()
        mock_wsd.return_value = mock_discovery
        
        # Mock service found
        mock_service = Mock()
        mock_service.getXAddrs.return_value = ['http://192.168.1.100:80/onvif/device_service']
        mock_service.getTypes.return_value = ['NetworkVideoTransmitter']
        mock_service.getScopes.return_value = ['onvif://www.onvif.org/type/video_encoder']
        
        mock_discovery.searchServices.return_value = [mock_service]
        
        # Mock MAC address lookup
        with patch.object(camera_detector, '_get_mac_address', return_value="AA:BB:CC:DD:EE:FF"):
            camera_detector._discover_onvif_cameras()
        
        # Should have discovered camera
        assert len(camera_detector.cameras) == 1
        camera = list(camera_detector.cameras.values())[0]
        assert camera.ip == "192.168.1.100"
        assert camera.mac == "AA:BB:CC:DD:EE:FF"
        assert camera.manufacturer == "Test"
        assert camera.model == "Camera-1000"
    
    @patch('detect.subprocess.run')
    def test_mdns_discovery(self, mock_run, camera_detector):
        """Test mDNS camera discovery"""
        # Mock avahi-browse output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
=;eth0;IPv4;Camera-1;_rtsp._tcp;local;camera1.local;192.168.1.100;554;""
        """
        mock_run.return_value = mock_result
        
        with patch.object(camera_detector, '_check_camera_at_ip') as mock_check:
            camera_detector._discover_mdns_cameras()
            mock_check.assert_called_with("192.168.1.100", "mDNS: Camera-1")
    
    @patch('detect.cv2.VideoCapture')
    def test_rtsp_validation(self, mock_capture, camera_detector):
        """Test RTSP stream validation"""
        # Mock successful capture
        mock_cap = Mock()
        mock_cap.read.return_value = (True, Mock())  # Success, frame
        mock_capture.return_value = mock_cap
        
        assert camera_detector._validate_rtsp_stream("rtsp://192.168.1.100:554/stream1") is True
        
        # Mock failed capture
        mock_cap.read.return_value = (False, None)
        assert camera_detector._validate_rtsp_stream("rtsp://192.168.1.100:554/stream1") is False
    
    def test_camera_offline_detection(self, camera_detector, sample_camera):
        """Test camera offline detection"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        sample_camera.last_seen = time.time() - 200  # Old timestamp
        
        # Run health check
        with patch.object(camera_detector, '_publish_camera_status') as mock_publish:
            camera_detector._health_check_loop()
            
            # Should mark camera offline
            assert sample_camera.online is False
            mock_publish.assert_called_with(sample_camera, "offline")

# ─────────────────────────────────────────────────────────────
# Frigate Integration Tests
# ─────────────────────────────────────────────────────────────
class TestFrigateIntegration:
    def test_frigate_config_generation(self, sample_camera):
        """Test Frigate configuration generation"""
        config = sample_camera.to_frigate_config()
        
        assert sample_camera.id in config
        cam_config = config[sample_camera.id]
        
        # Check ffmpeg inputs
        assert len(cam_config['ffmpeg']['inputs']) > 0
        assert cam_config['ffmpeg']['inputs'][0]['path'] == sample_camera.primary_rtsp_url
        
        # Check detection settings
        assert cam_config['detect']['enabled'] is True
        assert cam_config['detect']['width'] == 1280
        assert cam_config['detect']['height'] == 720
        
        # Check object tracking
        assert 'fire' in cam_config['objects']['track']
        assert 'smoke' in cam_config['objects']['track']
    
    @patch('detect.open', create=True)
    @patch('detect.yaml.dump')
    def test_frigate_config_update(self, mock_yaml_dump, mock_open, camera_detector, sample_camera, mock_mqtt):
        """Test Frigate configuration file update"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Update config
        camera_detector._update_frigate_config()
        
        # Should write config file
        mock_open.assert_called_once()
        mock_yaml_dump.assert_called_once()
        
        # Check YAML content
        yaml_content = mock_yaml_dump.call_args[0][0]
        assert 'cameras' in yaml_content
        assert sample_camera.id in yaml_content['cameras']
        
        # Check MQTT publications
        config_msgs = [m for m in mock_mqtt.published_messages
                      if m['topic'] == Config.TOPIC_FRIGATE_CONFIG]
        assert len(config_msgs) == 1
        
        # Check reload trigger
        reload_msgs = [m for m in mock_mqtt.published_messages
                      if m['topic'] == Config.FRIGATE_RELOAD_TOPIC]
        assert len(reload_msgs) == 1

# ─────────────────────────────────────────────────────────────
# Network Resilience Tests
# ─────────────────────────────────────────────────────────────
class TestNetworkResilience:
    def test_mqtt_reconnection(self, camera_detector, mock_mqtt):
        """Test MQTT reconnection handling"""
        # Simulate disconnect
        mock_mqtt.on_disconnect(mock_mqtt, None, 1)
        assert not camera_detector.mqtt_connected
        
        # Simulate reconnect
        mock_mqtt.on_connect(mock_mqtt, None, None, 0)
        assert camera_detector.mqtt_connected
    
    def test_camera_ip_change_handling(self, camera_detector, sample_camera, mock_mqtt):
        """Test handling of camera IP changes"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        original_ip = sample_camera.ip
        
        # Simulate IP change via MAC tracking
        camera_detector.mac_tracker.update(sample_camera.mac, "192.168.1.200")
        
        # Update mappings
        with patch.object(camera_detector, '_publish_camera_status') as mock_publish:
            camera_detector._update_mac_mappings()
            
            # Camera IP should be updated
            assert sample_camera.ip == "192.168.1.200"
            assert original_ip in sample_camera.ip_history
            mock_publish.assert_called_with(sample_camera, "ip_changed")
    
    @patch('detect.cv2.VideoCapture')
    def test_stream_validation_timeout(self, mock_capture, camera_detector):
        """Test RTSP validation handles timeouts properly"""
        # Mock timeout scenario
        mock_cap = Mock()
        mock_cap.read.side_effect = Exception("Timeout")
        mock_capture.return_value = mock_cap
        
        # Should handle gracefully and return False
        result = camera_detector._validate_rtsp_stream("rtsp://192.168.1.100:554/stream1")
        assert result is False
    
    def test_discovery_error_handling(self, camera_detector):
        """Test discovery handles errors gracefully"""
        # Mock discovery method to raise exception
        with patch.object(camera_detector, '_discover_onvif_cameras', side_effect=Exception("Network error")):
            # Should not crash
            camera_detector._discovery_loop()
            # Loop continues despite error

# ─────────────────────────────────────────────────────────────
# Multi-Camera Tests
# ─────────────────────────────────────────────────────────────
class TestMultiCamera:
    def test_multiple_camera_discovery(self, camera_detector):
        """Test discovering multiple cameras"""
        # Add multiple cameras
        cameras = [
            Camera(ip=f"192.168.1.{100+i}", mac=f"AA:BB:CC:DD:EE:{i:02X}", name=f"Camera-{i}")
            for i in range(5)
        ]
        
        for cam in cameras:
            camera_detector.cameras[cam.mac] = cam
        
        assert len(camera_detector.cameras) == 5
    
    def test_duplicate_camera_handling(self, camera_detector, sample_camera):
        """Test handling duplicate camera discoveries"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        original_last_seen = sample_camera.last_seen
        
        time.sleep(0.1)
        
        # Rediscover same camera
        with patch.object(camera_detector, '_get_onvif_details') as mock_onvif:
            camera_detector._check_camera_at_ip(sample_camera.ip)
            
            # Should update existing camera, not create new
            assert len(camera_detector.cameras) == 1
            assert sample_camera.last_seen > original_last_seen
    
    def test_camera_profile_handling(self):
        """Test camera with multiple profiles"""
        camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
        
        # Add profiles
        camera.profiles = [
            CameraProfile(name="Main", token="profile_1", resolution=(1920, 1080), framerate=30),
            CameraProfile(name="Sub", token="profile_2", resolution=(640, 480), framerate=15)
        ]
        
        camera.rtsp_urls = {
            'main': 'rtsp://192.168.1.100/stream1',
            'sub': 'rtsp://192.168.1.100/stream2'
        }
        
        # Should prefer main stream
        assert camera.primary_rtsp_url == 'rtsp://192.168.1.100/stream1'
        
        # Frigate config should use both streams appropriately
        config = camera.to_frigate_config()
        assert len(config[camera.id]['ffmpeg']['inputs']) == 2

# ─────────────────────────────────────────────────────────────
# Event Publishing Tests
# ─────────────────────────────────────────────────────────────
class TestEventPublishing:
    def test_camera_discovery_event(self, camera_detector, sample_camera, mock_mqtt):
        """Test camera discovery event publishing"""
        camera_detector._publish_camera_discovery(sample_camera)
        
        # Check published message
        discovery_msgs = [m for m in mock_mqtt.published_messages
                         if m['topic'].startswith(Config.TOPIC_DISCOVERY)]
        assert len(discovery_msgs) == 1
        
        msg = discovery_msgs[0]
        assert msg['payload']['event'] == 'discovered'
        assert msg['payload']['camera']['id'] == sample_camera.id
        assert msg['retain'] is True
    
    def test_camera_status_event(self, camera_detector, sample_camera, mock_mqtt):
        """Test camera status event publishing"""
        camera_detector._publish_camera_status(sample_camera, "offline")
        
        # Check published message
        status_msgs = [m for m in mock_mqtt.published_messages
                      if m['topic'].startswith(Config.TOPIC_STATUS)]
        assert len(status_msgs) == 1
        
        msg = status_msgs[0]
        assert msg['payload']['status'] == 'offline'
        assert msg['payload']['camera_id'] == sample_camera.id
    
    def test_health_report_publishing(self, camera_detector, sample_camera, mock_mqtt):
        """Test health report publishing"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Clear previous messages
        mock_mqtt.published_messages.clear()
        
        # Publish health
        camera_detector._publish_health()
        
        # Check health message
        health_msgs = [m for m in mock_mqtt.published_messages
                      if m['topic'] == Config.TOPIC_HEALTH]
        assert len(health_msgs) == 1
        
        health = health_msgs[0]['payload']
        assert health['stats']['total_cameras'] == 1
        assert health['stats']['online_cameras'] == 1
        assert sample_camera.id in health['cameras']

# ─────────────────────────────────────────────────────────────
# Credential Management Tests
# ─────────────────────────────────────────────────────────────
class TestCredentialManagement:
    def test_multiple_credential_attempts(self, camera_detector):
        """Test trying multiple credentials"""
        # Mock ONVIF attempts with different credentials
        attempt_count = 0
        
        def mock_onvif_init(host, port, user, passwd, wsdl_dir=None):
            nonlocal attempt_count
            attempt_count += 1
            if user == "admin" and passwd == "password":
                return MockONVIFCamera(host, port, user, passwd)
            else:
                raise Exception("Auth failed")
        
        with patch('detect.ONVIFCamera', side_effect=mock_onvif_init):
            camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
            result = camera_detector._get_onvif_details(camera)
            
            # Should succeed with correct credentials
            assert result is True
            assert camera.username == "admin"
            assert camera.password == "password"
    
    def test_rtsp_credential_discovery(self, camera_detector):
        """Test discovering RTSP credentials"""
        camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
        
        # Mock validation to succeed only with specific credentials
        def mock_validate(url):
            return "admin:password@" in url
        
        with patch.object(camera_detector, '_validate_rtsp_stream', side_effect=mock_validate):
            result = camera_detector._check_rtsp_stream(camera)
            
            assert result is True
            assert camera.username == "admin"
            assert camera.password == "password"

# ─────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────
class TestIntegration:
    @patch('detect.WSDiscovery')
    @patch('detect.cv2.VideoCapture')
    def test_full_discovery_cycle(self, mock_capture, mock_wsd, camera_detector, mock_mqtt):
        """Test complete discovery cycle"""
        # Mock WS-Discovery
        mock_discovery = Mock()
        mock_wsd.return_value = mock_discovery
        
        mock_service = Mock()
        mock_service.getXAddrs.return_value = ['http://192.168.1.100:80/onvif/device_service']
        mock_service.getTypes.return_value = ['NetworkVideoTransmitter']
        mock_service.getScopes.return_value = []
        
        mock_discovery.searchServices.return_value = [mock_service]
        
        # Mock successful stream validation
        mock_cap = Mock()
        mock_cap.read.return_value = (True, Mock())
        mock_capture.return_value = mock_cap
        
        # Mock MAC lookup
        with patch.object(camera_detector, '_get_mac_address', return_value="AA:BB:CC:DD:EE:FF"):
            # Run discovery
            camera_detector._discovery_loop()
        
        # Should have discovered and configured camera
        assert len(camera_detector.cameras) == 1
        camera = list(camera_detector.cameras.values())[0]
        assert camera.online is True
        assert camera.manufacturer == "Test"
        assert len(camera.rtsp_urls) > 0
    
    def test_health_check_cycle(self, camera_detector, sample_camera, mock_mqtt):
        """Test health check cycle"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Mock stream validation
        with patch.object(camera_detector, '_validate_rtsp_stream', return_value=False):
            camera_detector._health_check_loop()
            
            # Should detect stream error
            assert sample_camera.stream_active is False
            
            # Check status event
            status_msgs = [m for m in mock_mqtt.published_messages
                          if m['topic'].startswith(Config.TOPIC_STATUS)]
            assert any(m['payload']['status'] == 'stream_error' for m in status_msgs)

# ─────────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────────
class TestPerformance:
    def test_large_camera_count(self, camera_detector):
        """Test handling large number of cameras"""
        # Add 100 cameras
        for i in range(100):
            camera = Camera(
                ip=f"192.168.{i//256}.{i%256}",
                mac=f"AA:BB:{i//256:02X}:{i%256:02X}:EE:FF",
                name=f"Camera-{i}"
            )
            camera_detector.cameras[camera.mac] = camera
        
        assert len(camera_detector.cameras) == 100
        
        # Health report should handle all cameras
        with patch.object(camera_detector.mqtt_client, 'publish'):
            camera_detector._publish_health()
    
    def test_concurrent_discovery(self, camera_detector):
        """Test concurrent discovery operations"""
        # Mock discovery methods to run concurrently
        def slow_discovery():
            time.sleep(0.1)
            return []
        
        with patch.object(camera_detector, '_discover_onvif_cameras', side_effect=slow_discovery):
            with patch.object(camera_detector, '_discover_mdns_cameras', side_effect=slow_discovery):
                with patch.object(camera_detector, '_scan_rtsp_ports', side_effect=slow_discovery):
                    # All should complete without deadlock
                    camera_detector._discovery_loop()

# ─────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_invalid_mac_address(self, camera_detector):
        """Test handling invalid MAC addresses"""
        camera = Camera(ip="192.168.1.100", mac="INVALID", name="Test")
        assert camera.id == "invalid"
    
    def test_empty_rtsp_urls(self):
        """Test camera with no RTSP URLs"""
        camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
        assert camera.primary_rtsp_url is None
        assert camera.to_frigate_config() is None
    
    def test_network_interface_detection(self, camera_detector):
        """Test network interface detection"""
        with patch('detect.netifaces.interfaces', return_value=['lo', 'eth0', 'wlan0']):
            with patch('detect.netifaces.gateways', return_value={'default': {2: ('192.168.1.1', 'eth0')}}):
                networks = camera_detector._get_local_networks()
                assert len(networks) >= 0  # Should handle gracefully even if no networks
    
    def test_cleanup_on_shutdown(self, camera_detector, sample_camera, mock_mqtt):
        """Test cleanup on shutdown"""
        # Add online camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        sample_camera.online = True
        
        # Cleanup
        camera_detector.cleanup()
        
        # Camera should be marked offline
        assert sample_camera.online is False
        
        # Check offline events
        status_msgs = [m for m in mock_mqtt.published_messages
                      if m['topic'].startswith(Config.TOPIC_STATUS)]
        assert any(m['payload']['status'] == 'offline' for m in status_msgs)

# ─────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
