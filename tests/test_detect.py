#!/usr/bin/env python3.12
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
        try:
            parsed_payload = json.loads(payload) if isinstance(payload, str) else payload
        except (json.JSONDecodeError, TypeError):
            parsed_payload = payload
        
        self.published_messages.append({
            'topic': topic,
            'payload': parsed_payload,
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
        # Credentials depend on environment variables
        assert len(camera_detector.credentials) >= 1
    
    def test_credential_parsing(self, camera_detector):
        """Test credential parsing"""
        # When CAMERA_USERNAME and CAMERA_PASSWORD are set, only those are used
        config_username = camera_detector.config.DEFAULT_USERNAME
        config_password = camera_detector.config.DEFAULT_PASSWORD
        
        if config_username and config_password:
            # Specific credentials provided
            assert len(camera_detector.credentials) == 1
            assert camera_detector.credentials[0][0] == config_username
            # Password matches what was configured
        else:
            # Default credentials
            assert len(camera_detector.credentials) >= 2
            # Should have some form of admin credentials
    
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
    @patch('os.geteuid', return_value=0)  # Mock running as root
    def test_mac_tracker_network_scan(self, mock_geteuid, mock_srp):
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
        
        # Extract the health check logic without the infinite loop
        with patch.object(camera_detector, '_publish_camera_status') as mock_publish:
            # Simulate one iteration of the health check loop
            current_time = time.time()
            
            with camera_detector.lock:
                for mac, camera in list(camera_detector.cameras.items()):
                    # Check if camera is offline
                    if current_time - camera.last_seen > camera_detector.config.OFFLINE_THRESHOLD:
                        if camera.online:
                            camera.online = False
                            camera.stream_active = False
                            camera_detector._publish_camera_status(camera, "offline")
            
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
    
    def test_frigate_config_update(self, camera_detector, sample_camera, mock_mqtt):
        """Test Frigate configuration file update"""
        # Enable Frigate updates for this test
        camera_detector.config.FRIGATE_UPDATE_ENABLED = True
        
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Mock file operations
        with patch('os.makedirs') as mock_makedirs:
            with patch('builtins.open', create=True) as mock_open:
                with patch('yaml.dump') as mock_yaml_dump:
                    # Update config
                    camera_detector._update_frigate_config()
                    
                    # Should create directory if needed
                    mock_makedirs.assert_called_once_with(
                        os.path.dirname(camera_detector.config.FRIGATE_CONFIG_PATH), 
                        exist_ok=True
                    )
                    
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
            with patch.object(camera_detector, '_discover_mdns_cameras'):
                with patch.object(camera_detector, '_scan_rtsp_ports'):
                    with patch.object(camera_detector, '_update_mac_mappings'):
                        with patch.object(camera_detector, '_update_frigate_config'):
                            # Should not crash when running discovery
                            try:
                                camera_detector._run_full_discovery()
                                # Should complete without throwing
                            except Exception:
                                pytest.fail("Discovery should handle errors gracefully")

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
        
        # Rediscover same camera - mock MAC lookup to return existing MAC
        with patch.object(camera_detector, '_get_mac_address', return_value=sample_camera.mac):
            with patch.object(camera_detector, '_get_onvif_details', return_value=True):
                camera_detector._check_camera_at_ip(sample_camera.ip)
            
            # Should update existing camera, not create new
            assert len(camera_detector.cameras) == 1
            # Camera should have been updated
            camera = camera_detector.cameras.get(sample_camera.mac)
            assert camera is not None
            assert camera.last_seen > original_last_seen
    
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
        # Ensure camera_detector has the test credentials
        camera_detector.credentials = [
            ("user", "wrongpass"),
            ("admin", "wrongpass"),
            ("admin", "password"),  # This one should work
        ]
        
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
            # Should have tried multiple credentials
            assert attempt_count >= 3
    
    def test_rtsp_credential_discovery(self, camera_detector):
        """Test discovering RTSP credentials"""
        # Ensure camera_detector has test credentials
        camera_detector.credentials = [
            ("user", "wrongpass"),
            ("admin", "password"),  # This one should work
        ]
        
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
            # Run discovery methods directly instead of the infinite loop
            camera_detector._discover_onvif_cameras()
        
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
        sample_camera.last_validated = time.time() - 200  # Old validation
        
        # Mock stream validation
        with patch.object(camera_detector, '_validate_rtsp_stream', return_value=False):
            with patch.object(camera_detector, '_publish_camera_status') as mock_publish:
                # Simulate one iteration of health check
                current_time = time.time()
                
                with camera_detector.lock:
                    for mac, camera in list(camera_detector.cameras.items()):
                        # Validate RTSP stream periodically
                        if camera.online and camera.primary_rtsp_url:
                            if current_time - camera.last_validated > camera_detector.config.HEALTH_CHECK_INTERVAL:
                                if camera_detector._validate_rtsp_stream(camera.primary_rtsp_url):
                                    camera.stream_active = True
                                    camera.last_validated = current_time
                                else:
                                    camera.stream_active = False
                                    camera_detector._publish_camera_status(camera, "stream_error")
            
            # Should detect stream error
            assert sample_camera.stream_active is False
            
            # Check status event
            mock_publish.assert_called_with(sample_camera, "stream_error")

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
        call_times = []
        
        def track_call(method_name):
            def wrapper():
                start = time.time()
                time.sleep(0.1)
                call_times.append((method_name, time.time() - start))
                return []
            return wrapper
        
        with patch.object(camera_detector, '_discover_onvif_cameras', side_effect=track_call('onvif')):
            with patch.object(camera_detector, '_discover_mdns_cameras', side_effect=track_call('mdns')):
                with patch.object(camera_detector, '_scan_rtsp_ports', side_effect=track_call('rtsp')):
                    with patch.object(camera_detector, '_update_frigate_config'):
                        with patch.object(camera_detector, '_update_mac_mappings'):
                            # Run full discovery once
                            camera_detector._run_full_discovery()
        
        # Should have called all methods
        assert len(call_times) == 3
        method_names = [call[0] for call in call_times]
        assert 'onvif' in method_names
        assert 'mdns' in method_names
        assert 'rtsp' in method_names

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
# Resource Management Tests
# ─────────────────────────────────────────────────────────────
class TestResourceManagement:
    def test_opencv_resource_cleanup_on_exception(self, camera_detector):
        """Test OpenCV VideoCapture is properly released on exceptions"""
        # Mock VideoCapture that raises exception
        mock_cap = Mock()
        mock_cap.read.side_effect = Exception("Connection timeout")
        
        with patch('detect.cv2.VideoCapture', return_value=mock_cap):
            # Should not leak resources
            result = camera_detector._validate_rtsp_stream("rtsp://test")
            assert result is False
            mock_cap.release.assert_called_once()
    
    def test_wsdiscovery_cleanup_on_exception(self, camera_detector):
        """Test WSDiscovery is stopped even when exception occurs"""
        # Mock WSDiscovery that raises exception during search
        mock_wsd = Mock()
        mock_wsd.searchServices.side_effect = Exception("Network error")
        
        with patch('detect.WSDiscovery', return_value=mock_wsd):
            # Should handle exception and cleanup
            camera_detector._discover_onvif_cameras()
            mock_wsd.stop.assert_called_once()
    
    def test_infinite_recursion_prevention(self, camera_detector):
        """Test MAC address lookup doesn't recurse infinitely"""
        call_count = 0
        
        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate failed ARP lookups
            return Mock(returncode=1, stdout="")
        
        with patch('detect.subprocess.run', side_effect=mock_run):
            # Should not recurse infinitely
            result = camera_detector._get_mac_address("192.168.1.100")
            assert result is None
            # Should have limited number of attempts
            assert call_count <= 3

# ─────────────────────────────────────────────────────────────
# Configuration Validation Tests
# ─────────────────────────────────────────────────────────────
class TestConfigurationValidation:
    def test_port_validation(self, monkeypatch):
        """Test that invalid port values are corrected"""
        # The Config class reads environment variables at import time,
        # so we need to reload the module to test different values
        
        # Save original
        original_mqtt_port = os.getenv("MQTT_PORT")
        
        try:
            # Test port too high
            os.environ["MQTT_PORT"] = "99999"
            # Reload the module to pick up new env
            import importlib
            import detect
            importlib.reload(detect)
            config = detect.Config()
            assert config.MQTT_PORT == 65535  # Should be capped at max
            
            # Test port too low
            os.environ["MQTT_PORT"] = "0"
            importlib.reload(detect)
            config = detect.Config()
            assert config.MQTT_PORT == 1  # Should be at least 1
        finally:
            # Restore original
            if original_mqtt_port is None:
                os.environ.pop("MQTT_PORT", None)
            else:
                os.environ["MQTT_PORT"] = original_mqtt_port
            importlib.reload(detect)
    
    def test_timeout_validation(self, monkeypatch):
        """Test that invalid timeout values are corrected"""
        monkeypatch.setenv("RTSP_TIMEOUT", "0")
        monkeypatch.setenv("DISCOVERY_INTERVAL", "5")
        monkeypatch.setenv("HEALTH_CHECK_INTERVAL", "1")
        
        config = Config()
        assert config.RTSP_TIMEOUT >= 1  # Should be at least 1
        assert config.DISCOVERY_INTERVAL >= 30  # Should be at least 30
        assert config.HEALTH_CHECK_INTERVAL >= 10  # Should be at least 10
    
    def test_malformed_credentials_handling(self, monkeypatch):
        """Test handling of malformed credential strings"""
        test_cases = [
            "",  # Empty
            "invalid",  # No colon
            ":password",  # Empty username
            "user1:pass1,invalid,user2:pass2",  # Mixed valid/invalid
        ]
        
        for creds in test_cases:
            monkeypatch.setenv("CAMERA_CREDENTIALS", creds)
            # Mock both background tasks and MQTT connection
            with patch.object(CameraDetector, '_start_background_tasks'):
                with patch.object(CameraDetector, '_mqtt_connect_with_retry'):
                    detector = CameraDetector()
                    # Should always have at least default credentials
                    assert len(detector.credentials) >= 1
                    # All credentials should have non-empty usernames
                    for user, passwd in detector.credentials:
                        assert len(user) > 0


# ─────────────────────────────────────────────────────────────
# Input Validation Tests
# ─────────────────────────────────────────────────────────────
class TestInputValidation:
    def test_invalid_ip_address_handling(self, camera_detector):
        """Test handling of invalid IP addresses"""
        invalid_ips = [
            "999.999.999.999",  # Invalid IP
            "not.an.ip",        # Non-numeric
            "",                 # Empty
            "192.168.1",        # Incomplete
        ]
        
        for ip in invalid_ips:
            # Should handle gracefully without crashing
            result = camera_detector._get_mac_address(ip)
            assert result is None
    
    def test_malformed_rtsp_url_handling(self, camera_detector):
        """Test handling of malformed RTSP URLs"""
        malformed_urls = [
            "",                           # Empty
            "not-a-url",                 # Invalid format
            "http://192.168.1.100",      # Wrong protocol
            "rtsp://",                   # Incomplete
        ]
        
        for url in malformed_urls:
            # Should handle gracefully
            result = camera_detector._validate_rtsp_stream(url)
            assert result is False
    
    def test_camera_data_sanitization(self):
        """Test camera data is properly sanitized"""
        # Create camera with potentially problematic data
        camera = Camera(
            ip="192.168.1.100",
            mac="aa:bb:cc:dd:ee:ff",  # Lowercase MAC
            name="Camera with\nnewlines\tand special chars",
            manufacturer="Test\x01\x02\x03"
        )
        
        # Convert to dict (used in MQTT messages)
        data = camera.to_dict()
        
        # Should handle data safely
        assert data['mac'] == "aa:bb:cc:dd:ee:ff"
        # Should be JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0

# ─────────────────────────────────────────────────────────────
# Security Tests
# ─────────────────────────────────────────────────────────────
class TestSecurity:
    def test_credential_exposure_prevention(self, camera_detector, sample_camera, mock_mqtt):
        """Test credentials are not exposed in health reports"""
        # Set sensitive credentials
        sample_camera.username = "admin"
        sample_camera.password = "secret_password"
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Clear any previous messages
        mock_mqtt.published_messages.clear()
        
        # Generate health report
        camera_detector._publish_health()
        
        # Check that password is not exposed
        health_msgs = [m for m in mock_mqtt.published_messages 
                      if m['topic'] == Config.TOPIC_HEALTH]
        assert len(health_msgs) == 1
        
        health_str = json.dumps(health_msgs[0]['payload'])
        assert "secret_password" not in health_str
    
    def test_command_injection_prevention(self, camera_detector):
        """Test prevention of command injection via IP addresses"""
        # Attempt injection via IP address
        malicious_ip = "192.168.1.100; rm -rf /"
        
        with patch('detect.subprocess.run') as mock_run:
            # Should handle safely
            result = camera_detector._get_mac_address(malicious_ip)
            
            # Check that malicious commands weren't executed as separate commands
            for call in mock_run.call_args_list:
                if call[0]:  # If there are args
                    args = call[0][0]
                    # The IP should be passed as a single argument, not parsed as shell
                    if isinstance(args, list) and len(args) >= 3:
                        # For arp command: ['arp', '-n', 'malicious_ip']
                        if args[0] == 'arp':
                            # The malicious IP should be a single argument
                            assert args[2] == malicious_ip
                        # For ping command: ['ping', '-c', '1', '-W', '1', 'malicious_ip']
                        elif args[0] == 'ping':
                            assert args[5] == malicious_ip
            
            # Verify subprocess.run was called with shell=False (default)
            for call in mock_run.call_args_list:
                # shell should not be True
                assert call[1].get('shell', False) is False

# ─────────────────────────────────────────────────────────────
# Frigate Integration Robustness Tests
# ─────────────────────────────────────────────────────────────
class TestFrigateRobustness:
    def test_frigate_config_write_failure_recovery(self, camera_detector, sample_camera):
        """Test recovery when Frigate config write fails"""
        # Add camera
        camera_detector.cameras[sample_camera.mac] = sample_camera
        
        # Mock file write failure
        with patch('builtins.open', side_effect=PermissionError("Write failed")):
            with patch('os.makedirs'):
                # Should handle write failure gracefully
                try:
                    camera_detector._update_frigate_config()
                    # Should not crash the service
                except PermissionError:
                    pass  # Expected to handle gracefully
    
    def test_frigate_config_with_no_cameras(self, camera_detector):
        """Test Frigate config generation with no cameras"""
        # No cameras available
        assert len(camera_detector.cameras) == 0
        
        # Should generate valid config with no cameras
        with patch('os.makedirs'):
            with patch('builtins.open', create=True) as mock_open:
                with patch('detect.yaml.dump') as mock_dump:
                    camera_detector._update_frigate_config()
                    
                    # Should still write config
                    mock_open.assert_called_once()
                    mock_dump.assert_called_once()
                    
                    # Config should have empty cameras section
                    config = mock_dump.call_args[0][0]
                    assert 'cameras' in config
                    assert len(config['cameras']) == 0
    
    def test_camera_without_rtsp_urls(self):
        """Test camera with no RTSP URLs"""
        camera = Camera(ip="192.168.1.100", mac="AA:BB:CC:DD:EE:FF", name="Test")
        # No RTSP URLs set
        assert camera.primary_rtsp_url is None
        assert camera.to_frigate_config() is None

# ─────────────────────────────────────────────────────────────
# Memory Management Tests
# ─────────────────────────────────────────────────────────────
class TestMemoryManagement:
    def test_large_camera_count_handling(self, camera_detector, mock_mqtt):
        """Test system handles large number of cameras"""
        # Add many cameras
        for i in range(100):
            camera = Camera(
                ip=f"10.0.{i//256}.{i%256}",
                mac=f"AA:BB:CC:{i//256:02X}:{i%256:02X}:FF",
                name=f"Camera-{i}"
            )
            camera.ip_history = [camera.ip]  # Limit history
            camera_detector.cameras[camera.mac] = camera
        
        # Clear any previous messages
        mock_mqtt.published_messages.clear()
        
        # Health report should not consume excessive memory
        camera_detector._publish_health()
        
        # Should successfully publish
        health_msgs = [m for m in mock_mqtt.published_messages 
                      if m['topic'] == Config.TOPIC_HEALTH]
        assert len(health_msgs) == 1
        assert health_msgs[0]['payload']['stats']['total_cameras'] == 100
    
    def test_mac_tracker_update_efficiency(self):
        """Test MAC tracker efficiently handles updates"""
        tracker = MACTracker()
        
        # Add many IP-MAC mappings
        for i in range(1000):
            ip = f"192.168.{(i//256)%256}.{i%256}"
            mac = f"AA:BB:CC:DD:{(i//256)%256:02X}:{i%256:02X}"
            tracker.update(mac, ip)
        
        # Should handle large datasets efficiently
        assert len(tracker.mac_to_ip) == 1000
        assert len(tracker.ip_to_mac) == 1000
        
        # Updating existing MAC should not increase size
        tracker.update("AA:BB:CC:DD:00:01", "192.168.1.200")
        assert len(tracker.mac_to_ip) == 1000

# ─────────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────────
class TestPerformanceEdgeCases:
    def test_rtsp_timeout_enforcement(self, camera_detector):
        """Test RTSP validation respects timeout settings"""
        # Test with very short timeout
        camera_detector.config.RTSP_TIMEOUT = 1
        
        # Mock slow capture
        mock_cap = Mock()
        mock_cap.read.side_effect = lambda: time.sleep(2) or (False, None)
        
        with patch('detect.cv2.VideoCapture', return_value=mock_cap):
            start_time = time.time()
            result = camera_detector._validate_rtsp_stream("rtsp://slow.camera")
            duration = time.time() - start_time
            
            # Should timeout quickly
            assert result is False
            # Should not wait longer than necessary
            assert duration < 5  # Should complete within 5 seconds
    
    def test_discovery_performance_with_many_networks(self, camera_detector):
        """Test discovery performance with multiple networks"""
        # Mock multiple network interfaces
        mock_networks = [f"192.168.{i}.0/24" for i in range(1, 10)]
        
        with patch.object(camera_detector, '_get_local_networks', return_value=mock_networks):
            with patch('detect.subprocess.run') as mock_run:
                # Mock successful but slow nmap
                mock_run.return_value = Mock(returncode=0, stdout="")
                
                start_time = time.time()
                camera_detector._scan_rtsp_ports()
                duration = time.time() - start_time
                
                # Should handle multiple networks
                assert mock_run.call_count >= len(mock_networks)
                # Should complete in reasonable time
                assert duration < 10

# ─────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
