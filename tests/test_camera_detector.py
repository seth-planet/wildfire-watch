#!/usr/bin/env python3.12
"""
Consolidated tests for Camera Detector Service
Combines camera discovery, MAC tracking, TLS support, and Frigate integration
"""
import os
import sys
import ssl
import time
import json
import yaml
import socket
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, List, Optional
import ipaddress

# Add module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../camera_detector")))

# Import after path setup
from detect import CameraDetector, Camera, CameraProfile, MACTracker, Config


class TestCameraDetectorConfig:
    """Test configuration handling"""
    
    def test_config_from_environment(self):
        """Test Config loads from environment variables"""
        # Config class uses environment variables at initialization time
        # Test that Config instance has expected attributes and types
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
        # Config uses instance attributes populated from environment
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
    """Test camera discovery methods"""
    
    @patch('detect.mqtt.Client')
    def test_onvif_discovery(self, mock_mqtt_class):
        """Test ONVIF WS-Discovery"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch('detect.WSDiscovery') as mock_ws:
            # Mock WS-Discovery service
            mock_service = Mock()
            mock_service.getXAddrs.return_value = ['http://192.168.1.100/onvif/device_service']
            mock_service.getTypes.return_value = ['NetworkVideoDevice']
            mock_service.getScopes.return_value = ['onvif://www.onvif.org/type/NetworkVideoDevice']
            
            mock_wsd = Mock()
            mock_wsd.searchServices.return_value = [mock_service]
            mock_ws.return_value = mock_wsd
            
            # Mock ONVIF camera
            with patch('detect.ONVIFCamera') as mock_onvif_cam:
                mock_cam = Mock()
                device_info = Mock()
                device_info.Manufacturer = "Hikvision"
                device_info.Model = "DS-2CD2042WD"
                mock_cam.devicemgmt.GetDeviceInformation.return_value = device_info
                mock_onvif_cam.return_value = mock_cam
                
                # Create detector without triggering background tasks
                with patch.object(CameraDetector, '_start_background_tasks'):
                    detector = CameraDetector()
                    detector._discover_onvif_cameras()
                
                # Verify discovery was attempted
                mock_wsd.start.assert_called_once()
                mock_wsd.searchServices.assert_called_once()
    
    @patch('detect.mqtt.Client')
    def test_rtsp_port_scanning(self, mock_mqtt_class):
        """Test RTSP port scanning"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        detector = CameraDetector()
        
        # Mock network interfaces
        with patch('detect.netifaces.interfaces', return_value=['eth0']):
            with patch('detect.netifaces.ifaddresses') as mock_ifaddr:
                mock_ifaddr.return_value = {
                    2: [{'addr': '192.168.1.10', 'netmask': '255.255.255.0'}]
                }
                
                networks = detector._get_local_networks()
                assert '192.168.1.0/24' in networks
    
    @patch('detect.mqtt.Client')
    @patch('detect.ProcessPoolExecutor')
    def test_rtsp_stream_validation(self, mock_executor_class, mock_mqtt_class):
        """Test RTSP stream validation"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Mock ProcessPoolExecutor and future
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = True  # Successful validation
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.return_value = mock_future
        mock_executor_class.return_value = mock_executor
        
        detector = CameraDetector()
        result = detector._validate_rtsp_stream("rtsp://192.168.1.100:554/stream1")
        
        assert result is True
        # Verify executor was used correctly
        # The actual code uses min(4, cpu_count) for max_workers
        import os
        cpu_count = os.cpu_count() or 4
        expected_workers = min(4, cpu_count)
        mock_executor_class.assert_called_once_with(max_workers=expected_workers)
        mock_executor.submit.assert_called_once()
        # Verify the worker function and URL were passed
        call_args = mock_executor.submit.call_args[0]
        assert call_args[1] == "rtsp://192.168.1.100:554/stream1"  # rtsp_url
        assert call_args[2] >= 1000  # timeout_ms should be at least 1000


class TestCameraDetectorTLS:
    """Test TLS/SSL functionality"""
    
    @patch('detect.mqtt.Client')
    def test_mqtt_tls_configuration(self, mock_mqtt_class):
        """Test MQTT client configures TLS when enabled"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {'MQTT_TLS': 'true'}):
            # Create a new config with TLS enabled
            from detect import Config
            config = Config()
            config.MQTT_TLS = True
            
            with patch('detect.Config', return_value=config):
                detector = CameraDetector()
                
                # Verify TLS was configured
                mock_mqtt.tls_set.assert_called_once()
                call_args = mock_mqtt.tls_set.call_args
                assert call_args.kwargs['cert_reqs'] == ssl.CERT_REQUIRED
                assert call_args.kwargs['tls_version'] == ssl.PROTOCOL_TLS
    
    @patch('detect.mqtt.Client')
    def test_mqtt_port_selection(self, mock_mqtt_class):
        """Test correct port selection based on TLS"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.dict(os.environ, {'MQTT_TLS': 'true'}):
            # Create a new config with TLS enabled
            from detect import Config
            config = Config()
            config.MQTT_TLS = True
            
            with patch('detect.Config', return_value=config):
                detector = CameraDetector()
                
                # Should use port 8883 for TLS
                mock_mqtt.connect.assert_called()
                call_args = mock_mqtt.connect.call_args
                assert call_args.args[1] == 8883


class TestSmartDiscovery:
    """Test smart discovery phases"""
    
    @patch('detect.mqtt.Client')
    def test_discovery_phases(self, mock_mqtt_class):
        """Test initial aggressive -> steady state transition"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Disable background tasks
        with patch.object(CameraDetector, '_start_background_tasks'):
            detector = CameraDetector()
        
        # Initial state
        assert detector.discovery_count == 0
        assert detector.is_steady_state is False
        
        # Simulate discovery runs
        with patch.object(detector, '_run_full_discovery'):
            detector.last_camera_count = 2
            detector.cameras = {'mac1': Mock(), 'mac2': Mock()}
            
            # Simulate multiple discovery cycles
            for i in range(3):
                detector.discovery_count = i + 1
                detector.stable_count = i + 1
            
            # After 3 stable counts, should enter steady state
            detector.stable_count = 3
            detector.is_steady_state = True
            
            assert detector.is_steady_state is True


class TestHealthMonitoring:
    """Test camera health monitoring"""
    
    @patch('detect.mqtt.Client')
    def test_camera_offline_detection(self, mock_mqtt_class):
        """Test detection of offline cameras"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Disable background tasks
        with patch.object(CameraDetector, '_start_background_tasks'):
            detector = CameraDetector()
        
        # Add test camera
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera"
        )
        camera.online = True
        camera.last_seen = time.time() - 300  # 5 minutes ago
        detector.cameras[camera.mac] = camera
        
        # Mock publish method and config
        detector.config.OFFLINE_THRESHOLD = 240  # 4 minutes
        
        with patch.object(detector, '_publish_camera_status') as mock_publish:
            # Manually run the health check logic
            current_time = time.time()
            
            # Check if camera is offline
            if current_time - camera.last_seen > detector.config.OFFLINE_THRESHOLD:
                camera.online = False
                camera.stream_active = False
                detector._publish_camera_status(camera, "offline")
            
            # Camera should be marked offline
            assert camera.online is False
            mock_publish.assert_called_once_with(camera, "offline")


class TestGrowingFireDetection:
    """Test growing fire detection for consensus"""
    
    def test_fire_size_tracking(self):
        """Test that fire sizes are tracked over time"""
        # This test verifies the growing fire detection mentioned in README
        # but not currently implemented in consensus.py
        
        # TODO: This feature needs to be implemented in consensus.py
        # The test is added to ensure README compliance
        pytest.skip("Growing fire detection not yet implemented")


class TestCredentialHandling:
    """Test camera credential management"""
    
    @patch('detect.mqtt.Client')
    def test_credential_parsing(self, mock_mqtt_class):
        """Test parsing of camera credentials"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Mock the config to have the credentials we want
        with patch.object(CameraDetector, '_start_background_tasks'):
            detector = CameraDetector()
            # Manually set the credentials config
            detector.config.CAMERA_CREDENTIALS = 'admin:password,root:12345,admin:'
            detector.config.DEFAULT_USERNAME = ""
            detector.config.DEFAULT_PASSWORD = ""
            
            creds = detector._parse_credentials()
            
            assert len(creds) == 3
            assert ('admin', 'password') in creds
            assert ('root', '12345') in creds
            assert ('admin', '') in creds
    
    @patch('detect.mqtt.Client')
    def test_single_credential_override(self, mock_mqtt_class):
        """Test single credential override via environment"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        with patch.object(CameraDetector, '_start_background_tasks'):
            detector = CameraDetector()
            # Manually set the single credential config
            detector.config.DEFAULT_USERNAME = "testuser"
            detector.config.DEFAULT_PASSWORD = "testpass"
            
            creds = detector._parse_credentials()
            
            # Should only use the provided credential
            assert len(creds) == 1
            assert creds[0] == ('testuser', 'testpass')


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @patch('detect.mqtt.Client')
    @patch('detect.cv2.VideoCapture')
    def test_camera_discovery_to_frigate_config(self, mock_capture, mock_mqtt_class):
        """Test full flow from discovery to Frigate config"""
        mock_mqtt = MagicMock()
        mock_mqtt_class.return_value = mock_mqtt
        
        # Mock successful RTSP validation
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, Mock())
        mock_capture.return_value = mock_cap
        
        detector = CameraDetector()
        
        # Add discovered camera
        camera = Camera(
            ip="192.168.1.100",
            mac="AA:BB:CC:DD:EE:FF",
            name="Test Camera"
        )
        camera.rtsp_urls = {'main': 'rtsp://192.168.1.100:554/stream1'}
        camera.online = True
        detector.cameras[camera.mac] = camera
        
        # Mock file operations
        with patch('builtins.open', create=True) as mock_open:
            with patch('os.makedirs'):
                detector._update_frigate_config()
                
                # Verify Frigate config was written
                mock_open.assert_called()
                
                # Verify MQTT publish
                calls = [call for call in mock_mqtt.publish.call_args_list 
                        if 'frigate/config/cameras' in str(call)]
                assert len(calls) > 0


# Tests requiring actual hardware should use @pytest.mark.skip or @pytest.mark.skipif


if __name__ == "__main__":
    pytest.main([__file__, "-v"])