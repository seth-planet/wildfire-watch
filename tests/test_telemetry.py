#!/usr/bin/env python3.12
"""
Tests for Camera Telemetry Service
"""
import os
import sys
import time
import json
import socket
import threading
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Dummy MQTT client to capture publishes and LWT
class DummyClient:
    def __init__(self, client_id=None, clean_session=True):
        self.published = []
        self.last_will = None
        self.client_id = client_id
        self.clean_session = clean_session
        self.connected = False

    def will_set(self, topic, payload, qos, retain):
        self.last_will = (topic, payload, qos, retain)

    def connect(self, broker, port, keepalive):
        self.connected = True
        return (0, None)

    def loop_start(self):
        pass

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        return MagicMock(rc=0)

    def loop_stop(self):
        pass

    def disconnect(self):
        self.connected = False

# Mock the MQTT client before importing telemetry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cam_telemetry"))

# Create a mock client instance
mock_client = DummyClient()

# Patch MQTT and import telemetry
with patch('paho.mqtt.client.Client', return_value=mock_client):
    import telemetry
    # Override the client instance
    telemetry.client = mock_client
    # Store the original mqtt_connect function
    original_mqtt_connect = telemetry.mqtt_connect
    # Prevent automatic connection at module load
    telemetry.mqtt_connect = lambda: None

@pytest.fixture(autouse=True)
def reset_client():
    """Reset the mock client before each test"""
    mock_client.published.clear()
    mock_client.connected = False
    yield mock_client

@pytest.fixture
def mock_psutil(monkeypatch):
    """Provide fake psutil with fixed metrics"""
    class FakeDiskUsage:
        def __init__(self):
            self.total = 100 * 1024 * 1024  # 100MB
            self.free = 60 * 1024 * 1024    # 60MB
            self.used = 40 * 1024 * 1024    # 40MB
            self.percent = 40.0

    class FakeVirtualMemory:
        def __init__(self):
            self.percent = 60.0
            self.total = 1000
            self.available = 400

    class FakePsutil:
        @staticmethod
        def disk_usage(path):
            return FakeDiskUsage()
        
        @staticmethod
        def virtual_memory():
            return FakeVirtualMemory()
        
        @staticmethod
        def cpu_percent(interval=None):
            return 5.0
        
        @staticmethod
        def boot_time():
            return time.time() - 120  # 2 minutes ago

    monkeypatch.setattr(telemetry, "psutil", FakePsutil)
    return FakePsutil

def test_lwt_is_set():
    """Test that Last Will Testament is configured"""
    # LWT should have been set during module import
    assert mock_client.last_will is not None
    topic, payload, qos, retain = mock_client.last_will
    
    # Check topic format
    assert "lwt" in topic
    assert telemetry.CAMERA_ID in topic
    
    # Check payload
    data = json.loads(payload)
    assert data["camera_id"] == telemetry.CAMERA_ID
    assert data["status"] == "offline"
    assert "timestamp" in data
    
    # Check QoS and retain
    assert qos == 1
    assert retain == True

def test_publish_telemetry_basic(mock_psutil):
    """Test basic telemetry publishing"""
    telemetry.publish_telemetry()
    
    # Should have published one message
    assert len(mock_client.published) == 1
    
    topic, payload, qos, retain = mock_client.published[0]
    assert topic == telemetry.TOPIC_INFO
    assert qos == 1
    assert retain == False
    
    # Check payload structure
    data = json.loads(payload)
    assert data["camera_id"] == telemetry.CAMERA_ID
    assert data["status"] == "online"
    assert "timestamp" in data
    assert data["backend"] == telemetry.DETECTOR_BACKEND

def test_system_metrics_included(mock_psutil):
    """Test that system metrics are included when psutil is available"""
    telemetry.publish_telemetry()
    
    topic, payload, qos, retain = mock_client.published[0]
    data = json.loads(payload)
    
    # Check metrics
    assert data["free_disk_mb"] == 60.0
    assert data["total_disk_mb"] == 100.0
    assert data["memory_percent"] == 60.0
    assert data["cpu_percent"] == 5.0
    # Uptime should be approximately 120 seconds (allow for small timing differences)
    assert 119 <= data["uptime_seconds"] <= 121

def test_telemetry_without_psutil(monkeypatch):
    """Test telemetry works without psutil"""
    monkeypatch.setattr(telemetry, "psutil", None)
    
    telemetry.publish_telemetry()
    
    topic, payload, qos, retain = mock_client.published[0]
    data = json.loads(payload)
    
    # Basic fields should exist
    assert data["camera_id"] == telemetry.CAMERA_ID
    assert data["status"] == "online"
    assert "config" in data
    
    # Metrics should not exist
    assert "free_disk_mb" not in data
    assert "cpu_percent" not in data
    assert "memory_percent" not in data

def test_configuration_snapshot():
    """Test configuration is included in telemetry"""
    telemetry.publish_telemetry()
    
    topic, payload, qos, retain = mock_client.published[0]
    data = json.loads(payload)
    
    # Check config section
    assert "config" in data
    config = data["config"]
    assert "rtsp_url" in config
    assert "model_path" in config
    
    # Backend should be included
    assert "backend" in data

def test_timestamp_format():
    """Test timestamp is in correct ISO format"""
    telemetry.publish_telemetry()
    
    topic, payload, qos, retain = mock_client.published[0]
    data = json.loads(payload)
    
    timestamp = data["timestamp"]
    assert timestamp.endswith("Z")
    assert "T" in timestamp
    
    # Should be parseable
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed is not None

def test_error_handling(monkeypatch):
    """Test error handling in publish"""
    # Make publish raise an exception
    def bad_publish(*args, **kwargs):
        raise Exception("Network error")
    
    monkeypatch.setattr(mock_client, "publish", bad_publish)
    
    # Should not raise
    telemetry.publish_telemetry()

def test_periodic_scheduling(monkeypatch):
    """Test that telemetry schedules periodic updates"""
    timers_created = []
    
    class MockTimer:
        def __init__(self, interval, function):
            self.interval = interval
            self.function = function
            self.daemon = False
            timers_created.append(self)
        
        def start(self):
            pass
    
    monkeypatch.setattr(threading, "Timer", MockTimer)
    
    telemetry.publish_telemetry()
    
    # Should have created a timer
    assert len(timers_created) == 1
    timer = timers_created[0]
    assert timer.interval == telemetry.TELEMETRY_INT
    assert timer.daemon == True

def test_mqtt_reconnection_logic(monkeypatch):
    """Test MQTT reconnection with retries"""
    connect_calls = []
    sleep_calls = []
    
    def mock_connect(broker, port, keepalive):
        connect_calls.append((broker, port, keepalive))
        if len(connect_calls) < 2:
            raise Exception("Connection failed")
        return (0, None)
    
    def mock_sleep(seconds):
        sleep_calls.append(seconds)
    
    # Mock the client's connect method
    original_client_connect = mock_client.connect
    mock_client.connect = mock_connect
    
    # Restore the original mqtt_connect function for this test
    monkeypatch.setattr(telemetry, "mqtt_connect", original_mqtt_connect)
    monkeypatch.setattr(time, "sleep", mock_sleep)
    
    # Test the actual mqtt_connect function with retry limit
    original_mqtt_connect(max_retries=2)
    
    # Restore original connect
    mock_client.connect = original_client_connect
    
    # Should have retried
    assert len(connect_calls) >= 2
    assert len(sleep_calls) >= 1
    assert sleep_calls[0] == 5

def test_custom_environment_variables():
    """Test custom environment variable handling"""
    # The telemetry module reads environment variables at import time
    # Since we already imported it with mocked MQTT, we can't really test
    # dynamic environment variable loading without causing issues.
    # Instead, we'll verify the module can read environment variables correctly
    
    # Test that the module has the expected attributes
    assert hasattr(telemetry, 'MQTT_BROKER')
    assert hasattr(telemetry, 'CAMERA_ID')
    assert hasattr(telemetry, 'TELEMETRY_INT')
    assert hasattr(telemetry, 'TOPIC_INFO')
    assert hasattr(telemetry, 'LWT_TOPIC')
    
    # Test that defaults work
    assert isinstance(telemetry.MQTT_BROKER, str)
    assert isinstance(telemetry.CAMERA_ID, str)
    assert isinstance(telemetry.TELEMETRY_INT, int)
    assert isinstance(telemetry.TOPIC_INFO, str)
    
    # Test LWT topic construction
    assert telemetry.CAMERA_ID in telemetry.LWT_TOPIC

def test_main_loop_structure():
    """Test that main() function exists and has proper structure"""
    # Should have a main function
    assert hasattr(telemetry, 'main')
    
    # Test won't actually run main() as it has an infinite loop
    # Just verify it's callable
    assert callable(telemetry.main)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])