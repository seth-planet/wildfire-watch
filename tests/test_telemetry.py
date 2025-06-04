#!/usr/bin/env python3.12
import os
import time
import json
import socket
import threading
import pytest

# Import the telemetry module
from cam_telemetry import telemetry

# Dummy MQTT client to capture publishes and LWT
class DummyClient:
    def __init__(self):
        self.published = []
        self.last_will = None

    def will_set(self, topic, payload, qos, retain):
        self.last_will = (topic, payload, qos, retain)

    def connect(self, *args, **kwargs):
        pass

    def loop_start(self):
        pass

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

@pytest.fixture(autouse=True)
def patch_mqtt(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(telemetry, "client", dummy)
    # Reload connection logic to apply dummy client
    monkeypatch.setattr(telemetry, "mqtt_connect", lambda: None)
    return dummy

@pytest.fixture(autouse=True)
def patch_psutil(monkeypatch):
    # Provide fake psutil with fixed metrics
    class FakePS:
        @staticmethod
        def disk_usage(path):
            from collections import namedtuple
            DU = namedtuple("DU", ["total", "used", "free", "percent"])
            return DU(total=100*1024**2, used=40*1024**2, free=60*1024**2, percent=60.0)
        @staticmethod
        def virtual_memory():
            from collections import namedtuple
            VM = namedtuple("VM", ["total", "available", "percent", "used", "free",
                                   "active", "inactive", "buffers", "cached", "shared", "slab"])
            return VM(total=1000, available=600, percent=60.0,
                      used=400, free=600, active=0, inactive=0,
                      buffers=0, cached=0, shared=0, slab=0)
        @staticmethod
        def cpu_percent(interval=None):
            return 5.0
        @staticmethod
        def boot_time():
            return time.time() - 120  # 2 minutes ago
    monkeypatch.setattr(telemetry, "psutil", FakePS)
    return FakePS

def test_lwt_is_set(patch_mqtt):
    # Upon import, LWT should have been configured
    topic, payload, qos, retain = patch_mqtt.last_will
    assert telemetry.LWT_TOPIC in topic
    data = json.loads(payload)
    assert data["camera_id"] == telemetry.CAMERA_ID
    assert data["status"] == "offline"

def test_publish_telemetry_once(patch_mqtt, monkeypatch):
    # Reduce interval for test
    monkeypatch.setenv("TELEMETRY_INTERVAL", "1")
    # Reload module to pick up new interval
    import importlib
    importlib.reload(telemetry)
    # Trigger a single publish
    telemetry.publish_telemetry()
    # There should be exactly one publish record immediately
    assert len(patch_mqtt.published) == 1
    topic, payload, qos, retain = patch_mqtt.published[0]
    assert topic == telemetry.TOPIC_INFO
    data = json.loads(payload)
    # Check core fields
    assert data["camera_id"] == telemetry.CAMERA_ID
    assert data["status"] == "online"
    assert "free_disk_mb" in data and isinstance(data["free_disk_mb"], float)
    assert "cpu_percent" in data
    assert "uptime_seconds" in data
    assert data["config"]["rtsp_url"] == telemetry.RTSP_URL
    assert data["config"]["model_path"] == telemetry.MODEL_PATH

def test_periodic_publish(patch_mqtt, monkeypatch):
    # Shorter interval for quick test
    monkeypatch.setenv("TELEMETRY_INTERVAL", "1")
    import importlib
    importlib.reload(telemetry)
    # Start publishing
    telemetry.publish_telemetry()
    # Wait enough for 3 intervals
    time.sleep(3.5)
    # Expect at least 3 publishes
    assert len(patch_mqtt.published) >= 3

def test_network_failure_retry(monkeypatch, patch_mqtt):
    # Simulate failure in client.publish
    def bad_publish(topic, payload, qos=0, retain=False):
        raise RuntimeError("Network down")
    patch_mqtt.publish = bad_publish
    # Should not raise
    telemetry.publish_telemetry()
    # Sleep to allow scheduling (no exception)
    time.sleep(1)
