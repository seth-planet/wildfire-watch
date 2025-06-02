# tests/test_consensus.py

import os
import sys
import time
import json
import threading
import pytest

# Ensure our module is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))
import consensus

# ─────────────────────────────────────────────────────────────
# Dummy MQTT client to capture subscribes and publishes
# ─────────────────────────────────────────────────────────────
class DummyClient:
    def __init__(self):
        self.subscribed = []
        self.published = []
        self.will = None
        self.on_connect = None
        self.on_message = None

    def will_set(self, topic, payload, qos, retain):
        self.will = (topic, payload, qos, retain)

    def connect(self, *args, **kwargs):
        pass

    def loop_start(self):
        pass

    def subscribe(self, topics):
        # topics is list of (topic, qos)
        self.subscribed.extend([t for t, _ in topics])

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload, qos))

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

# ─────────────────────────────────────────────────────────────
# Fixtures: patching consensus.client and resetting state
# ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def patch_mqtt(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr(consensus, "client", dummy)
    # Prevent real reconnect loops
    monkeypatch.setattr(consensus, "mqtt_connect", lambda: None)
    # Manually invoke on_connect to subscribe to topics
    consensus.on_connect(dummy, None, None, 0)
    return dummy

@pytest.fixture(autouse=True)
def reset_state():
    # Clear in-memory state before each test
    consensus._camera_registry.clear()
    consensus._detection_hist.clear()
    consensus._object_history.clear()
    consensus._valid_cameras.clear()
    consensus._last_trigger = 0
    yield
    # no-op

# ─────────────────────────────────────────────────────────────
# Helper to simulate incoming MQTT messages
# ─────────────────────────────────────────────────────────────
class Msg:
    def __init__(self, topic, payload_dict):
        self.topic = topic
        self.payload = json.dumps(payload_dict)

def publish_msg(client, topic, payload):
    msg = Msg(topic, payload)
    client.on_message(client, None, msg)

# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_lwt_configuration(patch_mqtt):
    # Last Will should be set on DummyClient
    topic, payload, qos, retain = patch_mqtt.will
    assert consensus.LWT_TOPIC in topic
    data = json.loads(payload)
    assert data["camera_id"] == consensus.CAM_ID
    assert data["status"] == "offline"

def test_process_telemetry_updates_registry(patch_mqtt):
    # Simulate telemetry message
    now = time.time()
    publish_msg(patch_mqtt, consensus.TOPIC_TELEM, {"camera_id": "camA"})
    assert "camA" in consensus._camera_registry
    assert consensus._camera_registry["camA"] >= now

def test_legacy_detection_triggers_when_increase(monkeypatch, patch_mqtt):
    # Configure for quick test
    monkeypatch.setenv("CONSENSUS_THRESHOLD", "1")
    monkeypatch.setenv("DETECTION_WINDOW", "60")
    monkeypatch.setenv("INCREASE_COUNT", "2")
    monkeypatch.setenv("DETECTION_COOLDOWN", "0")
    # Reload config
    import importlib; importlib.reload(consensus)

    # Simulate two detections with increasing area
    payloads = [
        {"camera_id": "cam1", "bounding_box": [0,0,10,10]},   # area 100
        {"camera_id": "cam1", "bounding_box": [0,0,15,15]},   # area 225
        {"camera_id": "cam1", "bounding_box": [0,0,20,20]},   # area 400
    ]
    for p in payloads:
        publish_msg(patch_mqtt, consensus.TOPIC_DET, p)
        time.sleep(0.01)

    # Should have exactly one trigger
    triggers = [t for t,_p,_q in patch_mqtt.published if t == consensus.TOPIC_TRIGGER]
    assert len(triggers) == 1

def test_process_frigate_update_triggers(monkeypatch, patch_mqtt):
    # Configure threshold=1, increase_count=2, cooldown=0
    monkeypatch.setenv("CONSENSUS_THRESHOLD", "1")
    monkeypatch.setenv("DETECTION_WINDOW", "60")
    monkeypatch.setenv("INCREASE_COUNT", "2")
    monkeypatch.setenv("DETECTION_COOLDOWN", "0")
    import importlib; importlib.reload(consensus)

    # Simulate start/update events with increasing box sizes
    base = {"id": "obj1", "camera": "camF", "label": "fire"}
    sizes = [
        [0,0,10,10],   # area 100
        [0,0,12,12],   # area 144
        [0,0,15,15],   # area 225
    ]
    for s in sizes:
        event = {"type": "update", "after": {**base, "box": s}}
        publish_msg(patch_mqtt, consensus.FRIGATE_EVENTS_TOPIC, event)
        time.sleep(0.01)

    triggers = [t for t,_p,_q in patch_mqtt.published if t == consensus.TOPIC_TRIGGER]
    assert len(triggers) == 1
    # Validate payload contains our camera ID and valid_cameras list
    _, payload, _ = next(p for p in patch_mqtt.published if p[0] == consensus.TOPIC_TRIGGER)
    data = json.loads(payload)
    assert "valid_cameras" in data and "camF" in data["valid_cameras"]

def test_process_frigate_end_cleans_history(patch_mqtt):
    # Prepare object history
    consensus._object_history["objX"] = [(0,100),(1,200)]
    # Simulate end event
    event = {"type":"end", "after":{"id":"objX","camera":"camZ","label":"fire","box":[0,0,10,10]}}
    publish_msg(patch_mqtt, consensus.FRIGATE_EVENTS_TOPIC, event)
    # objX should be removed
    assert "objX" not in consensus._object_history
    
def test_cleanup_memory_removes_stale_and_keeps_fresh(monkeypatch, patch_mqtt):
    # Configure a very short window for testing
    monkeypatch.setenv("CAMERA_WINDOW", "1")  # DETECTION_WINDOW = 1s
    import importlib
    importlib.reload(consensus)

    now = consensus.now_ts()

    # Insert stale object history (older than 2× window = 2s)
    consensus._object_history["old_obj"] = [(now - 3, 100)]
    # And a fresh object history
    consensus._object_history["fresh_obj"] = [(now, 50)]

    # Insert stale camera registry & related hist
    consensus._camera_registry["old_cam"] = now - 3
    consensus._valid_cameras["old_cam"]   = now - 3
    consensus._detection_hist["old_cam"]  = [(now - 3, 100)]
    # And fresh camera entries
    consensus._camera_registry["fresh_cam"] = now
    consensus._valid_cameras["fresh_cam"]   = now
    consensus._detection_hist["fresh_cam"]  = [(now, 100)]

    # Run cleanup
    consensus.cleanup_memory()

    # Stale entries should be gone
    assert "old_obj" not in consensus._object_history
    assert "old_cam" not in consensus._camera_registry
    assert "old_cam" not in consensus._valid_cameras
    assert "old_cam" not in consensus._detection_hist

    # Fresh entries should remain
    assert "fresh_obj" in consensus._object_history
    assert "fresh_cam" in consensus._camera_registry
    assert "fresh_cam" in consensus._valid_cameras
    assert "fresh_cam" in consensus._detection_hist

def test_publish_health_schedules(monkeypatch, patch_mqtt):
    # Shorten interval
    monkeypatch.setenv("TELEMETRY_INTERVAL", "1")
    import importlib; importlib.reload(consensus)
    # Trigger
    consensus.publish_health()
    time.sleep(2.2)
    health_msgs = [m for m in patch_mqtt.published if m[0] == consensus.TOPIC_HEALTH]
    assert len(health_msgs) >= 2
    # Check structure
    _, payload, _ = health_msgs[0]
    data = json.loads(payload)
    assert data["camera_id"] == consensus.CAM_ID
    assert "health" in data and "online_cameras" in data["health"]
