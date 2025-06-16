#!/usr/bin/env python3
import os
import sys
import time
import json
import socket
import threading
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Try to import psutil for system metrics; if unavailable, metrics will be empty
try:
    import psutil
except ImportError:
    psutil = None

# ─────────────────────────────────────────────────────────────
#  Load environment
# ─────────────────────────────────────────────────────────────
load_dotenv()

MQTT_BROKER       = os.getenv("MQTT_BROKER", "mqtt_broker")
CAMERA_ID         = os.getenv("CAMERA_ID", socket.gethostname())
TELEMETRY_INT     = int(os.getenv("TELEMETRY_INTERVAL", "60"))
TOPIC_INFO        = os.getenv("TELEMETRY_TOPIC", "system/telemetry")
LWT_TOPIC         = os.getenv("LWT_TOPIC", f"{TOPIC_INFO}/{CAMERA_ID}/lwt")

# Include config snapshot
RTSP_URL          = os.getenv("RTSP_STREAM_URL", None)
MODEL_PATH        = os.getenv("MODEL_PATH", None)
DETECTOR_BACKEND  = os.getenv("DETECTOR", "unknown")

# ─────────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)

# ─────────────────────────────────────────────────────────────
#  MQTT setup with LWT
# ─────────────────────────────────────────────────────────────
client = mqtt.Client(client_id=CAMERA_ID, clean_session=True)
client.will_set(
    LWT_TOPIC,
    payload=json.dumps({
        "camera_id": CAMERA_ID,
        "status": "offline",
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }),
    qos=1,
    retain=True
)

def mqtt_connect(retry_count=0, max_retries=None):
    """Connect to MQTT broker with optional retry limit"""
    try:
        client.connect(MQTT_BROKER, 1883, keepalive=60)
        client.loop_start()
        logging.info(f"Connected to MQTT broker at {MQTT_BROKER}")
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {e}")
        if max_retries is not None and retry_count >= max_retries:
            logging.error(f"Max retries ({max_retries}) reached, giving up")
            raise
        time.sleep(5)
        mqtt_connect(retry_count + 1, max_retries)

mqtt_connect()

# ─────────────────────────────────────────────────────────────
#  System Metrics
# ─────────────────────────────────────────────────────────────
def get_system_metrics():
    metrics = {}
    if psutil:
        try:
            du = psutil.disk_usage("/")
            metrics["free_disk_mb"] = round(du.free / 1024 / 1024, 2)
            metrics["total_disk_mb"] = round(du.total / 1024 / 1024, 2)
            vm = psutil.virtual_memory()
            metrics["memory_percent"] = vm.percent
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            uptime = time.time() - psutil.boot_time()
            metrics["uptime_seconds"] = int(uptime)
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    return metrics

# ─────────────────────────────────────────────────────────────
#  Telemetry Publish
# ─────────────────────────────────────────────────────────────
def publish_telemetry():
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    payload = {
        "camera_id": CAMERA_ID,
        "timestamp": timestamp,
        "status": "online",
        "backend": DETECTOR_BACKEND,
        "config": {
            "rtsp_url": RTSP_URL,
            "model_path": MODEL_PATH
        }
    }
    payload.update(get_system_metrics())

    try:
        client.publish(TOPIC_INFO, json.dumps(payload), qos=1, retain=False)
        logging.info(f"Telemetry published: {payload}")
    except Exception as e:
        logging.error(f"Failed to publish telemetry: {e}")

    # Schedule next telemetry
    timer = threading.Timer(TELEMETRY_INT, publish_telemetry)
    timer.daemon = True
    timer.start()

# ─────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────
def main():
    # Initial telemetry
    publish_telemetry()

    # Keep running to allow scheduled publishes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Telemetry service interrupted, shutting down...")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()

