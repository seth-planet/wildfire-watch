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
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CAMERA_ID, clean_session=True)
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
        # Use MQTT_PORT environment variable if available, otherwise default to 1883
        port = int(os.getenv("MQTT_PORT", "1883"))
        client.connect(MQTT_BROKER, port, keepalive=60)
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
    """Get system metrics, structured for compatibility with tests"""
    metrics = {}
    if psutil:
        try:
            du = psutil.disk_usage("/")
            vm = psutil.virtual_memory()
            
            # Structure metrics for test compatibility
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": vm.percent,
                "disk_usage": {
                    "free": du.free,
                    "total": du.total,
                    "used": du.used,
                    "percent": round((du.used / du.total) * 100, 1)
                },
                "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 2),
                # Legacy fields for backward compatibility
                "free_disk_mb": round(du.free / 1024 / 1024, 2),
                "total_disk_mb": round(du.total / 1024 / 1024, 2),
                "uptime_seconds": int(time.time() - psutil.boot_time())
            }
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    return metrics

# ─────────────────────────────────────────────────────────────
#  Telemetry Publish
# ─────────────────────────────────────────────────────────────
def publish_telemetry():
    timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Get system metrics
    system_metrics = get_system_metrics()
    
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
    
    # Add system metrics both as structured field and top-level for compatibility
    if system_metrics:
        payload["system_metrics"] = system_metrics
        # Also add some metrics at top level for backward compatibility
        for key in ["cpu_percent", "memory_percent", "free_disk_mb", "total_disk_mb", "uptime_seconds"]:
            if key in system_metrics:
                payload[key] = system_metrics[key]

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

