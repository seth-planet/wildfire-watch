#!/usr/bin/env python3
"""System health monitoring service for Wildfire Watch edge devices.

This module implements a lightweight telemetry service that monitors system
health metrics and publishes them via MQTT for centralized monitoring. It's
designed to run alongside camera detection services on edge devices like
Raspberry Pi, providing visibility into resource usage and system status.

IMPORTANT: When running in Docker containers, psutil reports HOST system
metrics, not container-specific metrics. This is because psutil reads from
/proc/stat which provides system-wide statistics. For accurate container
metrics, external monitoring tools like cAdvisor or Docker stats are needed.

The service publishes periodic health updates including:
    - CPU usage percentage (host system, not container)
    - Memory usage percentage (host system, not container)
    - Disk space availability
    - System uptime
    - Service configuration snapshot
    - Online/offline status with Last Will Testament (LWT)

Communication Flow:
    1. Publishes health metrics to 'system/telemetry' every 60 seconds
    2. Includes configuration snapshot for debugging
    3. Publishes LWT to 'system/telemetry/{CAMERA_ID}/lwt' on disconnect
    4. No subscriptions - this is a publish-only monitoring service

MQTT Topics:
    Published:
        - system/telemetry: Periodic health metrics (not retained)
        - system/telemetry/{CAMERA_ID}/lwt: Last will testament (retained)
    
    Subscribed:
        - None

Thread Model:
    - Main thread: MQTT connection and infinite sleep loop
    - Timer threads: Scheduled telemetry publications (daemon threads)
    - No shared state requiring synchronization

Configuration:
    All settings via environment variables:
    - MQTT_BROKER: Broker hostname (default: 'mqtt_broker')
    - CAMERA_ID: Unique identifier (default: hostname)
    - TELEMETRY_INTERVAL: Seconds between updates (default: 60)
    - TELEMETRY_TOPIC: Base topic for metrics (default: 'system/telemetry')
    - LWT_TOPIC: Last will topic (default: '{TELEMETRY_TOPIC}/{CAMERA_ID}/lwt')

Example:
    Run standalone:
        $ python3 telemetry.py
        
    Run in Docker:
        $ docker-compose up cam-telemetry
        
    With custom interval:
        $ TELEMETRY_INTERVAL=30 python3 telemetry.py

Note:
    This service gracefully handles missing psutil by publishing empty metrics.
    It will continue running and publishing status even without system metrics.
"""
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
#  Global state
# ─────────────────────────────────────────────────────────────
active_timer = None
_shutdown_flag = False

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
    """Connect to MQTT broker with exponential backoff retry.
    
    Establishes connection to the MQTT broker and starts the network loop.
    Uses recursive retry with exponential backoff on connection failure.
    
    Args:
        retry_count: Current retry attempt number (used internally).
        max_retries: Maximum number of retry attempts. None for infinite retries.
    
    Raises:
        Exception: If connection fails after max_retries attempts.
    
    Side Effects:
        - Starts MQTT network loop thread via client.loop_start()
        - Logs connection status to stderr
        - Sleeps for 5 seconds between retry attempts
    
    Note:
        The max_retries parameter was added to prevent infinite recursion
        during testing. Production deployments typically use infinite retry
        to handle temporary network issues.
    """
    global _shutdown_flag
    if _shutdown_flag:
        logging.info("Telemetry shutdown flag set, skipping MQTT connection")
        return
        
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
        if not _shutdown_flag:  # Check again before sleeping
            time.sleep(5)
            mqtt_connect(retry_count + 1, max_retries)

# Only connect if not in testing mode
if not any('pytest' in arg for arg in sys.argv):
    mqtt_connect()

# ─────────────────────────────────────────────────────────────
#  System Metrics
# ─────────────────────────────────────────────────────────────
def get_system_metrics():
    """Collect system health metrics using psutil.
    
    Gathers CPU, memory, disk, and uptime metrics from the host system.
    When running in Docker, these metrics reflect the HOST system, not
    the container, because psutil reads from /proc/stat.
    
    Returns:
        dict: System metrics including:
            - cpu_percent: Current CPU usage (0-100)
            - memory_percent: Memory usage percentage
            - disk_usage: Dict with free, total, used, percent
            - uptime_hours: Hours since system boot
            - free_disk_mb: Free disk space in MB (legacy)
            - total_disk_mb: Total disk space in MB (legacy)
            - uptime_seconds: Seconds since boot (legacy)
        
        Returns empty dict if psutil is not available.
    
    Side Effects:
        - Logs errors if metric collection fails
        - First call to cpu_percent() may return 0.0
    
    Note:
        The legacy fields (free_disk_mb, total_disk_mb, uptime_seconds)
        are maintained for backward compatibility with older versions
        of the telemetry consumers.
        
        Docker consideration: These metrics are from the host OS, not
        container-specific. For container metrics, use Docker stats API
        or cAdvisor.
    """
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
    """Publish system telemetry to MQTT and schedule next publication.
    
    Collects current system metrics and publishes a comprehensive health
    status message to the MQTT broker. Uses threading.Timer to schedule
    the next publication, creating a repeating telemetry cycle.
    
    MQTT Message Format:
        {
            "camera_id": "hostname",
            "timestamp": "2024-01-01T12:00:00Z",
            "status": "online",
            "backend": "coral",
            "config": {
                "rtsp_url": "rtsp://...",
                "model_path": "/path/to/model"
            },
            "system_metrics": {
                "cpu_percent": 45.2,
                "memory_percent": 62.1,
                "disk_usage": {...},
                "uptime_hours": 24.5
            },
            // Legacy top-level fields
            "cpu_percent": 45.2,
            "memory_percent": 62.1,
            ...
        }
    
    Side Effects:
        - Publishes to MQTT topic 'system/telemetry' (QoS 1, not retained)
        - Creates daemon Timer thread for next publication
        - Logs telemetry payload and any errors
    
    Thread Safety:
        This function is called from Timer threads. Each invocation creates
        a new Timer for the next cycle. No shared state is modified.
    
    Note:
        Messages are not retained to avoid stale data. Consumers should
        track the last received timestamp to detect offline cameras.
        
        The dual metric format (system_metrics object + top-level fields)
        maintains compatibility with older telemetry consumers while
        providing a cleaner structure for new consumers.
    """
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
    global active_timer
    active_timer = threading.Timer(TELEMETRY_INT, publish_telemetry)
    active_timer.daemon = True
    active_timer.start()

# ─────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────
def main():
    """Main entry point for the telemetry service.
    
    Starts the telemetry publication cycle and keeps the process running
    to allow scheduled Timer threads to execute. Handles graceful shutdown
    on keyboard interrupt.
    
    The main thread simply sleeps in a loop while Timer threads handle
    the actual telemetry publications. This design allows the service to
    run with minimal resource usage.
    
    Side Effects:
        - Initiates first telemetry publication
        - Blocks in sleep loop until interrupted
        - Stops MQTT network loop on exit
        - Disconnects from MQTT broker on exit
    
    Signal Handling:
        - KeyboardInterrupt (Ctrl+C): Graceful shutdown
        - SIGTERM: Not explicitly handled, causes immediate exit
    
    Note:
        The infinite sleep loop is intentional. The actual work happens
        in Timer threads spawned by publish_telemetry(). The main thread
        just needs to stay alive to prevent process termination.
    """
    # Initial telemetry
    publish_telemetry()

    # Keep running to allow scheduled publishes
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Telemetry service interrupted, shutting down...")
    finally:
        # Cancel active timer if running
        global active_timer
        if active_timer and active_timer.is_alive():
            active_timer.cancel()
        client.loop_stop()
        client.disconnect()

def shutdown_telemetry():
    """Shutdown telemetry service and cleanup resources."""
    global active_timer, _shutdown_flag
    
    # Set shutdown flag to prevent further MQTT connections
    _shutdown_flag = True
    
    # Cancel active timer
    if active_timer and active_timer.is_alive():
        active_timer.cancel()
        active_timer = None
    
    # Disconnect MQTT client
    try:
        client.loop_stop()
        client.disconnect()
    except:
        pass  # Ignore errors during shutdown
    
    # Stop any background threads gracefully
    import threading
    for thread in threading.enumerate():
        if thread != threading.current_thread() and thread.name.startswith('Thread-'):
            if hasattr(thread, 'cancel'):
                thread.cancel()

if __name__ == "__main__":
    main()

