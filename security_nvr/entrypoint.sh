#!/bin/bash
# ===================================================================
# Entrypoint Script - Security NVR Service
# Auto-configures Frigate based on hardware and environment
# ===================================================================
set -e

echo "=================================================="
echo "Wildfire Watch - Security NVR Service"
echo "=================================================="

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Detect and configure hardware
log "Detecting hardware..."
python3 /scripts/hardware_detector.py --export
HARDWARE_CONFIG="/tmp/hardware_config.json"

if [ -f "$HARDWARE_CONFIG" ]; then
    log "Hardware detection complete"
    
    # Extract configuration values
    export DETECTOR_TYPE=$(jq -r '.recommended.detector_type' "$HARDWARE_CONFIG")
    export DETECTOR_DEVICE=$(jq -r '.recommended.detector_device' "$HARDWARE_CONFIG")
    export MODEL_PATH=$(jq -r '.recommended.model_path' "$HARDWARE_CONFIG")
    export HWACCEL_ARGS=$(jq -r '.recommended.hwaccel_args | @json' "$HARDWARE_CONFIG")
    export RECORD_CODEC=$(jq -r '.recommended.record_codec' "$HARDWARE_CONFIG")
    export RECORD_PRESET=$(jq -r '.recommended.record_preset' "$HARDWARE_CONFIG")
    export RECORD_QUALITY=$(jq -r '.recommended.record_quality' "$HARDWARE_CONFIG")
else
    log "WARNING: Hardware detection failed, using defaults"
    export DETECTOR_TYPE="cpu"
    export DETECTOR_DEVICE="0"
    export MODEL_PATH="/models/wildfire/wildfire_cpu.tflite"
    export HWACCEL_ARGS="[]"
    export RECORD_CODEC="copy"
    export RECORD_PRESET="fast"
    export RECORD_QUALITY="23"
fi

# Check for already mounted storage
log "Checking for available storage..."
# Look for already mounted drives or use the configured path
if [ -d "$USB_MOUNT_PATH" ] && [ -w "$USB_MOUNT_PATH" ]; then
    log "Using existing mount at $USB_MOUNT_PATH"
    # Ensure Frigate directory structure exists
    mkdir -p "$USB_MOUNT_PATH"/{recordings,clips,exports,logs} 2>/dev/null || true
else
    log "No writable storage at $USB_MOUNT_PATH, using local storage"
    # Use Frigate's default data directory which should be writable
    export USB_MOUNT_PATH="/tmp/frigate"
    mkdir -p "$USB_MOUNT_PATH"/{recordings,clips,exports,logs} 2>/dev/null || true
fi

# Get storage stats for the mount path
if [ -d "$USB_MOUNT_PATH" ]; then
    STORAGE_STATS=$(df -h "$USB_MOUNT_PATH" 2>/dev/null | tail -n 1 || echo "")
    if [ -n "$STORAGE_STATS" ]; then
        log "Storage statistics:"
        echo "$STORAGE_STATS"
    fi
fi

# Configure power mode
case "${POWER_MODE}" in
    "performance")
        log "Power mode: Performance"
        export DETECTION_FPS="10"
        export RECORD_QUALITY="21"
        ;;
    "powersave")
        log "Power mode: Power Save"
        export DETECTION_FPS="2"
        export RECORD_QUALITY="28"
        ;;
    *)
        log "Power mode: Balanced"
        # Use defaults
        ;;
esac

# Generate Frigate configuration
log "Generating Frigate configuration..."
if ! python3 /scripts/camera_manager.py generate-config 2>/dev/null; then
    log "WARNING: Camera manager failed, generating minimal config"
    # Create a minimal Frigate config for testing/development
    cat > /config/frigate.yml << EOF
mqtt:
  enabled: true
  host: ${FRIGATE_MQTT_HOST:-mqtt_broker}
  port: ${FRIGATE_MQTT_PORT:-1883}
  topic_prefix: frigate
  client_id: frigate
  stats_interval: 30

database:
  path: ${USB_MOUNT_PATH}/frigate.db

detectors:
  cpu1:
    type: cpu

cameras: {}

record:
  enabled: false
  path: ${USB_MOUNT_PATH}/recordings

snapshots:
  enabled: false
  path: ${USB_MOUNT_PATH}/snapshots

clips:
  enabled: false
  path: ${USB_MOUNT_PATH}/clips

exports:
  path: ${USB_MOUNT_PATH}/exports

objects:
  track:
    - fire
    - smoke

logger:
  default: ${LOG_LEVEL:-info}

ui:
  live_mode: mse
  timezone: UTC

birdseye:
  enabled: false
EOF
fi

# Validate configuration
if [ -f "/config/frigate.yml" ]; then
    log "Configuration ready"
else
    log "ERROR: Failed to create configuration"
    exit 1
fi

# Skip storage monitor since we're not mounting
log "Storage monitoring disabled (using existing mounts only)"
STORAGE_MONITOR_PID=""

# Start power manager in background if available
if [ -f "/scripts/power_manager.py" ]; then
    log "Starting power manager..."
    python3 /scripts/power_manager.py &
    POWER_MANAGER_PID=$!
else
    log "Power manager not available, skipping..."
    POWER_MANAGER_PID=""
fi

# Function to cleanup on exit
cleanup() {
    log "Shutting down services..."
    [ -n "$STORAGE_MONITOR_PID" ] && kill $STORAGE_MONITOR_PID 2>/dev/null || true
    [ -n "$POWER_MANAGER_PID" ] && kill $POWER_MANAGER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Log startup summary
log "=================================================="
log "Configuration Summary:"
log "  Detector: ${DETECTOR_TYPE}"
log "  Model: $(basename ${MODEL_PATH})"
log "  Hardware Accel: ${RECORD_CODEC}"
log "  Power Mode: ${POWER_MODE}"
log "  Storage: ${USB_MOUNT_PATH}"
log "  MQTT: ${FRIGATE_MQTT_HOST}:${FRIGATE_MQTT_PORT}"
log "=================================================="

# Start Frigate using the base image's init system
log "Starting Frigate NVR..."
# Call the original entrypoint from the base image
exec /init
