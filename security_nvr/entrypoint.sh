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

# Mount USB storage
log "Setting up storage..."
python3 /scripts/usb_manager.py mount

# Get storage stats
STORAGE_STATS=$(python3 /scripts/usb_manager.py stats)
if [ $? -eq 0 ]; then
    log "Storage mounted successfully"
    echo "$STORAGE_STATS"
else
    log "WARNING: No USB storage detected, using local storage"
    mkdir -p /media/frigate/{recordings,clips,exports,logs}
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
python3 /scripts/camera_manager.py generate-config

# Validate configuration
if [ -f "/config/frigate.yml" ]; then
    log "Configuration generated successfully"
else
    log "ERROR: Failed to generate configuration"
    exit 1
fi

# Start USB monitor in background
log "Starting USB storage monitor..."
python3 /scripts/usb_manager.py monitor &
USB_MONITOR_PID=$!

# Start power manager in background
log "Starting power manager..."
python3 /scripts/power_manager.py &
POWER_MANAGER_PID=$!

# Function to cleanup on exit
cleanup() {
    log "Shutting down services..."
    kill $USB_MONITOR_PID 2>/dev/null || true
    kill $POWER_MANAGER_PID 2>/dev/null || true
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

# Start Frigate
log "Starting Frigate NVR..."
exec python3 -m frigate
