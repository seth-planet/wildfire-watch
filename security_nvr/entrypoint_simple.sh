#!/bin/bash
# Simple entrypoint for Frigate
set -e

echo "=================================================="
echo "Wildfire Watch - Security NVR Service (Frigate)"
echo "=================================================="

# Create necessary directories
mkdir -p /config /media/frigate/{recordings,clips,exports}

# Check if config exists, if not create a minimal one
if [ ! -f "/config/config.yml" ]; then
    echo "Creating minimal Frigate config..."
    cat > /config/config.yml << EOF
mqtt:
  enabled: true
  host: ${FRIGATE_MQTT_HOST:-mqtt_broker}
  port: ${FRIGATE_MQTT_PORT:-1883}
  topic_prefix: frigate
  client_id: frigate
  stats_interval: 15

detectors:
  cpu1:
    type: cpu

cameras: {}

record:
  enabled: false

snapshots:
  enabled: false

objects:
  track:
    - fire
    - smoke

logger:
  default: ${LOG_LEVEL:-info}

# Required for web UI
ui:
  live_mode: mse
  timezone: UTC

# Disable features we don't need for testing
birdseye:
  enabled: false

# Add a go2rtc config for WebRTC
go2rtc:
  streams: {}
EOF
fi

echo "Installing socat for API proxy..."
apt-get update -qq && apt-get install -y -qq socat

echo "Starting API proxy..."
socat TCP-LISTEN:5000,fork,reuseaddr TCP:127.0.0.1:5001 &

echo "Starting Frigate..."
exec python3 -m frigate