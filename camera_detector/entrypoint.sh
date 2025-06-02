#!/bin/sh
# ===================================================================
# Entrypoint Script - Camera Detector Service
# Starts required services and runs camera discovery
# ===================================================================
set -e

# Start D-Bus daemon if not running (required for Avahi)
if [ ! -f /var/run/dbus/pid ]; then
    echo "Starting D-Bus daemon..."
    dbus-daemon --system --fork || true
fi

# Start Avahi daemon for mDNS discovery
if command -v avahi-daemon >/dev/null 2>&1; then
    echo "Starting Avahi daemon for mDNS support..."
    # Create required directories
    mkdir -p /var/run/avahi-daemon
    
    # Start Avahi (no-drop-root since we're in container)
    avahi-daemon --no-drop-root --daemonize --no-chroot || true
    
    # Give Avahi time to start
    sleep 2
    
    echo "Avahi daemon started"
else
    echo "Warning: Avahi not found, mDNS discovery disabled"
fi

# Set capabilities for network scanning if needed
if command -v setcap >/dev/null 2>&1; then
    setcap cap_net_raw+ep /usr/bin/python3.11 || true
    setcap cap_net_admin+ep /usr/bin/python3.11 || true
fi

# Log startup info
echo "=================================================="
echo "Wildfire Watch - Camera Detector Service"
echo "Node ID: ${BALENA_DEVICE_UUID:-$(hostname)}"
echo "Discovery Interval: ${DISCOVERY_INTERVAL:-300}s"
echo "MAC Tracking: ${MAC_TRACKING_ENABLED:-true}"
echo "MQTT Broker: ${MQTT_BROKER:-mqtt_broker}"
echo "=================================================="

# Execute the main command
exec "$@"
