#!/bin/sh
# ===================================================================
# Entrypoint Script - GPIO Trigger Service
# Starts Avahi daemon for mDNS support before running trigger service
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

# Log startup info
echo "=================================================="
echo "Wildfire Watch - GPIO Trigger Service"
echo "Node ID: ${BALENA_DEVICE_UUID:-$(hostname)}"
echo "GPIO Simulation: ${GPIO_SIMULATION:-false}"
echo "Max Engine Runtime: ${MAX_ENGINE_RUNTIME:-1800}s"
echo "MQTT Broker: ${MQTT_BROKER:-mqtt_broker}"
echo "=================================================="

# Execute the main command
exec "$@"