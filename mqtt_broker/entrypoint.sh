#!/bin/sh
# ===================================================================
# Entrypoint Script - MQTT Broker with mDNS Support
# ===================================================================
set -e

# Start D-Bus daemon (required for Avahi)
mkdir -p /var/run/dbus
if [ ! -f /var/run/dbus/pid ]; then
    dbus-daemon --system --fork
fi

# Wait for D-Bus to be ready
sleep 1

# Start Avahi mDNS daemon for service discovery
# This allows devices to find the broker at mqtt_broker.local
if command -v avahi-daemon >/dev/null 2>&1; then
    echo "Starting Avahi mDNS daemon..."
    avahi-daemon --no-drop-root --daemonize --no-chroot
    
    # Wait for Avahi to start
    sleep 2
    
    # Publish MQTT service via mDNS
    if command -v avahi-publish-service >/dev/null 2>&1; then
        echo "Publishing MQTT service via mDNS..."
        avahi-publish-service "Wildfire MQTT Broker" _mqtt._tcp 1883 "Wildfire Watch MQTT Broker" &
        avahi-publish-service "Wildfire MQTT-TLS Broker" _secure-mqtt._tcp 8883 "Wildfire Watch Secure MQTT" &
    fi
else
    echo "Warning: Avahi not available, mDNS discovery disabled"
fi

# Create required directories if they don't exist
mkdir -p /mosquitto/data /mosquitto/log

# Check for TLS certificates and configuration
if [ -f "/mnt/data/certs/ca.crt" ] && [ -f "/mnt/data/certs/server.crt" ] && [ -f "/mnt/data/certs/server.key" ]; then
    echo "TLS certificates found - secure MQTT will be available on port 8883"
    
    # If MQTT_TLS is enabled, use TLS configuration
    if [ "${MQTT_TLS}" = "true" ]; then
        echo "MQTT_TLS=true - Using TLS configuration"
        if [ -f "/mosquitto/config/mosquitto_tls.conf" ]; then
            export MOSQUITTO_CONF="/mosquitto/config/mosquitto_tls.conf"
        fi
    fi
else
    echo "Warning: TLS certificates not found at /mnt/data/certs/"
    echo "Only plain MQTT on port 1883 will be available"
fi

# Fix permissions
chown -R mosquitto:mosquitto /mosquitto/data /mosquitto/log

# Log startup information
echo "=================================================="
echo "Wildfire Watch MQTT Broker Starting"
echo "Plain MQTT: Port 1883"
echo "Secure MQTT: Port 8883 (if certs available)"
echo "WebSocket: Port 9001"
echo "Hostname: $(hostname)"
echo "=================================================="

# Execute mosquitto with the provided configuration
# If MQTT_TLS is true and TLS config exists, use it
if [ "${MQTT_TLS}" = "true" ] && [ -f "/mosquitto/config/mosquitto_tls.conf" ]; then
    echo "Starting Mosquitto with TLS configuration..."
    exec /usr/sbin/mosquitto -c /mosquitto/config/mosquitto_tls.conf
else
    exec "$@"
fi
