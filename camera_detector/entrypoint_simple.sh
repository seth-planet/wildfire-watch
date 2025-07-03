#!/bin/sh
# Simple entrypoint without dbus/avahi
set -e

echo "=================================================="
echo "Wildfire Watch - Camera Detector Service"
echo "Discovery Interval: ${DISCOVERY_INTERVAL:-300}s"
echo "MQTT Broker: ${MQTT_BROKER:-mqtt_broker}"
echo "=================================================="

# Execute the main command
exec "$@"