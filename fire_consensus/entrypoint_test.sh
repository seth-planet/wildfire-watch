#!/bin/sh
# ===================================================================
# Test Entrypoint Script - Fire Consensus Service
# Simplified version without D-Bus/Avahi for integration testing
# ===================================================================
set -e

# Log startup info
echo "=================================================="
echo "Wildfire Watch - Fire Consensus Service (TEST MODE)"
echo "Node ID: ${NODE_ID:-test-node}"
echo "Consensus Threshold: ${CONSENSUS_THRESHOLD:-2}"
echo "Detection Window: ${CAMERA_WINDOW:-10}s"
echo "MQTT Broker: ${MQTT_BROKER:-mqtt_broker}"
echo "MQTT Port: ${MQTT_PORT:-1883}"
echo "=================================================="

# Execute the main command
exec "$@"