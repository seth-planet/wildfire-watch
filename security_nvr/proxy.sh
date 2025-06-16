#!/bin/bash
# Simple reverse proxy to expose Frigate API
echo "Starting reverse proxy for Frigate API..."

# Install socat if not available
which socat > /dev/null || apt-get update -qq && apt-get install -y -qq socat

# Forward port 5000 to internal port 5001
socat TCP-LISTEN:5000,fork,reuseaddr TCP:127.0.0.1:5001 &
PROXY_PID=$!

# Start Frigate
exec "$@"