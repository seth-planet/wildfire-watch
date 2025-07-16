#!/bin/sh
set -e

echo "Starting web interface directly..."

# Set Python path
export PYTHONPATH=/:/app:$PYTHONPATH

# Start the web interface
exec python3.12 -m web_interface