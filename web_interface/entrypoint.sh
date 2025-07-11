#!/bin/sh
set -e

echo "Starting web_interface entrypoint..."

# Set Python path to include parent directory for imports
export PYTHONPATH=/:/app:$PYTHONPATH

# Skip D-Bus and Avahi for now - they're causing issues
# TODO: Fix Avahi startup in container environment

# Execute the command passed to the container
echo "Starting web interface..."
echo "Command to execute: $@"

# Check if we can find Python
which python || echo "python not found"
which python3.12 || echo "python3.12 not found"
which python3 || echo "python3 not found"

# Try to execute
exec "$@"