#!/bin/sh
set -e

echo "Starting web_interface entrypoint..."
echo "Container starting with PID $$"
echo "PATH: $PATH"
echo "Working directory: $(pwd)"
echo "Contents of /usr/local/bin: $(ls -la /usr/local/bin/)"

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

# Check if start.sh exists
if [ ! -f "/usr/local/bin/start.sh" ]; then
    echo "ERROR: start.sh not found at /usr/local/bin/start.sh"
    exit 1
fi

# Try to execute
exec "$@"