#!/bin/bash
set -e

# Start D-Bus daemon
echo "Starting D-Bus daemon..."
mkdir -p /var/run/dbus
dbus-daemon --system --fork

# Start Avahi daemon for mDNS support
echo "Starting Avahi daemon for mDNS support..."
avahi-daemon --daemonize --no-drop-root

# Give daemons time to start
sleep 2

# Execute the main command
exec "$@"