#!/bin/bash
# Script to fix Docker permissions for the current user

set -e

echo "Docker Permission Fix Script"
echo "============================"
echo

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please run this script as a regular user, not as root."
   exit 1
fi

# Check if docker group exists
if ! getent group docker > /dev/null 2>&1; then
    echo "Docker group doesn't exist. Creating it requires sudo..."
    echo "Run: sudo groupadd docker"
    exit 1
fi

# Check current user's groups
echo "Current user: $USER"
echo "Current groups: $(groups)"
echo

# Check if already in docker group
if groups | grep -q '\bdocker\b'; then
    echo "✓ You are already in the docker group!"
    echo
    echo "If you're still getting permission errors, try:"
    echo "1. Log out and log back in"
    echo "2. Run: newgrp docker"
    echo "3. Restart the Docker daemon: sudo systemctl restart docker"
else
    echo "✗ You are NOT in the docker group."
    echo
    echo "To fix this, run the following command:"
    echo "  sudo usermod -aG docker $USER"
    echo
    echo "Then EITHER:"
    echo "  1. Log out and log back in (recommended)"
    echo "  OR"
    echo "  2. Run: newgrp docker (temporary fix for current session)"
    echo
    echo "After that, verify with: docker ps"
fi

echo
echo "Additional checks:"
echo "=================="

# Check Docker socket permissions
echo -n "Docker socket permissions: "
ls -l /var/run/docker.sock

# Check if Docker service is running
echo -n "Docker service status: "
if systemctl is-active --quiet docker; then
    echo "✓ Running"
else
    echo "✗ Not running (run: sudo systemctl start docker)"
fi

# Check for nvidia-docker if GPU tests are needed
echo -n "NVIDIA Docker runtime: "
if docker info 2>/dev/null | grep -q nvidia; then
    echo "✓ Installed"
elif command -v nvidia-docker &> /dev/null; then
    echo "✓ nvidia-docker command found"
else
    echo "✗ Not installed (needed for GPU tests)"
    echo "  Install with: sudo apt-get install nvidia-docker2"
fi

echo
echo "For more information, see:"
echo "https://docs.docker.com/engine/install/linux-postinstall/"