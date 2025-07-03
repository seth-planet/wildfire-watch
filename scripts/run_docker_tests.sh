#!/bin/bash
# Script to run Docker-related tests with proper group permissions

set -e

echo "Docker Test Runner"
echo "=================="
echo

# Check if user is in docker group
if id -nG "$USER" | grep -qw "docker"; then
    echo "✓ User '$USER' is in docker group"
else
    echo "✗ User '$USER' is NOT in docker group"
    echo "  Run: sudo usermod -aG docker $USER"
    echo "  Then log out and back in"
    exit 1
fi

# Check if current session has docker group
if groups | grep -qw "docker"; then
    echo "✓ Current session has docker group"
    echo
    echo "Running tests directly..."
    python3.12 -m pytest tests/test_hardware_integration.py -v -k "docker" "$@"
else
    echo "✗ Current session doesn't have docker group"
    echo "  (User was added to group but session needs refresh)"
    echo
    echo "Running tests in new shell with docker group..."
    echo
    
    # Run tests in a new shell with docker group
    bash -c 'newgrp docker && python3.12 -m pytest tests/test_hardware_integration.py -v -k "docker" "$@"' -- "$@"
fi