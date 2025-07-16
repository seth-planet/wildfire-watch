#!/bin/bash
# Build Docker images required for tests

set -e

echo "Building Docker images for tests..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Build camera_detector image
if [ -f "camera_detector/Dockerfile" ]; then
    echo "Building camera_detector image..."
    docker build -t wildfire-watch/camera_detector:latest -f camera_detector/Dockerfile .
    echo "✓ camera_detector image built"
fi

# Build fire_consensus image
if [ -f "fire_consensus/Dockerfile" ]; then
    echo "Building fire_consensus image..."
    docker build -t wildfire-watch/fire_consensus:latest -f fire_consensus/Dockerfile .
    echo "✓ fire_consensus image built"
fi

# Build gpio_trigger image
if [ -f "gpio_trigger/Dockerfile" ]; then
    echo "Building gpio_trigger image..."
    # GPIO trigger needs platform arg
    docker build -t wildfire-watch/gpio_trigger:latest -f gpio_trigger/Dockerfile --build-arg PLATFORM=amd64 .
    echo "✓ gpio_trigger image built"
fi

# Pull required base images
echo "Pulling base images..."
docker pull eclipse-mosquitto:2.0 || true
docker pull eclipse-mosquitto:latest || true
docker pull ghcr.io/blakeblackshear/frigate:stable || true
docker pull python:3.12-slim || true

echo "✓ All images ready for testing"