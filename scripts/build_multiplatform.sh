#!/usr/bin/env bash
set -euo pipefail

# Build images for multiple platforms
PLATFORMS="linux/amd64,linux/arm64"
VERSION="${VERSION:-latest}"

echo "Building Wildfire Watch images for platforms: $PLATFORMS"

# Setup buildx
docker buildx create --name wildfire-builder --use || true
docker buildx inspect --bootstrap

# Build each service
for service in mqtt_broker camera_detector security_nvr fire_consensus gpio_trigger cam_telemetry; do
    echo "Building $service..."
    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "wildfire-watch/$service:$VERSION" \
        --push \
        "./$service"
done

echo "Build complete!"
