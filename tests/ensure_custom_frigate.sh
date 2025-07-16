#!/bin/bash
#
# Ensure custom Frigate image is available for testing
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRIGATE_PLUGIN_DIR="$PROJECT_ROOT/converted_models/frigate_yolo_plugin"

# Check if custom Frigate image exists
if docker images "frigate-yolo:dev" --format "{{.Repository}}:{{.Tag}}" | grep -q "frigate-yolo:dev"; then
    echo "✓ Custom Frigate image already exists"
    exit 0
fi

echo "Custom Frigate image not found. Building..."

# Check if build files exist
if [ ! -d "$FRIGATE_PLUGIN_DIR" ]; then
    echo "ERROR: Frigate plugin directory not found: $FRIGATE_PLUGIN_DIR"
    exit 1
fi

# Build the custom image
cd "$FRIGATE_PLUGIN_DIR"
if [ -f "build_custom_frigate.sh" ]; then
    echo "Running build script..."
    ./build_custom_frigate.sh
else
    echo "Build script not found, building manually..."
    docker build -f Dockerfile.patch -t frigate-yolo:dev .
fi

# Verify the image was built
if docker images "frigate-yolo:dev" --format "{{.Repository}}:{{.Tag}}" | grep -q "frigate-yolo:dev"; then
    echo "✓ Custom Frigate image built successfully"
else
    echo "✗ Failed to build custom Frigate image"
    exit 1
fi