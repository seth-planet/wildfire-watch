#!/bin/bash
# Wildfire Watch Model Deployment Script

set -e

echo "Wildfire Watch YOLO Model Deployment"
echo "===================================="

# Check if Frigate directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/frigate/directory"
    echo "Example: $0 /opt/frigate"
    exit 1
fi

FRIGATE_DIR="$1"
MODELS_DIR="$FRIGATE_DIR/models"
CONFIG_FILE="$FRIGATE_DIR/config.yml"

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Copy models
echo "Copying YOLO models..."
cp "converted_models/640x640/yolo8l_wildfire_640x640.onnx" "$MODELS_DIR/"
cp "converted_models/320x320/yolo8l_wildfire_320x320.onnx" "$MODELS_DIR/"

echo "✓ Models copied successfully"

echo ""
echo "Next steps:"
echo "1. Update your Frigate config.yml with the configuration from:"
echo "   output/FRIGATE_DEPLOYMENT_GUIDE.md"
echo "2. Restart Frigate: docker restart frigate"
echo ""
echo "Models available in $MODELS_DIR:"
ls -la "$MODELS_DIR"/yolo8l_wildfire_*.onnx
