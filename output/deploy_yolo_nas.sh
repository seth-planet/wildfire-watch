#!/bin/bash
# YOLO-NAS Deployment Script

set -e

echo "Deploying YOLO-NAS model to Frigate..."

# Check if Frigate directory provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/frigate/directory"
    exit 1
fi

FRIGATE_DIR="$1"
MODELS_DIR="$FRIGATE_DIR/models"

# Create models directory
mkdir -p "$MODELS_DIR"

# Copy model
cp "../output/yolo_nas_s_wildfire.onnx" "$MODELS_DIR/"

echo "✓ Model deployed to Frigate"
echo "Add configuration from ../output/frigate_yolo_nas_config.yml to your Frigate config.yml"
