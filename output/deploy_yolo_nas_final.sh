#!/bin/bash
# YOLO-NAS Final Deployment Script

set -e

echo "YOLO-NAS Wildfire Model Deployment"
echo "=================================="
echo "Model: yolo_nas_s_wildfire_final.onnx"
echo "Size: 51.5 MB"
echo "Classes: 32 (including Fire at class 26)"
echo ""

# Check if Frigate directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/frigate/directory"
    echo "Example: $0 /opt/frigate"
    echo "Example: $0 ./security_nvr"
    exit 1
fi

FRIGATE_DIR="$1"
MODELS_DIR="$FRIGATE_DIR/models"

# Create models directory
mkdir -p "$MODELS_DIR"

# Copy model
echo "Copying YOLO-NAS model..."
cp "../output/yolo_nas_s_wildfire_final.onnx" "$MODELS_DIR/"
echo "✓ Model copied successfully"

echo ""
echo "Next steps:"
echo "1. Add configuration from ../output/frigate_yolo_nas_final_config.yml to Frigate config.yml"
echo "2. Restart Frigate: docker restart frigate"
echo "3. Check for fire detection (class 26) in Frigate events"
echo ""
echo "Model deployed successfully!"
