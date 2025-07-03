#!/bin/bash
# Deploy Coral TPU model to Frigate

FRIGATE_CONFIG_DIR="/config"
FRIGATE_MODELS_DIR="/models"

echo "Deploying Coral TPU model to Frigate..."

# Copy model
if [ -f "yolov8n_320_int8_edgetpu.tflite" ]; then
    echo "Copying Edge TPU model..."
    cp yolov8n_320_int8_edgetpu.tflite "$FRIGATE_MODELS_DIR/"
else
    echo "ERROR: Edge TPU model not found"
    exit 1
fi

# Update Frigate config
echo "Updating Frigate configuration..."
cat frigate_coral_config.yml >> "$FRIGATE_CONFIG_DIR/config.yml"

echo "✓ Deployment complete"
echo "Restart Frigate to apply changes"
