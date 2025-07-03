#!/bin/bash
# Install Hailo HEF models for Frigate

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models/wildfire"
HEF_DIR="${PROJECT_ROOT}/converted_models/hailo_qat_output"

echo "=== Installing Hailo Models for Frigate ==="
echo

# Create models directory if it doesn't exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "Creating models directory: $MODELS_DIR"
    mkdir -p "$MODELS_DIR"
fi

# Check if HEF files exist
if [ ! -d "$HEF_DIR" ]; then
    echo "Error: HEF directory not found: $HEF_DIR"
    echo "Please run the Hailo conversion first."
    exit 1
fi

# Copy HEF files
echo "Copying HEF models..."
for hef_file in "$HEF_DIR"/*.hef; do
    if [ -f "$hef_file" ]; then
        filename=$(basename "$hef_file")
        echo "  - $filename"
        cp "$hef_file" "$MODELS_DIR/"
        
        # Also copy metadata
        json_file="${hef_file%.hef}.json"
        if [ -f "$json_file" ]; then
            cp "$json_file" "$MODELS_DIR/"
        fi
    fi
done

# Create a model index file
echo
echo "Creating model index..."
cat > "$MODELS_DIR/hailo_models.json" << EOF
{
  "models": {
    "yolo8l_fire_hailo8": {
      "path": "/models/wildfire/yolo8l_fire_640x640_hailo8_qat.hef",
      "target": "hailo8",
      "batch_size": 8,
      "input_size": [640, 640],
      "tops": 26,
      "description": "YOLOv8L fire detection optimized for Hailo-8 (26 TOPS)"
    },
    "yolo8l_fire_hailo8l": {
      "path": "/models/wildfire/yolo8l_fire_640x640_hailo8l_qat.hef",
      "target": "hailo8l",
      "batch_size": 8,
      "input_size": [640, 640],
      "tops": 13,
      "description": "YOLOv8L fire detection optimized for Hailo-8L M.2 (13 TOPS)"
    }
  },
  "default_model": "yolo8l_fire_hailo8l",
  "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# List installed models
echo
echo "Installed models:"
ls -lh "$MODELS_DIR"/*.hef 2>/dev/null || echo "No HEF files found"

echo
echo "=== Installation Complete ==="
echo
echo "To use Hailo models with Frigate:"
echo "1. Set FRIGATE_VARIANT=-h8l in your .env file"
echo "2. Set FRIGATE_DETECTOR=hailo8l"
echo "3. Ensure /dev/hailo0 is mapped in docker-compose.yml"
echo "4. Restart the security_nvr service"
echo
echo "Example .env configuration:"
echo "  FRIGATE_VARIANT=-h8l"
echo "  FRIGATE_DETECTOR=hailo8l"
echo "  MODEL_PATH=/models/wildfire/yolo8l_fire_640x640_hailo8l_qat.hef"