#!/bin/bash
#
# Build custom Frigate with YOLO EdgeTPU support
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="frigate-yolo"
IMAGE_TAG="dev"

echo "Building custom Frigate with YOLO EdgeTPU support..."
echo "=============================================="

# Step 1: Check if required files exist
echo "1. Checking required files..."
required_files=(
    "yolo_edgetpu.py"
    "Dockerfile.yolo"
    "fire_labels.txt"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$SCRIPT_DIR/$file" ]; then
        echo "ERROR: Required file missing: $file"
        exit 1
    fi
done
echo "✓ All required files present"

# Step 2: Copy YOLO models if they exist
echo -e "\n2. Checking for YOLO models..."
MODEL_DIR="$SCRIPT_DIR/../../frigate_models"
if [ -d "$MODEL_DIR" ]; then
    echo "✓ Found models directory: $MODEL_DIR"
    # List available models
    echo "Available models:"
    find "$MODEL_DIR" -name "*.tflite" -type f | while read -r model; do
        echo "  - $(basename "$model")"
    done
else
    echo "⚠ Models directory not found. You'll need to mount models when running container."
fi

# Step 3: Build Docker image
echo -e "\n3. Building Docker image..."
cd "$SCRIPT_DIR"
# Try patch-based approach first
if [ -f "Dockerfile.patch" ]; then
    echo "Using patch-based Dockerfile..."
    docker build -f Dockerfile.patch -t "${IMAGE_NAME}:${IMAGE_TAG}" .
else
    docker build -f Dockerfile.yolo -t "${IMAGE_NAME}:${IMAGE_TAG}" .
fi

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "✗ Docker build failed"
    exit 1
fi

# Step 4: Tag image
echo -e "\n4. Tagging image..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_NAME}:latest"
echo "✓ Tagged as ${IMAGE_NAME}:latest"

# Step 5: Verify image
echo -e "\n5. Verifying image..."
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" python3 -c "
import sys
sys.path.insert(0, '/opt/frigate')
try:
    from frigate.detectors.plugins import yolo_edgetpu
    print('✓ YOLO EdgeTPU plugin is available in image')
except ImportError as e:
    print(f'✗ Failed to import YOLO plugin: {e}')
    sys.exit(1)
"

# Step 6: Create test configuration
echo -e "\n6. Creating test configuration..."
TEST_CONFIG_DIR="$SCRIPT_DIR/test_config"
mkdir -p "$TEST_CONFIG_DIR"

cat > "$TEST_CONFIG_DIR/config.yml" << EOF
mqtt:
  enabled: true
  host: localhost
  port: 1883
  topic_prefix: frigate
  client_id: frigate_yolo_test

detectors:
  coral:
    type: yolo_edgetpu
    device: usb:0
    conf_threshold: 0.25
    iou_threshold: 0.45
    max_detections: 100

model:
  path: /models/yolo8l_fire_640x640_frigate_edgetpu.tflite
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 640
  height: 640
  labelmap_path: /models/fire_labels.txt

cameras:
  test_cam:
    enabled: false
    ffmpeg:
      inputs:
        - path: rtsp://127.0.0.1:554/null
          roles:
            - detect

logger:
  default: info
  logs:
    frigate.detectors.plugins.yolo_edgetpu: debug
EOF

echo "✓ Test configuration created at: $TEST_CONFIG_DIR/config.yml"

# Step 7: Display usage instructions
echo -e "\n=============================================="
echo "Build completed successfully!"
echo ""
echo "To run the custom Frigate:"
echo ""
echo "docker run -d \\"
echo "  --name frigate-yolo \\"
echo "  --restart unless-stopped \\"
echo "  --network host \\"
echo "  --privileged \\"
echo "  -v /dev/bus/usb:/dev/bus/usb \\"
echo "  -v $TEST_CONFIG_DIR:/config \\"
echo "  -v $MODEL_DIR:/models \\"
echo "  -v /dev/shm:/dev/shm \\"
echo "  ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To check logs:"
echo "docker logs -f frigate-yolo"
echo ""
echo "To stop and remove:"
echo "docker stop frigate-yolo && docker rm frigate-yolo"
echo "=============================================="