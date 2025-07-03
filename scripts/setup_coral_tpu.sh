#!/bin/bash
# Setup script for Coral TPU development environment

echo "=== Coral TPU Setup Script ==="
echo

# Check if running on the right architecture
ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" && "$ARCH" != "aarch64" ]]; then
    echo "ERROR: Unsupported architecture: $ARCH"
    echo "Coral TPU supports x86_64 and aarch64 (ARM64) only"
    exit 1
fi

# Check for Python 3.8
if ! command -v python3.8 &> /dev/null; then
    echo "ERROR: Python 3.8 not found"
    echo "Install with: sudo apt install python3.8 python3.8-pip python3.8-dev"
    exit 1
fi

echo "✓ Python 3.8 found"

# Install system dependencies
echo
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    libusb-1.0-0 \
    libc++1 \
    libc++abi1 \
    libunwind8 \
    libgcc1

# Add Coral repository
echo
echo "Adding Coral repository..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

# Install Edge TPU runtime
echo
echo "Installing Edge TPU runtime..."
# Use standard runtime (not max frequency) for stability
sudo apt-get install -y libedgetpu1-std

# Install Edge TPU compiler
echo
echo "Installing Edge TPU compiler..."
sudo apt-get install -y edgetpu-compiler

# Install Python packages for Python 3.8
echo
echo "Installing Python 3.8 packages..."

# Install tflite-runtime for Python 3.8
echo "Installing tflite-runtime..."
python3.8 -m pip install --upgrade pip
python3.8 -m pip install tflite-runtime

# Install pycoral for Python 3.8
echo "Installing pycoral..."
if [[ "$ARCH" == "x86_64" ]]; then
    python3.8 -m pip install pycoral
elif [[ "$ARCH" == "aarch64" ]]; then
    # For ARM64, we might need the specific wheel
    python3.8 -m pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_aarch64.whl
fi

# Install additional dependencies
python3.8 -m pip install numpy pillow opencv-python-headless

# Set up udev rules for USB Coral
echo
echo "Setting up USB permissions..."
sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"1a6e\", MODE=\"0666\"" > /etc/udev/rules.d/99-coral-edgetpu.rules'
sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"18d1\", MODE=\"0666\"" >> /etc/udev/rules.d/99-coral-edgetpu.rules'
sudo udevadm control --reload-rules && sudo udevadm trigger

# Check for Coral devices
echo
echo "Checking for Coral devices..."
python3.8 -c "
try:
    from pycoral.utils.edgetpu import list_edge_tpus
    tpus = list_edge_tpus()
    if tpus:
        print(f'✓ Found {len(tpus)} Coral TPU(s):')
        for i, tpu in enumerate(tpus):
            print(f'  TPU {i}: {tpu}')
    else:
        print('⚠ No Coral TPUs detected. Please connect your Coral device.')
except Exception as e:
    print(f'✗ Error checking for TPUs: {e}')
"

# Download test model
echo
echo "Downloading test model..."
TEST_MODEL_DIR="converted_models"
mkdir -p "$TEST_MODEL_DIR"

if [ ! -f "$TEST_MODEL_DIR/yolo8n_320_coral_int8_edgetpu.tflite" ]; then
    wget -q -O "$TEST_MODEL_DIR/yolo8n_320_coral_int8_edgetpu.tflite" \
        "https://huggingface.co/mailseth/wildfire-watch/resolve/main/yolo8n_320_coral_int8_edgetpu.tflite"
    echo "✓ Test model downloaded"
else
    echo "✓ Test model already exists"
fi

# Test the setup
echo
echo "Testing Coral TPU setup..."
python3.8 -c "
import sys
import time
import numpy as np

try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
    
    # Load test model
    interpreter = make_interpreter('$TEST_MODEL_DIR/yolo8n_320_coral_int8_edgetpu.tflite')
    interpreter.allocate_tensors()
    
    # Get input shape
    input_shape = interpreter.get_input_details()[0]['shape']
    height, width = input_shape[1:3]
    
    # Create test input
    test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Run inference
    common.set_input(interpreter, test_image)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    
    print(f'✓ Coral TPU test successful!')
    print(f'  Model: yolo8n_320_coral_int8_edgetpu.tflite')
    print(f'  Input size: {width}x{height}')
    print(f'  Inference time: {inference_time:.2f}ms')
    
except Exception as e:
    print(f'✗ Coral TPU test failed: {e}')
    sys.exit(1)
"

echo
echo "=== Coral TPU Setup Complete ==="
echo
echo "To run Coral TPU tests:"
echo "  python3.8 -m pytest tests/test_coral_tpu_hardware.py -v"
echo
echo "To use Coral TPU in your code:"
echo "  - Always use Python 3.8"
echo "  - Import: from pycoral.utils.edgetpu import make_interpreter"
echo "  - Models must be compiled with edgetpu_compiler"
echo