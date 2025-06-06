# Coral TPU Python 3.8 Requirements

## Important Notice

The Google Coral TPU Edge TPU runtime (`tflite_runtime`) currently requires **Python 3.8** for proper operation. This is due to binary compatibility requirements in the Edge TPU runtime library.

## Affected Components

### Hardware
- Coral USB Accelerator
- Coral PCIe Accelerator (M.2 A+E, B+M)
- Coral Dev Board
- Any system using Edge TPU

### Software
- `tflite_runtime` package
- Edge TPU runtime library (`libedgetpu.so`)
- Models compiled for Edge TPU (`.tflite` files with Edge TPU delegate)

## Installation Instructions

### Ubuntu/Debian Systems

```bash
# Install Python 3.8 if not present
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev

# Install required packages for Python 3.8
python3.8 -m pip install numpy pillow

# Install tflite_runtime for Python 3.8
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install python3-tflite-runtime
```

### Running Coral Code

When running any code that uses Coral TPU, use Python 3.8:

```bash
# Instead of:
python3 coral_inference.py

# Use:
python3.8 coral_inference.py
```

### In Docker Containers

If using Coral TPU in Docker containers, ensure the base image has Python 3.8:

```dockerfile
FROM python:3.8-slim
# Install tflite_runtime and other dependencies
```

## Workarounds for Other Python Versions

### Using Subprocess (Recommended)

For projects using Python 3.11+, you can run Coral inference in a subprocess:

```python
import subprocess
import json

def run_coral_inference(image_path):
    """Run Coral inference using Python 3.8 subprocess."""
    script = f'''
import tflite_runtime.interpreter as tflite
# Your inference code here
'''
    
    result = subprocess.run(
        ['python3.8', '-c', script],
        capture_output=True,
        text=True
    )
    
    return json.loads(result.stdout)
```

### Using a Microservice

Deploy Coral inference as a separate microservice running Python 3.8:

```yaml
# docker-compose.yml
services:
  coral-inference:
    build:
      context: .
      dockerfile: Dockerfile.coral  # Uses Python 3.8
    devices:
      - /dev/bus/usb:/dev/bus/usb  # For USB Coral
      - /dev/apex_0:/dev/apex_0    # For PCIe Coral
```

## Known Issues

1. **ImportError with Python 3.11+**: The `_pywrap_tensorflow_interpreter_wrapper` module is not compatible with Python versions above 3.8.

2. **Segmentation Faults**: Using wrong Python version can cause crashes during inference.

3. **Performance**: Running through subprocess adds ~10ms overhead compared to native calls.

## Future Updates

Google is working on updating the Edge TPU runtime for newer Python versions. Check the [Coral AI official documentation](https://coral.ai/docs/) for updates.

## Testing Your Setup

```bash
# Test if Coral is working with Python 3.8
python3.8 -c "
import tflite_runtime.interpreter as tflite
print('tflite_runtime imported successfully')
interpreter = tflite.Interpreter(
    model_path='test_model_edgetpu.tflite',
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
print('Coral TPU delegate loaded successfully')
"
```

## References

- [Coral AI Python Quickstart](https://coral.ai/docs/python-quickstart/)
- [Edge TPU Python API](https://coral.ai/docs/edgetpu/api-intro/)
- [TensorFlow Lite Python Guide](https://www.tensorflow.org/lite/guide/python)