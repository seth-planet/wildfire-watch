# YOLO-NAS to Frigate Converter

This converter transforms YOLO-NAS ONNX models into Frigate-compatible UINT8 TFLite format for use with Coral EdgeTPU.

## Features

- Converts YOLO-NAS ONNX models to TensorFlow SavedModel
- Applies UINT8 quantization (required by Frigate)
- Compiles models for EdgeTPU acceleration
- Includes caching to avoid repeated conversions
- Validates output tensor types

## Requirements

```bash
# Core requirements (Python 3.12)
pip3.12 install tensorflow onnx2tf

# Optional for EdgeTPU compilation
sudo apt-get install edgetpu-compiler

# Optional for validation (requires Python 3.8)
python3.8 -m pip install pycoral
```

## Usage

### Basic Conversion

```bash
# Convert YOLO-NAS fire model to Frigate format
python3.12 convert_yolo_nas_frigate.py \
    --onnx /home/seth/wildfire-watch/output/yolo_nas_s_wildfire_complete.onnx \
    --output-dir frigate_models \
    --size 640 \
    --calibration-dir /home/seth/wildfire-watch/converted_models/calibration_data_fire
```

### Command Line Options

- `--onnx`: Path to YOLO-NAS ONNX model (required)
- `--output-dir`: Output directory for converted models (default: `frigate_models`)
- `--size`: Model input size in pixels (default: 640)
- `--calibration-dir`: Directory with calibration images for better quantization
- `--no-cache`: Disable caching to force re-conversion

### Output Files

The converter generates:
1. `{model_name}_{size}x{size}_frigate.tflite` - UINT8 quantized TFLite model
2. `{model_name}_{size}x{size}_frigate_edgetpu.tflite` - EdgeTPU compiled model (if compiler available)

## Frigate Configuration

After conversion, add the model to your Frigate config:

```yaml
detectors:
  coral:
    type: edgetpu
    device: usb

model:
  path: /config/model_cache/yolo_nas_s_wildfire_complete_640x640_frigate_edgetpu.tflite
  input_tensor: nhwc
  input_pixel_format: rgb
  width: 640
  height: 640
  labelmap:
    0: fire
    1: smoke
```

## Technical Details

### UINT8 vs INT8

Frigate requires UINT8 input tensors (0-255 range) rather than INT8 (-128 to 127). This converter ensures proper quantization by:

1. Using `tf.lite.OpsSet.TFLITE_BUILTINS` instead of INT8-specific ops
2. Setting `inference_input_type = tf.uint8`
3. Providing a representative dataset for calibration

### Conversion Pipeline

1. **ONNX → TensorFlow SavedModel**: Uses onnx2tf for reliable conversion
2. **SavedModel → TFLite**: Applies UINT8 quantization with calibration data
3. **TFLite → EdgeTPU**: Compiles for Coral TPU acceleration

### Caching

The converter caches results based on:
- Model file path and modification time
- Input size
- Conversion parameters

This prevents redundant conversions when re-running the script.

## Troubleshooting

### "onnx2tf not available"
```bash
pip3.12 install onnx2tf
```

### "EdgeTPU compiler not found"
```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

### Model has INT8 input instead of UINT8
This usually happens when using INT8-specific ops. The converter automatically uses the correct ops for UINT8.

### Poor detection accuracy after quantization
Provide more calibration images that represent your target domain (fire/smoke images).

## Testing

Run the test script to verify conversion:

```bash
python3.12 test_yolo_nas_frigate_conversion.py
```

This will:
1. Convert the YOLO-NAS fire model
2. Verify output files exist
3. Report file sizes and status