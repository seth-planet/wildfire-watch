#!/usr/bin/env python3.8
"""
Convert YOLOv8 fire detection model to Coral Edge TPU format
Uses INT8 quantization for optimal Edge TPU performance
"""

import os
import sys
import subprocess
import numpy as np
import cv2
import onnx
import tensorflow as tf
from pathlib import Path
import tarfile
import urllib.request

def download_calibration_data():
    """Download wildfire calibration dataset if not present"""
    cal_dir = Path("converted_models/calibration_data_fire")
    
    if cal_dir.exists() and len(list(cal_dir.glob("*.jpg"))) > 0:
        print("✓ Calibration data already exists")
        return cal_dir
    
    print("Downloading wildfire calibration data...")
    tar_path = "wildfire_calibration_data.tar.gz"
    
    if not Path(tar_path).exists():
        url = "https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz?download=true"
        urllib.request.urlretrieve(url, tar_path)
    
    # Extract
    cal_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(cal_dir.parent)
    
    print(f"✓ Extracted calibration data to {cal_dir}")
    return cal_dir

def load_calibration_images(cal_dir, target_size=(320, 320), num_samples=100):
    """Load and preprocess calibration images"""
    images = []
    image_paths = list(Path(cal_dir).glob("*.jpg"))[:num_samples]
    
    print(f"Loading {len(image_paths)} calibration images...")
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            # Resize and convert to RGB
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    
    return np.array(images, dtype=np.uint8)

def representative_dataset_gen(calibration_images):
    """Generate representative dataset for quantization"""
    for img in calibration_images:
        # Normalize to [0, 1] for TFLite
        yield [img.astype(np.float32) / 255.0]

def convert_onnx_to_tflite(onnx_path, output_path, calibration_images):
    """Convert ONNX model to INT8 quantized TFLite"""
    print(f"\nConverting {onnx_path} to TFLite with INT8 quantization...")
    
    # First convert ONNX to TF SavedModel
    tf_dir = Path(onnx_path).parent / "tf_saved_model"
    
    print("Step 1: Converting ONNX to TensorFlow SavedModel...")
    cmd = [
        'python3.8', '-c', f'''
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("{onnx_path}")

# Convert to TF
tf_rep = prepare(onnx_model)
tf_rep.export_graph("{tf_dir}")

print("✓ Saved TensorFlow model to {tf_dir}")
'''
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting to TF: {result.stderr}")
        return False
    
    # Convert TF to TFLite with INT8 quantization
    print("\nStep 2: Converting to INT8 quantized TFLite...")
    
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(calibration_images)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Convert
    try:
        tflite_model = converter.convert()
        
        # Save
        tflite_path = output_path.replace('_edgetpu.tflite', '.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ Saved INT8 TFLite model to {tflite_path}")
        return tflite_path
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def compile_for_edge_tpu(tflite_path):
    """Compile TFLite model for Edge TPU"""
    print(f"\nStep 3: Compiling for Edge TPU...")
    
    output_path = tflite_path.replace('.tflite', '_edgetpu.tflite')
    
    # Use Edge TPU compiler
    cmd = ['edgetpu_compiler', '-s', tflite_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Edge TPU compilation successful")
        print(result.stdout)
        
        # The compiler creates the file with _edgetpu suffix
        compiled_path = tflite_path.replace('.tflite', '_edgetpu.tflite')
        if os.path.exists(compiled_path):
            return compiled_path
    else:
        print(f"Error compiling for Edge TPU: {result.stderr}")
    
    return None

def test_edge_tpu_model(model_path):
    """Test the Edge TPU model"""
    print(f"\nTesting Edge TPU model...")
    
    test_script = f'''
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import numpy as np
import time

# Load model
interpreter = make_interpreter("{model_path}")
interpreter.allocate_tensors()

# Get model info
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()

print(f"Model loaded successfully!")
print(f"Input shape: {{input_details['shape']}}")
print(f"Input dtype: {{input_details['dtype']}}")
print(f"Output count: {{len(output_details)}}")

# Test inference
height, width = input_details['shape'][1:3]
test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Warm up
for _ in range(5):
    common.set_input(interpreter, test_img)
    interpreter.invoke()

# Time inference
times = []
for _ in range(20):
    start = time.perf_counter()
    common.set_input(interpreter, test_img)
    interpreter.invoke()
    times.append((time.perf_counter() - start) * 1000)

print(f"\\nInference performance:")
print(f"  Average: {{np.mean(times):.2f}}ms")
print(f"  Min: {{np.min(times):.2f}}ms")
print(f"  Max: {{np.max(times):.2f}}ms")
'''
    
    result = subprocess.run(['python3.8', '-c', test_script], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"Error testing model: {result.stderr}")
        return False

def main():
    # Select the best fire model
    fire_models = [
        "converted_models/output/640x640/yolo8l_fire_640x640.onnx",
        "converted_models/output_int8/640x640/yolo8l_fire_int8_640x640.onnx",
    ]
    
    onnx_path = None
    for model in fire_models:
        if os.path.exists(model):
            onnx_path = model
            break
    
    if not onnx_path:
        print("No YOLOv8 fire ONNX model found!")
        return 1
    
    print(f"Using fire model: {onnx_path}")
    
    # Download calibration data
    cal_dir = download_calibration_data()
    
    # Load calibration images (320x320 for Edge TPU)
    calibration_images = load_calibration_images(cal_dir, target_size=(320, 320))
    print(f"✓ Loaded {len(calibration_images)} calibration images")
    
    # Output path
    output_dir = Path("converted_models")
    output_path = output_dir / "yolo8l_fire_320_edgetpu.tflite"
    
    # Convert to TFLite
    tflite_path = convert_onnx_to_tflite(onnx_path, str(output_path), calibration_images)
    if not tflite_path:
        print("Failed to convert to TFLite!")
        return 1
    
    # Compile for Edge TPU
    edgetpu_path = compile_for_edge_tpu(tflite_path)
    if not edgetpu_path:
        print("Failed to compile for Edge TPU!")
        return 1
    
    print(f"\n✓ Successfully created Edge TPU model: {edgetpu_path}")
    
    # Test the model
    if test_edge_tpu_model(edgetpu_path):
        print("\n✅ Edge TPU model is working correctly!")
        print(f"   Model path: {edgetpu_path}")
        print("   Ready for fire detection on Coral TPU")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())