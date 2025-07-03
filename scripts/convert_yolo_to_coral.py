#!/usr/bin/env python3.8
"""
Convert YOLOv8 PyTorch models to Coral Edge TPU format

This script handles the complete conversion pipeline:
1. PyTorch (.pt) → ONNX → TensorFlow → TFLite → Edge TPU
2. Downloads calibration dataset for INT8 quantization
3. Compiles for Edge TPU with edgetpu_compiler
4. Tests the model on Coral hardware if available

For ONNX models, use convert_to_coral.py instead.

Requirements:
- Python 3.8 (required for tflite_runtime)
- tensorflow==2.13.0
- onnx-tf
- edgetpu_compiler (for Edge TPU compilation)

Usage:
    python3.8 scripts/convert_yolo_to_coral.py model.pt --size 320
    python3.8 scripts/convert_yolo_to_coral.py yolov8n.pt --size 416 --output coral_models/
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil
import subprocess
import urllib.request
import tarfile

# Ensure Python 3.8
if sys.version_info[:2] != (3, 8):
    print(f"ERROR: This script requires Python 3.8, but running {sys.version}")
    print("Run with: python3.8 scripts/convert_yolo_to_coral.py")
    sys.exit(1)

try:
    import tensorflow as tf
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("Install with:")
    print("  python3.8 -m pip install tensorflow==2.13.0")
    print("  python3.8 -m pip install tflite-runtime")
    sys.exit(1)


def download_calibration_data(output_dir: Path) -> Path:
    """Download calibration dataset for quantization"""
    cal_dir = output_dir / "calibration_data"
    
    if cal_dir.exists() and len(list(cal_dir.glob("*.jpg"))) > 100:
        print(f"Calibration data already exists: {cal_dir}")
        return cal_dir
    
    print("Downloading calibration dataset...")
    cal_dir.mkdir(exist_ok=True)
    
    # Download wildfire calibration data
    url = "https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz"
    tar_path = output_dir / "calibration_data.tar.gz"
    
    try:
        urllib.request.urlretrieve(url, str(tar_path))
        print(f"Downloaded calibration data: {tar_path}")
        
        # Extract
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(cal_dir)
        
        # Clean up
        tar_path.unlink()
        
        # Check we have images
        images = list(cal_dir.glob("**/*.jpg")) + list(cal_dir.glob("**/*.png"))
        print(f"Found {len(images)} calibration images")
        
        if len(images) < 100:
            # If not enough images, download COCO samples
            print("Downloading additional COCO samples...")
            for i in range(100):
                img_url = f"https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
                img_path = cal_dir / f"coco_sample_{i:03d}.jpg"
                if not img_path.exists():
                    urllib.request.urlretrieve(img_url, str(img_path))
                if i > 0 and i % 20 == 0:
                    print(f"  Downloaded {i}/100 samples")
        
        return cal_dir
        
    except Exception as e:
        print(f"Error downloading calibration data: {e}")
        print("Creating synthetic calibration data...")
        
        # Create synthetic data as fallback
        for i in range(200):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(cal_dir / f"synthetic_{i:03d}.jpg"), img)
        
        return cal_dir


def convert_yolo_to_tflite(model_path: Path, output_dir: Path, model_size: int = 320):
    """Convert YOLOv8 model to TFLite with INT8 quantization"""
    
    print(f"\nConverting {model_path.name} to TFLite INT8 for Coral TPU")
    print(f"Target size: {model_size}x{model_size}")
    
    # Step 1: Export to ONNX if needed
    if model_path.suffix == '.pt':
        print("\nStep 1: Exporting PyTorch model to ONNX...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            onnx_path = output_dir / f"{model_path.stem}_{model_size}.onnx"
            model.export(format='onnx', imgsz=model_size, simplify=True)
            # Move exported file
            exported_onnx = model_path.parent / f"{model_path.stem}.onnx"
            if exported_onnx.exists():
                shutil.move(str(exported_onnx), str(onnx_path))
        except Exception as e:
            print(f"ERROR: Failed to export to ONNX: {e}")
            return None
    else:
        onnx_path = model_path
    
    # Step 2: Convert ONNX to TensorFlow SavedModel
    print("\nStep 2: Converting ONNX to TensorFlow SavedModel...")
    saved_model_dir = output_dir / "saved_model"
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))
        print(f"Saved TensorFlow model to: {saved_model_dir}")
    except ImportError:
        print("ERROR: onnx-tf not installed")
        print("Install with: python3.8 -m pip install onnx-tf")
        return None
    except Exception as e:
        print(f"ERROR: Failed to convert ONNX to TF: {e}")
        return None
    
    # Step 3: Download calibration data
    cal_dir = download_calibration_data(output_dir)
    
    # Step 4: Convert to TFLite with INT8 quantization
    print("\nStep 4: Converting to TFLite with INT8 quantization...")
    
    def representative_dataset():
        """Generator for calibration images"""
        images = list(cal_dir.glob("**/*.jpg")) + list(cal_dir.glob("**/*.png"))
        np.random.shuffle(images)
        
        for img_path in images[:200]:  # Use up to 200 images
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Resize to model input size
                img = cv2.resize(img, (model_size, model_size))
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # Add batch dimension
                img = np.expand_dims(img, axis=0)
                
                yield [img]
                
            except Exception as e:
                continue
    
    # Load SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    # Configure for INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Full integer quantization with TF Select ops support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Fallback
        tf.lite.OpsSet.SELECT_TF_OPS  # Support for SplitV and other TF ops
    ]
    
    # Ensure uint8 input/output for Edge TPU
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Advanced settings
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.allow_custom_ops = False
    
    # Convert
    print("Performing INT8 quantization (this may take several minutes)...")
    try:
        tflite_model = converter.convert()
        
        # Save quantized model
        tflite_path = output_dir / f"{model_path.stem}_{model_size}_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✓ Saved INT8 quantized model: {tflite_path}")
        
        # Verify model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nModel details:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shapes: {[out['shape'] for out in output_details]}")
        
        # Step 5: Compile for Edge TPU
        print("\nStep 5: Compiling for Edge TPU...")
        
        # Check if compiler is available
        result = subprocess.run(['which', 'edgetpu_compiler'], capture_output=True)
        if result.returncode != 0:
            print("WARNING: Edge TPU compiler not found")
            print("Install with: sudo apt-get install edgetpu-compiler")
            return tflite_path
        
        # Run compiler
        result = subprocess.run([
            'edgetpu_compiler',
            '-s',  # Show statistics
            '-m', '13',  # Model version 13
            str(tflite_path)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            edgetpu_path = output_dir / f"{model_path.stem}_{model_size}_int8_edgetpu.tflite"
            compiled_path = tflite_path.parent / f"{tflite_path.stem}_edgetpu.tflite"
            
            if compiled_path.exists():
                shutil.move(str(compiled_path), str(edgetpu_path))
                print(f"✓ Edge TPU model saved: {edgetpu_path}")
                
                # Parse stats
                for line in result.stdout.split('\n'):
                    if "Operations successfully mapped" in line or "%" in line:
                        print(f"  {line.strip()}")
                
                return edgetpu_path
            else:
                print("WARNING: Edge TPU compilation produced no output")
                return tflite_path
        else:
            print(f"WARNING: Edge TPU compilation failed: {result.stderr}")
            return tflite_path
            
    except Exception as e:
        print(f"ERROR: TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Cleanup
        if saved_model_dir.exists():
            shutil.rmtree(saved_model_dir)


def test_coral_model(model_path: Path):
    """Test the converted model on Coral TPU"""
    print(f"\nTesting model on Coral TPU: {model_path}")
    
    try:
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common
        
        # Load model
        interpreter = make_interpreter(str(model_path))
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        height, width = input_details[0]['shape'][1:3]
        
        # Create test image
        test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Run inference
        import time
        common.set_input(interpreter, test_img)
        
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"✓ Coral TPU test successful!")
        print(f"  Inference time: {inference_time:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Coral TPU test failed: {e}")
        return False


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLO models to Coral TPU format")
    parser.add_argument("model", nargs="?", help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("--size", type=int, default=320, choices=[320, 416, 640],
                       help="Model input size (default: 320)")
    parser.add_argument("--output", type=str, default="converted_models/coral",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Find model to convert
    if args.model:
        model_path = Path(args.model)
    else:
        # Look for existing models
        candidates = [
            Path("yolov8n.pt"),
            Path("yolov5s.pt"),
            Path("converted_models/output/yolo8l_fire_640x640.onnx"),
        ]
        
        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                break
        
        if not model_path:
            print("ERROR: No model specified and no default models found")
            print("Usage: python3.8 scripts/convert_yolo_to_coral.py <model_path>")
            sys.exit(1)
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    coral_model = convert_yolo_to_tflite(model_path, output_dir, args.size)
    
    if coral_model and coral_model.exists():
        print(f"\n✓ Conversion complete: {coral_model}")
        
        # Test on Coral TPU if available
        if "_edgetpu.tflite" in str(coral_model):
            test_coral_model(coral_model)
    else:
        print("\n✗ Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()