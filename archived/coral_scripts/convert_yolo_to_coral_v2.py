#!/usr/bin/env python3.8
"""
Convert YOLOv8 model to Coral TPU format with INT8 quantization
Version 2: Uses onnx2tf for better compatibility with complex models
Requires Python 3.8 for TensorFlow Lite compatibility
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
    print("Run with: python3.8 scripts/convert_yolo_to_coral_v2.py")
    sys.exit(1)

try:
    import tensorflow as tf
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("Install with:")
    print("  python3.8 -m pip install tensorflow==2.13.0")
    print("  python3.8 -m pip install tflite-runtime")
    print("  python3.8 -m pip install onnx2tf==1.17.5")
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
        
        return cal_dir
        
    except Exception as e:
        print(f"Error downloading calibration data: {e}")
        print("Creating synthetic calibration data...")
        
        # Create synthetic data as fallback
        for i in range(200):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(cal_dir / f"synthetic_{i:03d}.jpg"), img)
        
        return cal_dir


def convert_yolo_to_tflite_v2(model_path: Path, output_dir: Path, model_size: int = 320):
    """Convert YOLOv8 model to TFLite using onnx2tf for better compatibility"""
    
    print(f"\nConverting {model_path.name} to TFLite INT8 for Coral TPU (v2)")
    print(f"Target size: {model_size}x{model_size}")
    
    # Step 1: Export to ONNX if needed
    if model_path.suffix == '.pt':
        print("\nStep 1: Exporting PyTorch model to ONNX...")
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            onnx_path = output_dir / f"{model_path.stem}_{model_size}.onnx"
            
            # Export with specific settings for better compatibility
            model.export(
                format='onnx', 
                imgsz=model_size, 
                simplify=True,
                dynamic=False,  # Static shapes for TFLite
                opset=12  # Use older opset for better compatibility
            )
            
            # Move exported file
            exported_onnx = model_path.parent / f"{model_path.stem}.onnx"
            if exported_onnx.exists():
                shutil.move(str(exported_onnx), str(onnx_path))
        except Exception as e:
            print(f"ERROR: Failed to export to ONNX: {e}")
            return None
    else:
        onnx_path = model_path
    
    # Step 2: Convert ONNX to TensorFlow using onnx2tf
    print("\nStep 2: Converting ONNX to TensorFlow using onnx2tf...")
    saved_model_dir = output_dir / "saved_model"
    
    try:
        # Use onnx2tf command line for more control
        cmd = [
            sys.executable, "-m", "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(saved_model_dir),
            "-nuo",  # Do not use onnx_tf
            "-cotof",  # Constant output to TFLite Float32
            "-coton",  # Constant output to TFLite INT8
            "--disable_group_convolution",  # Disable group conv for better compatibility
            "--replace_splitv_to_split",  # Key flag to replace SplitV with Split
            "--non_verbose"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"ERROR: onnx2tf conversion failed: {result.stderr}")
            # Try alternative conversion method
            print("\nTrying alternative conversion with onnx-tf...")
            import onnx
            from onnx_tf.backend import prepare
            
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model, device='CPU', strict=False)
            tf_rep.export_graph(str(saved_model_dir))
        
        print(f"Saved TensorFlow model to: {saved_model_dir}")
        
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
    
    try:
        # Load SavedModel with compatibility options
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        # Configure for INT8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        
        # Edge TPU compatible settings - avoid SELECT_TF_OPS
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback only
        ]
        
        # Ensure uint8 input/output for Edge TPU
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Advanced settings for better compatibility
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.allow_custom_ops = False
        converter._experimental_lower_tensor_list_ops = False  # Avoid complex ops
        
        # Convert
        print("Performing INT8 quantization (this may take several minutes)...")
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
        
        # Run compiler with additional flags for complex models
        result = subprocess.run([
            'edgetpu_compiler',
            '-s',  # Show statistics
            '-m', '13',  # Model version 13
            '-a',  # Show all operations
            str(tflite_path)
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            edgetpu_path = output_dir / f"{model_path.stem}_{model_size}_int8_edgetpu.tflite"
            compiled_path = tflite_path.parent / f"{tflite_path.stem}_edgetpu.tflite"
            
            if compiled_path.exists():
                shutil.move(str(compiled_path), str(edgetpu_path))
                print(f"✓ Edge TPU model saved: {edgetpu_path}")
                
                # Parse and display compilation stats
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


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert YOLO models to Coral TPU format (v2)")
    parser.add_argument("model", nargs="?", help="Path to YOLO model (.pt or .onnx)")
    parser.add_argument("--size", type=int, default=320, choices=[320, 416, 640],
                       help="Model input size (default: 320)")
    parser.add_argument("--output", type=str, default="converted_models/coral",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Install onnx2tf if not available
    try:
        import onnx2tf
    except ImportError:
        print("Installing onnx2tf==1.17.5 for TensorFlow 2.13 compatibility...")
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx2tf==1.17.5"])
    
    # Find model to convert
    if args.model:
        model_path = Path(args.model)
    else:
        # Look for existing models
        candidates = [
            Path("yolov8l.pt"),
            Path("yolov8n.pt"),
            Path("converted_models/output/yolo8l_fire_640x640.onnx"),
        ]
        
        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                break
        
        if not model_path:
            print("ERROR: No model specified and no default models found")
            print("Usage: python3.8 scripts/convert_yolo_to_coral_v2.py <model_path>")
            sys.exit(1)
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert model
    coral_model = convert_yolo_to_tflite_v2(model_path, output_dir, args.size)
    
    if coral_model and coral_model.exists():
        print(f"\n✓ Conversion complete: {coral_model}")
    else:
        print("\n✗ Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()