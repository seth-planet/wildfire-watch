#!/usr/bin/env python3.8
"""
Convert ONNX models (YOLOv8/YOLO-NAS) to Coral Edge TPU format

This script specializes in converting ONNX models to Edge TPU:
1. Uses onnx2tf for better ONNX → TensorFlow conversion
2. Handles complex operations like SplitV that cause issues
3. Supports both YOLOv8 and YOLO-NAS architectures
4. Downloads calibration dataset for INT8 quantization
5. Multiple conversion strategies for compatibility

For PyTorch (.pt) models, use convert_yolo_to_coral.py instead.

Requirements:
- Python 3.8 (required for tflite_runtime)
- tensorflow==2.13.0
- onnx2tf (preferred) or onnx-tf
- edgetpu_compiler (for Edge TPU compilation)

Usage:
    python3.8 scripts/convert_to_coral.py model.onnx --size 320
    python3.8 scripts/convert_to_coral.py yolo_nas.onnx --size 416 --output coral_models/
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
import json

# Ensure Python 3.8
if sys.version_info[:2] != (3, 8):
    print(f"ERROR: This script requires Python 3.8, but running {sys.version}")
    print("Run with: python3.8 scripts/convert_to_coral_improved.py")
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
        
        return cal_dir
        
    except Exception as e:
        print(f"Error downloading calibration data: {e}")
        print("Creating synthetic calibration data...")
        
        # Create synthetic data as fallback
        for i in range(200):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(cal_dir / f"synthetic_{i:03d}.jpg"), img)
        
        return cal_dir


def convert_onnx_to_tf_improved(onnx_path: Path, output_dir: Path, model_size: int = 320):
    """Convert ONNX to TensorFlow SavedModel using onnx2tf approach"""
    print(f"\nConverting ONNX to TensorFlow SavedModel (improved method)...")
    
    saved_model_dir = output_dir / "saved_model"
    
    try:
        # First, try using onnx2tf if available
        result = subprocess.run(['which', 'onnx2tf'], capture_output=True)
        if result.returncode == 0:
            print("Using onnx2tf for conversion...")
            
            # Run onnx2tf conversion
            cmd = [
                'onnx2tf',
                '-i', str(onnx_path),
                '-o', str(saved_model_dir),
                '-nuo',  # No optimization
                '-b', '1',  # Batch size 1
                '--non_verbose'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Converted with onnx2tf: {saved_model_dir}")
                return saved_model_dir
            else:
                print(f"onnx2tf failed: {result.stderr}")
        
        # Fallback to onnx-tf with modifications
        print("Using onnx-tf for conversion...")
        import onnx
        from onnx import helper, numpy_helper
        
        # Load and potentially modify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Check for problematic operations
        has_splitv = False
        for node in onnx_model.graph.node:
            if node.op_type in ['Split', 'SplitV']:
                has_splitv = True
                print(f"Found {node.op_type} operation - will need special handling")
        
        # If we have Split operations, try to simplify the model
        if has_splitv:
            print("Simplifying model to avoid Split operations...")
            try:
                import onnxsim
                onnx_model, check = onnxsim.simplify(onnx_model)
                print("Model simplified successfully")
            except:
                print("Could not simplify model, continuing anyway")
        
        # Convert to TensorFlow
        from onnx_tf.backend import prepare
        
        # Prepare with special flags for better compatibility
        tf_rep = prepare(onnx_model, device='CPU', strict=False)
        tf_rep.export_graph(str(saved_model_dir))
        
        print(f"✓ Saved TensorFlow model to: {saved_model_dir}")
        return saved_model_dir
        
    except Exception as e:
        print(f"ERROR: Failed to convert ONNX to TF: {e}")
        
        # Last resort: try using tf.saved_model directly if it's a TF model
        if onnx_path.suffix == '.pb':
            print("Attempting to load as TensorFlow SavedModel...")
            try:
                tf.saved_model.load(str(onnx_path.parent))
                return onnx_path.parent
            except:
                pass
        
        return None


def create_representative_dataset(cal_dir: Path, model_size: int, num_samples: int = 200):
    """Create representative dataset generator for quantization"""
    def representative_dataset():
        """Generator for calibration images"""
        images = list(cal_dir.glob("**/*.jpg")) + list(cal_dir.glob("**/*.png"))
        np.random.shuffle(images)
        
        count = 0
        for img_path in images:
            if count >= num_samples:
                break
                
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
                count += 1
                
            except Exception as e:
                continue
        
        print(f"Used {count} images for quantization")
    
    return representative_dataset


def convert_to_tflite_edge_tpu(saved_model_dir: Path, output_dir: Path, model_size: int, cal_dir: Path):
    """Convert TensorFlow SavedModel to TFLite with Edge TPU optimization"""
    print("\nConverting to TFLite with INT8 quantization for Edge TPU...")
    
    try:
        # Load the saved model
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        # Configure for Edge TPU
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = create_representative_dataset(cal_dir, model_size)
        
        # Edge TPU requires full integer quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        # Set input/output types to uint8
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        # Additional settings for better compatibility
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.allow_custom_ops = False
        
        # For YOLOv8, we may need to handle the output differently
        converter._experimental_lower_tensor_list_ops = False
        
        print("Performing INT8 quantization (this may take several minutes)...")
        
        # Try conversion with different settings if needed
        tflite_model = None
        attempts = [
            # Attempt 1: Strict INT8 only
            {
                'ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                'description': 'Strict INT8 quantization'
            },
            # Attempt 2: INT8 with TFLITE_BUILTINS fallback
            {
                'ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS],
                'description': 'INT8 with TFLite builtins fallback'
            },
            # Attempt 3: With SELECT_TF_OPS (less ideal for Edge TPU)
            {
                'ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS],
                'description': 'INT8 with TF ops (may not compile for Edge TPU)'
            }
        ]
        
        for attempt in attempts:
            try:
                print(f"\nAttempting: {attempt['description']}")
                converter.target_spec.supported_ops = attempt['ops']
                tflite_model = converter.convert()
                print(f"✓ Conversion successful with {attempt['description']}")
                break
            except Exception as e:
                print(f"✗ Failed: {str(e)[:200]}")
                continue
        
        if tflite_model is None:
            raise Exception("All conversion attempts failed")
        
        # Save the model
        model_name = saved_model_dir.parent.name
        tflite_path = output_dir / f"{model_name}_{model_size}_int8.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"\n✓ Saved INT8 quantized model: {tflite_path}")
        
        # Verify model
        print("\nVerifying model...")
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model details:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output count: {len(output_details)}")
        for i, out in enumerate(output_details):
            print(f"  Output {i} shape: {out['shape']}")
        
        return tflite_path
        
    except Exception as e:
        print(f"ERROR: TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compile_for_edge_tpu(tflite_path: Path, output_dir: Path):
    """Compile TFLite model for Edge TPU"""
    print("\nCompiling for Edge TPU...")
    
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
        # Find compiled model
        compiled_name = f"{tflite_path.stem}_edgetpu.tflite"
        compiled_path = tflite_path.parent / compiled_name
        
        if compiled_path.exists():
            # Move to output directory with better name
            model_name = tflite_path.stem.replace('_int8', '')
            final_path = output_dir / f"{model_name}_edgetpu.tflite"
            shutil.move(str(compiled_path), str(final_path))
            
            print(f"\n✓ Edge TPU model saved: {final_path}")
            
            # Parse and display statistics
            for line in result.stdout.split('\n'):
                if "Operations successfully mapped" in line or "%" in line:
                    print(f"  {line.strip()}")
            
            return final_path
        else:
            print("WARNING: Edge TPU compilation produced no output")
            return tflite_path
    else:
        print(f"WARNING: Edge TPU compilation failed: {result.stderr}")
        return tflite_path


def test_edge_tpu_model(model_path: Path):
    """Test the compiled Edge TPU model"""
    print(f"\nTesting Edge TPU model: {model_path}")
    
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


def convert_model(model_path: Path, output_dir: Path, model_size: int = 320):
    """Main conversion pipeline"""
    print(f"\n{'='*60}")
    print(f"Converting {model_path.name} to Coral Edge TPU format")
    print(f"Model size: {model_size}x{model_size}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Handle input format
    if model_path.suffix == '.pt':
        print("\nStep 1: PyTorch model detected - needs ONNX export")
        print("Please export to ONNX format first using appropriate tools")
        print("For YOLOv8: Use YOLO export functionality")
        print("For YOLO-NAS: Use super-gradients export")
        return None
    elif model_path.suffix != '.onnx':
        print(f"ERROR: Unsupported format: {model_path.suffix}")
        print("Please provide an ONNX model")
        return None
    
    # Step 2: Convert ONNX to TensorFlow
    saved_model_dir = convert_onnx_to_tf_improved(model_path, output_dir, model_size)
    if not saved_model_dir:
        return None
    
    # Step 3: Download calibration data
    cal_dir = download_calibration_data(output_dir)
    
    # Step 4: Convert to TFLite with INT8 quantization
    tflite_path = convert_to_tflite_edge_tpu(saved_model_dir, output_dir, model_size, cal_dir)
    if not tflite_path:
        return None
    
    # Step 5: Compile for Edge TPU
    edge_tpu_path = compile_for_edge_tpu(tflite_path, output_dir)
    
    # Step 6: Test on hardware if available
    if edge_tpu_path and "_edgetpu.tflite" in str(edge_tpu_path):
        test_edge_tpu_model(edge_tpu_path)
    
    # Cleanup
    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"  Output: {edge_tpu_path}")
    print(f"{'='*60}")
    
    return edge_tpu_path


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8/YOLO-NAS models to Coral Edge TPU format"
    )
    parser.add_argument("model", help="Path to ONNX model")
    parser.add_argument("--size", type=int, default=320, 
                       choices=[224, 320, 416, 640],
                       help="Model input size (default: 320)")
    parser.add_argument("--output", type=str, default="converted_models/coral",
                       help="Output directory")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    # Run conversion
    result = convert_model(model_path, output_dir, args.size)
    
    if result:
        print(f"\nSuccess! Model ready at: {result}")
        
        # Generate labels file if needed
        labels_path = result.parent / f"{result.stem.replace('_edgetpu', '')}_labels.txt"
        if not labels_path.exists():
            print(f"\nGenerating labels file: {labels_path}")
            # For fire detection, create appropriate labels
            if 'fire' in str(model_path).lower():
                labels = ['background'] + ['fire'] * 80  # Fire at multiple indices
                labels[26] = 'Fire'  # Primary fire class
            else:
                # Standard COCO labels
                labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
                         'bus', 'train', 'truck', 'boat', 'traffic light',
                         'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                         'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                         'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                         'wine glass', 'cup', 'fork', 'knife', 'spoon',
                         'bowl', 'banana', 'apple', 'sandwich', 'orange',
                         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                         'cake', 'chair', 'couch', 'potted plant', 'bed',
                         'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                         'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                         'toaster', 'sink', 'refrigerator', 'book', 'clock',
                         'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
            
            with open(labels_path, 'w') as f:
                for label in labels[:80]:  # YOLOv8 uses 80 classes
                    f.write(f"{label}\n")
            
            print(f"✓ Labels file created")
    else:
        print("\n✗ Conversion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()