#!/usr/bin/env python3.8
"""Manual test of Coral TPU functionality"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_coral_basics():
    """Test basic Coral TPU functionality"""
    print("=== Testing Coral TPU Basics ===\n")
    
    # Check hardware
    try:
        from pycoral.utils.edgetpu import list_edge_tpus
        tpus = list_edge_tpus()
        print(f"✓ Found {len(tpus)} Coral TPU(s):")
        for i, tpu in enumerate(tpus):
            print(f"  TPU {i}: {tpu}")
    except Exception as e:
        print(f"✗ Error detecting TPUs: {e}")
        return False
    
    # Check for model
    model_paths = [
        "converted_models/yolo8n_320_coral_int8_edgetpu.tflite",
        "converted_models/yolov8n_320_edgetpu.tflite",
        "converted_models/yolov8n_640_edgetpu.tflite",
        "models/yolov8n_edgetpu.tflite"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("\n✗ No Edge TPU model found")
        print("  Looking for models in:")
        for path in model_paths:
            print(f"    - {path}")
        return False
    
    print(f"\n✓ Found model: {model_path}")
    
    # Load model
    try:
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common
        
        print("\nLoading model on TPU...")
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✓ Model loaded successfully!")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input dtype: {input_details[0]['dtype']}")
        print(f"  Output shapes: {[out['shape'] for out in output_details]}")
        
        # Run test inference
        height, width = input_details[0]['shape'][1:3]
        print(f"\nRunning test inference ({width}x{height})...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(5):
            common.set_input(interpreter, test_image)
            interpreter.invoke()
        
        # Measure performance
        times = []
        for _ in range(20):
            start = time.perf_counter()
            common.set_input(interpreter, test_image)
            interpreter.invoke()
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        print(f"✓ Inference successful!")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Min time: {np.min(times):.2f}ms")
        print(f"  Max time: {np.max(times):.2f}ms")
        
        return avg_time < 30  # Should be under 30ms
        
    except Exception as e:
        print(f"\n✗ Error during model loading/inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_compilation():
    """Test Edge TPU compilation"""
    print("\n\n=== Testing Edge TPU Compilation ===\n")
    
    import subprocess
    
    # Check compiler
    result = subprocess.run(['which', 'edgetpu_compiler'], capture_output=True)
    if result.returncode != 0:
        print("✗ Edge TPU compiler not found")
        return False
    
    print("✓ Edge TPU compiler found")
    
    # Find a TFLite model to compile
    model_dir = Path('converted_models')
    tflite_models = list(model_dir.glob('*.tflite'))
    tflite_models = [m for m in tflite_models if '_edgetpu' not in m.name]
    
    if not tflite_models:
        print("✗ No TFLite models found to compile")
        # Try to download one
        print("\nTrying to download a test model...")
        try:
            import urllib.request
            url = "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1.tflite"
            test_model = model_dir / "test_model.tflite"
            urllib.request.urlretrieve(url, str(test_model))
            tflite_models = [test_model]
            print(f"✓ Downloaded test model: {test_model}")
        except Exception as e:
            print(f"✗ Failed to download test model: {e}")
            return False
    
    source_model = tflite_models[0]
    print(f"\nCompiling {source_model.name} for Edge TPU...")
    
    # Run compiler
    result = subprocess.run([
        'edgetpu_compiler',
        '-s',  # Show statistics
        str(source_model)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Compilation successful!")
        print("\nCompiler output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"  {line}")
        
        # Check if output exists
        compiled_path = source_model.parent / f"{source_model.stem}_edgetpu.tflite"
        if compiled_path.exists():
            print(f"\n✓ Compiled model created: {compiled_path}")
            return True
    else:
        print(f"✗ Compilation failed: {result.stderr}")
    
    return False


if __name__ == "__main__":
    print(f"Python version: {sys.version}\n")
    
    # Test basics
    basics_ok = test_coral_basics()
    
    # Test compilation
    compile_ok = test_model_compilation()
    
    print("\n\n=== Summary ===")
    print(f"Basic functionality: {'✓ PASS' if basics_ok else '✗ FAIL'}")
    print(f"Compilation: {'✓ PASS' if compile_ok else '✗ FAIL'}")
    
    if basics_ok and compile_ok:
        print("\n✓ All Coral TPU tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)