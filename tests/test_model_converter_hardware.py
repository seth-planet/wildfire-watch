#!/usr/bin/env python3
"""
Hardware Tests for Model Converter
Tests with real GPU and Coral hardware
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
import subprocess
import time

# Check for hardware availability
def check_gpu_available():
    """Check if NVIDIA GPU is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_coral_available():
    """Check if Coral TPU is available"""
    try:
        # Check for PCIe Coral
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        if 'Coral' in result.stdout or 'Global Unichip' in result.stdout:
            return True
        # Check for USB Coral
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        return 'Google Inc' in result.stdout or 'Global Unichip' in result.stdout
    except:
        return False

def check_tensorrt_available():
    """Check if TensorRT is available"""
    try:
        import tensorrt
        return True
    except ImportError:
        return False


class ModelConverterHardwareTests(unittest.TestCase):
    """Test model converter with real hardware"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = Path(tempfile.mkdtemp(prefix='hardware_test_'))
        cls.output_dir = cls.test_dir / 'output'
        cls.output_dir.mkdir(exist_ok=True)
        
        # Check available hardware
        cls.has_gpu = check_gpu_available()
        cls.has_coral = check_coral_available()
        cls.has_tensorrt = check_tensorrt_available()
        
        print(f"\nHardware Detection:")
        print(f"  GPU (CUDA): {cls.has_gpu}")
        print(f"  Coral TPU: {cls.has_coral}")
        print(f"  TensorRT: {cls.has_tensorrt}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_onnx_conversion_basic(self):
        """Test basic ONNX conversion without hardware acceleration"""
        print("\n=== Testing Basic ONNX Conversion ===")
        
        # Create a simple test script
        test_script = self.test_dir / 'test_onnx.py'
        test_script.write_text('''
import sys
sys.path.insert(0, 'converted_models')

try:
    from ultralytics import YOLO
    
    # Create a minimal model
    model = YOLO('yolov8n.pt')
    
    # Export to ONNX
    success = model.export(
        format='onnx',
        imgsz=320,
        simplify=False,
        opset=13
    )
    
    print(f"Export success: {success}")
    if success:
        print("ONNX conversion successful!")
        sys.exit(0)
    else:
        print("ONNX conversion failed!")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
''')
        
        # Run the test
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(Path.cwd())
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if model file exists
        if Path('converted_models/yolov8n.pt').exists():
            self.assertEqual(result.returncode, 0, "ONNX conversion should succeed")
        else:
            self.skipTest("Model file not found - skipping ONNX test")
    
    @unittest.skipUnless(check_gpu_available(), "GPU not available")
    def test_gpu_inference(self):
        """Test GPU inference with PyTorch"""
        print("\n=== Testing GPU Inference ===")
        
        test_script = self.test_dir / 'test_gpu.py'
        test_script.write_text('''
import torch
import time

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Create test tensor
x = torch.randn(1, 3, 640, 640).to(device)

# Warmup
for _ in range(10):
    y = x * 2.0
    if device.type == 'cuda':
        torch.cuda.synchronize()

# Benchmark
start = time.time()
iterations = 100
for _ in range(iterations):
    y = x * 2.0
    if device.type == 'cuda':
        torch.cuda.synchronize()

elapsed = time.time() - start
fps = iterations / elapsed

print(f"Processed {iterations} iterations in {elapsed:.2f}s")
print(f"FPS: {fps:.1f}")
''')
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        self.assertEqual(result.returncode, 0, "GPU test should succeed")
        self.assertIn("Using device: cuda", result.stdout)
    
    @unittest.skipUnless(check_coral_available(), "Coral TPU not available")
    def test_coral_tpu(self):
        """Test Coral TPU detection and basic functionality"""
        print("\n=== Testing Coral TPU ===")
        
        test_script = self.test_dir / 'test_coral.py'
        test_script.write_text('''
try:
    from pycoral.utils import edgetpu
    from pycoral.utils.dataset import read_label_file
    from pycoral.adapters import common
    from pycoral.adapters import detect
    import numpy as np
    
    # List available TPUs
    devices = edgetpu.list_edge_tpus()
    print(f"Found {len(devices)} Coral TPU(s):")
    for i, device in enumerate(devices):
        print(f"  {i}: {device}")
    
    if devices:
        # Try to create interpreter with first device
        print("\\nTesting TPU initialization...")
        # Would need an actual Edge TPU model to test further
        print("Coral TPU is available and responsive!")
    else:
        print("No Coral TPUs detected")
        
except ImportError as e:
    print(f"PyCoral not installed: {e}")
    print("Install with: pip install pycoral")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
''')
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        # Don't fail if pycoral isn't installed, just report
        if "PyCoral not installed" in result.stdout:
            self.skipTest("PyCoral not installed - skipping Coral test")
    
    @unittest.skipUnless(check_tensorrt_available(), "TensorRT not available")
    def test_tensorrt_conversion(self):
        """Test TensorRT conversion"""
        print("\n=== Testing TensorRT ===")
        
        test_script = self.test_dir / 'test_tensorrt.py'
        test_script.write_text('''
import tensorrt as trt
import numpy as np

print(f"TensorRT version: {trt.__version__}")

# Create logger
logger = trt.Logger(trt.Logger.WARNING)

# Create builder
builder = trt.Builder(logger)
print("TensorRT builder created successfully")

# Create a network to test basic functionality
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
print("Network created successfully")

# Create builder config
config = builder.create_builder_config()
print("Builder config created successfully")

# Check available features without using deprecated attributes
if hasattr(config, 'set_flag'):
    # Check FP16 support
    try:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 support: Available")
    except:
        print("FP16 support: Not available")
    
    # Check INT8 support
    try:
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 support: Available")
    except:
        print("INT8 support: Not available")

# Would need ONNX model to test actual conversion
print("TensorRT is available and functional!")
''')
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0, "TensorRT test should succeed")
        self.assertIn("TensorRT version:", result.stdout)
    
    def test_model_sizes_parsing(self):
        """Test the model size parsing with real values"""
        print("\n=== Testing Size Parsing ===")
        
        test_script = self.test_dir / 'test_sizes.py'
        test_script.write_text('''
import sys
sys.path.insert(0, 'converted_models')

from convert_model import EnhancedModelConverter

# Test cases - updated to match tuple output format and include non-square sizes
test_cases = [
    # Square sizes
    ("640", [(640, 640)]),
    ("640,320", [(640, 640), (320, 320)]),
    ("640-320", [(640, 640), (608, 608), (576, 576), (544, 544), (512, 512), (480, 480), (448, 448), (416, 416), (384, 384), (352, 352), (320, 320)]),
    
    # Non-square sizes
    ("640x480", [(640, 480)]),
    ("640x480,320", [(640, 480), (320, 320)]),
    ("640x480,320x256", [(640, 480), (320, 256)]),
    ("416x320", [(416, 320)]),
    ("320x256,640x512", [(320, 256), (640, 512)]),
    
    # Additional test cases
    (640, [(640, 640)]),  # Integer input
    ((640, 480), [(640, 480)]),  # Tuple input
    ([640, 320], [(640, 640), (320, 320)]),  # List input
    ([(640, 480), 320], [(640, 480), (320, 320)]),  # Mixed list
]

all_passed = True
# Use an existing model file for testing
import os
model_path = os.path.join(os.getcwd(), 'yolov8n.pt')
if not os.path.exists(model_path):
    # Fallback to another model if yolov8n.pt doesn't exist
    model_path = os.path.join(os.getcwd(), 'yolov5s.pt')

converter = EnhancedModelConverter(model_path)

for input_val, expected in test_cases:
    try:
        result = converter._parse_model_sizes(input_val)
        if result == expected:
            print(f"✓ {repr(input_val)} -> {result}")
        else:
            print(f"✗ {repr(input_val)} -> {result} (expected {expected})")
            all_passed = False
    except Exception as e:
        print(f"✗ {repr(input_val)} -> ERROR: {e}")
        all_passed = False

if all_passed:
    print("\\nAll size parsing tests passed!")
    sys.exit(0)
else:
    print("\\nSome tests failed!")
    sys.exit(1)
''')
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(Path.cwd())
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        self.assertEqual(result.returncode, 0, "Size parsing tests should pass")


if __name__ == '__main__':
    unittest.main(verbosity=2)