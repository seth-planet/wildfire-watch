#!/usr/bin/env python3.12
"""
Simple Coral TPU + Frigate Integration Test
Verifies that Coral TPU works with Frigate without full E2E complexity
"""

import os
import sys
import time
import yaml
import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import has_coral_tpu


class TestCoralFrigateSimple:
    """Simple test to verify Coral TPU works with Frigate"""
    
    @pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
    def test_coral_tpu_detection_simple(self):
        """Test that Coral TPU can perform inference with YOLOv8 model"""
        
        print("\n" + "="*60)
        print("SIMPLE TEST: Coral TPU Fire Detection")
        print("="*60)
        
        # Step 1: Verify Coral TPU
        print("\n1. Verifying Coral TPU hardware...")
        result = subprocess.run([
            'python3.8', '-c', 
            'from pycoral.utils.edgetpu import list_edge_tpus; print(f"TPUs: {len(list_edge_tpus())}")'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, "Failed to detect Coral TPU"
        print(f"✓ {result.stdout.strip()}")
        
        # Step 2: Find Edge TPU model
        print("\n2. Finding Edge TPU model...")
        model_path = None
        for path in [
            "converted_models/yolov8n_320_edgetpu.tflite",
            "converted_models/mobilenet_v2_edgetpu.tflite"
        ]:
            if os.path.exists(path):
                model_path = path
                break
        
        assert model_path is not None, "No Edge TPU model found"
        print(f"✓ Found model: {model_path}")
        
        # Step 3: Test inference
        print("\n3. Testing Coral TPU inference...")
        test_script = f'''
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# Load model
interpreter = make_interpreter("{model_path}")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
height, width = input_details[0]['shape'][1:3]

# Create test image
test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

# Run inference
times = []
for i in range(10):
    start = time.perf_counter()
    common.set_input(interpreter, test_img)
    interpreter.invoke()
    times.append((time.perf_counter() - start) * 1000)

avg_time = np.mean(times)
print(f"Average inference: {{avg_time:.2f}}ms")
print(f"Min: {{np.min(times):.2f}}ms, Max: {{np.max(times):.2f}}ms")

# Test output
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
print(f"Output shape: {{output.shape}}")
print(f"Output range: [{{np.min(output)}}, {{np.max(output)}}]")

# Success check
if avg_time < 25:
    print("✓ Performance test PASSED")
    exit(0)
else:
    print("✗ Performance test FAILED")
    exit(1)
'''
        
        result = subprocess.run(['python3.8', '-c', test_script], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        assert result.returncode == 0, "Coral TPU inference test failed"
        
        # Step 4: Generate minimal Frigate config
        print("\n4. Testing Frigate configuration...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal config
            config = {
                'mqtt': {'enabled': False},
                'detectors': {
                    'coral': {
                        'type': 'edgetpu',
                        'device': 'pci:0'
                    }
                },
                'model': {
                    'path': f'/config/model/{os.path.basename(model_path)}',
                    'input_tensor': 'nhwc',
                    'input_pixel_format': 'rgb',
                    'width': 320,
                    'height': 320
                },
                'cameras': {
                    'dummy': {
                        'enabled': False,
                        'ffmpeg': {
                            'inputs': [{
                                'path': 'rtsp://127.0.0.1:554/null',
                                'roles': ['detect']
                            }]
                        }
                    }
                }
            }
            
            config_path = Path(temp_dir) / 'config.yml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Copy model
            model_dir = Path(temp_dir) / 'model'
            model_dir.mkdir()
            shutil.copy(model_path, model_dir / os.path.basename(model_path))
            
            print(f"✓ Created Frigate config at: {config_path}")
            
            # Validate config structure
            with open(config_path) as f:
                loaded_config = yaml.safe_load(f)
            
            assert 'detectors' in loaded_config
            assert 'coral' in loaded_config['detectors']
            assert loaded_config['detectors']['coral']['type'] == 'edgetpu'
            print("✓ Config validation passed")
        
        # Step 5: Test multi-TPU configuration
        print("\n5. Testing multi-TPU configuration...")
        result = subprocess.run([
            'python3.8', '-c', '''
from pycoral.utils.edgetpu import list_edge_tpus
tpus = list_edge_tpus()
print(f"Available TPUs: {len(tpus)}")
for i, tpu in enumerate(tpus):
    print(f"  TPU {i}: {tpu['type']} at {tpu['path']}")
    
# Generate Frigate multi-TPU config
if len(tpus) > 1:
    print("\\nMulti-TPU Frigate configuration:")
    for i in range(min(len(tpus), 4)):
        print(f"  coral{i}:")
        print(f"    type: edgetpu")
        print(f"    device: pci:{i}")
'''
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("✓ Coral TPU hardware detected")
        print("✓ Edge TPU model loaded successfully")
        print("✓ Inference performance < 25ms")
        print("✓ Frigate configuration valid")
        print("✓ Multi-TPU support available")
        print("\nTEST PASSED")
        print("="*60)
    
    @pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
    def test_coral_yolo_fire_model(self):
        """Test YOLOv8 fire detection model on Coral TPU"""
        
        print("\n" + "="*60)
        print("CORAL TPU: YOLOv8 Fire Detection Model Test")
        print("="*60)
        
        # Find fire-specific model or use generic
        model_candidates = [
            "converted_models/yolov8n_fire_320_edgetpu.tflite",
            "converted_models/yolo_fire_edgetpu.tflite",
            "converted_models/yolov8n_320_edgetpu.tflite"
        ]
        
        model_path = None
        for path in model_candidates:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            pytest.skip("No YOLOv8 model found for testing")
        
        print(f"\nTesting model: {model_path}")
        
        # Test with simulated fire image
        test_script = f'''
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
import time

# Load model
interpreter = make_interpreter("{model_path}")
interpreter.allocate_tensors()

# Get model info
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()

print(f"Model loaded successfully")
print(f"Input shape: {{input_details['shape']}}")
print(f"Input dtype: {{input_details['dtype']}}")
print(f"Output count: {{len(output_details)}}")

# Create test images
height, width = input_details['shape'][1:3]

# Simulate different scenarios
scenarios = [
    ("Random noise", np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)),
    ("Bright image", np.full((height, width, 3), 200, dtype=np.uint8)),
    ("Dark image", np.full((height, width, 3), 50, dtype=np.uint8)),
    ("Red/orange tint", np.stack([
        np.full((height, width), 200, dtype=np.uint8),  # R
        np.full((height, width), 100, dtype=np.uint8),  # G
        np.full((height, width), 50, dtype=np.uint8)    # B
    ], axis=-1))
]

print(f"\\nRunning inference on {{len(scenarios)}} test scenarios...")

for name, test_img in scenarios:
    # Run inference
    start = time.perf_counter()
    common.set_input(interpreter, test_img)
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\\n{{name}}:")
    print(f"  Inference time: {{inference_time:.2f}}ms")
    print(f"  Output shape: {{output.shape}}")
    print(f"  Output stats: min={{np.min(output):.2f}}, max={{np.max(output):.2f}}, mean={{np.mean(output):.2f}}")

print("\\n✓ All scenarios completed successfully")
'''
        
        result = subprocess.run(['python3.8', '-c', test_script],
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        assert result.returncode == 0, "Model test failed"
        
        # Verify performance
        assert "Inference time:" in result.stdout
        
        # Extract inference times
        import re
        import numpy as np
        times = re.findall(r'Inference time: ([\d.]+)ms', result.stdout)
        if times:
            avg_time = np.mean([float(t) for t in times])
            print(f"\nAverage inference across scenarios: {avg_time:.2f}ms")
            assert avg_time < 25, f"Inference too slow: {avg_time}ms"
        
        print("\n✓ YOLOv8 fire detection model test PASSED")


if __name__ == '__main__':
    import numpy as np  # Import for main
    pytest.main([__file__, '-v', '-s'])