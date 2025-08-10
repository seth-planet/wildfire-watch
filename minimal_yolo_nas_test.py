#!/usr/bin/env python3.10
"""Minimal test to reproduce YOLO-NAS shape error"""

import sys
import os
import traceback

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Wrap dict to catch shape access
class DebugDict(dict):
    def __getattr__(self, name):
        if name == 'shape':
            print("\n!!! dict.shape accessed !!!")
            print(f"Dict contents: {dict(self)}")
            traceback.print_stack()
            raise AttributeError(f"'dict' object has no attribute 'shape'")
        return super().__getattribute__(name)

# Monkey patch dict creation in key modules
import builtins
original_dict = builtins.dict

def debug_dict(*args, **kwargs):
    return DebugDict(*args, **kwargs)

try:
    # Temporarily replace dict
    builtins.dict = debug_dict
    
    print("Starting minimal test...")
    
    # Import modules that might use dict
    from converted_models.inference_runner import InferenceRunner
    
    # Create a minimal test case
    runner = InferenceRunner()
    
    # Create a test output that's a dict (like ONNX might return)
    test_output = {'output': {'predictions': None}}  # This will be a DebugDict
    
    # Try to parse it
    print("\nTrying to parse dict output...")
    try:
        # This should trigger our _find_first_numpy_array fix
        import numpy as np
        test_output_array = np.random.rand(1, 6, 8400)  # Typical YOLO output shape
        test_dict_output = {'output': test_output_array}
        
        # Call the method that was fixed
        result = runner._parse_onnx_outputs(test_dict_output, (640, 640))
        print(f"Parse successful: {len(result)} detections")
        
    except AttributeError as e:
        print(f"\nAttributeError in parsing: {e}")
    
    print("\nNow testing if the error comes from elsewhere...")
    
    # Test other possible locations
    # 1. Check if it's in the model prediction parsing
    test_prediction = {
        'bboxes_xyxy': [[100, 100, 200, 200]],
        'confidence': [0.9],
        'labels': [0]
    }
    
    # This might trigger shape access if the code expects arrays
    print("\nTesting prediction parsing...")
    
except Exception as e:
    print(f"\nCaught exception: {type(e).__name__}: {e}")
    traceback.print_exc()

finally:
    # Restore original dict
    builtins.dict = original_dict
    print("\nTest complete")