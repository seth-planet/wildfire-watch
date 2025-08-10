#!/usr/bin/env python3.10
"""Trace .shape attribute access to find where the error occurs"""

import sys
import os
import types

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global to track shape accesses
shape_accesses = []

def trace_shape_access(frame, event, arg):
    """Trace function to catch .shape attribute access"""
    if event == 'call':
        code = frame.f_code
        filename = code.co_filename
        
        # Only trace our code
        if "wildfire-watch" in filename and "converted_models" in filename:
            # Look for .shape in the code
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    line_no = frame.f_lineno
                    if 0 <= line_no - 1 < len(lines):
                        line = lines[line_no - 1]
                        if '.shape' in line:
                            shape_accesses.append({
                                'file': filename,
                                'line': line_no,
                                'code': line.strip(),
                                'function': code.co_name
                            })
            except:
                pass
    
    return trace_shape_access

# Monkey patch to intercept attribute access
original_getattr = object.__getattribute__

def patched_getattr(obj, name):
    if name == 'shape' and isinstance(obj, dict):
        # Found it! Print the stack trace
        import traceback
        print("\n!!! FOUND IT: dict.shape access !!!")
        print(f"Dict object: {obj}")
        print("\nStack trace:")
        traceback.print_stack()
        raise AttributeError("'dict' object has no attribute 'shape'")
    return original_getattr(obj, name)

# Apply monkey patch to dict class
dict.__getattribute__ = patched_getattr

try:
    # Set up tracing
    sys.settrace(trace_shape_access)
    
    # Now run the problematic imports and initialization
    print("Starting trace...")
    
    from converted_models.unified_yolo_trainer import UnifiedYOLOTrainer
    from converted_models.model_validator import ModelAccuracyValidator
    from converted_models.model_exporter import ModelExporter
    from converted_models.inference_runner import InferenceRunner
    
    print("Imports complete, creating instances...")
    
    # Create instances
    validator = ModelAccuracyValidator()
    exporter = ModelExporter()
    inference_runner = InferenceRunner(confidence_threshold=0.25)
    
    print("\n.shape accesses found during initialization:")
    for access in shape_accesses:
        print(f"  {access['file']}:{access['line']} in {access['function']}()")
        print(f"    {access['code']}")
    
except AttributeError as e:
    print(f"\nCaught AttributeError: {e}")
    
except Exception as e:
    print(f"\nOther error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Remove monkey patch
    dict.__getattribute__ = original_getattr
    sys.settrace(None)