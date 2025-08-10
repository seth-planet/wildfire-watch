#!/usr/bin/env python3.12
"""
Convert fire detection models to Frigate-compatible UINT8 format.

This script ensures models have UINT8 input tensors as required by Frigate's
EdgeTPU detector implementation.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_frigate_uint8(model_path: str, output_dir: str = None):
    """Convert a model to have UINT8 input for Frigate compatibility."""
    
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / "frigate_compatible"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = model_path.stem
    output_path = output_dir / f"{model_name}_frigate.tflite"
    
    logger.info(f"Converting {model_path} to Frigate-compatible UINT8 format...")
    
    # First, check if it's already a TFLite model
    if model_path.suffix == '.tflite':
        # Check current input type
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        
        if input_details[0]['dtype'] == np.uint8:
            logger.info("Model already has UINT8 input, copying...")
            shutil.copy(model_path, output_path)
            return output_path
        
        logger.info(f"Model has {input_details[0]['dtype']} input, need to reconvert...")
        
        # For EdgeTPU models, we can't reconvert directly
        if '_edgetpu' in model_name:
            logger.warning("EdgeTPU models cannot be reconverted. Need original model.")
            # Try to find the non-EdgeTPU version
            non_edgetpu_path = model_path.parent / model_name.replace('_edgetpu', '') + '.tflite'
            if non_edgetpu_path.exists():
                model_path = non_edgetpu_path
                logger.info(f"Using non-EdgeTPU model: {model_path}")
            else:
                logger.error("Cannot find non-EdgeTPU model for reconversion")
                return None
    
    # If we have a PyTorch model, convert it first
    if model_path.suffix in ['.pt', '.pth']:
        logger.info("Converting PyTorch model to ONNX first...")
        onnx_path = output_dir / f"{model_name}.onnx"
        
        # Export to ONNX
        script = f'''
import torch
from ultralytics import YOLO

model = YOLO("{model_path}")
model.export(format='onnx', imgsz=640, simplify=True)
'''
        subprocess.run([sys.executable, '-c', script], check=True)
        
        # Find the exported ONNX
        onnx_files = list(model_path.parent.glob(f"{model_name}*.onnx"))
        if onnx_files:
            onnx_path = onnx_files[0]
            shutil.move(onnx_path, output_dir / onnx_path.name)
            model_path = output_dir / onnx_path.name
        else:
            logger.error("Failed to export to ONNX")
            return None
    
    # Convert to TensorFlow SavedModel if needed
    if model_path.suffix == '.onnx':
        logger.info("Converting ONNX to TensorFlow SavedModel...")
        saved_model_dir = output_dir / f"{model_name}_saved_model"
        
        script = f'''
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

onnx_model = onnx.load("{model_path}")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("{saved_model_dir}")
'''
        try:
            subprocess.run([sys.executable, '-c', script], check=True)
        except:
            logger.error("Failed to convert ONNX to SavedModel")
            return None
    else:
        saved_model_dir = model_path
    
    # Now convert to TFLite with UINT8 quantization
    logger.info("Converting to TFLite with UINT8 quantization...")
    
    # Create calibration dataset
    def representative_dataset():
        """Generate representative data for quantization calibration."""
        for _ in range(100):
            # Generate random images normalized to [0, 1]
            data = np.random.rand(1, 640, 640, 3).astype(np.float32)
            yield [data]
    
    # Convert with UINT8 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Force UINT8 input/output
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Use integer only operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    
    # Ensure quantization
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Verify the conversion
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        
        logger.info(f"Converted model input type: {input_details[0]['dtype']}")
        logger.info(f"Converted model saved to: {output_path}")
        
        if input_details[0]['dtype'] == np.uint8:
            # Now compile for EdgeTPU
            logger.info("Compiling for EdgeTPU...")
            edgetpu_path = output_dir / f"{model_name}_frigate_edgetpu.tflite"
            
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',
                '-o', str(output_dir),
                str(output_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find the compiled model
                compiled_path = output_path.parent / f"{output_path.stem}_edgetpu.tflite"
                if compiled_path.exists():
                    shutil.move(compiled_path, edgetpu_path)
                    logger.info(f"EdgeTPU model saved to: {edgetpu_path}")
                    return edgetpu_path
            else:
                logger.warning(f"EdgeTPU compilation failed: {result.stderr}")
                return output_path
        else:
            logger.error(f"Conversion failed - model still has {input_details[0]['dtype']} input")
            return None
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return None

def main():
    """Convert fire detection models to Frigate-compatible format."""
    
    # Models to convert
    models_to_convert = [
        "converted_models/coral_fire/saved_model_v2/yolo8l_fire_640x640_full_integer_quant.tflite",
        "converted_models/yolov8n_416.tflite",
    ]
    
    output_dir = Path("converted_models/frigate_compatible")
    output_dir.mkdir(exist_ok=True)
    
    for model_path in models_to_convert:
        if os.path.exists(model_path):
            logger.info(f"\nProcessing {model_path}...")
            result = convert_to_frigate_uint8(model_path, output_dir)
            if result:
                logger.info(f"✓ Successfully converted to: {result}")
            else:
                logger.error(f"✗ Failed to convert {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}")
    
    logger.info("\nConversion complete!")
    
    # Show all converted models
    logger.info("\nFrigate-compatible models:")
    for tflite_file in output_dir.glob("*.tflite"):
        logger.info(f"  - {tflite_file}")

if __name__ == "__main__":
    main()