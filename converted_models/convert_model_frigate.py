#!/usr/bin/env python3.12
"""
Frigate-specific model converter for UINT8 EdgeTPU models.

This converter ensures models have UINT8 input tensors as required by
Frigate's EdgeTPU detector implementation.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import shutil
import time
import hashlib
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class FrigateModelConverter:
    """Convert models specifically for Frigate EdgeTPU with UINT8 input."""
    
    def __init__(self, cache_dir: str = "frigate_model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_cache_metadata()
    
    def load_cache_metadata(self):
        """Load cache metadata."""
        if self.cache_metadata_file.exists():
            with open(self.cache_metadata_file, 'r') as f:
                self.cache_metadata = json.load(f)
        else:
            self.cache_metadata = {}
    
    def save_cache_metadata(self):
        """Save cache metadata."""
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(self.cache_metadata, f, indent=2)
    
    def get_cache_key(self, model_path: Path, size: tuple) -> str:
        """Generate cache key for model."""
        # Include file modification time in cache key
        stat = model_path.stat()
        key_parts = [
            str(model_path),
            str(size),
            str(stat.st_mtime),
            str(stat.st_size),
            "frigate_uint8"
        ]
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def get_cached_model(self, cache_key: str) -> Path:
        """Check if model exists in cache."""
        if cache_key in self.cache_metadata:
            cached_path = Path(self.cache_metadata[cache_key]['path'])
            if cached_path.exists():
                logger.info(f"Found cached model: {cached_path}")
                return cached_path
        return None
    
    def convert_to_frigate_tflite(self, saved_model_path: Path, output_path: Path, size: tuple = (640, 640)) -> bool:
        """Convert SavedModel to Frigate-compatible TFLite with UINT8 input."""
        
        logger.info(f"Converting {saved_model_path} to Frigate TFLite...")
        logger.info(f"Target size: {size}")
        
        try:
            # Create representative dataset for quantization
            def representative_dataset():
                """Generate representative data for quantization calibration."""
                # Use more samples for better quantization
                for _ in range(200):
                    # Generate random images in [0, 1] range
                    data = np.random.rand(1, size[1], size[0], 3).astype(np.float32)
                    yield [data]
            
            # Load the SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            # Configure for UINT8 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # CRITICAL: Use standard ops, not INT8-specific ops
            # This is key to getting UINT8 input instead of INT8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            
            # Force UINT8 input/output types
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            # Additional settings for better compatibility
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            converter.allow_custom_ops = False
            
            # Convert the model
            logger.info("Starting TFLite conversion (this may take a while)...")
            start_time = time.time()
            tflite_model = converter.convert()
            conversion_time = time.time() - start_time
            logger.info(f"Conversion completed in {conversion_time:.1f} seconds")
            
            # Verify the model has UINT8 input
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_dtype = input_details[0]['dtype']
            logger.info(f"Model input type: {input_dtype}")
            logger.info(f"Model input shape: {input_details[0]['shape']}")
            logger.info(f"Model output type: {output_details[0]['dtype']}")
            
            if input_dtype != np.uint8:
                logger.error(f"ERROR: Model has {input_dtype} input, expected uint8")
                return False
            
            # Save the model
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Saved UINT8 model to: {output_path}")
            logger.info(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compile_for_edgetpu(self, tflite_path: Path, output_dir: Path) -> Path:
        """Compile TFLite model for EdgeTPU."""
        
        logger.info(f"Compiling {tflite_path} for EdgeTPU...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run EdgeTPU compiler
        result = subprocess.run([
            'edgetpu_compiler',
            '-s',  # Show statistics
            '-m', '13',  # Model version 13
            '-o', str(output_dir),
            str(tflite_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"EdgeTPU compilation failed: {result.stderr}")
            return None
        
        # Find the compiled model
        compiled_name = tflite_path.stem + "_edgetpu.tflite"
        compiled_path = output_dir / compiled_name
        
        if compiled_path.exists():
            logger.info(f"EdgeTPU model compiled successfully: {compiled_path}")
            logger.info(f"Compiler output:\n{result.stdout}")
            return compiled_path
        else:
            logger.error("Compiled model not found")
            return None
    
    def convert_fire_model(self, model_name: str, size: tuple = (640, 640), use_cache: bool = True) -> dict:
        """Convert a fire detection model for Frigate."""
        
        results = {
            'success': False,
            'tflite_path': None,
            'edgetpu_path': None,
            'cached': False
        }
        
        # Define paths
        if model_name == "yolo8l_fire":
            saved_model_path = Path("coral_fire/saved_model_v2")
            output_name = f"yolo8l_fire_{size[0]}x{size[1]}_frigate"
        else:
            logger.error(f"Unknown model: {model_name}")
            return results
        
        # Check cache
        cache_key = self.get_cache_key(saved_model_path, size)
        if use_cache:
            cached_edgetpu = self.get_cached_model(cache_key + "_edgetpu")
            if cached_edgetpu:
                results['success'] = True
                results['edgetpu_path'] = cached_edgetpu
                results['cached'] = True
                return results
        
        # Convert to TFLite
        output_dir = Path("frigate_models")
        tflite_path = output_dir / f"{output_name}.tflite"
        
        if self.convert_to_frigate_tflite(saved_model_path, tflite_path, size):
            results['tflite_path'] = tflite_path
            
            # Compile for EdgeTPU
            edgetpu_path = self.compile_for_edgetpu(tflite_path, output_dir)
            if edgetpu_path:
                results['edgetpu_path'] = edgetpu_path
                results['success'] = True
                
                # Update cache
                self.cache_metadata[cache_key + "_edgetpu"] = {
                    'path': str(edgetpu_path),
                    'created': time.time(),
                    'model': model_name,
                    'size': size
                }
                self.save_cache_metadata()
        
        return results
    
    def validate_with_pycoral(self, model_path: Path) -> bool:
        """Validate model with pycoral."""
        
        logger.info(f"Validating {model_path} with pycoral...")
        
        script = f'''
import sys
sys.path.insert(0, '.')
from pycoral.utils.edgetpu import make_interpreter
import numpy as np

try:
    interpreter = make_interpreter("{model_path}")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {{input_details[0]['shape']}}")
    print(f"Input dtype: {{input_details[0]['dtype']}}")
    print(f"Output shape: {{output_details[0]['shape']}}")
    print(f"Output dtype: {{output_details[0]['dtype']}}")
    
    if input_details[0]['dtype'] == np.uint8:
        print("SUCCESS: Model has UINT8 input")
    else:
        print(f"ERROR: Model has {{input_details[0]['dtype']}} input")
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
'''
        
        result = subprocess.run(
            ['python3.8', '-c', script],
            capture_output=True,
            text=True
        )
        
        logger.info(f"Validation output:\n{result.stdout}")
        if result.stderr:
            logger.error(f"Validation errors:\n{result.stderr}")
        
        return result.returncode == 0

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert models for Frigate EdgeTPU")
    parser.add_argument('--model', default='yolo8l_fire', help='Model name to convert')
    parser.add_argument('--size', type=int, default=640, help='Model input size')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache')
    args = parser.parse_args()
    
    # Check Python version
    if sys.version_info[:2] != (3, 12):
        logger.warning("This script is designed for Python 3.12")
    
    # Create converter
    converter = FrigateModelConverter()
    
    # Convert model
    size = (args.size, args.size)
    logger.info(f"Converting {args.model} at size {size}")
    
    results = converter.convert_fire_model(
        args.model,
        size=size,
        use_cache=not args.no_cache
    )
    
    if results['success']:
        logger.info("Conversion successful!")
        
        # Validate if not cached
        if not results['cached'] and results['edgetpu_path']:
            if converter.validate_with_pycoral(results['edgetpu_path']):
                logger.info("Model validation passed!")
            else:
                logger.error("Model validation failed!")
                return 1
    else:
        logger.error("Conversion failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())