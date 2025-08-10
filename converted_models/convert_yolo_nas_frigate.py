#!/usr/bin/env python3.12
"""
Convert YOLO-NAS fire models to Frigate-compatible UINT8 TFLite format.

This converter takes YOLO-NAS ONNX models and converts them to TFLite with:
- UINT8 input tensors (required by Frigate)
- Proper quantization using representative dataset
- EdgeTPU compilation support
- Validation of output tensor types
"""

import os
import sys
import subprocess
import logging
import argparse
import tempfile
import shutil
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class YOLONASFrigateConverter:
    """Convert YOLO-NAS models to Frigate-compatible TFLite format."""
    
    def __init__(self, cache_dir: str = "frigate_model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_metadata_file = self.cache_dir / "yolo_nas_cache.json"
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
        stat = model_path.stat()
        key_parts = [
            str(model_path),
            str(size),
            str(stat.st_mtime),
            str(stat.st_size),
            "yolo_nas_frigate_uint8"
        ]
        return "_".join(key_parts).replace("/", "_").replace(" ", "_")
    
    def get_cached_model(self, cache_key: str) -> Optional[Path]:
        """Check if model exists in cache."""
        if cache_key in self.cache_metadata:
            cached_path = Path(self.cache_metadata[cache_key]['path'])
            if cached_path.exists():
                logger.info(f"Found cached model: {cached_path}")
                return cached_path
        return None
    
    def convert_onnx_to_tensorflow(self, onnx_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert ONNX to TensorFlow SavedModel format."""
        logger.info(f"Converting ONNX to TensorFlow SavedModel...")
        
        saved_model_path = output_dir / f"{onnx_path.stem}_saved_model"
        
        # Create conversion script
        script = f'''
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try onnx2tf first (more reliable)
    try:
        import onnx2tf
        import onnx
        
        # Load and check ONNX model
        model = onnx.load('{onnx_path}')
        
        # Convert using onnx2tf
        onnx2tf.convert(
            input_onnx_file_path='{onnx_path}',
            output_folder_path='{saved_model_path}',
            non_verbose=True,
            output_signaturedefs=True,
            output_integer_quantized_tflite=False,
            batch_size=1  # Frigate uses batch size 1
        )
        print("SUCCESS")
        sys.exit(0)
    except ImportError:
        print("onnx2tf not available, trying onnx-tf...")
    
    # Try onnx-tf as fallback
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        model = onnx.load('{onnx_path}')
        
        # Convert to TensorFlow
        tf_rep = prepare(model)
        tf_rep.export_graph('{saved_model_path}')
        print("SUCCESS")
        sys.exit(0)
    except ImportError:
        print("onnx-tf not available")
    
    # Final fallback - use tf2onnx in reverse (if available)
    try:
        import tf2onnx
        import tensorflow as tf
        
        # This is a workaround - we'll need to manually create the SavedModel
        print("ERROR: Direct ONNX to TF conversion not available")
        sys.exit(1)
    except:
        pass
        
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("ERROR: No conversion method available")
sys.exit(1)
'''
        
        # Run conversion script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            temp_script = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and saved_model_path.exists():
                logger.info(f"Successfully converted ONNX to SavedModel: {saved_model_path}")
                return saved_model_path
            else:
                logger.error(f"ONNX to TF conversion failed: {result.stderr}")
                if "onnx2tf not available" in result.stdout:
                    logger.error("Please install onnx2tf: pip install onnx2tf")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("ONNX to TF conversion timed out")
            return None
        finally:
            os.unlink(temp_script)
    
    def convert_to_frigate_tflite(self, saved_model_path: Path, output_path: Path, 
                                  size: tuple = (640, 640), calibration_dir: Optional[Path] = None) -> bool:
        """Convert SavedModel to Frigate-compatible TFLite with UINT8 input."""
        
        logger.info(f"Converting to Frigate TFLite with UINT8 input...")
        logger.info(f"Target size: {size}")
        
        try:
            # Create representative dataset for quantization
            def representative_dataset():
                """Generate representative data for quantization calibration."""
                
                if calibration_dir and calibration_dir.exists():
                    # Use real calibration images if available
                    import cv2
                    from PIL import Image
                    
                    image_files = list(calibration_dir.glob("*.jpg"))
                    image_files.extend(list(calibration_dir.glob("*.jpeg")))
                    image_files.extend(list(calibration_dir.glob("*.png")))
                    
                    if image_files:
                        logger.info(f"Using {len(image_files)} calibration images")
                        
                        for img_path in image_files[:200]:  # Use up to 200 images
                            try:
                                img = Image.open(img_path).convert('RGB')
                                img = img.resize((size[0], size[1]), Image.BILINEAR)
                                img_array = np.array(img, dtype=np.float32) / 255.0
                                yield [img_array.reshape(1, size[1], size[0], 3)]
                            except Exception as e:
                                logger.warning(f"Error loading calibration image {img_path}: {e}")
                                continue
                        return
                
                # Fallback to synthetic data
                logger.info("Using synthetic calibration data")
                for i in range(200):
                    # Generate diverse synthetic images
                    if i % 4 == 0:
                        # Random noise
                        data = np.random.rand(1, size[1], size[0], 3).astype(np.float32)
                    elif i % 4 == 1:
                        # Gradient patterns
                        x = np.linspace(0, 1, size[0])
                        y = np.linspace(0, 1, size[1])
                        xx, yy = np.meshgrid(x, y)
                        data = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
                        data = data.reshape(1, size[1], size[0], 3).astype(np.float32)
                    elif i % 4 == 2:
                        # Checkerboard pattern
                        data = np.zeros((size[1], size[0], 3), dtype=np.float32)
                        data[::32, ::32] = 1.0
                        data = data.reshape(1, size[1], size[0], 3)
                    else:
                        # Solid colors
                        color = np.random.rand(3)
                        data = np.full((1, size[1], size[0], 3), color, dtype=np.float32)
                    
                    yield [data]
            
            # Load the SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            # Configure for UINT8 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            
            # CRITICAL: Use standard ops for UINT8 input (not INT8)
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
            logger.info(f"Model output shape: {output_details[0]['shape']}")
            
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
    
    def compile_for_edgetpu(self, tflite_path: Path, output_dir: Path) -> Optional[Path]:
        """Compile TFLite model for EdgeTPU."""
        
        logger.info(f"Compiling {tflite_path} for EdgeTPU...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if EdgeTPU compiler is available
        if subprocess.run(['which', 'edgetpu_compiler'], capture_output=True).returncode != 0:
            logger.warning("EdgeTPU compiler not found. Please install with: curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -")
            return None
        
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
    
    def validate_with_pycoral(self, model_path: Path) -> bool:
        """Validate model with pycoral."""
        
        logger.info(f"Validating {model_path} with pycoral...")
        
        script = f'''
import sys
sys.path.insert(0, '.')
try:
    from pycoral.utils.edgetpu import make_interpreter
    import numpy as np
    
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
    
    def convert_yolo_nas_fire(self, onnx_path: Path, output_dir: Path, 
                             size: tuple = (640, 640), calibration_dir: Optional[Path] = None,
                             use_cache: bool = True) -> dict:
        """Convert YOLO-NAS fire detection model for Frigate."""
        
        results = {
            'success': False,
            'tflite_path': None,
            'edgetpu_path': None,
            'cached': False
        }
        
        # Check cache
        cache_key = self.get_cache_key(onnx_path, size)
        if use_cache:
            cached_edgetpu = self.get_cached_model(cache_key + "_edgetpu")
            if cached_edgetpu:
                results['success'] = True
                results['edgetpu_path'] = cached_edgetpu
                results['tflite_path'] = Path(str(cached_edgetpu).replace("_edgetpu.tflite", ".tflite"))
                results['cached'] = True
                return results
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Convert ONNX to TensorFlow SavedModel
        saved_model_path = self.convert_onnx_to_tensorflow(onnx_path, output_dir)
        if not saved_model_path:
            logger.error("Failed to convert ONNX to TensorFlow")
            return results
        
        try:
            # Step 2: Convert to TFLite with UINT8 quantization
            model_name = onnx_path.stem
            tflite_path = output_dir / f"{model_name}_{size[0]}x{size[1]}_frigate.tflite"
            
            if self.convert_to_frigate_tflite(saved_model_path, tflite_path, size, calibration_dir):
                results['tflite_path'] = tflite_path
                
                # Step 3: Compile for EdgeTPU
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
                    self.cache_metadata[cache_key + "_tflite"] = {
                        'path': str(tflite_path),
                        'created': time.time(),
                        'model': model_name,
                        'size': size
                    }
                    self.save_cache_metadata()
                    
                    # Validate if pycoral is available
                    if subprocess.run(['python3.8', '-c', 'import pycoral'], 
                                    capture_output=True).returncode == 0:
                        if self.validate_with_pycoral(edgetpu_path):
                            logger.info("Model validation with pycoral passed!")
                        else:
                            logger.warning("Model validation with pycoral failed")
                else:
                    logger.warning("EdgeTPU compilation failed, but TFLite model is ready")
                    results['success'] = True  # TFLite conversion succeeded
            
        finally:
            # Cleanup SavedModel
            if saved_model_path.exists():
                shutil.rmtree(saved_model_path)
                logger.info(f"Cleaned up temporary SavedModel: {saved_model_path}")
        
        return results

def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(description="Convert YOLO-NAS models for Frigate EdgeTPU")
    parser.add_argument('--onnx', type=str, 
                       default='/home/seth/wildfire-watch/output/yolo_nas_s_wildfire_complete.onnx',
                       help='Path to YOLO-NAS ONNX model')
    parser.add_argument('--output-dir', type=str, default='frigate_models',
                       help='Output directory for converted models')
    parser.add_argument('--size', type=int, default=640, 
                       help='Model input size (square)')
    parser.add_argument('--calibration-dir', type=str,
                       help='Directory containing calibration images')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Disable cache')
    args = parser.parse_args()
    
    # Check Python version
    if sys.version_info[:2] != (3, 12):
        logger.warning("This script is designed for Python 3.12")
    
    # Check if ONNX file exists
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return 1
    
    # Setup paths
    output_dir = Path(args.output_dir)
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None
    
    # Create converter
    converter = YOLONASFrigateConverter()
    
    # Convert model
    size = (args.size, args.size)
    logger.info(f"Converting {onnx_path.name} at size {size}")
    
    results = converter.convert_yolo_nas_fire(
        onnx_path,
        output_dir,
        size=size,
        calibration_dir=calibration_dir,
        use_cache=not args.no_cache
    )
    
    if results['success']:
        logger.info("Conversion successful!")
        logger.info(f"TFLite model: {results['tflite_path']}")
        if results['edgetpu_path']:
            logger.info(f"EdgeTPU model: {results['edgetpu_path']}")
        
        # Print usage instructions
        logger.info("\n" + "="*60)
        logger.info("To use with Frigate, add to your config:")
        logger.info(f"detectors:")
        logger.info(f"  coral:")
        logger.info(f"    type: edgetpu")
        logger.info(f"    device: usb")
        logger.info(f"model:")
        if results['edgetpu_path']:
            logger.info(f"  path: /config/model_cache/{results['edgetpu_path'].name}")
        else:
            logger.info(f"  path: /config/model_cache/{results['tflite_path'].name}")
        logger.info(f"  input_tensor: nhwc")
        logger.info(f"  input_pixel_format: rgb")
        logger.info(f"  width: {size[0]}")
        logger.info(f"  height: {size[1]}")
        logger.info("="*60)
    else:
        logger.error("Conversion failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())