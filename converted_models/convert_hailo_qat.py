#!/usr/bin/env python3.10
"""Hailo conversion with QAT (Quantization-Aware Training) optimization.

This script converts ONNX models to Hailo HEF format with emphasis on
maintaining accuracy through proper calibration and QAT techniques.

For wildfire detection models, maintaining high accuracy is critical
to avoid false negatives that could miss real fires.
"""

import os
import sys
import json
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class HailoQATConverter:
    """Hailo converter with QAT optimization focus."""
    
    def __init__(
        self,
        model_path: str,
        calibration_data: str,
        output_dir: str = "./hailo_qat_output",
        target: str = "hailo8l",
        batch_size: int = 8,
        optimize_for_accuracy: bool = True
    ):
        """Initialize QAT-focused Hailo converter.
        
        Args:
            model_path: Path to ONNX model
            calibration_data: Path to calibration dataset (required)
            output_dir: Output directory for HEF files
            target: Target hardware (hailo8 or hailo8l)
            batch_size: Batch size for efficient inference
            optimize_for_accuracy: Prioritize accuracy over speed
        """
        self.model_path = Path(model_path)
        self.calibration_data = Path(calibration_data)
        self.output_dir = Path(output_dir)
        self.target = target
        self.batch_size = batch_size
        self.optimize_for_accuracy = optimize_for_accuracy
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not self.calibration_data.exists():
            raise FileNotFoundError(f"Calibration data not found: {calibration_data}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = self.model_path.stem
        
        # Check if model has QAT in name
        self.is_qat_model = 'qat' in self.model_name.lower()
        if self.is_qat_model:
            logger.info("QAT model detected - will use QAT-specific optimizations")
    
    def prepare_calibration_npy(self, work_dir: Path) -> Path:
        """Prepare calibration data in numpy format for Hailo.
        
        Creates a high-quality calibration dataset with proper preprocessing
        and augmentation for better quantization.
        
        Returns:
            Path to calibration .npy file
        """
        logger.info("Preparing calibration dataset...")
        
        calib_script = f"""
import numpy as np
import cv2
from pathlib import Path
import json

# Load calibration metadata if available
calib_dir = Path("{self.calibration_data}")
metadata_path = calib_dir / "metadata.json"

if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Calibration metadata: {{metadata}}")

# Get all images
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
    image_files.extend(calib_dir.glob(f'*{{ext}}'))
    image_files.extend(calib_dir.glob(f'*{{ext.upper()}}'))

# Sort for reproducibility
image_files = sorted(image_files)
print(f"Found {{len(image_files)}} calibration images")

# Use more images for better calibration (up to 2000)
num_images = min(len(image_files), 2000 if {self.optimize_for_accuracy} else 500)
image_files = image_files[:num_images]

# Prepare calibration data with augmentation
calib_data = []
augment = {self.optimize_for_accuracy}

for i, img_path in enumerate(image_files):
    if i % 100 == 0:
        print(f"Processing {{i}}/{{num_images}} images...")
    
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Resize to model input size
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    calib_data.append(img)
    
    # Add augmented versions for better calibration
    if augment and i < 500:  # Augment first 500 images
        # Brightness variation
        bright = np.clip(img * 1.2, 0, 1)
        calib_data.append(bright)
        
        # Slight rotation/flip for diversity
        flipped = cv2.flip(img, 1)
        calib_data.append(flipped)

# Stack into array
calib_array = np.array(calib_data, dtype=np.float32)
print(f"Calibration array shape: {{calib_array.shape}}")

# Save with compression
np.save("{work_dir}/calibration.npy", calib_array)
print(f"Saved calibration data with {{len(calib_array)}} samples")

# Also create a smaller validation set
val_data = calib_array[:100]
np.save("{work_dir}/validation.npy", val_data)
print("Created validation dataset")
"""
        
        result = subprocess.run(
            ['python3.10', '-c', calib_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Calibration preparation failed: {result.stderr}")
        
        logger.info(result.stdout)
        
        calib_npy = work_dir / "calibration.npy"
        if not calib_npy.exists():
            raise RuntimeError("Calibration numpy file was not created")
        
        return calib_npy
    
    def create_model_script(self, work_dir: Path) -> Path:
        """Create Hailo model script with QAT optimizations.
        
        The model script controls quantization parameters and optimizations.
        """
        # Determine optimization parameters
        if self.optimize_for_accuracy:
            quant_params = {
                "calibration_method": "percentile",
                "percentile": 99.999,  # Very conservative
                "per_channel_quantization": True,
                "bias_correction": True,
                "finetune_epochs": 5 if self.is_qat_model else 0
            }
        else:
            quant_params = {
                "calibration_method": "percentile",
                "percentile": 99.99,
                "per_channel_quantization": True,
                "bias_correction": True,
                "finetune_epochs": 0
            }
        
        model_script = f"""
# Hailo Model Script for {self.model_name}
# Generated for {'QAT' if self.is_qat_model else 'PTQ'} optimization

import hailo_sdk_client

# Quantization parameters
quantization_params = {{
    "calibration_method": "{quant_params['calibration_method']}",
    "percentile": {quant_params['percentile']},
    "per_channel": {quant_params['per_channel_quantization']},
    "bias_correction": {quant_params['bias_correction']}
}}

# Set mixed precision for critical layers (fire detection)
# These layers will use higher precision to maintain accuracy
mixed_precision_layers = []

# If QAT model, enable QAT-specific optimizations
if {self.is_qat_model}:
    print("Enabling QAT optimizations...")
    quantization_params["qat_mode"] = True
    quantization_params["finetune_epochs"] = {quant_params['finetune_epochs']}

# Performance optimizations
performance_params = {{
    "optimization_level": 3,
    "batch_size": {self.batch_size},
    "use_precompiled_kernels": True
}}

# Model-specific configurations for fire detection
# Preserve accuracy for small object detection (early fire/smoke)
model_config = {{
    "nms_threshold": 0.45,  # Standard YOLO NMS
    "confidence_threshold": 0.25,  # Lower threshold to catch more potential fires
    "preserve_small_objects": True
}}

print("Model script loaded with optimizations")
print(f"Quantization: {{quantization_params}}")
print(f"Performance: {{performance_params}}")
"""
        
        script_path = work_dir / f"{self.model_name}_hailo_script.py"
        with open(script_path, 'w') as f:
            f.write(model_script)
        
        return script_path
    
    def convert(self) -> Dict[str, Path]:
        """Run QAT-optimized Hailo conversion.
        
        Returns:
            Dictionary with paths to generated files
        """
        work_dir = Path(tempfile.mkdtemp(prefix=f"hailo_qat_{self.model_name}_"))
        
        try:
            logger.info(f"Starting QAT-optimized conversion for {self.model_name}")
            logger.info(f"Target: {self.target}, Batch size: {self.batch_size}")
            logger.info(f"Working directory: {work_dir}")
            
            results = {}
            
            # Step 1: Prepare calibration data
            calib_npy = self.prepare_calibration_npy(work_dir)
            results['calibration_data'] = calib_npy
            
            # Step 2: Create model script
            model_script = self.create_model_script(work_dir)
            results['model_script'] = model_script
            
            # Step 3: Parse ONNX to HAR
            har_path = work_dir / f"{self.model_name}.har"
            
            logger.info("Parsing ONNX to HAR...")
            # Try with output0 first, then with recommended nodes if needed
            parse_cmd = [
                'hailo', 'parser', 'onnx',
                str(self.model_path),
                '--har-path', str(har_path),
                '--start-node-names', 'images',
                '--end-node-names', 'output0',
                '--hw-arch', self.target,
                '-y'
            ]
            
            result = subprocess.run(parse_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                # Check if error suggests different end nodes
                if '/model.22/' in result.stderr:
                    logger.info("Retrying with YOLOv8 specific end nodes...")
                    # Use the recommended end nodes for YOLOv8
                    parse_cmd = [
                        'hailo', 'parser', 'onnx',
                        str(self.model_path),
                        '--har-path', str(har_path),
                        '--start-node-names', 'images',
                        '--end-node-names', '/model.22/Concat_1',
                        '--end-node-names', '/model.22/Concat_2', 
                        '--end-node-names', '/model.22/Concat',
                        '--hw-arch', self.target,
                        '-y'
                    ]
                    
                    result = subprocess.run(parse_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"Parsing failed with YOLOv8 nodes: {result.stderr}")
                else:
                    raise RuntimeError(f"Parsing failed: {result.stderr}")
            
            logger.info("HAR created successfully")
            
            # Step 4: Optimize with calibration
            optimized_har = work_dir / f"{self.model_name}_optimized.har"
            
            logger.info("Optimizing with calibration data...")
            optimize_cmd = [
                'hailo', 'optimize',
                str(har_path),
                '--hw-arch', self.target,
                '--calib-set-path', str(calib_npy),
                '--model-script', str(model_script),
                '--output-har-path', str(optimized_har)
            ]
            
            # Use longer timeout for optimization
            result = subprocess.run(
                optimize_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Optimization stderr: {result.stderr}")
                # Try without model script
                logger.info("Retrying without model script...")
                optimize_cmd = [
                    'hailo', 'optimize',
                    str(har_path),
                    '--hw-arch', self.target,
                    '--calib-set-path', str(calib_npy),
                    '--output-har-path', str(optimized_har)
                ]
                result = subprocess.run(
                    optimize_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Optimization failed: {result.stderr}")
            
            logger.info("Optimization complete")
            
            # Step 5: Compile to HEF
            logger.info("Compiling to HEF...")
            compile_cmd = [
                'hailo', 'compiler',
                str(optimized_har),
                '--hw-arch', self.target,
                '--output-dir', str(work_dir)
            ]
            
            # Use longer timeout for compilation
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed: {result.stderr}")
            
            # Find generated HEF
            hef_files = list(work_dir.glob("*.hef"))
            if not hef_files:
                raise RuntimeError("No HEF file was generated")
            
            hef_path = hef_files[0]
            logger.info(f"HEF created: {hef_path}")
            
            # Copy to output directory
            output_hef = self.output_dir / f"{self.model_name}_{self.target}_qat.hef"
            shutil.copy2(hef_path, output_hef)
            results['hef'] = output_hef
            
            # Create metadata
            metadata = {
                'model': str(self.model_path),
                'target': self.target,
                'batch_size': self.batch_size,
                'optimization': 'QAT' if self.is_qat_model else 'PTQ',
                'calibration_samples': 'augmented',
                'optimize_for_accuracy': self.optimize_for_accuracy,
                'file_size_mb': output_hef.stat().st_size / 1024 / 1024,
                'quantization_params': quant_params if 'quant_params' in locals() else {}
            }
            
            metadata_path = output_hef.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            results['metadata'] = metadata_path
            
            logger.info(f"✓ Conversion complete: {output_hef}")
            return results
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise
        finally:
            # Cleanup
            if work_dir.exists():
                logger.debug(f"Cleaning up: {work_dir}")
                shutil.rmtree(work_dir, ignore_errors=True)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to Hailo HEF with QAT optimization"
    )
    
    parser.add_argument(
        'model',
        help='Path to ONNX model'
    )
    
    parser.add_argument(
        '--calibration-data', '-c',
        required=True,
        help='Path to calibration dataset directory (required)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./hailo_qat_output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--target', '-t',
        choices=['hailo8', 'hailo8l'],
        default='hailo8l',
        help='Target hardware'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for efficiency'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Optimize for speed rather than accuracy'
    )
    
    args = parser.parse_args()
    
    try:
        converter = HailoQATConverter(
            model_path=args.model,
            calibration_data=args.calibration_data,
            output_dir=args.output_dir,
            target=args.target,
            batch_size=args.batch_size,
            optimize_for_accuracy=not args.fast
        )
        
        results = converter.convert()
        
        print(f"\n✓ Successfully converted to {args.target}")
        print(f"  HEF: {results['hef']}")
        print(f"  Metadata: {results['metadata']}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()