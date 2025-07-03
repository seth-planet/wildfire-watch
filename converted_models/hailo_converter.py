#!/usr/bin/env python3.10
"""Hailo Model Converter - Converts ONNX models to HEF format.

This module implements the actual Hailo conversion workflow using the Hailo SDK.
It performs the three-stage conversion process:
1. Parse: ONNX → HAR (Hailo Archive)
2. Optimize: Quantization and optimization
3. Compile: HAR → HEF (Hailo Executable Format)

The converter supports both Hailo-8 (26 TOPS) and Hailo-8L (13 TOPS) targets
with appropriate optimizations for each.

Usage:
    python3.10 hailo_converter.py model.onnx --output model.hef
    python3.10 hailo_converter.py model.onnx --target hailo8l --calibration-data ./images/

Requirements:
    - Python 3.10 (required by Hailo SDK)
    - hailo_dataflow_compiler (DFC)
    - Calibration dataset for quantization
"""

import os
import sys
import json
import shutil
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class HailoConverter:
    """Converts ONNX models to Hailo HEF format.
    
    This class handles the complete conversion pipeline from ONNX to HEF,
    including quantization calibration and optimization for specific Hailo
    hardware targets.
    
    Attributes:
        model_path: Path to input ONNX model
        output_path: Path for output HEF file
        target: Hailo hardware target ('hailo8' or 'hailo8l')
        calibration_data: Path to calibration dataset
        batch_size: Batch size for conversion (always 1 for Frigate)
        optimization_level: DFC optimization level (0-3)
    """
    
    # Hailo hardware specifications
    HAILO_TARGETS = {
        'hailo8': {
            'arch': 'hailo8',
            'tops': 26,
            'optimization_level': 3,  # Maximum performance
            'resources_mode': 'performance'
        },
        'hailo8l': {
            'arch': 'hailo8l',
            'tops': 13,
            'optimization_level': 2,  # Balanced for lower power
            'resources_mode': 'normal'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        target: str = 'hailo8',
        calibration_data: Optional[str] = None,
        batch_size: int = 1,
        optimization_level: Optional[int] = None
    ):
        """Initialize Hailo converter.
        
        Args:
            model_path: Path to ONNX model
            output_path: Output HEF path (auto-generated if not provided)
            target: Target hardware ('hailo8' or 'hailo8l')
            calibration_data: Path to calibration images
            batch_size: Batch size (default 1 for Frigate compatibility)
            optimization_level: Override default optimization level
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set output path
        if output_path:
            self.output_path = Path(output_path)
        else:
            self.output_path = self.model_path.with_suffix(f'.{target}.hef')
        
        # Validate target
        if target not in self.HAILO_TARGETS:
            raise ValueError(f"Invalid target: {target}. Must be one of {list(self.HAILO_TARGETS.keys())}")
        
        self.target = target
        self.target_config = self.HAILO_TARGETS[target]
        self.calibration_data = Path(calibration_data) if calibration_data else None
        self.batch_size = batch_size
        self.optimization_level = optimization_level or self.target_config['optimization_level']
        
        # Working directory for intermediate files
        self.work_dir = Path(tempfile.mkdtemp(prefix='hailo_convert_'))
        logger.info(f"Working directory: {self.work_dir}")
    
    def check_prerequisites(self) -> bool:
        """Check if Hailo SDK is installed and accessible.
        
        Returns:
            True if all prerequisites are met
            
        Raises:
            RuntimeError: If Hailo SDK is not found
        """
        # Check for hailo_dataflow_compiler
        try:
            result = subprocess.run(
                ['hailo', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Hailo SDK found: {result.stdout.strip()}")
                return True
            else:
                raise RuntimeError("Hailo SDK not properly installed")
        except FileNotFoundError:
            raise RuntimeError(
                "Hailo dataflow compiler not found. "
                "Please ensure Hailo SDK is installed and in PATH. "
                "Install with: pip install hailo_dataflow_compiler"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Hailo SDK check timed out")
    
    def prepare_calibration_dataset(self) -> Path:
        """Prepare calibration dataset for quantization.
        
        Creates a calibration dataset configuration file that the Hailo
        compiler can use for INT8 quantization.
        
        Returns:
            Path to calibration config file
            
        Raises:
            ValueError: If calibration data directory is invalid
        """
        if not self.calibration_data or not self.calibration_data.exists():
            raise ValueError(f"Invalid calibration data path: {self.calibration_data}")
        
        # Get list of image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.calibration_data.glob(f'*{ext}'))
            image_files.extend(self.calibration_data.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No images found in {self.calibration_data}")
        
        logger.info(f"Found {len(image_files)} calibration images")
        
        # Create calibration config
        calib_config = {
            "dataset_name": "wildfire_calibration",
            "path": str(self.calibration_data),
            "format": "image",
            "num_images": min(len(image_files), 2000),  # Use up to 2000 images
            "preprocessing": {
                "input_shape": None,  # Will be determined from model
                "normalize": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [255.0, 255.0, 255.0]
                },
                "data_type": "float32"
            }
        }
        
        calib_config_path = self.work_dir / "calibration_config.json"
        with open(calib_config_path, 'w') as f:
            json.dump(calib_config, f, indent=2)
        
        return calib_config_path
    
    def parse_onnx_to_har(self) -> Path:
        """Parse ONNX model to HAR (Hailo Archive).
        
        This is the first stage of conversion where the ONNX model is
        parsed and converted to Hailo's intermediate representation.
        
        Returns:
            Path to generated HAR file
            
        Raises:
            RuntimeError: If parsing fails
        """
        har_path = self.work_dir / f"{self.model_path.stem}.har"
        
        logger.info("Stage 1: Parsing ONNX to HAR...")
        
        # Build parse command
        cmd = [
            'hailo', 'parser', 'onnx',
            str(self.model_path),
            '--har-path', str(har_path),
            '--start-node-names', 'images',  # Frigate standard input
            '--end-node-names', 'output0,output1,output2',  # All YOLO outputs for multi-scale detection
            '--net-name', self.model_path.stem,
            '--hw-arch', self.target_config['arch'],
            '--tensor-shapes', f'images=[1,3,640,640]',  # Batch size 1 for Frigate compatibility
            '-y'  # Auto-yes to recommendations
        ]
        
        logger.debug(f"Parse command: {' '.join(cmd)}")
        
        # Run parser
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=self.work_dir
            )
            
            if result.returncode != 0:
                logger.error(f"Parser stdout: {result.stdout}")
                logger.error(f"Parser stderr: {result.stderr}")
                raise RuntimeError(f"ONNX parsing failed: {result.stderr}")
            
            if not har_path.exists():
                raise RuntimeError("HAR file was not created")
            
            logger.info(f"HAR created: {har_path} ({har_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return har_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("ONNX parsing timed out after 10 minutes")
    
    def apply_yolo_postprocessing(self, har_path: Path) -> None:
        """Apply YOLO NMS post-processing to the HAR model.
        
        This adds YOLO-specific NMS (Non-Maximum Suppression) post-processing
        to enable on-device detection output instead of raw feature maps.
        
        Args:
            har_path: Path to HAR file to modify
        """
        logger.info("Applying YOLO NMS post-processing...")
        
        # Create NMS configuration
        nms_config = {
            "nms_scores_th": 0.3,  # Confidence threshold
            "nms_iou_th": 0.6,     # IoU threshold for NMS
            "nms_max_predictions": 300,  # Max detections per image
            "nms_classes": 5,      # Number of classes (fire, smoke, person, vehicle, wildlife)
            "engine": "hailo"      # Run NMS on Hailo device for best performance
        }
        
        # Save NMS config
        config_path = self.work_dir / "nms_config.json"
        with open(config_path, 'w') as f:
            json.dump(nms_config, f, indent=2)
        
        # Apply NMS post-processing using model script
        cmd = [
            'hailo', 'model-script',
            str(har_path),
            '--cmd', 'nms_postprocess',
            '--config-path', str(config_path),
            '--onnx-path', str(self.model_path),  # Original ONNX for reference
            '--postprocess-type', 'yolov8',  # YOLOv8 style NMS
            '-o', str(har_path)  # Overwrite with NMS-enabled version
        ]
        
        logger.debug(f"NMS command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.work_dir
            )
            
            if result.returncode != 0:
                logger.warning(f"NMS application failed: {result.stderr}")
                logger.warning("Proceeding without NMS (may affect detection output)")
            else:
                logger.info("YOLO NMS post-processing applied successfully")
                
        except subprocess.TimeoutExpired:
            logger.warning("NMS application timed out")
        except Exception as e:
            logger.warning(f"NMS application error: {e}")
    
    def optimize_har(self, har_path: Path) -> Path:
        """Optimize HAR with quantization.
        
        This stage performs INT8 quantization using the calibration dataset
        and applies various optimizations for the target hardware.
        
        Args:
            har_path: Path to input HAR file
            
        Returns:
            Path to optimized HAR file
            
        Raises:
            RuntimeError: If optimization fails
        """
        optimized_har = self.work_dir / f"{self.model_path.stem}_optimized.har"
        
        logger.info("Stage 2: Optimizing with quantization...")
        
        # Prepare calibration if available
        calib_args = []
        if self.calibration_data:
            try:
                calib_config = self.prepare_calibration_dataset()
                calib_args = [
                    '--calibration-dataset', str(self.calibration_data),
                    '--calibration-dataset-format', 'image'
                ]
            except Exception as e:
                logger.warning(f"Calibration preparation failed: {e}")
                logger.warning("Proceeding without calibration (may reduce accuracy)")
        
        # Build optimization command
        cmd = [
            'hailo', 'optimize',
            str(har_path),
            '--har-path', str(optimized_har),
            '--use-random-calib-set',  # Fallback if no calibration provided
            '--quantization-precision', 'int8',
            '--optimization-level', str(self.optimization_level),
            '--resources-mode', self.target_config['resources_mode']
            # Removed --no-nms to enable on-device YOLO NMS post-processing
        ] + calib_args
        
        logger.debug(f"Optimize command: {' '.join(cmd)}")
        
        # Run optimizer
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=self.work_dir
            )
            
            if result.returncode != 0:
                logger.error(f"Optimizer stdout: {result.stdout}")
                logger.error(f"Optimizer stderr: {result.stderr}")
                raise RuntimeError(f"Optimization failed: {result.stderr}")
            
            if not optimized_har.exists():
                raise RuntimeError("Optimized HAR was not created")
            
            logger.info(f"Optimized HAR: {optimized_har} ({optimized_har.stat().st_size / 1024 / 1024:.1f} MB)")
            return optimized_har
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Optimization timed out after 30 minutes")
    
    def compile_har_to_hef(self, har_path: Path) -> Path:
        """Compile HAR to HEF (Hailo Executable Format).
        
        Final stage that compiles the optimized HAR into a HEF file that
        can be loaded and executed on Hailo hardware.
        
        Args:
            har_path: Path to optimized HAR file
            
        Returns:
            Path to compiled HEF file
            
        Raises:
            RuntimeError: If compilation fails
        """
        hef_path = self.work_dir / f"{self.model_path.stem}_{self.target}.hef"
        
        logger.info(f"Stage 3: Compiling for {self.target}...")
        
        # Build compile command
        cmd = [
            'hailo', 'compile',
            str(har_path),
            '--hw-arch', self.target_config['arch'],
            '--output-dir', str(self.work_dir)
        ]
        
        logger.debug(f"Compile command: {' '.join(cmd)}")
        
        # Run compiler
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=self.work_dir
            )
            
            if result.returncode != 0:
                logger.error(f"Compiler stdout: {result.stdout}")
                logger.error(f"Compiler stderr: {result.stderr}")
                raise RuntimeError(f"Compilation failed: {result.stderr}")
            
            if not hef_path.exists():
                raise RuntimeError("HEF file was not created")
            
            logger.info(f"HEF compiled: {hef_path} ({hef_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return hef_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compilation timed out after 30 minutes")
    
    def convert(self) -> Path:
        """Run complete conversion pipeline.
        
        Executes all three stages of conversion and copies the final HEF
        to the output location.
        
        Returns:
            Path to final HEF file
            
        Raises:
            RuntimeError: If any conversion stage fails
        """
        try:
            # Check prerequisites
            self.check_prerequisites()
            
            # Stage 1: Parse ONNX to HAR
            har_path = self.parse_onnx_to_har()
            
            # Stage 1.5: Apply YOLO NMS post-processing
            self.apply_yolo_postprocessing(har_path)
            
            # Stage 2: Optimize with quantization
            optimized_har = self.optimize_har(har_path)
            
            # Stage 3: Compile to HEF
            hef_path = self.compile_har_to_hef(optimized_har)
            
            # Copy to final location
            logger.info(f"Copying HEF to: {self.output_path}")
            shutil.copy2(hef_path, self.output_path)
            
            # Create metadata file
            metadata = {
                'source_model': str(self.model_path),
                'target': self.target,
                'target_specs': self.target_config,
                'optimization_level': self.optimization_level,
                'batch_size': self.batch_size,
                'calibration_used': self.calibration_data is not None,
                'file_size_mb': self.output_path.stat().st_size / 1024 / 1024
            }
            
            metadata_path = self.output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Conversion complete! Output: {self.output_path}")
            return self.output_path
            
        finally:
            # Cleanup working directory
            if self.work_dir.exists():
                logger.debug(f"Cleaning up: {self.work_dir}")
                shutil.rmtree(self.work_dir, ignore_errors=True)
    
    def validate_conversion(self) -> bool:
        """Validate the converted HEF file.
        
        Uses inspect_hef.py to verify the model structure matches
        Frigate's expectations.
        
        Returns:
            True if validation passes
        """
        # Check if inspect_hef.py exists
        inspect_script = Path(__file__).parent.parent / 'scripts' / 'inspect_hef.py'
        if not inspect_script.exists():
            logger.warning("inspect_hef.py not found, skipping validation")
            return True
        
        try:
            result = subprocess.run(
                ['python3.10', str(inspect_script), str(self.output_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("HEF validation passed")
                return True
            else:
                logger.warning(f"HEF validation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            return False


def main():
    """Command-line interface for Hailo converter."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to Hailo HEF format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'model',
        help='Path to ONNX model file'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output HEF file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--target', '-t',
        choices=['hailo8', 'hailo8l'],
        default='hailo8',
        help='Target Hailo hardware (default: hailo8)'
    )
    
    parser.add_argument(
        '--calibration-data', '-c',
        help='Path to calibration dataset directory'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size (default: 8 for efficient inference)'
    )
    
    parser.add_argument(
        '--optimization-level',
        type=int,
        choices=[0, 1, 2, 3],
        help='Override optimization level (0-3)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate HEF after conversion'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create converter
        converter = HailoConverter(
            model_path=args.model,
            output_path=args.output,
            target=args.target,
            calibration_data=args.calibration_data,
            batch_size=args.batch_size,
            optimization_level=args.optimization_level
        )
        
        # Run conversion
        output_path = converter.convert()
        
        # Validate if requested
        if args.validate:
            if converter.validate_conversion():
                print(f"\n✓ Conversion successful: {output_path}")
                sys.exit(0)
            else:
                print(f"\n⚠ Conversion completed with warnings: {output_path}")
                sys.exit(1)
        else:
            print(f"\n✓ Conversion complete: {output_path}")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()