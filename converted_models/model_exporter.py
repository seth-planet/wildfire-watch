#!/usr/bin/env python3.10
"""
Model Exporter for YOLO-NAS Models
Handles exporting trained models to various formats (ONNX, HEF, etc.)
"""
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import tempfile
import json

# Configure logging
logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles exporting YOLO-NAS models to various formats for deployment"""
    
    def __init__(self, logger_name: str = None):
        """Initialize the model exporter
        
        Args:
            logger_name: Optional custom logger name
        """
        self.logger = logging.getLogger(logger_name or __name__)
        
    def export_to_onnx(
        self,
        model_path: Path,
        output_dir: Path,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 11,
        simplify: bool = True
    ) -> Path:
        """Export PyTorch model to ONNX format.
        
        Args:
            model_path: Path to the PyTorch model checkpoint
            output_dir: Directory to save the ONNX model
            input_size: Model input size (height, width)
            opset_version: ONNX opset version to use
            simplify: Whether to simplify the ONNX model
            
        Returns:
            Path to the exported ONNX model
            
        Raises:
            ImportError: If required dependencies are not available
            RuntimeError: If export fails
        """
        self.logger.info("Exporting model to ONNX format")
        
        try:
            import torch
            from super_gradients.training import models
        except ImportError as e:
            raise ImportError(f"PyTorch or super-gradients not available: {e}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        model = models.get('yolo_nas_s', num_classes=32, checkpoint_path=str(model_path))
        model.eval()
        
        # Move to CPU for ONNX export
        model = model.cpu()
        
        # Prepare dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        
        # Export to ONNX
        onnx_path = output_dir / 'yolo_nas_qat.onnx'
        
        self.logger.info(f"Exporting to {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images'],  # Frigate-compatible name
            output_names=['output0'],  # Frigate-compatible name
            dynamic_axes={
                'images': {0: 'batch'},
                'output0': {0: 'batch'}
            }
        )
        
        # Optionally simplify the model
        if simplify:
            try:
                import onnx
                import onnxsim
                
                self.logger.info("Simplifying ONNX model")
                model_onnx = onnx.load(str(onnx_path))
                model_simp, check = onnxsim.simplify(model_onnx)
                
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    self.logger.info("ONNX model simplified successfully")
                else:
                    self.logger.warning("ONNX simplification check failed, using original model")
                    
            except ImportError:
                self.logger.info("onnx-simplifier not available, skipping simplification")
            except Exception as e:
                self.logger.warning(f"ONNX simplification failed: {e}")
        
        # Verify the exported model exists and has reasonable size
        if not onnx_path.exists():
            raise RuntimeError("ONNX export failed - file not created")
            
        model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"ONNX model exported successfully: {model_size_mb:.1f} MB")
        
        if model_size_mb < 1:
            raise RuntimeError(f"ONNX model suspiciously small: {model_size_mb:.1f} MB")
        
        return onnx_path
    
    def convert_to_hailo_hef(
        self,
        onnx_path: Path,
        output_dir: Path,
        calibration_data: Path,
        hailo_arch: str = "hailo8l",
        timeout_seconds: int = 7200,
        optimization_level: int = 2
    ) -> Path:
        """
        Convert ONNX model to Hailo HEF format using Docker.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory for HEF
            calibration_data: Path to calibration dataset
            hailo_arch: Target Hailo architecture (hailo8, hailo8l)
            timeout_seconds: Conversion timeout (default 2 hours)
            optimization_level: Optimization level 0-4 (higher = more optimization)
            
        Returns:
            Path to generated HEF file
            
        Raises:
            RuntimeError: If conversion fails
            subprocess.TimeoutExpired: If conversion times out
        """
        self.logger.info(f"Starting Hailo HEF conversion for {hailo_arch}")
        
        # Ensure paths exist
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not calibration_data.exists():
            raise FileNotFoundError(f"Calibration data not found: {calibration_data}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create conversion script
        conversion_script = output_dir / 'hailo_conversion.py'
        script_content = f'''#!/usr/bin/env python3
import hailo_model_optimization as hmo
import numpy as np
from pathlib import Path
import sys
import json

# Parse arguments
onnx_path = sys.argv[1]
output_path = sys.argv[2]
calib_path = sys.argv[3]

print(f"Converting {{onnx_path}} to HEF format")
print(f"Target architecture: {hailo_arch}")
print(f"Optimization level: {optimization_level}")

try:
    # Create runner with specific architecture
    runner = hmo.ModelRunner(
        model_path=onnx_path,
        hw_arch="{hailo_arch}"
    )
    
    # Load calibration dataset
    print("Loading calibration dataset...")
    calib_dataset = hmo.CalibrationDataset(
        calib_path,
        preprocessor=lambda x: x.astype(np.float32) / 255.0
    )
    
    # Configure optimization
    optimization_config = {{
        "optimization_level": {optimization_level},
        "compression_level": 1,  # Enable compression
        "batch_size": 8
    }}
    
    # Optimize model with QAT-aware settings
    print("Optimizing model...")
    quantized_model = runner.optimize(
        calib_dataset,
        optimization_config=optimization_config
    )
    
    # Compile to HEF
    print("Compiling to HEF...")
    hef_path = runner.compile(output_path)
    
    # Save metadata
    metadata = {{
        "architecture": "{hailo_arch}",
        "optimization_level": {optimization_level},
        "input_size": [640, 640],
        "num_classes": 32
    }}
    
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"HEF saved to: {{hef_path}}")
    print(f"Metadata saved to: {{metadata_path}}")
    
except Exception as e:
    print(f"ERROR: Conversion failed: {{e}}")
    raise
'''
        
        with open(conversion_script, 'w') as f:
            f.write(script_content)
        conversion_script.chmod(0o755)
        
        # Prepare Docker command
        docker_cmd = [
            'docker', 'run',
            '--rm',
            '-v', f'{output_dir.absolute()}:/workspace',
            '-v', f'{calibration_data.absolute()}:/calibration',
            '-v', f'{onnx_path.parent.absolute()}:/models',
            'hailo-ai/hailo-dataflow-compiler:latest',
            'python3', '/workspace/hailo_conversion.py',
            f'/models/{onnx_path.name}',
            '/workspace/yolo_nas_qat.hef',
            '/calibration'
        ]
        
        self.logger.info(f"Running Hailo conversion with timeout of {timeout_seconds/60:.0f} minutes")
        self.logger.debug(f"Docker command: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=True
            )
            
            self.logger.info("Hailo conversion completed successfully")
            if result.stdout:
                self.logger.debug(f"Conversion output:\n{result.stdout}")
            
            # Check for output HEF
            hef_path = output_dir / 'yolo_nas_qat.hef'
            if not hef_path.exists():
                raise RuntimeError("HEF file not generated")
            
            # Verify HEF file size
            hef_size_mb = hef_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"HEF file size: {hef_size_mb:.1f} MB")
            
            if hef_size_mb < 1:  # Suspiciously small
                raise RuntimeError(f"HEF file too small: {hef_size_mb:.1f} MB")
            
            # Check for metadata file
            metadata_path = output_dir / 'yolo_nas_qat.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Model metadata: {metadata}")
            
            return hef_path
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Hailo conversion timed out after {timeout_seconds/60:.0f} minutes")
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Hailo conversion failed with return code {e.returncode}")
            if e.stderr:
                self.logger.error(f"Error output:\n{e.stderr}")
            raise RuntimeError(f"Hailo conversion failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during Hailo conversion: {e}")
            raise
    
    def export_to_tflite(
        self,
        model_path: Path,
        output_dir: Path,
        input_size: Tuple[int, int] = (640, 640),
        quantization: str = 'int8',
        calibration_data: Optional[Path] = None
    ) -> Path:
        """Export model to TensorFlow Lite format.
        
        Args:
            model_path: Path to the model checkpoint
            output_dir: Directory to save the TFLite model
            input_size: Model input size (height, width)
            quantization: Quantization type ('fp32', 'fp16', 'int8')
            calibration_data: Path to calibration data for INT8 quantization
            
        Returns:
            Path to the exported TFLite model
        """
        self.logger.info(f"Exporting model to TFLite format with {quantization} quantization")
        
        # First export to ONNX as intermediate format
        onnx_path = self.export_to_onnx(model_path, output_dir, input_size)
        
        # Then convert ONNX to TFLite (implementation would go here)
        # This is a placeholder - actual implementation would use TF/TFLite converters
        
        tflite_path = output_dir / f'yolo_nas_{quantization}.tflite'
        self.logger.info(f"TFLite export to {tflite_path} would be implemented here")
        
        return tflite_path
    
    def export_to_tensorrt(
        self,
        onnx_path: Path,
        output_dir: Path,
        precision: str = 'fp16',
        workspace_size: int = 4096,
        calibration_data: Optional[Path] = None
    ) -> Path:
        """Export ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory to save the TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            workspace_size: GPU workspace size in MB
            calibration_data: Path to calibration data for INT8
            
        Returns:
            Path to the exported TensorRT engine
        """
        self.logger.info(f"Exporting model to TensorRT with {precision} precision")
        
        # This is a placeholder - actual implementation would use TensorRT Python API
        engine_path = output_dir / f'yolo_nas_{precision}.engine'
        self.logger.info(f"TensorRT export to {engine_path} would be implemented here")
        
        return engine_path
    
    def validate_export(self, exported_path: Path, format_type: str) -> bool:
        """Validate an exported model file.
        
        Args:
            exported_path: Path to the exported model
            format_type: Format type ('onnx', 'hef', 'tflite', 'tensorrt')
            
        Returns:
            True if validation passes
        """
        if not exported_path.exists():
            self.logger.error(f"Exported file does not exist: {exported_path}")
            return False
        
        file_size_mb = exported_path.stat().st_size / (1024 * 1024)
        self.logger.info(f"{format_type.upper()} file size: {file_size_mb:.1f} MB")
        
        # Basic size validation
        min_sizes = {
            'onnx': 10.0,    # ONNX models are typically 10-50 MB
            'hef': 5.0,      # HEF files are compressed, typically 5-20 MB
            'tflite': 2.0,   # TFLite INT8 models are small, 2-10 MB
            'tensorrt': 5.0  # TensorRT engines vary, typically 5-30 MB
        }
        
        min_size = min_sizes.get(format_type, 1.0)
        if file_size_mb < min_size:
            self.logger.warning(f"{format_type.upper()} file suspiciously small: {file_size_mb:.1f} MB < {min_size} MB")
            return False
        
        # Format-specific validation
        if format_type == 'onnx':
            try:
                import onnx
                model = onnx.load(str(exported_path))
                onnx.checker.check_model(model)
                self.logger.info("ONNX model validation passed")
                return True
            except Exception as e:
                self.logger.error(f"ONNX validation failed: {e}")
                return False
        
        # Other formats would have their specific validation
        return True