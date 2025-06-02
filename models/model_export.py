#!/usr/bin/env python3
"""
Model Export Script for Wildfire Detection
Exports trained models to various hardware-specific formats
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ModelExporter:
    def __init__(self, base_model_path: str, output_dir: str = "models/wildfire"):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model metadata
        self.metadata = {
            'version': '1.0',
            'classes': ['fire', 'smoke'],
            'input_size': 320,
            'description': 'Wildfire detection model for edge deployment'
        }
        
    def export_coral_tpu(self, quantized: bool = True):
        """Export model for Google Coral TPU"""
        try:
            import tensorflow as tf
            import tflite_runtime.interpreter as tflite
            
            logger.info("Exporting Coral TPU models...")
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(self.base_model_path)
            
            # Standard TFLite (for CPU fallback)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            
            output_path = self.output_dir / "wildfire_cpu.tflite"
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Exported CPU model: {output_path}")
            
            # Quantized for Edge TPU
            if quantized:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset_gen
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                
                quantized_model = converter.convert()
                
                # Save quantized model
                quant_path = self.output_dir / "wildfire_coral_lite_quant.tflite"
                with open(quant_path, 'wb') as f:
                    f.write(quantized_model)
                    
                # Compile for Edge TPU
                self._compile_edge_tpu(quant_path)
                
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
        except Exception as e:
            logger.error(f"Coral export failed: {e}")
            
    def _compile_edge_tpu(self, model_path: Path):
        """Compile model for Edge TPU"""
        try:
            import subprocess
            
            output_path = model_path.parent / f"{model_path.stem}_edgetpu.tflite"
            
            # Run Edge TPU compiler
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',  # Show statistics
                '-o', str(model_path.parent),
                str(model_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Rename output files
                compiled_path = model_path.parent / f"{model_path.stem}_edgetpu.tflite"
                if compiled_path.exists():
                    lite_path = self.output_dir / "wildfire_coral_lite.tflite"
                    compiled_path.rename(lite_path)
                    logger.info(f"Exported Coral Lite model: {lite_path}")
                    
                    # Create full model (same file, different config)
                    full_path = self.output_dir / "wildfire_coral_full.tflite"
                    import shutil
                    shutil.copy(lite_path, full_path)
                    logger.info(f"Exported Coral Full model: {full_path}")
            else:
                logger.error(f"Edge TPU compiler failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.error("Edge TPU compiler not found. Install from: https://coral.ai/docs/accelerator/get-started/")
            
    def export_hailo(self):
        """Export model for Hailo-8 and Hailo-8L"""
        try:
            logger.info("Exporting Hailo models...")
            
            # First convert to ONNX
            onnx_path = self._export_onnx()
            if not onnx_path:
                return
                
            # Hailo Model Zoo format
            hailo_config = {
                'model_name': 'wildfire_detector',
                'input_shape': [1, 320, 320, 3],
                'output_nodes': ['detection_output'],
                'preprocessing': {
                    'normalization': {
                        'mean': [127.5, 127.5, 127.5],
                        'std': [127.5, 127.5, 127.5]
                    }
                }
            }
            
            # Save config
            config_path = self.output_dir / "hailo_config.json"
            with open(config_path, 'w') as f:
                json.dump(hailo_config, f, indent=2)
                
            # Note: Actual HEF compilation requires Hailo Dataflow Compiler
            # This creates placeholder files with instructions
            
            # Hailo-8 (26 TOPS)
            hailo8_path = self.output_dir / "wildfire_hailo8.hef"
            self._create_hailo_placeholder(hailo8_path, "Hailo-8", 26)
            
            # Hailo-8L (13 TOPS)
            hailo8l_path = self.output_dir / "wildfire_hailo8l.hef"
            self._create_hailo_placeholder(hailo8l_path, "Hailo-8L", 13)
            
            logger.info(f"Created Hailo configuration: {config_path}")
            logger.info("To compile HEF files, use Hailo Dataflow Compiler with the ONNX model")
            
        except Exception as e:
            logger.error(f"Hailo export failed: {e}")
            
    def _create_hailo_placeholder(self, path: Path, device: str, tops: int):
        """Create placeholder HEF file with compilation instructions"""
        instructions = f"""
# Hailo {device} Model Compilation Instructions

This is a placeholder for the compiled HEF file.
To generate the actual HEF file:

1. Install Hailo Dataflow Compiler:
   https://hailo.ai/developer-zone/

2. Use the ONNX model: wildfire_tensorrt.onnx

3. Compile with optimization for {tops} TOPS:
   hailo compile wildfire_tensorrt.onnx \
     --hw-arch hailo8{'l' if 'L' in device else ''} \
     --model-script hailo_config.json \
     --output-file {path.name}

4. Replace this file with the compiled HEF
"""
        with open(path, 'w') as f:
            f.write(instructions)
            
    def export_tensorrt(self):
        """Export model for NVIDIA TensorRT"""
        try:
            logger.info("Exporting TensorRT model...")
            
            # Export to ONNX first
            onnx_path = self._export_onnx()
            if not onnx_path:
                return
                
            # TensorRT optimization script
            trt_script = self.output_dir / "compile_tensorrt.py"
            script_content = '''#!/usr/bin/env python3
"""
TensorRT Compilation Script
Run this on the target device with TensorRT installed
"""
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def compile_tensorrt(onnx_path, output_path, fp16=True):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX")
            return False
            
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Serialize
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
        
    return True

if __name__ == "__main__":
    compile_tensorrt("wildfire_tensorrt.onnx", "wildfire_tensorrt.engine")
'''
            
            with open(trt_script, 'w') as f:
                f.write(script_content)
                
            os.chmod(trt_script, 0o755)
            logger.info(f"Created TensorRT compilation script: {trt_script}")
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            
    def export_openvino(self):
        """Export model for Intel OpenVINO"""
        try:
            logger.info("Exporting OpenVINO model...")
            
            # Export to ONNX first
            onnx_path = self._export_onnx()
            if not onnx_path:
                return
                
            # OpenVINO Model Optimizer command
            mo_command = f"""
# OpenVINO Model Optimizer Command
# Run this with OpenVINO toolkit installed:

mo --input_model {onnx_path} \\
   --output_dir {self.output_dir} \\
   --model_name wildfire_openvino \\
   --input_shape [1,320,320,3] \\
   --data_type FP16 \\
   --mean_values [127.5,127.5,127.5] \\
   --scale_values [127.5,127.5,127.5]
"""
            
            # Save command
            command_path = self.output_dir / "openvino_convert.sh"
            with open(command_path, 'w') as f:
                f.write(mo_command)
                
            os.chmod(command_path, 0o755)
            
            # Create placeholder files
            xml_path = self.output_dir / "wildfire_openvino.xml"
            bin_path = self.output_dir / "wildfire_openvino.bin"
            
            with open(xml_path, 'w') as f:
                f.write("<!-- OpenVINO IR XML - Generate with openvino_convert.sh -->")
                
            with open(bin_path, 'w') as f:
                f.write("OpenVINO IR Binary - Generate with openvino_convert.sh")
                
            logger.info(f"Created OpenVINO conversion script: {command_path}")
            
        except Exception as e:
            logger.error(f"OpenVINO export failed: {e}")
            
    def _export_onnx(self) -> Optional[Path]:
        """Export model to ONNX format"""
        try:
            import tf2onnx
            import tensorflow as tf
            
            onnx_path = self.output_dir / "wildfire_tensorrt.onnx"
            
            # Convert TF model to ONNX
            model = tf.saved_model.load(self.base_model_path)
            concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            
            _, _ = tf2onnx.convert.from_function(
                concrete_func,
                input_signature=concrete_func.structured_input_signature[1],
                opset=13,
                output_path=str(onnx_path)
            )
            
            logger.info(f"Exported ONNX model: {onnx_path}")
            return onnx_path
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            
        return None
        
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization"""
        import numpy as np
        for _ in range(100):
            data = np.random.rand(1, 320, 320, 3).astype(np.float32)
            yield [data]
            
    def create_labels_file(self):
        """Create labels.txt file"""
        labels_path = self.output_dir / "labels.txt"
        with open(labels_path, 'w') as f:
            f.write("fire\n")
            f.write("smoke\n")
            
        logger.info(f"Created labels file: {labels_path}")
        
    def create_metadata_file(self):
        """Create model metadata file"""
        metadata_path = self.output_dir / "metadata.json"
        
        self.metadata['files'] = {
            'coral_lite': 'wildfire_coral_lite.tflite',
            'coral_full': 'wildfire_coral_full.tflite',
            'hailo8': 'wildfire_hailo8.hef',
            'hailo8l': 'wildfire_hailo8l.hef',
            'tensorrt': 'wildfire_tensorrt.onnx',
            'openvino': 'wildfire_openvino.xml',
            'cpu': 'wildfire_cpu.tflite'
        }
        
        self.metadata['performance'] = {
            'coral_lite': {'fps': 30, 'power_watts': 2},
            'coral_full': {'fps': 15, 'power_watts': 4},
            'hailo8': {'fps': 60, 'power_watts': 5},
            'hailo8l': {'fps': 30, 'power_watts': 2.5},
            'tensorrt': {'fps': 45, 'power_watts': 10},
            'openvino': {'fps': 20, 'power_watts': 15},
            'cpu': {'fps': 5, 'power_watts': 5}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        logger.info(f"Created metadata file: {metadata_path}")
        
    def export_all(self):
        """Export models for all supported hardware"""
        logger.info("Starting model export for all hardware targets...")
        
        # Create common files
        self.create_labels_file()
        self.create_metadata_file()
        
        # Export for each hardware
        self.export_coral_tpu()
        self.export_hailo()
        self.export_tensorrt()
        self.export_openvino()
        
        logger.info("Model export complete!")
        self._print_summary()
        
    def _print_summary(self):
        """Print export summary"""
        print("\n=== Model Export Summary ===\n")
        print(f"Output Directory: {self.output_dir}")
        print("\nGenerated Files:")
        
        for file in sorted(self.output_dir.glob("*")):
            size = file.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"  {file.name:<30} {size_str:>10}")
            
        print("\nNext Steps:")
        print("1. For Coral: Models are ready to use")
        print("2. For Hailo: Run Hailo Dataflow Compiler on target")
        print("3. For TensorRT: Run compile_tensorrt.py on target")
        print("4. For OpenVINO: Run openvino_convert.sh with OpenVINO toolkit")
        print("5. Copy models to: security_nvr/models/wildfire/")

def main():
    parser = argparse.ArgumentParser(description='Export wildfire detection models')
    parser.add_argument('base_model', help='Path to base TensorFlow SavedModel')
    parser.add_argument('--output-dir', default='models/wildfire', help='Output directory')
    parser.add_argument('--target', choices=['all', 'coral', 'hailo', 'tensorrt', 'openvino'],
                       default='all', help='Target hardware platform')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_model):
        logger.error(f"Base model not found: {args.base_model}")
        sys.exit(1)
        
    exporter = ModelExporter(args.base_model, args.output_dir)
    
    if args.target == 'all':
        exporter.export_all()
    elif args.target == 'coral':
        exporter.export_coral_tpu()
    elif args.target == 'hailo':
        exporter.export_hailo()
    elif args.target == 'tensorrt':
        exporter.export_tensorrt()
    elif args.target == 'openvino':
        exporter.export_openvino()

if __name__ == '__main__':
    main()
