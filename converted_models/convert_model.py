CALIBRATION_PATH="{Path(self.calibration_data).absolute()}"
OUTPUT_DIR="{self.output_dir.absolute()}"

# Step 1: Parse ONNX model
echo "Parsing ONNX model..."
hailo parser onnx $ONNX_PATH --end-node-names {' '.join(self._get_yolo_output_names())} --start-node-names images

# Step 2: Optimize model
echo "Optimizing model..."
hailo optimize $MODEL_NAME.har --use-random-calib-set --calib-set-size 100

# Step 3: Compile for Hailo-8
echo "Compiling for Hailo-8..."
hailo compiler $MODEL_NAME_optimized.har --hw-arch hailo8

# Step 4: Compile for Hailo-8L
echo "Compiling for Hailo-8L..."
hailo compiler $MODEL_NAME_optimized.har --hw-arch hailo8l

# Move outputs
mv *.hef $OUTPUT_DIR/

echo "Conversion complete!"
echo "Hailo-8 model: $OUTPUT_DIR/${{MODEL_NAME}}_hailo8.hef"
echo "Hailo-8L model: $OUTPUT_DIR/${{MODEL_NAME}}_hailo8l.hef"
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_hailo.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _check_hailo_tools(self) -> bool:
        """Check if Hailo tools are available"""
        try:
            result = subprocess.run(['hailo', '--version'], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _run_hailo_conversion(self, onnx_path: Path, config_path: Path) -> Optional[Path]:
        """Run Hailo conversion if tools available"""
        try:
            # This would require running inside Hailo Docker
            # For now, we just prepare the files
            logger.info("Hailo conversion requires Hailo Docker environment")
            return None
        except Exception as e:
            logger.error(f"Hailo conversion error: {e}")
            return None
    
    def convert_to_openvino(self, onnx_path: Optional[Path] = None) -> Dict[str, Path]:
        """Convert model to OpenVINO format"""
        logger.info("Converting to OpenVINO format...")
        
        results = {}
        
        # Use provided ONNX or convert
        if not onnx_path:
            onnx_path = self.convert_to_onnx()
        
        # Create conversion script
        script = f"""#!/bin/bash
# OpenVINO Model Conversion Script

# Check if OpenVINO is installed
if ! command -v mo &> /dev/null; then
    echo "OpenVINO Model Optimizer not found!"
    echo "Please install OpenVINO toolkit from: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
    exit 1
fi

# Convert model
mo \\
    --input_model {onnx_path.absolute()} \\
    --output_dir {self.output_dir.absolute()} \\
    --model_name {self.model_name}_openvino \\
    --input_shape [1,3,{self.model_info['input_size']},{self.model_info['input_size']}] \\
    --data_type FP16 \\
    --mean_values [0,0,0] \\
    --scale_values [255,255,255]

echo "OpenVINO conversion complete!"
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_openvino.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        results['script'] = script_path
        
        # Try to run if OpenVINO available
        try:
            result = subprocess.run(['mo', '--help'], capture_output=True)
            if result.returncode == 0:
                # Run conversion
                subprocess.run(['bash', str(script_path)], check=True)
                
                # Check for output files
                xml_path = self.output_dir / f"{self.model_name}_openvino.xml"
                bin_path = self.output_dir / f"{self.model_name}_openvino.bin"
                
                if xml_path.exists() and bin_path.exists():
                    results['xml'] = xml_path
                    results['bin'] = bin_path
                    logger.info(f"OpenVINO model saved: {xml_path}")
        except FileNotFoundError:
            logger.warning("OpenVINO tools not found. Please run the conversion script manually.")
        except Exception as e:
            logger.error(f"OpenVINO conversion error: {e}")
        
        return results
    
    def convert_to_tensorrt(self, onnx_path: Optional[Path] = None) -> Dict[str, Path]:
        """Create TensorRT conversion script"""
        logger.info("Creating TensorRT conversion script...")
        
        results = {}
        
        # Use provided ONNX or convert
        if not onnx_path:
            onnx_path = self.convert_to_onnx()
        
        # Create Python conversion script
        script = f"""#!/usr/bin/env python3
\"\"\"
TensorRT Conversion Script for {self.model_name}
Run this on the target device with TensorRT installed
\"\"\"
import tensorrt as trt
import numpy as np

def convert_onnx_to_tensorrt(
    onnx_path="{onnx_path.absolute()}",
    engine_path="{self.output_dir.absolute() / f'{self.model_name}_tensorrt.engine'}",
    fp16=True,
    workspace_size=1 << 30
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    print(f"Parsing ONNX model: {{onnx_path}}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    # Enable FP16 if requested and supported
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision")
    
    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    # Save engine
    print(f"Saving engine to: {{engine_path}}")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print("TensorRT conversion complete!")
    return True

if __name__ == "__main__":
    try:
        import tensorrt as trt
        convert_onnx_to_tensorrt()
    except ImportError:
        print("TensorRT not installed!")
        print("Please install TensorRT from: https://developer.nvidia.com/tensorrt")
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_tensorrt.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        results['script'] = script_path
        
        logger.info(f"TensorRT conversion script saved: {script_path}")
        logger.info("Run this script on your target device with TensorRT installed")
        
        return results
    
    def generate_frigate_config(self) -> Path:
        """Generate Frigate configuration for the model"""
        logger.info("Generating Frigate configuration...")
        
        config = {
            'model': {
                'path': f"/models/{self.model_name}.tflite",  # Default to TFLite
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': self.model_info['input_size'],
                'height': self.model_info['input_size'],
                'labels': f"/models/{self.model_name}_labels.txt"
            },
            'objects': {
                'track': self.model_info['classes'][:10],  # Track first 10 classes
                'filters': {}
            }
        }
        
        # Add filters for fire-related classes
        fire_classes = ['fire', 'smoke', 'flame', 'wildfire']
        for cls in self.model_info['classes']:
            if any(fire_word in cls.lower() for fire_word in fire_classes):
                config['objects']['filters'][cls] = {
                    'min_area': 300,
                    'max_area': 100000,
                    'min_score': 0.6,
                    'threshold': 0.7
                }
        
        # Save config
        config_path = self.output_dir / f"{self.model_name}_frigate_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Save labels file
        labels_path = self.output_dir / f"{self.model_name}_labels.txt"
        with open(labels_path, 'w') as f:
            for cls in self.model_info['classes']:
                f.write(f"{cls}\n")
        
        logger.info(f"Frigate config saved: {config_path}")
        logger.info(f"Labels file saved: {labels_path}")
        
        return config_path
    
    def convert_all(self) -> Dict[str, Dict]:
        """Convert to all supported formats"""
        logger.info("Converting to all formats...")
        
        results = {
            'model_info': self.model_info,
            'outputs': {}
        }
        
        # Convert to ONNX (base for other formats)
        onnx_path = self.convert_to_onnx()
        results['outputs']['onnx'] = {'path': onnx_path}
        
        # Convert to TFLite/Coral
        tflite_results = self.convert_to_tflite()
        if tflite_results:
            results['outputs']['tflite'] = tflite_results
        
        # Convert to Hailo
        hailo_results = self.convert_to_hailo(onnx_path)
        if hailo_results:
            results['outputs']['hailo'] = hailo_results
        
        # Convert to OpenVINO
        openvino_results = self.convert_to_openvino(onnx_path)
        if openvino_results:
            results['outputs']['openvino'] = openvino_results
        
        # Create TensorRT script
        tensorrt_results = self.convert_to_tensorrt(onnx_path)
        if tensorrt_results:
            results['outputs']['tensorrt'] = tensorrt_results
        
        # Generate Frigate config
        frigate_config = self.generate_frigate_config()
        results['outputs']['frigate'] = {'config': frigate_config}
        
        # Save summary
        summary_path = self.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Conversion summary saved: {summary_path}")
        
        # Create README
        self._create_readme(results)
        
        return results
    
    def _create_readme(self, results: Dict):
        """Create README with deployment instructions"""
        readme = f"""# {self.model_name} - Converted Models

## Model Information
- **Type**: {self.model_info['type']}
- **Classes**: {self.model_info['num_classes']}
- **Input Size**: {self.model_info['input_size']}x{self.model_info['input_size']}

## Converted Formats

### 1. ONNX
- **File**: `{self.model_name}.onnx`
- **Use**: Base format for other conversions

### 2. TensorFlow Lite (Coral)
- **CPU Model**: `{self.model_name}_cpu.tflite`
- **Quantized Model**: `{self.model_name}_quant.tflite`
- **Edge TPU Model**: `{self.model_name}_edgetpu.tflite`

### 3. Hailo
- **Config**: `{self.model_name}_hailo_config.json`
- **Script**: `convert_{self.model_name}_hailo.sh`
- Run the script inside Hailo Docker to generate HEF files

### 4. OpenVINO
- **Script**: `convert_{self.model_name}_openvino.sh`
- Run with OpenVINO toolkit installed

### 5. TensorRT
- **Script**: `convert_{self.model_name}_tensorrt.py`
- Run on target device with TensorRT

## Deployment with Frigate

1. Copy model files to Frigate container:
   ```bash
   docker cp {self.model_name}.tflite frigate:/models/
   docker cp {self.model_name}_labels.txt frigate:/models/
   ```

2. Update Frigate config with generated settings from `{self.model_name}_frigate_config.yml`

3. Restart Frigate to load new model

## Class Labels
"""
        
        # Add class list
        for i, cls in enumerate(self.model_info['classes']):
            readme += f"- {i}: {cls}\n"
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        logger.info(f"README saved: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO models for edge deployment')
    parser.add_argument('model', help='Path to .pt model file')
    parser.add_argument('--output-dir', default='converted_models', help='Output directory')
    parser.add_argument('--name', help='Model name (default: from filename)')
    parser.add_argument('--calibration-data', help='Path to calibration images')
    parser.add_argument('--formats', nargs='+', 
                       choices=['all', 'onnx', 'tflite', 'hailo', 'openvino', 'tensorrt'],
                       default=['all'], help='Formats to convert to')
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    # Create converter
    converter = ModelConverter(
        model_path=args.model,
        output_dir=args.output_dir,
        model_name=args.name,
        calibration_data=args.calibration_data
    )
    
    # Convert based on requested formats
    if 'all' in args.formats:
        converter.convert_all()
    else:
        results = {'model_info': converter.model_info, 'outputs': {}}
        
        if 'onnx' in args.formats:
            results['outputs']['onnx'] = converter.convert_to_onnx()
        
        if 'tflite' in args.formats:
            results['outputs']['tflite'] = converter.convert_to_tflite()
        
        if 'hailo' in args.formats:
            results['outputs']['hailo'] = converter.convert_to_hailo()
        
        if 'openvino' in args.formats:
            results['outputs']['openvino'] = converter.convert_to_openvino()
        
        if 'tensorrt' in args.formats:
            results['outputs']['tensorrt'] = converter.convert_to_tensorrt()
    
    logger.info("Conversion complete!")
    logger.info(f"Output directory: {converter.output_dir}")

if __name__ == '__main__':
    main()#!/usr/bin/env python3
"""
Wildfire Watch Model Converter
Converts YOLOv8/v9/v11 PyTorch models to deployment formats for edge devices
Supports: Hailo (HEF), Coral (TFLite), ONNX, OpenVINO, TensorRT
"""
import os
import sys
import json
import yaml
import shutil
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

class ModelConverter:
    """Converts YOLO models to various edge deployment formats"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "converted_models",
        model_name: Optional[str] = None,
        calibration_data: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.model_path.stem
        
        # Load YOLO model
        logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Extract model info
        self.model_info = self._extract_model_info()
        
        # Calibration data
        self.calibration_data = calibration_data
        if not self.calibration_data:
            logger.warning("No calibration data provided. Using default COCO samples.")
            self._download_calibration_data()
    
    def _extract_model_info(self) -> Dict:
        """Extract model information from YOLO model"""
        info = {
            "name": self.model_name,
            "type": "yolov8",  # Default, will be updated
            "input_size": 640,  # Default
            "classes": [],
            "num_classes": 0,
        }
        
        # Try to determine model type
        if hasattr(self.model, 'model'):
            model_cfg = str(self.model.model)
            if 'yolov8' in model_cfg.lower():
                info['type'] = 'yolov8'
            elif 'yolov9' in model_cfg.lower():
                info['type'] = 'yolov9'
            elif 'yolo11' in model_cfg.lower() or 'yolov11' in model_cfg.lower():
                info['type'] = 'yolo11'
        
        # Get class names
        if hasattr(self.model, 'names'):
            info['classes'] = list(self.model.names.values())
            info['num_classes'] = len(info['classes'])
        
        # Get input size
        if hasattr(self.model.model, 'args'):
            info['input_size'] = getattr(self.model.model.args, 'imgsz', 640)
        
        logger.info(f"Model info: {info['type']} with {info['num_classes']} classes")
        return info
    
    def _download_calibration_data(self):
        """Download sample calibration data if not provided"""
        self.calibration_data = self.output_dir / "calibration_data"
        if self.calibration_data.exists():
            return
        
        logger.info("Downloading COCO calibration samples...")
        self.calibration_data.mkdir(exist_ok=True)
        
        # Download script
        download_script = """
import urllib.request
import zipfile
import os

# Download COCO val2017 samples (first 1000 images)
url = "http://images.cocodataset.org/zips/val2017.zip"
zip_path = "coco_val2017.zip"

print("Downloading COCO validation images...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting images...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract first 1000 images
    members = zip_ref.namelist()[:1000]
    for member in members:
        if member.endswith('.jpg'):
            zip_ref.extract(member, '.')

os.remove(zip_path)
print("Calibration data ready!")
"""
        
        script_path = self.calibration_data / "download.py"
        with open(script_path, 'w') as f:
            f.write(download_script)
        
        try:
            subprocess.run([sys.executable, str(script_path)], 
                         cwd=str(self.calibration_data), check=True)
            
            # Move images to calibration folder
            val_dir = self.calibration_data / "val2017"
            if val_dir.exists():
                for img in val_dir.glob("*.jpg"):
                    shutil.move(str(img), str(self.calibration_data))
                shutil.rmtree(val_dir)
        except Exception as e:
            logger.error(f"Failed to download calibration data: {e}")
            logger.info("Please provide calibration images manually")
    
    def convert_to_onnx(self, opset: int = 11) -> Path:
        """Convert model to ONNX format"""
        logger.info("Converting to ONNX...")
        
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        # Export using ultralytics
        self.model.export(
            format='onnx',
            opset=opset,
            simplify=True,
            dynamic=False,
            batch=1,
            imgsz=self.model_info['input_size']
        )
        
        # Move to output directory
        exported_onnx = self.model_path.parent / f"{self.model_path.stem}.onnx"
        if exported_onnx.exists():
            shutil.move(str(exported_onnx), str(onnx_path))
        
        logger.info(f"ONNX model saved: {onnx_path}")
        return onnx_path
    
    def convert_to_tflite(self, quantize: bool = True) -> Dict[str, Path]:
        """Convert model to TFLite format for Coral TPU"""
        logger.info("Converting to TFLite...")
        
        results = {}
        
        # First convert to TensorFlow SavedModel
        logger.info("Exporting to TensorFlow SavedModel...")
        self.model.export(
            format='saved_model',
            imgsz=self.model_info['input_size']
        )
        
        # Get SavedModel path
        saved_model_path = self.model_path.parent / f"{self.model_path.stem}_saved_model"
        
        if not saved_model_path.exists():
            logger.error("SavedModel export failed")
            return results
        
        try:
            import tensorflow as tf
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            # CPU version (float32)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_model = converter.convert()
            
            cpu_path = self.output_dir / f"{self.model_name}_cpu.tflite"
            with open(cpu_path, 'wb') as f:
                f.write(tflite_model)
            results['cpu'] = cpu_path
            logger.info(f"CPU TFLite saved: {cpu_path}")
            
            if quantize:
                # Quantized version for Edge TPU
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset_gen
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
                
                quantized_model = converter.convert()
                
                quant_path = self.output_dir / f"{self.model_name}_quant.tflite"
                with open(quant_path, 'wb') as f:
                    f.write(quantized_model)
                results['quantized'] = quant_path
                logger.info(f"Quantized TFLite saved: {quant_path}")
                
                # Compile for Edge TPU
                edge_tpu_path = self._compile_edge_tpu(quant_path)
                if edge_tpu_path:
                    results['edge_tpu'] = edge_tpu_path
            
            # Cleanup
            shutil.rmtree(saved_model_path)
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
        
        return results
    
    def _representative_dataset_gen(self):
        """Generate representative dataset for quantization"""
        import tensorflow as tf
        
        # Use calibration images
        image_paths = list(Path(self.calibration_data).glob("*.jpg"))[:100]
        
        for img_path in image_paths:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.model_info['input_size'], self.model_info['input_size']])
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.expand_dims(img, 0)
            yield [img]
    
    def _compile_edge_tpu(self, tflite_path: Path) -> Optional[Path]:
        """Compile TFLite model for Coral Edge TPU"""
        try:
            logger.info("Compiling for Edge TPU...")
            
            output_path = self.output_dir / f"{self.model_name}_edgetpu.tflite"
            
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',  # Show statistics
                '-o', str(self.output_dir),
                str(tflite_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find compiled model
                compiled_path = self.output_dir / f"{tflite_path.stem}_edgetpu.tflite"
                if compiled_path.exists():
                    shutil.move(str(compiled_path), str(output_path))
                    logger.info(f"Edge TPU model saved: {output_path}")
                    return output_path
            else:
                logger.error(f"Edge TPU compilation failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.warning("Edge TPU compiler not found. Install from: https://coral.ai/docs/accelerator/get-started/")
            
        return None
    
    def convert_to_hailo(self, onnx_path: Optional[Path] = None) -> Dict[str, Path]:
        """Convert model to Hailo format"""
        logger.info("Converting to Hailo format...")
        
        results = {}
        
        # Use provided ONNX or convert
        if not onnx_path:
            onnx_path = self.convert_to_onnx(opset=11)
        
        # Create Hailo configuration
        config_path = self._create_hailo_config()
        results['config'] = config_path
        
        # Create conversion script
        script_path = self._create_hailo_script(onnx_path, config_path)
        results['script'] = script_path
        
        # Try to run conversion if Hailo tools available
        if self._check_hailo_tools():
            hef_path = self._run_hailo_conversion(onnx_path, config_path)
            if hef_path:
                results['hef'] = hef_path
        else:
            logger.warning("Hailo tools not found. Please run the conversion script manually.")
            logger.info(f"Conversion script saved: {script_path}")
        
        return results
    
    def _create_hailo_config(self) -> Path:
        """Create Hailo model configuration"""
        config = {
            "model_name": self.model_name,
            "input_shape": [1, 3, self.model_info['input_size'], self.model_info['input_size']],
            "output_names": self._get_yolo_output_names(),
            "classes": self.model_info['classes'],
            "num_classes": self.model_info['num_classes'],
            "anchors": self._get_yolo_anchors(),
            "preprocessing": {
                "input_type": "image",
                "input_format": "RGB",
                "mean_values": [0, 0, 0],
                "scale_values": [255.0, 255.0, 255.0]
            }
        }
        
        config_path = self.output_dir / f"{self.model_name}_hailo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def _get_yolo_output_names(self) -> List[str]:
        """Get YOLO output layer names based on model type"""
        if self.model_info['type'] == 'yolov8':
            return [
                "/model.22/cv2.0/cv2.0.2/Conv",
                "/model.22/cv3.0/cv3.0.2/Conv",
                "/model.22/cv2.1/cv2.1.2/Conv",
                "/model.22/cv3.1/cv3.1.2/Conv",
                "/model.22/cv2.2/cv2.2.2/Conv",
                "/model.22/cv3.2/cv3.2.2/Conv"
            ]
        elif self.model_info['type'] == 'yolo11':
            # YOLO11 uses similar structure
            return [
                "/model.22/cv2.0/cv2.0.2/Conv",
                "/model.22/cv3.0/cv3.0.2/Conv",
                "/model.22/cv2.1/cv2.1.2/Conv",
                "/model.22/cv3.1/cv3.1.2/Conv",
                "/model.22/cv2.2/cv2.2.2/Conv",
                "/model.22/cv3.2/cv3.2.2/Conv"
            ]
        else:
            # Default fallback
            return ["output1", "output2", "output3"]
    
    def _get_yolo_anchors(self) -> List[List[float]]:
        """Get YOLO anchor boxes"""
        # YOLOv8/v11 doesn't use anchors (anchor-free)
        if self.model_info['type'] in ['yolov8', 'yolo11']:
            return []
        
        # Default anchors for older YOLO versions
        return [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ]
    
    def _create_hailo_script(self, onnx_path: Path, config_path: Path) -> Path:
        """Create Hailo conversion script"""
        script = f"""#!/bin/bash
# Hailo Model Conversion Script for {self.model_name}
# Generated by Wildfire Watch Model Converter

# Check if running in Hailo Docker
if [ ! -f /opt/hailo/hailo_sw_suite_version ]; then
    echo "ERROR: This script must be run inside Hailo Docker container"
    echo "Please follow instructions in README.md"
    exit 1
fi

# Configuration
MODEL_NAME="{self.model_name}"
ONNX_PATH="{onnx_path.absolute()}"
CONFIG_PATH="{config_path.absolute()}"
