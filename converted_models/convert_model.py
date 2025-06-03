#!/usr/bin/env python3
"""
Wildfire Watch Model Converter
Converts YOLO PyTorch models to deployment formats for edge devices
Supports: Hailo (HEF), Coral (TFLite), ONNX, OpenVINO, TensorRT

This is a wrapper script that can use GPL/AGPL tools for conversion
but doesn't link them into the main codebase.
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
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Model download URLs (update with actual HuggingFace URLs)
MODEL_URLS = {
    'yolov8n': 'https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt',
    'yolov9s': 'https://huggingface.co/merve/yolov9/resolve/main/yolov9s.pt',
    'fire_detector_v1': 'https://huggingface.co/wildfire-watch/fire-detector/resolve/main/best.pt',
}

# Calibration dataset URL
CALIBRATION_URL = 'https://huggingface.co/datasets/wildfire-watch/calibration-images/resolve/main/calibration_images.tgz'

class ModelConverter:
    """Converts YOLO models to various edge deployment formats"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "converted_models",
        model_name: Optional[str] = None,
        calibration_data: Optional[str] = None,
        model_size: Tuple[int, int] = (640, 640)
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_size = model_size
        
        # Determine model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self.model_path.stem
        
        # Model info (will be extracted)
        self.model_info = {
            "name": self.model_name,
            "type": "yolo",
            "input_size": model_size,
            "classes": [],
            "num_classes": 0,
        }
        
        # Calibration data
        self.calibration_data = calibration_data
        if not self.calibration_data:
            logger.info("No calibration data provided. Downloading default dataset...")
            self._download_calibration_data()
    
    def _download_calibration_data(self):
        """Download sample calibration data"""
        self.calibration_data = self.output_dir / "calibration_data"
        if self.calibration_data.exists() and len(list(self.calibration_data.glob("*.jpg"))) > 100:
            logger.info("Calibration data already exists")
            return
        
        self.calibration_data.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Downloading calibration dataset from {CALIBRATION_URL}")
            tar_path = self.output_dir / "calibration.tgz"
            urllib.request.urlretrieve(CALIBRATION_URL, tar_path)
            
            # Extract
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.calibration_data)
            
            tar_path.unlink()
            logger.info("Calibration data ready")
        except Exception as e:
            logger.warning(f"Failed to download calibration data: {e}")
            logger.info("Creating synthetic calibration data...")
            self._create_synthetic_calibration()
    
    def _create_synthetic_calibration(self):
        """Create synthetic calibration images"""
        try:
            import PIL.Image
            
            for i in range(100):
                # Create random image
                img_array = np.random.randint(0, 255, (*self.model_size, 3), dtype=np.uint8)
                img = PIL.Image.fromarray(img_array)
                img.save(self.calibration_data / f"synthetic_{i:04d}.jpg")
            
            logger.info("Created 100 synthetic calibration images")
        except ImportError:
            logger.error("PIL not installed. Cannot create calibration data.")
    
    def _extract_model_info_external(self) -> Dict:
        """Extract model info using external script to avoid GPL linking"""
        script = f"""
import json
import sys
sys.path.insert(0, '{self.model_path.parent}')

try:
    # Try ultralytics (GPL)
    from ultralytics import YOLO
    model = YOLO('{self.model_path}')
    info = {{
        'type': 'yolov8',
        'classes': list(model.names.values()) if hasattr(model, 'names') else [],
        'num_classes': len(model.names) if hasattr(model, 'names') else 0,
    }}
except:
    # Try to load with torch directly
    import torch
    model = torch.load('{self.model_path}', map_location='cpu')
    if 'model' in model and hasattr(model['model'], 'names'):
        info = {{
            'type': 'yolo',
            'classes': model['model'].names,
            'num_classes': len(model['model'].names),
        }}
    else:
        info = {{'type': 'unknown', 'classes': [], 'num_classes': 0}}

print(json.dumps(info))
"""
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                self.model_info.update(info)
                logger.info(f"Model info: {info['type']} with {info['num_classes']} classes")
            else:
                logger.warning(f"Could not extract model info: {result.stderr}")
        except Exception as e:
            logger.warning(f"Model info extraction failed: {e}")
    
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
    onnx_path="{onnx_path}",
    engine_path="{self.output_dir}/{self.model_name}_tensorrt.engine",
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
        print("Install from: https://developer.nvidia.com/tensorrt")
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_tensorrt.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        results['script'] = script_path
        
        logger.info(f"TensorRT conversion script saved: {script_path}")
        logger.info("Run this script on your target device with TensorRT installed")
        
        return results
    
    def generate_frigate_config(self) -> Path:
        """Generate Frigate configuration for the model"""
        logger.info("Generating Frigate configuration...")
        
        # Extract model info if not done
        if not self.model_info['classes']:
            self._extract_model_info_external()
        
        # Determine which model file to use based on available hardware
        config = {
            'model': {
                'path': f"/models/{self.model_name}.tflite",  # Default to TFLite
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': self.model_size[0],
                'height': self.model_size[1],
                'labels': f"/models/{self.model_name}_labels.txt"
            },
            'objects': {
                'track': [],
                'filters': {}
            }
        }
        
        # Add tracking for detected classes
        # Common object classes we want to track
        track_classes = ['fire', 'smoke', 'person', 'car', 'truck', 'wildlife', 'animal', 'bird']
        
        for cls in self.model_info['classes']:
            cls_lower = cls.lower()
            # Track if it's a class we're interested in
            for track_cls in track_classes:
                if track_cls in cls_lower:
                    config['objects']['track'].append(cls)
                    
                    # Add specific filters for fire-related classes
                    if 'fire' in cls_lower or 'smoke' in cls_lower:
                        config['objects']['filters'][cls] = {
                            'min_area': 300,
                            'max_area': 100000,
                            'min_score': 0.6,
                            'threshold': 0.7
                        }
                    elif 'person' in cls_lower:
                        config['objects']['filters'][cls] = {
                            'min_area': 1000,
                            'max_area': 50000,
                            'min_score': 0.5,
                            'threshold': 0.7
                        }
                    elif 'car' in cls_lower or 'truck' in cls_lower:
                        config['objects']['filters'][cls] = {
                            'min_area': 5000,
                            'max_area': 200000,
                            'min_score': 0.6,
                            'threshold': 0.7
                        }
                    break
        
        # If no specific classes found, track first 5 classes
        if not config['objects']['track'] and self.model_info['classes']:
            config['objects']['track'] = self.model_info['classes'][:5]
        
        # Save config snippet
        config_path = self.output_dir / f"{self.model_name}_frigate_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
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
        
        # Extract model info first
        self._extract_model_info_external()
        
        # Convert to ONNX (base for other formats)
        try:
            onnx_path = self.convert_to_onnx()
            results['outputs']['onnx'] = {'path': onnx_path}
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            onnx_path = None
        
        # Convert to TFLite/Coral
        try:
            tflite_results = self.convert_to_tflite()
            if tflite_results:
                results['outputs']['tflite'] = tflite_results
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
        
        # Convert to Hailo
        if onnx_path:
            try:
                hailo_results = self.convert_to_hailo(onnx_path)
                if hailo_results:
                    results['outputs']['hailo'] = hailo_results
            except Exception as e:
                logger.error(f"Hailo preparation failed: {e}")
        
        # Convert to OpenVINO
        if onnx_path:
            try:
                openvino_results = self.convert_to_openvino(onnx_path)
                if openvino_results:
                    results['outputs']['openvino'] = openvino_results
            except Exception as e:
                logger.error(f"OpenVINO preparation failed: {e}")
        
        # Create TensorRT script
        if onnx_path:
            try:
                tensorrt_results = self.convert_to_tensorrt(onnx_path)
                if tensorrt_results:
                    results['outputs']['tensorrt'] = tensorrt_results
            except Exception as e:
                logger.error(f"TensorRT preparation failed: {e}")
        
        # Generate Frigate config
        try:
            frigate_config = self.generate_frigate_config()
            results['outputs']['frigate'] = {'config': frigate_config}
        except Exception as e:
            logger.error(f"Frigate config generation failed: {e}")
        
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
- **Input Size**: {self.model_size[0]}x{self.model_size[1]}
- **Labels**: {', '.join(self.model_info['classes'][:10])}{'...' if len(self.model_info['classes']) > 10 else ''}

## Converted Formats

### 1. ONNX
- **File**: `{self.model_name}.onnx`
- **Use**: Universal format, base for other conversions
- **Deployment**: Can be used directly with ONNX Runtime

### 2. TensorFlow Lite (Coral TPU)
"""
        
        if 'tflite' in results['outputs']:
            tflite_out = results['outputs']['tflite']
            if 'cpu' in tflite_out:
                readme += f"- **CPU Model**: `{tflite_out['cpu'].name}` - Float32 precision\n"
            if 'quantized' in tflite_out:
                readme += f"- **Quantized Model**: `{tflite_out['quantized'].name}` - INT8 precision\n"
            if 'edge_tpu' in tflite_out:
                readme += f"- **Edge TPU Model**: `{tflite_out['edge_tpu'].name}` - Optimized for Coral\n"
        
        readme += """
### 3. Hailo
"""
        if 'hailo' in results['outputs']:
            hailo_out = results['outputs']['hailo']
            readme += f"- **Config**: `{hailo_out['config'].name}`\n"
            readme += f"- **Script**: `{hailo_out['script'].name}`\n"
            readme += "- Run the script inside Hailo Docker to generate HEF files\n"
        else:
            readme += "- Conversion script created - requires Hailo SDK\n"
        
        readme += """
### 4. OpenVINO
"""
        if 'openvino' in results['outputs'] and 'xml' in results['outputs']['openvino']:
            openvino_out = results['outputs']['openvino']
            readme += f"- **Model**: `{openvino_out['xml'].name}`\n"
            readme += f"- **Weights**: `{openvino_out['bin'].name}`\n"
        else:
            readme += "- Conversion script created - requires OpenVINO toolkit\n"
        
        readme += """
### 5. TensorRT
"""
        if 'tensorrt' in results['outputs']:
            readme += f"- **Script**: `{results['outputs']['tensorrt']['script'].name}`\n"
            readme += "- Run on target device with TensorRT installed\n"
        
        readme += f"""
## Deployment with Frigate

1. Copy model files to Frigate container:
   ```bash
   # For Coral TPU
   docker cp {self.model_name}_edgetpu.tflite frigate:/models/
   docker cp {self.model_name}_labels.txt frigate:/models/
   
   # For CPU/GPU
   docker cp {self.model_name}_cpu.tflite frigate:/models/
   ```

2. Update Frigate config with generated settings from `{self.model_name}_frigate_config.yml`

3. Select appropriate model based on hardware:
   - Coral TPU: Use `{self.model_name}_edgetpu.tflite`
   - CPU: Use `{self.model_name}_cpu.tflite`
   - Hailo: Use `{self.model_name}_hailo8.hef` or `{self.model_name}_hailo8l.hef`
   - GPU: Use TensorRT engine after conversion

4. Restart Frigate to load new model

## Hardware-Specific Notes

### Coral TPU
- Requires Edge TPU runtime installed
- Best performance with USB 3.0
- Use INT8 quantized model

### Hailo
- Requires Hailo RT installed on device
- Hailo-8: 26 TOPS performance
- Hailo-8L: 13 TOPS performance

### TensorRT (NVIDIA)
- Must compile on target device
- Engine files are not portable
- Supports FP16 optimization

### OpenVINO (Intel)
- Works on Intel CPUs, GPUs, VPUs
- Good balance of performance and compatibility

## Performance Optimization

- For real-time detection: Use {self.model_size[0]}x{self.model_size[0]} input
- For higher accuracy: Increase input size to 1280x1280
- For faster inference: Reduce to 320x320 or 416x416

## Troubleshooting

1. **Model not loading**: Check file paths and permissions
2. **Poor detection**: Verify labels match your use case
3. **Slow performance**: Ensure using hardware-accelerated model
4. **Memory issues**: Reduce batch size or input resolution

## License

Model conversions inherit the license of the original model.
Conversion scripts are provided under MIT license.
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        logger.info(f"README saved: {readme_path}")

def download_model(model_name: str, output_path: Path) -> Path:
    """Download pre-trained model from HuggingFace"""
    if model_name in MODEL_URLS:
        url = MODEL_URLS[model_name]
        logger.info(f"Downloading {model_name} from {url}")
        
        model_path = output_path / f"{model_name}.pt"
        if not model_path.exists():
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Downloaded to {model_path}")
        else:
            logger.info(f"Model already exists: {model_path}")
        
        return model_path
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert YOLO models for edge deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert local model
  python convert_model.py path/to/model.pt
  
  # Download and convert pre-trained model
  python convert_model.py --download yolov8n
  
  # Convert with custom settings
  python convert_model.py model.pt --name fire_v1 --size 416
  
  # Convert to specific formats only
  python convert_model.py model.pt --formats onnx tflite hailo
        """
    )
    
    parser.add_argument('model', nargs='?', help='Path to .pt model file')
    parser.add_argument('--download', choices=list(MODEL_URLS.keys()),
                       help='Download and convert pre-trained model')
    parser.add_argument('--output-dir', default='converted_models',
                       help='Output directory (default: converted_models)')
    parser.add_argument('--name', help='Model name (default: from filename)')
    parser.add_argument('--size', type=int, default=640,
                       help='Model input size (default: 640)')
    parser.add_argument('--calibration-data',
                       help='Path to calibration images directory')
    parser.add_argument('--formats', nargs='+',
                       choices=['all', 'onnx', 'tflite', 'hailo', 'openvino', 'tensorrt'],
                       default=['all'], help='Formats to convert to')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.download:
        model_path = download_model(args.download, Path(args.output_dir) / 'downloads')
    elif args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)
    else:
        parser.error("Either provide a model path or use --download")
    
    # Create converter
    converter = ModelConverter(
        model_path=model_path,
        output_dir=args.output_dir,
        model_name=args.name,
        calibration_data=args.calibration_data,
        model_size=(args.size, args.size)
    )
    
    # Convert based on requested formats
    if 'all' in args.formats:
        converter.convert_all()
    else:
        results = {'model_info': converter.model_info, 'outputs': {}}
        
        if 'onnx' in args.formats:
            onnx_path = converter.convert_to_onnx()
            results['outputs']['onnx'] = {'path': onnx_path}
        
        if 'tflite' in args.formats:
            results['outputs']['tflite'] = converter.convert_to_tflite()
        
        if 'hailo' in args.formats:
            results['outputs']['hailo'] = converter.convert_to_hailo()
        
        if 'openvino' in args.formats:
            results['outputs']['openvino'] = converter.convert_to_openvino()
        
        if 'tensorrt' in args.formats:
            results['outputs']['tensorrt'] = converter.convert_to_tensorrt()
        
        # Always generate Frigate config
        frigate_config = converter.generate_frigate_config()
        results['outputs']['frigate'] = {'config': frigate_config}
        
        # Save summary
        summary_path = converter.output_dir / "conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    logger.info("Conversion complete!")
    logger.info(f"Output directory: {converter.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Model: {converter.model_name}")
    print(f"Output: {converter.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
onnx(self, opset: int = 13, simplify: bool = True) -> Path:
        """Convert model to ONNX format using external script"""
        logger.info("Converting to ONNX...")
        
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        # First try using ultralytics export (external process to avoid GPL)
        export_script = f"""
try:
    from ultralytics import YOLO
    model = YOLO('{self.model_path}')
    model.export(
        format='onnx',
        opset={opset},
        simplify={simplify},
        dynamic=False,
        batch=1,
        imgsz={self.model_size[0]}
    )
    # Move output
    import shutil
    exported = '{self.model_path}'.replace('.pt', '.onnx')
    shutil.move(exported, '{onnx_path}')
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {{e}}")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', export_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            logger.info(f"ONNX model saved: {onnx_path}")
            return onnx_path
        
        # Fallback: Try direct torch export
        logger.warning("Ultralytics export failed, trying direct torch export...")
        torch_export_script = f"""
import torch
import torch.onnx

# Load model
model = torch.load('{self.model_path}', map_location='cpu')
if 'model' in model:
    model = model['model']

# Create dummy input
dummy_input = torch.randn(1, 3, {self.model_size[0]}, {self.model_size[1]})

# Export
torch.onnx.export(
    model,
    dummy_input,
    '{onnx_path}',
    opset_version={opset},
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={{'images': {{0: 'batch_size'}}, 'output': {{0: 'batch_size'}}}}
)
print("SUCCESS")
"""
        
        result = subprocess.run(
            [sys.executable, '-c', torch_export_script],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and onnx_path.exists():
            logger.info(f"ONNX model saved: {onnx_path}")
            return onnx_path
        else:
            logger.error(f"ONNX export failed: {result.stderr}")
            raise RuntimeError("Failed to export ONNX model")
    
    def convert_to_tflite(self, quantize: bool = True) -> Dict[str, Path]:
        """Convert model to TFLite format for Coral TPU"""
        logger.info("Converting to TFLite...")
        
        results = {}
        
        # We need ONNX first
        if not (self.output_dir / f"{self.model_name}.onnx").exists():
            self.convert_to_onnx()
        
        # Convert ONNX to TF using external script
        convert_script = f"""
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np

# Load ONNX model
onnx_model = onnx.load('{self.output_dir}/{self.model_name}.onnx')
tf_rep = prepare(onnx_model)

# Export to SavedModel
tf_rep.export_graph('{self.output_dir}/{self.model_name}_saved_model')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('{self.output_dir}/{self.model_name}_saved_model')

# CPU version
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('{self.output_dir}/{self.model_name}_cpu.tflite', 'wb') as f:
    f.write(tflite_model)

print("CPU model done")

# Quantized version for Edge TPU
def representative_dataset():
    import glob
    images = glob.glob('{self.calibration_data}/*.jpg')[:100]
    for img_path in images:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [{self.model_size[0]}, {self.model_size[1]}])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        yield [img]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_model = converter.convert()
with open('{self.output_dir}/{self.model_name}_quant.tflite', 'wb') as f:
    f.write(quantized_model)

print("SUCCESS")
"""
        
        try:
            # Try to run conversion
            result = subprocess.run(
                [sys.executable, '-c', convert_script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                results['cpu'] = self.output_dir / f"{self.model_name}_cpu.tflite"
                results['quantized'] = self.output_dir / f"{self.model_name}_quant.tflite"
                
                # Try to compile for Edge TPU
                edge_tpu_path = self._compile_edge_tpu(results['quantized'])
                if edge_tpu_path:
                    results['edge_tpu'] = edge_tpu_path
                
                logger.info("TFLite conversion complete")
            else:
                logger.error(f"TFLite conversion failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("TFLite conversion timed out")
        except Exception as e:
            logger.error(f"TFLite conversion error: {e}")
        
        return results
    
    def _compile_edge_tpu(self, tflite_path: Path) -> Optional[Path]:
        """Compile TFLite model for Coral Edge TPU"""
        try:
            logger.info("Compiling for Edge TPU...")
            
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',
                '-o', str(self.output_dir),
                str(tflite_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                compiled_path = self.output_dir / f"{tflite_path.stem}_edgetpu.tflite"
                if compiled_path.exists():
                    final_path = self.output_dir / f"{self.model_name}_edgetpu.tflite"
                    shutil.move(str(compiled_path), str(final_path))
                    logger.info(f"Edge TPU model saved: {final_path}")
                    return final_path
            else:
                logger.warning(f"Edge TPU compilation failed: {result.stderr}")
                
        except FileNotFoundError:
            logger.warning("Edge TPU compiler not found. Install from: https://coral.ai/docs/accelerator/get-started/")
            
        return None
    
    def convert_to_hailo(self, onnx_path: Optional[Path] = None) -> Dict[str, Path]:
        """Convert model to Hailo format"""
        logger.info("Converting to Hailo format...")
        
        results = {}
        
        # Use provided ONNX or convert
        if not onnx_path:
            onnx_path = self.convert_to_onnx()
        
        # Extract model info if not done
        if not self.model_info['classes']:
            self._extract_model_info_external()
        
        # Create Hailo configuration
        config = {
            "name": self.model_name,
            "input_layers": ["images"],
            "output_layers": self._get_output_layer_names(),
            "dataset": {
                "path": str(self.calibration_data),
                "format": "image"
            },
            "preprocessing": {
                "input_shape": [1, 3, self.model_size[0], self.model_size[1]],
                "normalization": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [255.0, 255.0, 255.0]
                }
            },
            "quantization": {
                "precision": "int8",
                "calib_set_size": 100
            }
        }
        
        config_path = self.output_dir / f"{self.model_name}_hailo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        results['config'] = config_path
        
        # Create conversion script
        script_path = self._create_hailo_script(onnx_path, config_path)
        results['script'] = script_path
        
        # Check if Hailo SDK is available
        if self._check_hailo_sdk():
            # Try to run conversion
            hef_paths = self._run_hailo_conversion(onnx_path, config_path)
            if hef_paths:
                results.update(hef_paths)
        else:
            logger.info("Hailo SDK not found. Please run the conversion script manually after installing the SDK.")
            logger.info(f"Script location: {script_path}")
        
        return results
    
    def _get_output_layer_names(self) -> List[str]:
        """Get output layer names for YOLO model"""
        # These are typical YOLO output layer patterns
        # Adjust based on actual model architecture
        return [
            "output",
            "output0",
            "output1",
            "output2"
        ]
    
    def _check_hailo_sdk(self) -> bool:
        """Check if Hailo SDK is installed"""
        try:
            result = subprocess.run(['hailo', '--version'], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _create_hailo_script(self, onnx_path: Path, config_path: Path) -> Path:
        """Create Hailo conversion script"""
        script = f"""#!/bin/bash
# Hailo Model Conversion Script for {self.model_name}

# This script should be run inside the Hailo SDK Docker container
# docker run -it --rm -v $(pwd):/workspace hailo/hailo_sdk:latest

# Set paths
ONNX_PATH="/workspace/{onnx_path.relative_to(Path.cwd())}"
CONFIG_PATH="/workspace/{config_path.relative_to(Path.cwd())}"
OUTPUT_DIR="/workspace/{self.output_dir.relative_to(Path.cwd())}"

# Parse ONNX to HAR
echo "Parsing ONNX model..."
hailo parser onnx "$ONNX_PATH" --config "$CONFIG_PATH" --hw-arch hailo8

# Optimize for Hailo-8
echo "Optimizing for Hailo-8..."
hailo optimize {self.model_name}.har --hw-arch hailo8 --calib-set-size 100

# Compile for Hailo-8
echo "Compiling for Hailo-8..."
hailo compiler {self.model_name}_optimized.har --hw-arch hailo8

# Compile for Hailo-8L
echo "Compiling for Hailo-8L..."
hailo compiler {self.model_name}_optimized.har --hw-arch hailo8l

# Move outputs
mv *.hef "$OUTPUT_DIR/"

echo "Conversion complete!"
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_hailo.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        return script_path
    
    def _run_hailo_conversion(self, onnx_path: Path, config_path: Path) -> Optional[Dict[str, Path]]:
        """Run Hailo conversion if SDK available"""
        # This is a placeholder - actual implementation would run the conversion
        # inside Hailo Docker container
        logger.info("Hailo conversion requires running inside Hailo SDK Docker container")
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
    echo "Install from: https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
    exit 1
fi

# Convert model
mo \\
    --input_model {onnx_path} \\
    --output_dir {self.output_dir} \\
    --model_name {self.model_name}_openvino \\
    --data_type FP16 \\
    --input_shape [1,3,{self.model_size[0]},{self.model_size[1]}] \\
    --mean_values [0,0,0] \\
    --scale_values [255,255,255]

echo "OpenVINO conversion complete!"
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_openvino.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        results['script'] = script_path
        
        # Try to run if OpenVINO available
        try:
            result = subprocess.run(['mo', '--help'], capture_output=True)
            if result.returncode == 0:
                # Run conversion
                subprocess.run(['bash', str(script_path)], check=True)
                
                xml_path = self.output_dir / f"{self.model_name}_openvino.xml"
                bin_path = self.output_dir / f"{self.model_name}_openvino.bin"
                
                if xml_path.exists() and bin_path.exists():
                    results['xml'] = xml_path
                    results['bin'] = bin_path
                    logger.info("OpenVINO conversion complete")
        except:
            logger.info("OpenVINO tools not found. Run the script manually after installing OpenVINO.")
        
        return results
    
    def convert_to_
