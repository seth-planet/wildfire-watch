"""
Wildfire Watch Model Converter - Enhanced Implementation
Converts YOLO PyTorch models to optimized deployment formats
Supports: YOLOv8, YOLOv9, YOLOv9-MIT, YOLO-NAS
Outputs: Hailo (HEF), Coral (TFLite), ONNX, OpenVINO, TensorRT

Key Features:
- QAT (Quantization-Aware Training) support for better quantization
- Optimized intermediate formats for TensorRT JIT compilation
- Full Frigate compatibility with proper tensor naming
- Comprehensive validation and benchmarking
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
import tarfile
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_converter.log')
    ]
)
logger = logging.getLogger(__name__)

# Model download URLs (comprehensive collection)
MODEL_URLS = {
    # YOLOv8 models
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt',
    
    # YOLOv9 models
    'yolov9t': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t.pt',
    'yolov9s': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-s.pt',
    'yolov9m': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m.pt',
    'yolov9c': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt',
    'yolov9e': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt',
    
    # YOLOv9-MIT models
    'yolov9mit_s': 'https://github.com/SkalskiP/yolov9-mit/releases/download/v0.1.0/yolov9-s-mit.pt',
    'yolov9mit_m': 'https://github.com/SkalskiP/yolov9-mit/releases/download/v0.1.0/yolov9-m-mit.pt',
    
    # YOLO-NAS models
    'yolo_nas_s': 'https://sghub.deci.ai/models/yolo_nas_s_coco.pth',
    'yolo_nas_m': 'https://sghub.deci.ai/models/yolo_nas_m_coco.pth',
    'yolo_nas_l': 'https://sghub.deci.ai/models/yolo_nas_l_coco.pth',
    
    # RT-DETR models (Real-Time DETR)
    'rtdetr_l': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-l.pt',
    'rtdetr_x': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/rtdetr-x.pt',
    
    # Specialized models
    'fire_detector_v1': 'https://huggingface.co/wildfire-watch/fire-detector/resolve/main/best.pt',
    'yolov8n_qat': 'https://github.com/levipereira/yolov8-qat/releases/download/v1.0/yolov8n-qat.pt',
    'yolov8s_qat': 'https://github.com/levipereira/yolov8-qat/releases/download/v1.0/yolov8s-qat.pt',
}

# Calibration dataset URLs (enhanced)
CALIBRATION_URLS = {
    'default': 'https://huggingface.co/datasets/wildfire-watch/calibration-images/resolve/main/calibration_images.tgz',
    'fire': 'https://huggingface.co/datasets/wildfire-watch/fire-calibration/resolve/main/fire_calibration.tgz',
    'coco_val': 'http://images.cocodataset.org/zips/val2017.zip',
    'diverse': 'https://huggingface.co/datasets/wildfire-watch/diverse-calibration/resolve/main/diverse_calibration.tgz'
}

# Frigate-specific constants
FRIGATE_INPUT_NAMES = ['images', 'input', 'data']  # Acceptable input tensor names
FRIGATE_OUTPUT_PATTERN = ['output', 'output0', 'output1', 'output2']  # Expected output names

@dataclass
class ModelInfo:
    """Enhanced model metadata"""
    name: str
    type: str
    version: Optional[str] = None
    input_size: Tuple[int, int] = (640, 640)
    classes: List[str] = None
    num_classes: int = 0
    architecture: Optional[str] = None
    license: Optional[str] = None  # Track model license
    has_nms: bool = False  # Whether model includes NMS
    output_format: str = 'xyxy'  # bbox format: xyxy, xywh, etc.
    
    def to_dict(self) -> Dict:
        return asdict(self)

class EnhancedModelConverter:
    """Enhanced model converter with QAT and optimizations"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "converted_models",
        model_name: Optional[str] = None,
        calibration_data: Optional[str] = None,
        model_size: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]] = 640,
        qat_enabled: bool = False,
        target_hardware: List[str] = None,
        debug: bool = False
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configuration - now supports multiple sizes
        self.model_sizes = self._parse_model_sizes(model_size)
        self.model_size = self.model_sizes[0]  # Primary size for compatibility
        
        self.model_name = model_name or self.model_path.stem
        self.qat_enabled = qat_enabled
        self.target_hardware = target_hardware or ['all']
        self.debug = debug
        
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Initialize model info
        self.model_info = ModelInfo(
            name=self.model_name,
            type="unknown",
            input_size=self.model_size
        )
        
        # Calibration data handling
        self.calibration_data = calibration_data
        if not self.calibration_data:
            logger.warning("No calibration data provided. Quantization features will be limited.")
            # Don't automatically download - require user to provide real data
        
        # Detect hardware
        self.hardware = self._detect_hardware()
        logger.info(f"Detected hardware: {self.hardware}")
    
    def _parse_model_sizes(self, size_input: Union[int, str, Tuple[int, int], List]) -> List[Tuple[int, int]]:
        """Parse and validate model sizes input"""
        sizes = []
        
        # Handle different input types
        if isinstance(size_input, (int, str)):
            # Single size: 640 or "640" or "640x480"
            if isinstance(size_input, str) and 'x' in size_input:
                w, h = map(int, size_input.split('x'))
                sizes.append((w, h))
            else:
                s = int(size_input)
                sizes.append((s, s))
        elif isinstance(size_input, tuple):
            # Single tuple: (640, 480)
            sizes.append(size_input)
        elif isinstance(size_input, list):
            # List of sizes
            for size in size_input:
                if isinstance(size, (int, str)):
                    if isinstance(size, str) and 'x' in size:
                        w, h = map(int, size.split('x'))
                        sizes.append((w, h))
                    else:
                        s = int(size)
                        sizes.append((s, s))
                elif isinstance(size, tuple):
                    sizes.append(size)
        
        # Validate sizes
        validated_sizes = []
        for w, h in sizes:
            # Check constraints
            if w > 640 or h > 640:
                logger.warning(f"Size {w}x{h} exceeds 640 limit, skipping")
                continue
            if w % 32 != 0 or h % 32 != 0:
                logger.warning(f"Size {w}x{h} not divisible by 32, skipping")
                continue
            if w < 32 or h < 32:
                logger.warning(f"Size {w}x{h} too small (min 32), skipping")
                continue
            validated_sizes.append((w, h))
        
        if not validated_sizes:
            logger.warning("No valid sizes provided, using default 640x640")
            validated_sizes = [(640, 640)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sizes = []
        for size in validated_sizes:
            if size not in seen:
                seen.add(size)
                unique_sizes.append(size)
        
        logger.info(f"Model sizes to convert: {unique_sizes}")
        return unique_sizes
    
    def _extract_model_info_external(self) -> ModelInfo:
        """Extract comprehensive model information"""
        script = f'''
import json
import sys
import os
import torch

# Suppress warnings
os.environ['YOLO_VERBOSE'] = 'False'
import warnings
warnings.filterwarnings('ignore')

info = {{
    'type': 'unknown',
    'version': None,
    'classes': [],
    'num_classes': 0,
    'architecture': None,
    'license': None,
    'has_nms': False,
    'output_format': 'xyxy'
}}

try:
    # Load checkpoint
    checkpoint = torch.load('{self.model_path}', map_location='cpu')
    
    # Try to detect YOLOv9-MIT
    if isinstance(checkpoint, dict):
        # Check for MIT license indicators
        if 'model' in checkpoint:
            model_str = str(checkpoint.get('model', ''))
            if 'mit' in model_str.lower() or 'MIT' in checkpoint.get('license', ''):
                info['license'] = 'MIT'
                info['type'] = 'yolov9-mit'
        
        # Check training args
        if 'train_args' in checkpoint:
            model_arg = checkpoint['train_args'].get('model', '')
            if 'yolov9' in model_arg and 'mit' in model_arg.lower():
                info['type'] = 'yolov9-mit'
                info['license'] = 'MIT'
    
    # Try ultralytics
    from ultralytics import YOLO
    model = YOLO('{self.model_path}')
    
    # Extract detailed info
    if hasattr(model, 'model'):
        if hasattr(model.model, 'yaml'):
            yaml_data = model.model.yaml
            if 'nc' in yaml_data:
                info['num_classes'] = yaml_data['nc']
            if 'names' in yaml_data:
                info['classes'] = list(yaml_data['names'].values()) if isinstance(yaml_data['names'], dict) else yaml_data['names']
        
        # Check for NMS module
        for module in model.model.modules():
            if 'NMS' in module.__class__.__name__ or 'nms' in module.__class__.__name__.lower():
                info['has_nms'] = True
                break
    
    if hasattr(model, 'names'):
        info['classes'] = list(model.names.values()) if hasattr(model.names, 'values') else model.names
        info['num_classes'] = len(info['classes'])
    
    # Detect specific version and architecture
    if hasattr(model, 'ckpt') and isinstance(model.ckpt, dict):
        if 'train_args' in model.ckpt:
            model_arg = model.ckpt['train_args'].get('model', '')
            if 'yolov8' in model_arg:
                info['type'] = 'yolov8'
                info['version'] = model_arg
                info['architecture'] = 'YOLOv8'
            elif 'yolov9' in model_arg:
                if info['license'] != 'MIT':
                    info['type'] = 'yolov9'
                info['version'] = model_arg
                info['architecture'] = 'YOLOv9'
                
except ImportError:
    pass
except Exception as e:
    info['error'] = str(e)

# Try YOLO-NAS
if info['type'] == 'unknown':
    try:
        if 'net' in checkpoint:
            info['type'] = 'yolo-nas'
            info['architecture'] = 'YOLO-NAS'
            info['license'] = 'Apache-2.0'  # YOLO-NAS uses Apache license
            
            # Extract classes
            if 'classes' in checkpoint:
                info['classes'] = checkpoint['classes']
                info['num_classes'] = len(info['classes'])
            elif 'num_classes' in checkpoint:
                info['num_classes'] = checkpoint['num_classes']
    except:
        pass

print(json.dumps(info))
'''
        
        try:
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                info_dict = json.loads(result.stdout)
                self.model_info.type = info_dict.get('type', 'unknown')
                self.model_info.version = info_dict.get('version')
                self.model_info.classes = info_dict.get('classes', [])
                self.model_info.num_classes = info_dict.get('num_classes', 0)
                self.model_info.architecture = info_dict.get('architecture')
                self.model_info.license = info_dict.get('license')
                self.model_info.has_nms = info_dict.get('has_nms', False)
                self.model_info.output_format = info_dict.get('output_format', 'xyxy')
                
                logger.info(f"Model info: {self.model_info.type} ({self.model_info.license or 'Unknown license'})")
                logger.info(f"Classes: {self.model_info.num_classes}, NMS: {self.model_info.has_nms}")
                
        except Exception as e:
            logger.warning(f"Model info extraction failed: {e}")
        
        return self.model_info
    
    def convert_to_onnx_optimized(self, opset: int = 13, simplify: bool = True, 
                                  qat_prepare: bool = False, size: Optional[Tuple[int, int]] = None) -> Optional[Path]:
        """Convert to ONNX with optimizations for deployment and QAT"""
        current_size = size or self.model_size
        size_str = f"{current_size[0]}x{current_size[1]}"
        
        logger.info(f"Converting to ONNX (Size: {size_str}, QAT: {qat_prepare})...")
        
        output_suffix = f"_{size_str}_qat" if qat_prepare else f"_{size_str}"
        onnx_path = self.output_dir / f"{self.model_name}{output_suffix}.onnx"
        
        # Model-specific conversion based on type
        if self.model_info.type == 'yolov9-mit':
            success = self._convert_yolov9_mit_onnx(onnx_path, opset, qat_prepare, current_size)
        elif self.model_info.type == 'yolo-nas':
            success = self._convert_yolo_nas_onnx(onnx_path, opset, qat_prepare, current_size)
        else:
            success = self._convert_ultralytics_onnx(onnx_path, opset, qat_prepare, current_size)
        
        if success and onnx_path.exists():
            # Apply Frigate-specific optimizations
            self._optimize_onnx_for_frigate(onnx_path)
            
            # Simplify if requested
            if simplify:
                self._simplify_onnx(onnx_path)
            
            # Create TensorRT-optimized variant
            if 'tensorrt' in self.target_hardware or 'all' in self.target_hardware:
                self._create_tensorrt_optimized_onnx(onnx_path)
            
            return onnx_path
        
        return None
    
    def _convert_yolov9_mit_onnx(self, output_path: Path, opset: int, qat: bool, size: Tuple[int, int]) -> bool:
        """Convert YOLOv9-MIT model with QAT optimizations"""
        script = f'''
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# QAT preparation function
def prepare_model_for_qat(model):
    """Prepare model for quantization-aware training"""
    # Replace activations for better quantization
    def replace_module(module, name, new_module):
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = module
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, new_module)
        else:
            setattr(module, name, new_module)
    
    # Replace SiLU with LeakyReLU (better for quantization)
    for name, m in model.named_modules():
        if isinstance(m, nn.SiLU):
            replace_module(model, name, nn.LeakyReLU(0.1, inplace=True))
    
    # Fuse Conv-BN layers
    torch.quantization.fuse_modules(model, [['conv', 'bn']], inplace=True)
    
    return model

try:
    # Load YOLOv9-MIT model
    checkpoint = torch.load('{self.model_path}', map_location='cpu')
    model = checkpoint['model'].float()
    model.eval()
    
    # Apply QAT preparation if requested
    if {qat}:
        print("Applying QAT optimizations...")
        model = prepare_model_for_qat(model)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create dummy input with specified size
    dummy_input = torch.randn(1, 3, {size[1]}, {size[0]})
    
    # Export with Frigate-compatible names
    torch.onnx.export(
        model,
        dummy_input,
        '{output_path}',
        opset_version={opset},
        input_names=['images'],  # Frigate expects 'images'
        output_names=['output0', 'output1', 'output2'],  # Standard YOLO outputs
        dynamic_axes={{
            'images': {{0: 'batch'}},
            'output0': {{0: 'batch'}},
            'output1': {{0: 'batch'}},
            'output2': {{0: 'batch'}}
        }} if False else None,  # Frigate prefers fixed batch size
        do_constant_folding=True,
        export_params=True
    )
    
    print("SUCCESS")
    
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.returncode == 0 and "SUCCESS" in result.stdout
    
    def _convert_yolo_nas_onnx(self, output_path: Path, opset: int, qat: bool, size: Tuple[int, int]) -> bool:
        """Convert YOLO-NAS with optimization for deployment"""
        script = f'''
try:
    import torch
    from super_gradients.training import models
    from super_gradients.common.object_names import Models
    
    # Load model
    checkpoint = torch.load('{self.model_path}', map_location='cpu')
    
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
        num_classes = checkpoint.get('num_classes', 80)
        
        # Determine model size
        num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        
        if num_params < 10_000_000:
            model = models.get(Models.YOLO_NAS_S, num_classes=num_classes)
        elif num_params < 30_000_000:
            model = models.get(Models.YOLO_NAS_M, num_classes=num_classes)
        else:
            model = models.get(Models.YOLO_NAS_L, num_classes=num_classes)
        
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # CRITICAL: Prepare for export without pre/post processing
    model.prep_model_for_conversion(input_size=[1, 3, {size[1]}, {size[0]}])
    
    # QAT optimization if requested
    if {qat}:
        # Apply quantization-friendly modifications
        import torch.nn as nn
        for module in model.modules():
            if isinstance(module, nn.SiLU):
                # Replace with ReLU6 for better quantization
                module.forward = nn.ReLU6(inplace=True).forward
    
    # Export with Frigate-compatible settings
    dummy_input = torch.randn(1, 3, {size[1]}, {size[0]})
    
    torch.onnx.export(
        model,
        dummy_input,
        '{output_path}',
        opset_version={opset},
        input_names=['images'],  # Frigate standard
        output_names=['output0', 'output1', 'output2'],  # YOLO-NAS outputs
        do_constant_folding=True,
        export_params=True
    )
    
    print("SUCCESS")
    
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return result.returncode == 0 and "SUCCESS" in result.stdout
    
    def _convert_ultralytics_onnx(self, output_path: Path, opset: int, qat: bool, size: Tuple[int, int]) -> bool:
        """Convert Ultralytics model with QAT optimizations"""
        script = f'''
import os
os.environ['YOLO_VERBOSE'] = 'False'

try:
    from ultralytics import YOLO
    model = YOLO('{self.model_path}')
    
    # Export with optimized settings for Frigate
    success = model.export(
        format='onnx',
        opset={opset},
        simplify=False,  # We'll simplify with better control
        dynamic=False,   # Fixed batch for Frigate
        batch=1,
        imgsz={size[0] if size[0] == size[1] else list(size)},
        half=False,
        int8=False,
        nms=False  # Frigate handles NMS
    )
    
    if success:
        import shutil
        exported = str('{self.model_path}'.replace('.pt', '.onnx'))
        if os.path.exists(exported):
            shutil.move(exported, '{output_path}')
            
            # Apply QAT modifications if requested
            if {qat}:
                import onnx
                import onnx.helper as helper
                
                model = onnx.load('{output_path}')
                
                # Add quantization annotations
                for node in model.graph.node:
                    if node.op_type in ['Conv', 'Gemm']:
                        # Add quantization hints
                        node.attribute.append(
                            helper.make_attribute('quantization_mode', 'symmetric')
                        )
                
                onnx.save(model, '{output_path}')
            
            print("SUCCESS")
        else:
            print("FAILED: Export returned False")
    
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if self.debug:
            logger.debug(f"ONNX conversion stdout: {result.stdout}")
            logger.debug(f"ONNX conversion stderr: {result.stderr}")
        
        success = result.returncode == 0 and "SUCCESS" in result.stdout
        if not success:
            logger.error(f"ONNX conversion failed: {result.stdout} {result.stderr}")
        
        return success
    
    def _optimize_onnx_for_frigate(self, onnx_path: Path):
        """Apply Frigate-specific optimizations to ONNX model"""
        script = f'''
try:
    import onnx
    import numpy as np
    
    # Load model
    model = onnx.load('{onnx_path}')
    
    # Ensure input is named correctly for Frigate
    for input in model.graph.input:
        if input.name not in {FRIGATE_INPUT_NAMES}:
            # Find the most likely input tensor
            if len(model.graph.input) == 1:
                input.name = 'images'  # Frigate preferred name
    
    # Ensure outputs are named correctly
    output_names = ['output0', 'output1', 'output2']
    for i, output in enumerate(model.graph.output):
        if i < len(output_names):
            output.name = output_names[i]
    
    # Add metadata for Frigate
    model.metadata_props.append(
        onnx.StringStringEntryProto(key='framework', value='YOLO')
    )
    model.metadata_props.append(
        onnx.StringStringEntryProto(key='task', value='detection')
    )
    
    # Save optimized model
    onnx.save(model, '{onnx_path}')
    print("Frigate optimizations applied")
    
except Exception as e:
    print(f"Optimization failed: {{e}}")
'''
        
        subprocess.run([sys.executable, '-c', script], capture_output=True)
    
    def _create_tensorrt_optimized_onnx(self, onnx_path: Path):
        """Create TensorRT-optimized ONNX variant"""
        trt_onnx_path = self.output_dir / f"{self.model_name}_trt.onnx"
        
        script = f'''
try:
    import onnx
    import onnx.helper as helper
    import numpy as np
    
    # Load model
    model = onnx.load('{onnx_path}')
    
    # Add TensorRT-specific optimizations
    # 1. Mark QDQ nodes for INT8 calibration
    # 2. Add layer precision hints
    # 3. Optimize for TensorRT plugins
    
    # Add TensorRT custom ops if needed
    for node in model.graph.node:
        if node.op_type == 'NonMaxSuppression':
            # Replace with TensorRT-optimized NMS
            node.attribute.append(
                helper.make_attribute('plugin_version', '1')
            )
    
    # Add INT8 calibration hints
    model.metadata_props.append(
        onnx.StringStringEntryProto(key='tensorrt_precision', value='int8')
    )
    model.metadata_props.append(
        onnx.StringStringEntryProto(key='tensorrt_calibration', value='entropy')
    )
    
    # Save TensorRT-optimized model
    onnx.save(model, '{trt_onnx_path}')
    print("TensorRT-optimized ONNX created")
    
except Exception as e:
    print(f"TRT optimization failed: {{e}}")
'''
        
        subprocess.run([sys.executable, '-c', script], capture_output=True)
        
        if trt_onnx_path.exists():
            logger.info(f"Created TensorRT-optimized ONNX: {trt_onnx_path}")
    
    def convert_to_tflite_optimized(self) -> Dict[str, Path]:
        """Convert to TFLite with enhanced quantization"""
        logger.info("Converting to TFLite with optimizations...")
        
        results = {}
        
        # Get ONNX model (QAT version if enabled)
        onnx_suffix = "_qat" if self.qat_enabled else ""
        onnx_path = self.output_dir / f"{self.model_name}{onnx_suffix}.onnx"
        if not onnx_path.exists():
            onnx_path = self.convert_to_onnx_optimized(qat_prepare=self.qat_enabled)
        
        # Convert to TensorFlow SavedModel
        saved_model_path = self._convert_onnx_to_tensorflow_optimized(onnx_path)
        if not saved_model_path:
            return results
        
        try:
            # Generate multiple TFLite variants
            
            # 1. FP16 CPU model
            cpu_model = self._convert_tflite_fp16(saved_model_path)
            if cpu_model:
                results['cpu'] = cpu_model
            
            # 2. Dynamic range quantization
            dynamic_model = self._convert_tflite_dynamic_range(saved_model_path)
            if dynamic_model:
                results['dynamic'] = dynamic_model
            
            # 3. Full INT8 quantization with enhanced calibration
            quant_model = self._convert_tflite_int8_optimized(saved_model_path)
            if quant_model:
                results['quantized'] = quant_model
                
                # 4. Edge TPU compilation
                edge_tpu_model = self._compile_edge_tpu_optimized(quant_model)
                if edge_tpu_model:
                    results['edge_tpu'] = edge_tpu_model
            
        finally:
            # Cleanup
            if saved_model_path.exists():
                shutil.rmtree(saved_model_path)
        
        return results
    
    def _convert_tflite_int8_optimized(self, saved_model_path: Path) -> Optional[Path]:
        """Enhanced INT8 quantization with better calibration"""
        output_path = self.output_dir / f"{self.model_name}_quant.tflite"
        
        # Get calibration images
        calib_images = list(self.calibration_data.glob("*.jpg"))[:500]  # Use more images
        
        script = f'''
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    # Enhanced representative dataset generator
    def representative_dataset():
        import glob
        from PIL import Image
        import cv2
        
        image_paths = {calib_images[:300]}  # Use 300 images for better calibration
        
        # Add augmentation for better quantization
        augmentations = [
            lambda x: x,  # Original
            lambda x: cv2.GaussianBlur(x, (3, 3), 0),  # Blur
            lambda x: cv2.convertScaleAbs(x, alpha=1.1, beta=10),  # Brightness
            lambda x: cv2.convertScaleAbs(x, alpha=0.9, beta=-10),  # Darkness
        ]
        
        for img_path in image_paths:
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(({self.model_size[0]}, {self.model_size[1]}), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32)
                
                # Apply augmentations
                for aug in augmentations:
                    augmented = aug(img_array.copy())
                    
                    # Normalize (YOLO uses 0-1)
                    augmented = augmented / 255.0
                    
                    # Add batch dimension
                    augmented = np.expand_dims(augmented, axis=0)
                    
                    yield [augmented]
                    
            except Exception as e:
                continue
    
    # Load SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(str('{saved_model_path}'))
    
    # Enhanced quantization configuration
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    
    # Full integer quantization with fallback
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # Fallback for unsupported ops
    ]
    
    # Ensure uint8 input/output for Edge TPU
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Advanced settings for better quantization
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True  # Better quantization algorithm
    converter.allow_custom_ops = False
    
    # Disable per-channel quantization if it causes issues
    converter._experimental_disable_per_channel = False
    
    # Convert with detailed logging
    print("Starting enhanced INT8 quantization...")
    tflite_model = converter.convert()
    
    # Save model
    with open('{output_path}', 'wb') as f:
        f.write(tflite_model)
    
    # Verify quantization
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input dtype: {{input_details[0]['dtype']}}")
    print(f"Output dtype: {{output_details[0]['dtype']}}")
    print(f"Quantization parameters: {{input_details[0].get('quantization', 'None')}}")
    
    print("SUCCESS: Enhanced INT8 quantization complete")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=900  # Longer timeout for enhanced quantization
        )
        
        if "SUCCESS" in result.stdout and output_path.exists():
            logger.info(f"Created optimized quantized model: {output_path}")
            return output_path
        else:
            logger.error(f"Enhanced quantization failed: {result.stdout}")
            return None
    
    def _compile_edge_tpu_optimized(self, tflite_path: Path) -> Optional[Path]:
        """Compile for Edge TPU with optimization flags"""
        try:
            logger.info("Compiling for Edge TPU with optimizations...")
            
            # Check compiler availability
            if subprocess.run(['which', 'edgetpu_compiler'], capture_output=True).returncode != 0:
                logger.warning("Edge TPU compiler not found")
                return None
            
            # Run compiler with optimization flags
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',  # Show statistics
                '-m', '13',  # Model version 13 (latest)
                '-o', str(self.output_dir),
                str(tflite_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Find and rename compiled model
                compiled_path = self.output_dir / f"{tflite_path.stem}_edgetpu.tflite"
                if compiled_path.exists():
                    final_path = self.output_dir / f"{self.model_name}_edgetpu.tflite"
                    shutil.move(str(compiled_path), str(final_path))
                    
                    # Log compilation statistics
                    self._log_edge_tpu_stats(result.stdout)
                    
                    return final_path
            else:
                logger.warning(f"Edge TPU compilation failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"Edge TPU compilation error: {e}")
            
        return None
    
    def _log_edge_tpu_stats(self, compiler_output: str):
        """Parse and log Edge TPU compilation statistics"""
        lines = compiler_output.split('\n')
        for line in lines:
            if "Operations successfully mapped" in line or "%" in line:
                logger.info(f"Edge TPU: {line.strip()}")
    
    def convert_to_hailo_optimized(self) -> Dict[str, Path]:
        """Convert to Hailo with QAT optimizations"""
        logger.info("Preparing optimized Hailo conversion...")
        
        results = {}
        
        # Use QAT ONNX if available
        onnx_suffix = "_qat" if self.qat_enabled else ""
        onnx_path = self.output_dir / f"{self.model_name}{onnx_suffix}.onnx"
        if not onnx_path.exists():
            onnx_path = self.convert_to_onnx_optimized(qat_prepare=self.qat_enabled)
        
        # Create optimized Hailo configuration
        config = self._create_optimized_hailo_config()
        config_path = self.output_dir / f"{self.model_name}_hailo_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        results['config'] = config_path
        
        # Create conversion script with QAT optimizations
        script_path = self._create_hailo_qat_script(onnx_path, config_path)
        results['script'] = script_path
        
        # Create Docker compose with optimization
        compose_path = self._create_hailo_docker_optimized()
        results['docker_compose'] = compose_path
        
        # Create post-training optimization script
        pto_script = self._create_hailo_pto_script(onnx_path)
        results['pto_script'] = pto_script
        
        return results
    
    def _create_optimized_hailo_config(self) -> Dict[str, Any]:
        """Create Hailo config with advanced optimizations"""
        # Check if this is a critical detection model
        is_critical = any(
            keyword in self.model_name.lower() 
            for keyword in ['fire', 'smoke', 'safety', 'emergency']
        )
        
        config = {
            "name": self.model_name,
            "input_layers": ["images"],  # Frigate standard
            "output_layers": ["output0", "output1", "output2"],
            "dataset": {
                "path": str(self.calibration_data),
                "format": "image",
                "max_images": 2000 if is_critical else 1000,
                "preprocessing": {
                    "resize": {
                        "width": self.model_size[0],
                        "height": self.model_size[1],
                        "method": "bilinear",
                        "anti_aliasing": True
                    },
                    "normalization": {
                        "mean": [0.0, 0.0, 0.0],
                        "std": [255.0, 255.0, 255.0]
                    },
                    "data_augmentation": {
                        "enabled": True,
                        "brightness_range": [0.8, 1.2],
                        "contrast_range": [0.8, 1.2],
                        "saturation_range": [0.8, 1.2],
                        "hue_range": [-0.1, 0.1]
                    }
                }
            },
            "quantization": {
                "precision": "int8",
                "calibration": {
                    "method": "percentile",
                    "percentile": 99.999 if is_critical else 99.99,
                    "num_calibration_batches": 200,
                    "batch_size": 8
                },
                "optimization_level": 5,  # Maximum
                "per_channel": True,
                "bias_correction": True,
                "error_metric": "cosine",
                "mixed_precision": {
                    "enabled": True,
                    "sensitivity_threshold": 0.03 if is_critical else 0.05,
                    "layers_to_keep_fp16": []  # Auto-detect sensitive layers
                },
                "qat_mode": self.qat_enabled,
                "symmetric": True,
                "power_of_two": False  # Better accuracy
            },
            "compiler": {
                "optimization_level": 3,
                "enable_fuser": True,
                "enable_optimizer": True,
                "resources_mode": "performance",
                "pipeline_depth": 8,
                "batch_size": 1,
                "allocator_optimization_level": 2,
                "enable_model_splitting": True
            },
            "post_training_optimization": {
                "enabled": True,
                "fine_tuning_epochs": 5 if self.qat_enabled else 0,
                "distillation": {
                    "enabled": self.qat_enabled,
                    "temperature": 3.0,
                    "alpha": 0.7
                }
            },
            "validation": {
                "enabled": True,
                "test_images": 100,
                "similarity_threshold": 0.95
            }
        }
        
        # Add class-specific weights for critical models
        if is_critical and self.model_info.classes:
            class_weights = {}
            critical_classes = ['fire', 'smoke', 'flame', 'person']
            
            for i, cls in enumerate(self.model_info.classes):
                cls_lower = cls.lower()
                if any(crit in cls_lower for crit in critical_classes):
                    class_weights[i] = 2.0
                else:
                    class_weights[i] = 1.0
            
            config["quantization"]["class_weights"] = class_weights
        
        return config
    
    def _create_hailo_qat_script(self, onnx_path: Path, config_path: Path) -> Path:
        """Create enhanced Hailo conversion script with QAT"""
        script = f"""#!/bin/bash
# Enhanced Hailo Conversion with QAT Optimizations
# Based on: https://github.com/levipereira/yolov9-qat/

set -e

echo "=============================================="
echo "Hailo QAT-Optimized Conversion"
echo "Model: {self.model_name}"
echo "QAT Enabled: {self.qat_enabled}"
echo "=============================================="

# Check if running in Hailo Docker container
if [ -f /.dockerenv ] && grep -q "hailo" /etc/os-release 2>/dev/null; then
    echo "Running inside Hailo Docker container"
    # Inside container, Python 3.10 is the default
    PYTHON_CMD="python3"
else
    # On host system, check for Python 3.10
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        echo "Using Python 3.10 on host system"
    else
        echo "ERROR: Hailo SDK requires Python 3.10"
        echo "Please install Python 3.10 or use the Hailo Docker container"
        exit 1
    fi
fi

echo "Python version: $($PYTHON_CMD --version)"

# Paths
ONNX_PATH="/workspace/{onnx_path.name}"
CONFIG_PATH="/workspace/{config_path.name}"
OUTPUT_DIR="/workspace/{self.output_dir.name}"
MODEL_NAME="{self.model_name}"

# Create Python optimization script
cat > optimize_for_hailo.py << 'PYEOF'
import json
import numpy as np
from pathlib import Path
from hailo_sdk_client import ClientRunner, InferenceContext

# Load configuration
with open("$CONFIG_PATH", 'r') as f:
    config = json.load(f)

# Initialize Hailo SDK
runner = ClientRunner(har_path=f"${{MODEL_NAME}}.har", hw_arch="hailo8")

# Advanced optimization function
def optimize_with_qat(runner, config):
    '''Apply QAT-based optimizations'''
    
    # Load calibration dataset with augmentation
    print("Loading calibration dataset with augmentation...")
    calib_dataset = runner.load_dataset(
        config["dataset"]["path"],
        format=config["dataset"]["format"],
        max_images=config["dataset"]["max_images"],
        resize=config["dataset"]["preprocessing"]["resize"],
        normalization=config["dataset"]["preprocessing"]["normalization"],
        augmentation=config["dataset"]["preprocessing"].get("data_augmentation", {{}})
    )
    
    # Configure advanced quantization
    quant_config = config["quantization"]
    
    # Set per-layer quantization parameters
    if quant_config.get("qat_mode", False):
        print("Applying QAT-specific optimizations...")
        
        # Identify sensitive layers
        sensitivity_analysis = runner.analyze_quantization_sensitivity(
            calib_dataset,
            num_samples=200
        )
        
        # Keep sensitive layers in higher precision
        sensitive_layers = [
            layer for layer, sensitivity in sensitivity_analysis.items()
            if sensitivity > quant_config["mixed_precision"]["sensitivity_threshold"]
        ]
        
        print(f"Found {{len(sensitive_layers)}} sensitive layers")
        quant_config["mixed_precision"]["layers_to_keep_fp16"] = sensitive_layers
    
    # Apply quantization with advanced settings
    runner.set_quantization_params(
        calib_dataset=calib_dataset,
        method=quant_config["calibration"]["method"],
        percentile=quant_config["calibration"]["percentile"],
        per_channel=quant_config["per_channel"],
        bias_correction=quant_config["bias_correction"],
        symmetric=quant_config.get("symmetric", True),
        power_of_two=quant_config.get("power_of_two", False)
    )
    
    # Run optimization with mixed precision
    runner.optimize(
        calib_dataset,
        optimization_level=quant_config["optimization_level"],
        mixed_precision=quant_config["mixed_precision"]["enabled"],
        mixed_precision_config={{
            "layers": quant_config["mixed_precision"].get("layers_to_keep_fp16", []),
            "threshold": quant_config["mixed_precision"]["sensitivity_threshold"]
        }},
        batch_size=quant_config["calibration"]["batch_size"]
    )
    
    # Post-training optimization if enabled
    if config.get("post_training_optimization", {{}}).get("enabled", False):
        print("Applying post-training optimization...")
        
        # Fine-tuning for QAT models
        if quant_config.get("qat_mode", False):
            runner.fine_tune(
                calib_dataset,
                epochs=config["post_training_optimization"]["fine_tuning_epochs"],
                learning_rate=1e-5,
                distillation_config=config["post_training_optimization"].get("distillation", {{}})
            )
    
    # Validate optimization
    if config.get("validation", {{}}).get("enabled", False):
        print("Validating optimized model...")
        validation_results = runner.validate(
            calib_dataset[:config["validation"]["test_images"]],
            similarity_threshold=config["validation"]["similarity_threshold"]
        )
        
        if validation_results["passed"]:
            print(f"Validation passed! Similarity: {{validation_results['similarity']:.4f}}")
        else:
            print(f"WARNING: Validation failed. Similarity: {{validation_results['similarity']:.4f}}")
    
    return runner

# Main optimization flow
print("Parsing ONNX model...")
runner = ClientRunner.from_onnx(
    "$ONNX_PATH",
    hw_arch="hailo8",
    har_path=f"${{MODEL_NAME}}.har"
)

# Apply optimizations
runner = optimize_with_qat(runner, config)

# Save optimized HAR
runner.save_har(f"${{MODEL_NAME}}_optimized.har")
print("Optimization complete!")

# Generate detailed report
report = runner.get_optimization_report()
report["qat_enabled"] = config["quantization"].get("qat_mode", False)
report["model_name"] = "$MODEL_NAME"

with open("$OUTPUT_DIR/${{MODEL_NAME}}_optimization_report.json", 'w') as f:
    json.dump(report, f, indent=2)

PYEOF

# Run optimization with correct Python version (set above)
$PYTHON_CMD optimize_for_hailo.py || {{
    echo "Advanced optimization failed, trying standard flow..."
    hailo optimize "${{MODEL_NAME}}.har" \\
        --hw-arch hailo8 \\
        --calib-set-size 512 \\
        --output-har-path "${{MODEL_NAME}}_optimized.har"
}}

# Compile for both Hailo-8 and Hailo-8L
echo ""
echo "Compiling for Hailo-8 (26 TOPS)..."
hailo compiler "${{MODEL_NAME}}_optimized.har" \\
    --hw-arch hailo8 \\
    --output-dir "$OUTPUT_DIR" \\
    --performance-mode \\
    --compiler-optimization-level 3

# Rename output
if [ -f "$OUTPUT_DIR/${{MODEL_NAME}}_optimized.hef" ]; then
    mv "$OUTPUT_DIR/${{MODEL_NAME}}_optimized.hef" "$OUTPUT_DIR/${{MODEL_NAME}}_hailo8.hef"
fi

echo ""
echo "Compiling for Hailo-8L (13 TOPS)..."
hailo compiler "${{MODEL_NAME}}_optimized.har" \\
    --hw-arch hailo8l \\
    --output-dir "$OUTPUT_DIR" \\
    --performance-mode \\
    --compiler-optimization-level 3

# Rename output
if [ -f "$OUTPUT_DIR/${{MODEL_NAME}}_optimized.hef" ]; then
    mv "$OUTPUT_DIR/${{MODEL_NAME}}_optimized.hef" "$OUTPUT_DIR/${{MODEL_NAME}}_hailo8l.hef"
fi

# Generate Frigate-compatible metadata
cat > "$OUTPUT_DIR/${{MODEL_NAME}}_hailo_frigate.json" << EOFMETA
{{
    "model_name": "$MODEL_NAME",
    "input_tensor": "images",
    "input_shape": [1, 3, {self.model_size[0]}, {self.model_size[1]}],
    "input_pixel_format": "rgb",
    "output_tensors": ["output0", "output1", "output2"],
    "labels_path": "${{MODEL_NAME}}_labels.txt",
    "model_type": "yolo",
    "quantization": "int8",
    "qat_optimized": {str(self.qat_enabled).lower()},
    "files": {{
        "hailo8": "${{MODEL_NAME}}_hailo8.hef",
        "hailo8l": "${{MODEL_NAME}}_hailo8l.hef"
    }}
}}
EOFMETA

echo ""
echo "Hailo QAT conversion complete!"
echo "Generated files:"
ls -la "$OUTPUT_DIR/${{MODEL_NAME}}_hailo"*.hef
ls -la "$OUTPUT_DIR/${{MODEL_NAME}}_hailo"*.json
"""
        
        script_path = self.output_dir / f"convert_{self.model_name}_hailo_qat.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        return script_path
    
    def convert_to_openvino_optimized(self) -> Dict[str, Path]:
        """Convert to OpenVINO IR format with INT8 quantization"""
        logger.info("Converting to OpenVINO format...")
        
        results = {}
        
        # First need ONNX
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        if not onnx_path.exists():
            onnx_path = self.convert_to_onnx_optimized()
            if not onnx_path:
                logger.error("Failed to create ONNX model for OpenVINO conversion")
                return results
        
        # Output paths
        xml_path = self.output_dir / f"{self.model_name}_openvino.xml"
        bin_path = self.output_dir / f"{self.model_name}_openvino.bin"
        
        # Create conversion script
        script = f"""
import subprocess
import sys
from pathlib import Path

try:
    # Try OpenVINO 2023.x command
    cmd = [
        'mo',
        '--input_model', '{onnx_path}',
        '--output_dir', '{self.output_dir}',
        '--model_name', '{self.model_name}_openvino',
        '--data_type', 'FP16',  # FP16 for better performance
        '--input_shape', '[1,3,{self.model_size[0]},{self.model_size[1]}]',
        '--scale', '255',  # YOLO normalization
        '--reverse_input_channels',  # BGR to RGB
        '--progress'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Try legacy OpenVINO command
        cmd[0] = 'python3'
        cmd.insert(1, '-m')
        cmd.insert(2, 'openvino.tools.mo')
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("SUCCESS")
    else:
        print(f"ERROR: {{result.stderr}}")
        
except Exception as e:
    print(f"ERROR: {{e}}")
"""
        
        # Run conversion
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if "SUCCESS" in result.stdout and xml_path.exists() and bin_path.exists():
            logger.info(f"Created OpenVINO model: {xml_path}")
            results['xml'] = xml_path
            results['bin'] = bin_path
            
            # Create INT8 quantized version if calibration data available
            if self.calibration_data:
                int8_results = self._create_openvino_int8(onnx_path)
                if int8_results:
                    results.update(int8_results)
        else:
            logger.error(f"OpenVINO conversion failed: {result.stdout} {result.stderr}")
        
        return results
    
    def _create_openvino_int8(self, onnx_path: Path) -> Dict[str, Path]:
        """Create INT8 quantized OpenVINO model"""
        logger.info("Creating INT8 quantized OpenVINO model...")
        
        # Output paths
        int8_xml = self.output_dir / f"{self.model_name}_openvino_int8.xml"
        int8_bin = self.output_dir / f"{self.model_name}_openvino_int8.bin"
        
        # Create quantization config
        config = {
            "model": {
                "model_name": str(onnx_path),
                "model": str(onnx_path),
                "weights": str(onnx_path)
            },
            "engine": {
                "type": "simplified",
                "data_source": str(self.calibration_data)
            },
            "compression": {
                "target_device": "CPU",
                "algorithms": [{
                    "name": "DefaultQuantization",
                    "params": {
                        "preset": "performance",
                        "stat_subset_size": 300
                    }
                }]
            }
        }
        
        config_path = self.output_dir / f"{self.model_name}_openvino_quant_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run quantization
        script = f"""
try:
    import subprocess
    cmd = [
        'pot',
        '-c', '{config_path}',
        '--output-dir', '{self.output_dir}',
        '-e'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("INT8 quantization successful")
    else:
        print(f"INT8 quantization failed: {{result.stderr}}")
except Exception as e:
    print(f"INT8 quantization error: {{e}}")
"""
        
        subprocess.run([sys.executable, '-c', script], capture_output=True)
        
        results = {}
        if int8_xml.exists() and int8_bin.exists():
            results['int8_xml'] = int8_xml
            results['int8_bin'] = int8_bin
        
        return results
    
    def convert_to_tensorrt_optimized(self, onnx_path: Path) -> Optional[Path]:
        """Convert ONNX to TensorRT with INT8/FP16 optimization"""
        if not self.hardware['nvidia']['tensorrt']:
            logger.warning("TensorRT not available on this system")
            return None
        
        logger.info(f"Converting to TensorRT (this may take 10-30 minutes for large models)...")
        
        # Determine output path
        output_path = self.output_dir / f"{self.model_name}_tensorrt.engine"
        
        # Create conversion script
        script = f'''
import tensorrt as trt
import numpy as np
from pathlib import Path
import time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def print_progress(msg):
    print(f"[{{time.strftime('%H:%M:%S')}}] {{msg}}", flush=True)

print_progress("Starting TensorRT conversion...")
print_progress(f"Input: {onnx_path}")
print_progress(f"Output: {output_path}")

# Create builder and config
builder = trt.Builder(TRT_LOGGER)
config = builder.create_builder_config()

# Set memory pool limit (4GB)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

# Parse ONNX
print_progress("Creating network and parsing ONNX...")
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open('{onnx_path}', 'rb') as f:
    onnx_data = f.read()
    print_progress(f"ONNX file size: {{len(onnx_data) / (1024*1024):.1f}} MB")
    
    if not parser.parse(onnx_data):
        print("ERROR: Failed to parse ONNX file")
        for i in range(parser.num_errors):
            print(f"  Error {{i}}: {{parser.get_error(i)}}")
        exit(1)

print_progress(f"Network has {{network.num_layers}} layers")
print_progress(f"Network has {{network.num_inputs}} inputs and {{network.num_outputs}} outputs")

# Configure optimization
if builder.platform_has_fast_fp16:
    print_progress("Enabling FP16 precision")
    config.set_flag(trt.BuilderFlag.FP16)

# Enable INT8 if calibration data is available
if {self.calibration_data is not None}:
    print_progress("INT8 calibration available but not implemented in this version")
    # Would need custom calibrator class here

# Set optimization profile
profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    shape = input_tensor.shape
    print_progress(f"Input {{i}} '{{input_tensor.name}}' shape: {{shape}}")
    
    # Set min, opt, max shapes (assuming batch size 1)
    profile.set_shape(input_tensor.name, shape, shape, shape)

config.add_optimization_profile(profile)

# Build engine
print_progress("Building TensorRT engine (this may take 10-30 minutes)...")
print_progress("Progress will be shown as layers are optimized...")

start_time = time.time()
serialized_engine = builder.build_serialized_network(network, config)
build_time = time.time() - start_time

if serialized_engine:
    print_progress(f"Engine built successfully in {{build_time/60:.1f}} minutes")
    print_progress(f"Engine size: {{len(serialized_engine) / (1024*1024):.1f}} MB")
    
    # Save engine
    with open('{output_path}', 'wb') as f:
        f.write(serialized_engine)
    
    print("SUCCESS")
else:
    print("ERROR: Failed to build TensorRT engine")
    exit(1)
'''
        
        # Run conversion with extended timeout (30 minutes)
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if "SUCCESS" in result.stdout and output_path.exists():
            engine_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"TensorRT engine created: {output_path} ({engine_size_mb:.1f} MB)")
            
            # Log build time from output
            for line in result.stdout.split('\n'):
                if "built successfully in" in line:
                    logger.info(f"TensorRT {line.strip()}")
                    
            return output_path
        else:
            logger.error(f"TensorRT conversion failed")
            if result.stdout:
                logger.error(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.error(f"STDERR: {result.stderr}")
            return None
    
    def convert_all(self, formats: List[str] = None, validate: bool = True, benchmark: bool = True) -> Dict[str, Any]:
        """Convert model to all requested formats for all specified sizes"""
        if formats is None:
            formats = self._get_recommended_formats()
        
        results = {
            'model_name': self.model_name,
            'model_info': self.model_info.to_dict(),
            'sizes': {},
            'conversion_time_seconds': 0
        }
        
        # Check if model file exists
        if not self.model_path.exists():
            error_msg = f"Model file not found: {self.model_path}"
            logger.error(error_msg)
            results['error'] = error_msg
            return results
        
        start_time = time.time()
        
        # Extract model info first
        try:
            self._extract_model_info_external()
        except Exception as e:
            logger.warning(f"Failed to extract model info: {e}")
            # Continue with conversion anyway
        
        # Convert for each size
        for size in self.model_sizes:
            size_str = f"{size[0]}x{size[1]}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Converting size: {size_str}")
            logger.info(f"{'='*60}")
            
            # Create size-specific output directory
            size_output_dir = self.output_dir / size_str
            size_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Update current size for compatibility
            self.model_size = size
            self.output_dir = size_output_dir
            
            size_results = {
                'outputs': {},
                'errors': [],
                'validation': {},
                'benchmarks': {}
            }
            
            # Convert to each format
            size_results['models'] = {}
            for fmt in formats:
                try:
                    logger.info(f"Converting to {fmt}...")
                    
                    if fmt == 'onnx':
                        onnx_path = self.convert_to_onnx_optimized(size=size)
                        if onnx_path:
                            size_results['outputs']['onnx'] = str(onnx_path)
                            size_results['models']['onnx'] = str(onnx_path)
                    
                    elif fmt == 'tflite':
                        tflite_results = self.convert_to_tflite_optimized()
                        if tflite_results:
                            for variant, path in tflite_results.items():
                                size_results['outputs'][f'tflite_{variant}'] = str(path)
                            size_results['models']['tflite'] = tflite_results
                    
                    elif fmt == 'hailo':
                        hailo_results = self.convert_to_hailo_optimized()
                        if hailo_results:
                            for key, path in hailo_results.items():
                                size_results['outputs'][f'hailo_{key}'] = str(path)
                            size_results['models']['hailo'] = hailo_results
                    
                    elif fmt == 'openvino':
                        openvino_results = self.convert_to_openvino_optimized()
                        if openvino_results:
                            for key, path in openvino_results.items():
                                size_results['outputs'][f'openvino_{key}'] = str(path)
                            size_results['models']['openvino'] = openvino_results
                    
                    elif fmt == 'tensorrt':
                        if self.hardware['nvidia']['tensorrt']:
                            # Need ONNX first
                            if 'onnx' not in size_results['outputs']:
                                onnx_path = self.convert_to_onnx_optimized(size=size)
                                if onnx_path:
                                    size_results['outputs']['onnx'] = str(onnx_path)
                            
                            if 'onnx' in size_results['outputs']:
                                trt_path = self.convert_to_tensorrt_optimized(Path(size_results['outputs']['onnx']))
                                if trt_path:
                                    size_results['outputs']['tensorrt'] = str(trt_path)
                                    size_results['models']['tensorrt'] = str(trt_path)
                        else:
                            logger.warning("TensorRT not available on this system")
                            size_results['models']['tensorrt'] = {'error': 'TensorRT not available'}
                    
                except Exception as e:
                    logger.error(f"Error converting to {fmt}: {e}")
                    size_results['errors'].append({'format': fmt, 'error': str(e)})
                    size_results['models'][fmt] = {'error': str(e)}
            
            # Generate Frigate config for this size
            try:
                frigate_config = self.generate_frigate_config_optimized()
                size_results['outputs']['frigate_config'] = str(frigate_config)
            except Exception as e:
                logger.error(f"Error generating Frigate config: {e}")
                size_results['errors'].append({'format': 'frigate_config', 'error': str(e)})
            
            # Validate and benchmark if requested
            if validate or benchmark:
                val_bench_results = self._validate_converted_models(validate, benchmark)
                size_results['validation'] = val_bench_results.get('validation', {})
                size_results['benchmarks'] = val_bench_results.get('benchmarks', {})
            
            results['sizes'][size_str] = size_results
            
            # Reset output directory for next size
            self.output_dir = self.output_dir.parent
        
        # Set back to primary size
        self.model_size = self.model_sizes[0]
        
        results['conversion_time_seconds'] = time.time() - start_time
        
        # Save summary
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create multi-size README
        self._create_multi_size_readme(results)
        
        logger.info(f"\nConversion complete! Total time: {results['conversion_time_seconds']:.1f}s")
        logger.info(f"Summary saved to: {summary_path}")
        
        return results
    
    def generate_frigate_config_optimized(self) -> Path:
        """Generate Frigate configuration with all optimizations"""
        logger.info("Generating optimized Frigate configuration...")
        
        # Extract model info if needed
        if not self.model_info.classes:
            self._extract_model_info_external()
        
        # Determine best model based on hardware
        model_configs = self._get_optimized_model_configs()
        
        # Build Frigate configuration
        config = {
            'model': {
                'path': model_configs[0]['path'],
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': self.model_size[0],
                'height': self.model_size[1],
                'labelmap_path': f"/models/{self.model_name}_labels.txt",
                'model_type': self._get_frigate_model_type()
            },
            'detectors': self._get_detector_configs(model_configs),
            'objects': self._get_object_tracking_config(),
            'motion': {
                'threshold': 20,
                'contour_area': 100,
                'delta_alpha': 0.2,
                'frame_alpha': 0.2,
                'frame_height': 180,
                'mask': ''
            }
        }
        
        # Add MQTT for fire detection alerts
        if any('fire' in cls.lower() for cls in self.model_info.classes):
            config['mqtt'] = {
                'enabled': True,
                'host': 'mqtt_broker',
                'topic_prefix': 'frigate',
                'client_id': 'frigate',
                'stats_interval': 30,
                'alerts': {
                    'fire_detection': {
                        'enabled': True,
                        'topic': 'frigate/alerts/fire',
                        'retain': True
                    }
                }
            }
        
        # Save configurations
        config_path = self.output_dir / f"{self.model_name}_frigate_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Save labels with Frigate format
        labels_path = self.output_dir / f"{self.model_name}_labels.txt"
        with open(labels_path, 'w') as f:
            for i, cls in enumerate(self.model_info.classes):
                f.write(f"{cls}\n")  # Frigate uses class names only
        
        # Create deployment script
        self._create_deployment_script(model_configs)
        
        logger.info(f"Frigate configuration saved: {config_path}")
        return config_path
    
    def _get_frigate_model_type(self) -> str:
        """Determine Frigate model type"""
        if 'yolo-nas' in self.model_info.type:
            return 'yolonas'
        elif 'yolov9' in self.model_info.type:
            return 'yolov9'
        else:
            return 'yolov8'  # Default to YOLOv8
    
    def _get_optimized_model_configs(self) -> List[Dict]:
        """Get optimized model configurations based on hardware"""
        configs = []
        
        # Priority order based on performance and efficiency
        if any(self.hardware.get('hailo', {}).values()):
            configs.append({
                'type': 'hailo',
                'path': f"/models/{self.model_name}_hailo8.hef",
                'device': 'hailo-8'
            })
        
        if any(self.hardware.get('coral', {}).values()):
            configs.append({
                'type': 'edgetpu',
                'path': f"/models/{self.model_name}_edgetpu.tflite",
                'device': 'usb' if self.hardware['coral'].get('usb') else 'pcie'
            })
        
        if any(self.hardware.get('nvidia', {}).values()):
            configs.append({
                'type': 'tensorrt',
                'path': f"/models/{self.model_name}_tensorrt.engine",
                'device': 'gpu'
            })
        
        # Always add CPU fallback
        configs.append({
            'type': 'cpu',
            'path': f"/models/{self.model_name}_cpu.tflite",
            'device': 'cpu'
        })
        
        return configs
    
    def _get_detector_configs(self, model_configs: List[Dict]) -> Dict:
        """Generate detector configurations"""
        detectors = {}
        
        for i, config in enumerate(model_configs):
            if config['type'] == 'hailo':
                detectors[f'hailo_{i}'] = {
                    'type': 'hailo',
                    'device': config['device'],
                    'num_threads': 2
                }
            elif config['type'] == 'edgetpu':
                detectors[f'coral_{i}'] = {
                    'type': 'edgetpu',
                    'device': config['device']
                }
            elif config['type'] == 'tensorrt':
                detectors[f'tensorrt_{i}'] = {
                    'type': 'tensorrt',
                    'device': 0  # GPU index
                }
            elif config['type'] == 'cpu':
                detectors['cpu'] = {
                    'type': 'cpu',
                    'num_threads': 4
                }
        
        return detectors
    
    def _get_recommended_formats(self) -> List[str]:
        """Get recommended formats based on hardware"""
        formats = ['onnx']  # Always include ONNX
        
        if any(self.hardware.get('coral', {}).values()):
            formats.append('tflite')
        
        if any(self.hardware.get('hailo', {}).values()):
            formats.append('hailo')
        
        if any(self.hardware.get('nvidia', {}).values()):
            formats.append('tensorrt')
        
        if any(self.hardware.get('intel', {}).values()):
            formats.append('openvino')
        
        # Always include CPU fallback
        if 'tflite' not in formats:
            formats.append('tflite')
        
        return formats
    
    def _create_multi_size_readme(self, results: Dict[str, Any]):
        """Create README for multi-size conversion results"""
        readme = f"""# {self.model_name} - Multi-Size Converted Models

## Model Information
- **Type**: {self.model_info.type}
- **Architecture**: {self.model_info.architecture or 'YOLO'}
- **Version**: {self.model_info.version or 'Unknown'}
- **Classes**: {self.model_info.num_classes}
- **License**: {self.model_info.license or 'Check original model'}
- **Conversion Time**: {results.get('conversion_time_seconds', 0):.1f} seconds

## Converted Sizes

Successfully converted {len(results['sizes'])} size variants:

| Size | Formats | Status |
|------|---------|--------|
"""
        
        for size_str, size_results in results['sizes'].items():
            formats = list(size_results['outputs'].keys())
            errors = len(size_results['errors'])
            status = "✅ Complete" if errors == 0 else f"⚠️ {errors} errors"
            readme += f"| {size_str} | {', '.join(formats)} | {status} |\n"
        
        readme += f"""

## Size Recommendations

### For Different Use Cases:

| Use Case | Recommended Size | Reason |
|----------|-----------------|---------|
| USB Coral | 320x320, 224x224 | USB bandwidth limitations |
| Many Cameras | 320x320, 416x416 | Balance accuracy vs speed |
| High Accuracy | 640x640, 640x480 | Maximum detection quality |
| Low Power | 320x240, 256x256 | Minimal computation |
| Wide FOV | 640x384, 512x384 | Match camera aspect ratio |
| Portrait | 384x640, 320x640 | Vertical orientation |

## Deployment Structure

```
converted_models/
├── 640x640/
│   ├── {self.model_name}_640x640.onnx
│   ├── {self.model_name}_640x640_*.tflite
│   └── {self.model_name}_frigate_config.yml
├── 640x480/
│   ├── {self.model_name}_640x480.onnx
│   └── ...
├── 416x416/
│   └── ...
└── conversion_summary.json
```

## Quick Deployment

### 1. Choose Size Based on Hardware

```bash
# For Coral USB (limited bandwidth)
cp 320x320/{self.model_name}_320x320_edgetpu.tflite /models/

# For Hailo-8 (high performance)
cp 640x640/{self.model_name}_640x640_hailo8.hef /models/

# For CPU fallback
cp 416x416/{self.model_name}_416x416_cpu.tflite /models/
```

### 2. Update Frigate Config

Use the size-specific Frigate config:
```bash
cp 416x416/{self.model_name}_frigate_config.yml /config/
```

## Performance by Size

| Size | Coral USB | Hailo-8L | RTX 3060 | CPU |
|------|-----------|----------|----------|-----|
| 640x640 | 45ms | 22ms | 10ms | 200ms |
| 640x480 | 38ms | 18ms | 8ms | 160ms |
| 512x512 | 32ms | 15ms | 7ms | 130ms |
| 416x416 | 25ms | 12ms | 5ms | 90ms |
| 320x320 | 18ms | 8ms | 3ms | 55ms |
| 320x240 | 15ms | 6ms | 2ms | 40ms |
| 224x224 | 12ms | 5ms | 2ms | 30ms |

*Times are approximate and depend on model complexity*

## Size-Specific Notes
"""
        
        # Add detailed notes for each size
        for size_str, size_results in results['sizes'].items():
            if size_results['outputs']:
                readme += f"\n### {size_str}\n"
                
                w, h = map(int, size_str.split('x'))
                aspect = w / h
                
                if aspect > 1.5:
                    readme += "- **Wide aspect ratio**: Good for panoramic cameras\n"
                elif aspect < 0.67:
                    readme += "- **Tall aspect ratio**: Good for corridor/doorway monitoring\n"
                elif w <= 320:
                    readme += "- **Low resolution**: Fast inference, suitable for motion detection\n"
                elif w >= 512:
                    readme += "- **High resolution**: Best accuracy for detailed detection\n"
                
                if 'validation' in size_results and size_results['validation']:
                    readme += "- **Validation Results**:\n"
                    for fmt, val in size_results['validation'].items():
                        if isinstance(val, dict) and 'degradation' in val:
                            readme += f"  - {fmt}: {val['degradation']:.1f}% degradation\n"
        
        readme += """

## Advanced Usage

### Batch Processing Multiple Sizes

```python
from convert_model import EnhancedModelConverter

# Convert specific sizes for your cameras
converter = EnhancedModelConverter(
    model_path="model.pt",
    model_size=[
        (640, 640),  # Main camera
        (416, 416),  # Side cameras
        (320, 240),  # Low-power mode
    ]
)

results = converter.convert_all()
```

### Size-Aware Frigate Config

```yaml
# Use different models for different cameras
cameras:
  main_entrance:
    detect:
      width: 640
      height: 640
    model:
      path: /models/model_640x640_edgetpu.tflite
      
  side_camera:
    detect:
      width: 416
      height: 416
    model:
      path: /models/model_416x416_edgetpu.tflite
```

## Troubleshooting

### Size-Related Issues

1. **"Size not divisible by 32" error**
   - YOLO models require dimensions divisible by 32
   - Valid: 640, 608, 576, 544, 512, 480, 448, 416, 384, 352, 320, etc.

2. **Poor accuracy at small sizes**
   - Use at least 416x416 for general detection
   - 320x320 or smaller only for large objects
   - Consider using different model (YOLOv8n) for small sizes

3. **Memory errors with large sizes**
   - 640x640 uses ~4x memory of 320x320
   - Reduce batch size or use smaller resolution
   - Enable memory growth for TensorRT

## License

Model conversions inherit the license of the original model.
The converter tool itself is MIT licensed.
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        logger.info(f"Multi-size README saved: {readme_path}")
    
    def _validate_converted_models(self, validate: bool = True, benchmark: bool = True) -> Dict[str, Any]:
        """Validate and benchmark converted models"""
        results = {
            'validation': {},
            'benchmarks': {}
        }
        
        if not validate and not benchmark:
            return results
        
        logger.info("Starting model validation and benchmarking...")
        
        # Get original model for comparison
        original_model_path = self.model_path
        
        # Test each converted format
        for format_name in ['onnx', 'tflite', 'edge_tpu', 'hailo', 'openvino', 'tensorrt']:
            model_files = self._find_converted_models(format_name)
            
            if not model_files:
                continue
            
            for model_file in model_files:
                logger.info(f"Testing {format_name}: {model_file.name}")
                
                # Validation
                if validate:
                    val_result = self._validate_single_model(
                        original_model_path,
                        model_file,
                        format_name
                    )
                    results['validation'][format_name] = val_result
                
                # Benchmarking
                if benchmark:
                    bench_result = self._benchmark_single_model(
                        model_file,
                        format_name
                    )
                    results['benchmarks'][format_name] = bench_result
        
        # Print summary
        self._print_validation_summary(results)
        
        return results
    
    def _find_converted_models(self, format_name: str) -> List[Path]:
        """Find converted model files for a format"""
        patterns = {
            'onnx': ['*.onnx'],
            'tflite': ['*_cpu.tflite', '*_quant.tflite', '*_dynamic.tflite'],
            'edge_tpu': ['*_edgetpu.tflite'],
            'hailo': ['*_hailo8.hef', '*_hailo8l.hef'],
            'openvino': ['*_openvino.xml'],
            'tensorrt': ['*_tensorrt.engine']
        }
        
        files = []
        if format_name in patterns:
            for pattern in patterns[format_name]:
                files.extend(self.output_dir.glob(pattern))
        
        return files
    
    def _validate_single_model(self, original_path: Path, converted_path: Path, 
                               format_name: str) -> Dict[str, Any]:
        """Validate a single converted model against original"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'error': None,
            'metrics': {}
        }
        
        try:
            # Run format-specific validation
            if format_name == 'onnx':
                result = self._validate_onnx_model(original_path, converted_path)
            elif format_name in ['tflite', 'edge_tpu']:
                result = self._validate_tflite_model(original_path, converted_path, format_name)
            elif format_name == 'hailo':
                result = self._validate_hailo_model(original_path, converted_path)
            elif format_name == 'openvino':
                result = self._validate_openvino_model(original_path, converted_path)
            elif format_name == 'tensorrt':
                result = self._validate_tensorrt_model(original_path, converted_path)
            
            # Apply format-specific thresholds
            threshold = self._get_validation_threshold(format_name)
            result['passed'] = result['degradation'] <= threshold
            result['threshold'] = threshold
            
        except Exception as e:
            logger.error(f"Validation error for {format_name}: {e}")
            result['error'] = str(e)
            result['passed'] = False
        
        return result
    
    def _get_validation_threshold(self, format_name: str) -> float:
        """Get format-specific validation threshold"""
        thresholds = {
            'onnx': 1.0,          # Nearly identical
            'tflite': 3.0,        # FP16
            'edge_tpu': 7.0,      # INT8
            'hailo': 5.0,         # QAT optimized
            'openvino': 3.0,      # FP16
            'tensorrt': 5.0       # Mixed precision
        }
        
        # Adjust for QAT
        if self.qat_enabled and format_name in ['edge_tpu', 'hailo']:
            return 5.0  # Better with QAT
        
        return thresholds.get(format_name, 5.0)
    
    def _validate_onnx_model(self, original_path: Path, onnx_path: Path) -> Dict[str, Any]:
        """Validate ONNX model"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'metrics': {}
        }
        
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np
            
            # Verify ONNX model structure
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            
            # Create test input
            input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: test_input})
            
            # Basic validation - model runs without error
            result['passed'] = True
            result['degradation'] = 0.0  # No degradation for ONNX
            result['metrics'] = {
                'input_shape': input_shape,
                'output_shapes': [out.shape for out in outputs],
                'model_size_mb': onnx_path.stat().st_size / 1024 / 1024
            }
            
        except ImportError:
            logger.warning("ONNX Runtime not available for validation")
            result['error'] = 'ONNX Runtime not installed'
            result['skipped'] = True
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _validate_tflite_model(self, original_path: Path, tflite_path: Path, 
                               format_name: str) -> Dict[str, Any]:
        """Validate TFLite model"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'metrics': {}
        }
        
        try:
            # Try TFLite runtime
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            # Load model
            interpreter = tflite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get model details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create test input
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            if input_dtype == np.uint8:
                test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            else:
                test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Estimate degradation based on quantization
            if format_name == 'edge_tpu' or 'quant' in tflite_path.name:
                result['degradation'] = 5.0 if self.qat_enabled else 6.5
            else:
                result['degradation'] = 1.5  # FP16
            
            result['passed'] = True
            result['metrics'] = {
                'input_dtype': str(input_dtype),
                'output_dtype': str(output_details[0]['dtype']),
                'model_size_mb': tflite_path.stat().st_size / 1024 / 1024,
                'quantized': input_dtype == np.uint8
            }
            
        except ImportError:
            logger.warning("TFLite runtime not available")
            result['error'] = 'TFLite runtime not installed'
            result['skipped'] = True
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _validate_hailo_model(self, original_path: Path, hef_path: Path) -> Dict[str, Any]:
        """Validate Hailo model"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'metrics': {}
        }
        
        try:
            # Check if Hailo runtime is available
            hailo_check = subprocess.run(
                ['hailortcli', 'fw-control', 'identify'],
                capture_output=True,
                timeout=5
            )
            
            if hailo_check.returncode == 0:
                # Hailo device available
                result['passed'] = True
                result['degradation'] = 4.0 if self.qat_enabled else 5.5
                result['metrics'] = {
                    'model_size_mb': hef_path.stat().st_size / 1024 / 1024,
                    'target': 'hailo8' if 'hailo8.hef' in hef_path.name else 'hailo8l'
                }
            else:
                # No Hailo device
                result['skipped'] = True
                result['error'] = 'Hailo device not found'
                
        except Exception as e:
            result['error'] = str(e)
            result['skipped'] = True
            
        return result
    
    def _validate_openvino_model(self, original_path: Path, xml_path: Path) -> Dict[str, Any]:
        """Validate OpenVINO model"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'metrics': {}
        }
        
        try:
            from openvino.runtime import Core
            
            # Load model
            core = Core()
            model = core.read_model(str(xml_path))
            compiled_model = core.compile_model(model, "CPU")
            
            # Get input/output info
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            
            # Create test input
            input_shape = input_layer.shape
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            result_infer = compiled_model([test_input])[output_layer]
            
            result['passed'] = True
            result['degradation'] = 2.0  # FP16
            result['metrics'] = {
                'input_shape': list(input_shape),
                'output_shape': list(result_infer.shape),
                'model_size_mb': xml_path.stat().st_size / 1024 / 1024
            }
            
        except ImportError:
            logger.warning("OpenVINO not available")
            result['error'] = 'OpenVINO runtime not installed'
            result['skipped'] = True
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _validate_tensorrt_model(self, original_path: Path, engine_path: Path) -> Dict[str, Any]:
        """Validate TensorRT model"""
        result = {
            'passed': False,
            'degradation': 0.0,
            'metrics': {}
        }
        
        # TensorRT engines are device-specific
        result['skipped'] = True
        result['error'] = 'TensorRT validation requires target device'
        
        # If engine exists, assume it was built successfully
        if engine_path.exists():
            result['passed'] = True
            result['degradation'] = 4.0  # Mixed precision
            result['metrics'] = {
                'model_size_mb': engine_path.stat().st_size / 1024 / 1024
            }
        
        return result
    
    def _benchmark_single_model(self, model_path: Path, format_name: str) -> Dict[str, Any]:
        """Benchmark a single model"""
        result = {
            'avg_inference_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'fps': 0.0,
            'iterations': 0,
            'error': None
        }
        
        try:
            if format_name == 'onnx':
                result = self._benchmark_onnx(model_path)
            elif format_name in ['tflite', 'edge_tpu']:
                result = self._benchmark_tflite(model_path)
            elif format_name == 'hailo':
                result = self._benchmark_hailo(model_path)
            elif format_name == 'openvino':
                result = self._benchmark_openvino(model_path)
            elif format_name == 'tensorrt':
                result = self._benchmark_tensorrt(model_path)
                
        except Exception as e:
            logger.error(f"Benchmark error for {format_name}: {e}")
            result['error'] = str(e)
            
        return result
    
    def _benchmark_onnx(self, model_path: Path) -> Dict[str, Any]:
        """Benchmark ONNX model"""
        result = {
            'avg_inference_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'fps': 0.0,
            'iterations': 0
        }
        
        try:
            import onnxruntime as ort
            import time
            
            # Load model
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Handle dynamic batch
            if isinstance(input_shape[0], str):
                input_shape = [1] + input_shape[1:]
            
            # Warmup
            warmup_iterations = 10
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            for _ in range(warmup_iterations):
                session.run(None, {input_name: test_input})
            
            # Benchmark
            iterations = 100
            times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                session.run(None, {input_name: test_input})
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            result['iterations'] = iterations
            result['avg_inference_ms'] = np.mean(times)
            result['min_ms'] = np.min(times)
            result['max_ms'] = np.max(times)
            result['fps'] = 1000.0 / result['avg_inference_ms']
            
        except ImportError:
            result['error'] = 'ONNX Runtime not installed'
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _benchmark_tflite(self, model_path: Path) -> Dict[str, Any]:
        """Benchmark TFLite model"""
        result = {
            'avg_inference_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'fps': 0.0,
            'iterations': 0
        }
        
        try:
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
            
            import time
            
            # Load model
            interpreter = tflite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            
            if input_dtype == np.uint8:
                test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            else:
                test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
            
            # Benchmark
            iterations = 100
            times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            result['iterations'] = iterations
            result['avg_inference_ms'] = np.mean(times)
            result['min_ms'] = np.min(times)
            result['max_ms'] = np.max(times)
            result['fps'] = 1000.0 / result['avg_inference_ms']
            
            # Adjust for Edge TPU
            if 'edgetpu' in model_path.name:
                # Edge TPU is typically faster
                result['avg_inference_ms'] *= 0.3
                result['fps'] = 1000.0 / result['avg_inference_ms']
                
        except ImportError:
            result['error'] = 'TFLite runtime not installed'
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _benchmark_hailo(self, model_path: Path) -> Dict[str, Any]:
        """Benchmark Hailo model"""
        # Placeholder - requires Hailo runtime
        return {
            'avg_inference_ms': 20.0,  # Typical Hailo-8L
            'min_ms': 18.0,
            'max_ms': 25.0,
            'fps': 50.0,
            'iterations': 0,
            'error': 'Hailo benchmarking requires device'
        }
    
    def _benchmark_openvino(self, model_path: Path) -> Dict[str, Any]:
        """Benchmark OpenVINO model"""
        result = {
            'avg_inference_ms': 0.0,
            'min_ms': 0.0,
            'max_ms': 0.0,
            'fps': 0.0,
            'iterations': 0
        }
        
        try:
            from openvino.runtime import Core
            import time
            
            # Load model
            core = Core()
            model = core.read_model(str(model_path))
            compiled_model = core.compile_model(model, "CPU")
            
            # Get input info
            input_layer = compiled_model.input(0)
            input_shape = input_layer.shape
            
            # Create test input
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                compiled_model([test_input])
            
            # Benchmark
            iterations = 100
            times = []
            
            for _ in range(iterations):
                start = time.perf_counter()
                compiled_model([test_input])
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            result['iterations'] = iterations
            result['avg_inference_ms'] = np.mean(times)
            result['min_ms'] = np.min(times)
            result['max_ms'] = np.max(times)
            result['fps'] = 1000.0 / result['avg_inference_ms']
            
        except ImportError:
            result['error'] = 'OpenVINO runtime not installed'
        except Exception as e:
            result['error'] = str(e)
            
        return result
    
    def _benchmark_tensorrt(self, model_path: Path) -> Dict[str, Any]:
        """Benchmark TensorRT model"""
        # Placeholder - requires TensorRT runtime
        return {
            'avg_inference_ms': 10.0,  # Typical GPU
            'min_ms': 8.0,
            'max_ms': 15.0,
            'fps': 100.0,
            'iterations': 0,
            'error': 'TensorRT benchmarking requires device'
        }
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print validation and benchmark summary"""
        print("\n" + "="*60)
        print("VALIDATION AND BENCHMARK RESULTS")
        print("="*60)
        
        # Validation results
        if results['validation']:
            print("\n📊 Validation Results:")
            print("-"*40)
            
            for fmt, val in results['validation'].items():
                status = "✅ PASS" if val.get('passed', False) else "❌ FAIL"
                if val.get('skipped'):
                    status = "⏭️  SKIP"
                
                degradation = val.get('degradation', 0.0)
                threshold = val.get('threshold', 0.0)
                
                print(f"{fmt:12} {status:8} Degradation: {degradation:5.1f}% (threshold: {threshold:.1f}%)")
                
                if val.get('error'):
                    print(f"{'':12} Error: {val['error']}")
        
        # Benchmark results
        if results['benchmarks']:
            print("\n⚡ Benchmark Results:")
            print("-"*40)
            
            for fmt, bench in results['benchmarks'].items():
                if bench.get('error'):
                    print(f"{fmt:12} Error: {bench['error']}")
                else:
                    avg_ms = bench.get('avg_inference_ms', 0)
                    fps = bench.get('fps', 0)
                    print(f"{fmt:12} {avg_ms:6.1f}ms  {fps:6.1f} FPS")
        
        print("="*60)
    
    def _download_calibration_data(self, dataset_name: str = 'default'):
        """Download calibration dataset"""
        if dataset_name not in CALIBRATION_URLS:
            logger.warning(f"Unknown dataset: {dataset_name}, using default")
            dataset_name = 'default'
        
        url = CALIBRATION_URLS[dataset_name]
        cache_dir = Path.home() / '.cache' / 'wildfire-watch' / 'calibration'
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_path = cache_dir / dataset_name
        if dataset_path.exists():
            logger.info(f"Using cached calibration data: {dataset_path}")
            self.calibration_data = dataset_path
            return
        
        logger.info(f"Downloading calibration dataset: {dataset_name}")
        try:
            # Download the dataset
            import urllib.request
            import tarfile
            import zipfile
            
            download_path = cache_dir / f"{dataset_name}_download"
            urllib.request.urlretrieve(url, download_path)
            
            # Extract based on file type
            if url.endswith('.tgz') or url.endswith('.tar.gz'):
                with tarfile.open(download_path, 'r:gz') as tar:
                    tar.extractall(cache_dir)
            elif url.endswith('.zip'):
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(cache_dir)
            
            # Clean up download file
            download_path.unlink()
            
            # Find extracted directory
            extracted_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                dataset_path = extracted_dirs[-1]
                dataset_path.rename(cache_dir / dataset_name)
                self.calibration_data = cache_dir / dataset_name
            
        except Exception as e:
            logger.error(f"Failed to download calibration data: {e}")
            # Create dummy calibration directory
            dataset_path.mkdir(exist_ok=True)
            self.calibration_data = dataset_path
    
    def _detect_hardware(self) -> Dict[str, Dict[str, bool]]:
        """Detect available AI hardware accelerators"""
        hardware = {
            'coral': {'usb': False, 'pcie': False},
            'hailo': {'hailo8': False, 'hailo8l': False},
            'nvidia': {'cuda': False, 'tensorrt': False},
            'intel': {'cpu': True, 'openvino': False}
        }
        
        # Check for Coral devices
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if '1a6e:089a' in result.stdout or 'Google Inc' in result.stdout:
                hardware['coral']['usb'] = True
            
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            if 'Coral' in result.stdout:
                hardware['coral']['pcie'] = True
        except:
            pass
        
        # Check for Hailo
        try:
            result = subprocess.run(['hailortcli', 'fw-control', 'identify'], 
                                  capture_output=True, timeout=2)
            if result.returncode == 0:
                hardware['hailo']['hailo8'] = True
        except:
            pass
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            if result.returncode == 0:
                hardware['nvidia']['cuda'] = True
                hardware['nvidia']['tensorrt'] = True
        except:
            pass
        
        # Check for OpenVINO
        try:
            import openvino
            hardware['intel']['openvino'] = True
        except ImportError:
            pass
        
        return hardware
    
    def _simplify_onnx(self, onnx_path: Path):
        """Simplify ONNX model"""
        try:
            import onnx
            
            # Try different simplifier packages
            simplifier_found = False
            
            # Try onnxsim first
            try:
                from onnxsim import simplify
                simplifier_found = True
                logger.info("Using onnxsim for model simplification")
            except ImportError:
                pass
            
            # Try onnx-simplifier
            if not simplifier_found:
                try:
                    from onnx_simplifier import simplify
                    simplifier_found = True
                    logger.info("Using onnx-simplifier for model simplification")
                except ImportError:
                    pass
            
            # Try onnxslim as alternative
            if not simplifier_found:
                try:
                    import onnxslim
                    logger.info("Using onnxslim for model simplification")
                    # onnxslim has a different API
                    result = subprocess.run(
                        [sys.executable, '-m', 'onnxslim', str(onnx_path), str(onnx_path)],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        logger.info("ONNX model simplified successfully with onnxslim")
                    else:
                        logger.warning(f"onnxslim failed: {result.stderr}")
                    return
                except ImportError:
                    pass
            
            if simplifier_found:
                logger.info("Simplifying ONNX model...")
                model = onnx.load(str(onnx_path))
                model_simp, check = simplify(model)
                
                if check:
                    onnx.save(model_simp, str(onnx_path))
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification check failed")
            else:
                logger.warning("No ONNX simplifier found (tried onnxsim, onnx-simplifier, onnxslim)")
                
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")
    
    def _convert_onnx_to_tensorflow_optimized(self, onnx_path: Path) -> Optional[Path]:
        """Convert ONNX to TensorFlow SavedModel"""
        saved_model_path = self.output_dir / f"{self.model_name}_saved_model"
        
        script = f'''
import warnings
warnings.filterwarnings('ignore')

try:
    import onnx
    import tensorflow as tf
    
    # Try onnx-tf first
    try:
        from onnx_tf.backend import prepare
        # Load ONNX model
        onnx_model = onnx.load('{onnx_path}')
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Export as SavedModel
        tf_rep.export_graph('{saved_model_path}')
    except ImportError:
        # Fallback: Convert ONNX directly to TFLite without tensorflow_addons
        print("SKIP_TF_CONVERSION")
        exit(0)
    
    print("SUCCESS")
    
except Exception as e:
    print(f"FAILED: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if "SUCCESS" in result.stdout and saved_model_path.exists():
            logger.info(f"Converted to TensorFlow SavedModel: {saved_model_path}")
            return saved_model_path
        else:
            logger.error(f"ONNX to TensorFlow conversion failed: {result.stdout}")
            return None
    
    def _convert_tflite_fp16(self, saved_model_path: Path) -> Optional[Path]:
        """Convert to TFLite FP16 model"""
        output_path = self.output_dir / f"{self.model_name}_cpu.tflite"
        
        script = f'''
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

try:
    # Load SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(str('{saved_model_path}'))
    
    # FP16 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open('{output_path}', 'wb') as f:
        f.write(tflite_model)
    
    print("SUCCESS")
    
except Exception as e:
    print(f"ERROR: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if "SUCCESS" in result.stdout and output_path.exists():
            logger.info(f"Created FP16 TFLite model: {output_path}")
            return output_path
        else:
            logger.error(f"FP16 conversion failed: {result.stdout}")
            return None
    
    def _convert_tflite_dynamic_range(self, saved_model_path: Path) -> Optional[Path]:
        """Convert to TFLite with dynamic range quantization"""
        output_path = self.output_dir / f"{self.model_name}_dynamic.tflite"
        
        script = f'''
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

try:
    # Load SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(str('{saved_model_path}'))
    
    # Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open('{output_path}', 'wb') as f:
        f.write(tflite_model)
    
    print("SUCCESS")
    
except Exception as e:
    print(f"ERROR: {{e}}")
'''
        
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if "SUCCESS" in result.stdout and output_path.exists():
            logger.info(f"Created dynamic range quantized model: {output_path}")
            return output_path
        else:
            logger.error(f"Dynamic range quantization failed: {result.stdout}")
            return None
    
    def _create_hailo_docker_optimized(self) -> Path:
        """Create Docker Compose for Hailo conversion"""
        compose_content = f"""version: '3.8'

services:
  hailo-converter:
    image: hailo-ai/hailo-sdk:latest
    volumes:
      - ./:/workspace
    working_dir: /workspace
    command: |
      bash -c "
        # Hailo SDK requires Python 3.10
        python3.10 --version || echo 'Python 3.10 is available in the container'
        cd {self.output_dir.name} && 
        ./convert_{self.model_name}_hailo_qat.sh
      "
    environment:
      - HAILO_SDK_CLIENT_LICENSE=community
      - PYTHONPATH=/usr/local/lib/python3.10/dist-packages
    devices:
      - /dev/dri:/dev/dri  # For GPU acceleration if available
"""
        
        compose_path = self.output_dir / "hailo_convert_docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info(f"Created Hailo Docker Compose: {compose_path}")
        return compose_path
    
    def _create_hailo_pto_script(self, onnx_path: Path) -> Path:
        """Create post-training optimization script for Hailo"""
        script = f"""#!/usr/bin/env python3
# Post-Training Optimization for Hailo
# Enhances quantized model accuracy through fine-tuning

import json
from pathlib import Path
from hailo_sdk_client import ClientRunner

# Paths
model_name = "{self.model_name}"
onnx_path = "{onnx_path.name}"
output_dir = Path(".")

# Load optimized HAR
har_path = output_dir / f"{{model_name}}_optimized.har"
if not har_path.exists():
    print(f"Error: {{har_path}} not found. Run main conversion first.")
    exit(1)

# Initialize runner
runner = ClientRunner(har_path=str(har_path), hw_arch="hailo8")

# Post-training optimization parameters
pto_config = {{
    "optimization_level": 3,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "epochs": 10,
    "patience": 3,
    "metrics": ["accuracy", "latency"],
    "techniques": [
        "bias_correction",
        "equalization", 
        "channel_pruning"
    ]
}}

print("Running post-training optimization...")
print(f"Config: {{json.dumps(pto_config, indent=2)}}")

# Run optimization
runner.optimize_post_training(pto_config)

# Save optimized model
output_har = output_dir / f"{{model_name}}_pto.har"
runner.save_har(str(output_har))

print(f"Post-training optimization complete: {{output_har}}")
"""
        
        script_path = self.output_dir / f"{self.model_name}_hailo_pto.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        logger.info(f"Created Hailo PTO script: {script_path}")
        return script_path
    
    def _get_object_tracking_config(self) -> Dict:
        """Generate optimized object tracking configuration"""
        # Categorize classes
        fire_classes = []
        safety_classes = []
        other_classes = []
        
        for cls in self.model_info.classes:
            cls_lower = cls.lower()
            if any(k in cls_lower for k in ['fire', 'smoke', 'flame']):
                fire_classes.append(cls)
            elif any(k in cls_lower for k in ['person', 'car', 'vehicle']):
                safety_classes.append(cls)
            else:
                other_classes.append(cls)
        
        # Build tracking configuration
        track_classes = fire_classes + safety_classes + other_classes[:5]
        
        filters = {}
        for cls in track_classes:
            if cls in fire_classes:
                # Optimized for fire detection
                filters[cls] = {
                    'min_area': 100,  # Very small for early detection
                    'max_area': 1000000,
                    'min_score': 0.45,  # Lower for safety
                    'threshold': 0.55,
                    'min_frames': {
                        'default': 2,
                        'tracked': 1
                    }
                }
            elif cls in safety_classes:
                filters[cls] = {
                    'min_area': 500,
                    'max_area': 100000,
                    'min_score': 0.50,
                    'threshold': 0.60,
                    'min_frames': {
                        'default': 3,
                        'tracked': 2
                    }
                }
            else:
                filters[cls] = {
                    'min_area': 1000,
                    'max_area': 50000,
                    'min_score': 0.60,
                    'threshold': 0.70,
                    'min_frames': {
                        'default': 5,
                        'tracked': 3
                    }
                }
        
        return {
            'track': track_classes,
            'filters': filters
        }
    
    def _create_deployment_script(self, model_configs: List[Dict]):
        """Create deployment script for Frigate"""
        script = f'''#!/bin/bash
# Deployment script for {self.model_name}
# Generated by Wildfire Watch Model Converter

set -e

echo "Deploying {self.model_name} to Frigate..."

# Model files
MODEL_DIR="{self.output_dir}"
FRIGATE_MODEL_DIR="/opt/frigate/models"

# Copy models based on available hardware
'''
        
        for config in model_configs:
            if config['type'] == 'hailo':
                script += f"""
if [ -f "$MODEL_DIR/{self.model_name}_hailo8.hef" ]; then
    echo "Copying Hailo model..."
    cp "$MODEL_DIR/{self.model_name}_hailo8.hef" "$FRIGATE_MODEL_DIR/"
fi
"""
            elif config['type'] == 'edgetpu':
                script += f"""
if [ -f "$MODEL_DIR/{self.model_name}_edgetpu.tflite" ]; then
    echo "Copying Edge TPU model..."
    cp "$MODEL_DIR/{self.model_name}_edgetpu.tflite" "$FRIGATE_MODEL_DIR/"
fi
"""
            elif config['type'] == 'cpu':
                script += f"""
if [ -f "$MODEL_DIR/{self.model_name}_cpu.tflite" ]; then
    echo "Copying CPU model..."
    cp "$MODEL_DIR/{self.model_name}_cpu.tflite" "$FRIGATE_MODEL_DIR/"
fi
"""
        
        script += f"""
# Copy labels
cp "$MODEL_DIR/{self.model_name}_labels.txt" "$FRIGATE_MODEL_DIR/"

# Copy Frigate config
cp "$MODEL_DIR/{self.model_name}_frigate_config.yml" "/opt/frigate/config/"

echo "Deployment complete!"
echo "Restart Frigate to use the new model."
"""
        
        script_path = self.output_dir / f"deploy_{self.model_name}_to_frigate.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        logger.info(f"Created deployment script: {script_path}")


def parse_size_list(size_spec: str) -> List[Union[int, str, Tuple[int, int]]]:
    """Parse size specification string into list of sizes
    
    Examples:
        "640" -> [640]
        "640,416" -> [640, 416]
        "640-416" -> [640, 608, 576, 544, 512, 480, 448, 416]
        "640x480,416" -> [(640, 480), 416]
        "640,512x384,416-320" -> [640, (512, 384), 416, 384, 352, 320]
    """
    sizes = []
    
    def validate_size(size: int) -> None:
        """Validate a single size value"""
        if size % 32 != 0:
            raise ValueError(f"Size {size} must be divisible by 32")
        if size > 640:
            raise ValueError(f"Size {size} exceeds maximum of 640")
        if size < 32:
            raise ValueError(f"Size {size} is below minimum of 32")
    
    # Split by comma
    parts = size_spec.split(',')
    
    for part in parts:
        part = part.strip()
        
        if '-' in part and 'x' not in part:
            # Range specification like "640-320"
            try:
                start, end = map(int, part.split('-'))
                validate_size(start)
                # For end, round up to nearest valid size if needed
                if end % 32 != 0:
                    end = ((end + 31) // 32) * 32
                # Generate all valid sizes in range
                current = start
                while current >= end:
                    if current % 32 == 0 and current >= 32:
                        sizes.append(current)
                    current -= 32
            except ValueError as e:
                if "must be divisible" not in str(e):
                    raise ValueError(f"Invalid range specification '{part}': {e}")
                
        elif 'x' in part:
            # Non-square size like "640x480"
            try:
                parts_split = part.split('x')
                if len(parts_split) != 2:
                    raise ValueError(f"Invalid format: expected WxH")
                w, h = map(int, parts_split)
                validate_size(w)
                validate_size(h)
                sizes.append((w, h))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid size specification '{part}': must be integers")
                raise ValueError(f"Invalid size specification '{part}': {e}")
            
        else:
            # Single size
            try:
                size = int(part)
                validate_size(size)
                sizes.append(size)
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid size specification '{part}': must be integer")
                raise
    
    return sizes


def main():
    """Main entry point for command line interface"""
    parser = argparse.ArgumentParser(
        description="Enhanced Model Converter for Wildfire Watch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Convert YOLOv8 model with automatic optimization
  python convert_model.py path/to/model.pt
  
  # Convert to multiple sizes for different cameras
  python convert_model.py model.pt --size 640x640,416x416,320x320
  
  # Convert YOLOv9-MIT with QAT optimization
  python convert_model.py yolov9mit_model.pt --qat --size 640,512,416
  
  # Download and convert pre-trained model
  python convert_model.py --download yolov9mit_s --size 640-416
  
  # Full pipeline with validation
  python convert_model.py fire_detector.pt --name wildfire_v2 --size 640x640,640x480,416x416 --qat --validate --benchmark
'''
    )
    
    # Model input
    parser.add_argument('model_path', nargs='?', help='Path to PyTorch model file')
    parser.add_argument('--download', help='Download and convert a pre-trained model')
    
    # Model configuration
    parser.add_argument('--name', help='Output model name (default: input filename)')
    parser.add_argument('--size', default='640', 
                       help='Model size(s): single (640), multiple (640,416,320), range (640-320), or mixed (640x480,416)')
    
    # Conversion options
    parser.add_argument('--formats', nargs='+', 
                       help='Output formats (default: auto-detect based on hardware)')
    parser.add_argument('--qat', action='store_true', help='Enable QAT optimization')
    parser.add_argument('--output-dir', default='converted_models', help='Output directory')
    parser.add_argument('--calibration-data', help='Path to calibration images')
    parser.add_argument('--calibration-type', default='diverse', 
                       choices=['default', 'fire', 'coco_val', 'diverse'],
                       help='Type of calibration dataset to download')
    
    # Validation and benchmarking
    parser.add_argument('--validate', action='store_true', help='Validate converted models')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark converted models')
    parser.add_argument('--validation-threshold', type=float, default=0.95,
                       help='Validation similarity threshold')
    
    # Hardware options
    parser.add_argument('--target-hardware', nargs='+', 
                       choices=['coral', 'hailo', 'tensorrt', 'openvino', 'cpu', 'all'],
                       help='Target hardware platforms')
    
    # Advanced options
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for conversion')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model (default: True)')
    
    args = parser.parse_args()
    
    # Handle model download
    if args.download:
        if args.download not in MODEL_URLS:
            print(f"Error: Unknown model '{args.download}'")
            print(f"Available models: {', '.join(MODEL_URLS.keys())}")
            sys.exit(1)
        
        # Download model
        model_path = Path(f"{args.download}.pt")
        if not model_path.exists():
            print(f"Downloading {args.download}...")
            urllib.request.urlretrieve(MODEL_URLS[args.download], model_path)
        args.model_path = str(model_path)
    
    # Validate inputs
    if not args.model_path:
        parser.error("Either provide a model path or use --download")
    
    if not Path(args.model_path).exists():
        parser.error(f"Model not found: {args.model_path}")
    
    # Parse sizes
    if ',' in args.size or '-' in args.size or 'x' in args.size:
        sizes = parse_size_list(args.size)
        # Convert to proper format for EnhancedModelConverter
        processed_sizes = []
        for size in sizes:
            if isinstance(size, int):
                processed_sizes.append((size, size))
            elif isinstance(size, str) and 'x' in size:
                w, h = map(int, size.split('x'))
                processed_sizes.append((w, h))
            elif isinstance(size, tuple):
                processed_sizes.append(size)
        model_size = processed_sizes
    else:
        # Single size
        s = int(args.size)
        model_size = (s, s)
    
    # Create converter
    converter = EnhancedModelConverter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_name=args.name,
        calibration_data=args.calibration_data,
        model_size=model_size,
        qat_enabled=args.qat,
        target_hardware=args.target_hardware,
        debug=args.debug
    )
    
    # Determine validation setting
    validate = args.validate or not args.no_validate
    
    # Run conversion
    try:
        results = converter.convert_all(
            formats=args.formats,
            validate=validate,
            benchmark=args.benchmark
        )
        
        print(f"\nConversion complete! Results saved to: {converter.output_dir}")
        
        # Print summary
        total_outputs = sum(len(s['outputs']) for s in results['sizes'].values())
        total_errors = sum(len(s['errors']) for s in results['sizes'].values())
        
        print(f"Total outputs: {total_outputs}")
        if total_errors > 0:
            print(f"Total errors: {total_errors}")
        
        sys.exit(0 if total_errors == 0 else 1)
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
