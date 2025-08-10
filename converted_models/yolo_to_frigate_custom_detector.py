#!/usr/bin/env python3
"""
Create a custom Frigate detector for YOLOv8 models with [1, 36, 8400] output format.

This is the most practical solution - a custom detector that Frigate can use directly.
"""

import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yolov8_frigate_detector(model_path: str, output_dir: str, num_classes: int = 32):
    """
    Create a custom Frigate detector for YOLOv8 models.
    
    Args:
        model_path: Path to the YOLOv8 TFLite model
        output_dir: Directory to save the detector and instructions
        num_classes: Number of classes in the model (default 32 for fire detection)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detector code that handles YOLOv8 output format
    detector_code = f'''#!/usr/bin/env python3
"""
Custom YOLOv8 detector for Frigate NVR.
Handles YOLOv8 models with [1, 36, 8400] output format.

Place this file in Frigate's detector directory:
/config/custom_detectors/yolov8_frigate.py
"""

import numpy as np
import logging
from typing import List, Tuple

# Try both import paths for compatibility
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Import Frigate's detection API
try:
    from frigate.detectors.detection_api import DetectionApi
except ImportError:
    # Fallback for testing outside Frigate
    class DetectionApi:
        type_key = "base"
        def detect_raw(self, tensor_input): pass

logger = logging.getLogger(__name__)


class YOLOv8Detector(DetectionApi):
    """
    Custom detector for YOLOv8 models with transposed output format.
    
    Handles models that output [1, 36, 8400] where:
    - 36 = 4 bbox coords + 32 classes
    - 8400 = number of predictions
    """
    
    type_key = "yolov8"
    
    def __init__(self, det_device=None, model_config=None):
        """Initialize the YOLOv8 detector."""
        self.model_config = model_config or {{}}
        
        # Get model path
        model_path = self.model_config.get("path", "/models/yolov8.tflite")
        
        # Initialize interpreter
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Log model info
            logger.info(f"YOLOv8 detector initialized with model: {{model_path}}")
            logger.info(f"Input shape: {{self.input_details[0]['shape']}}")
            logger.info(f"Output shape: {{self.output_details[0]['shape']}}")
            
            # Get quantization parameters if model is quantized
            output_detail = self.output_details[0]
            if output_detail['quantization'] and len(output_detail['quantization']) >= 2:
                self.output_scale = output_detail['quantization'][0]
                self.output_zero_point = output_detail['quantization'][1]
                self.is_quantized = output_detail['dtype'] == np.uint8
                logger.info(f"Model is quantized: scale={{self.output_scale}}, zero={{self.output_zero_point}}")
            else:
                self.output_scale = 1.0
                self.output_zero_point = 0
                self.is_quantized = False
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 detector: {{e}}")
            raise
        
        # Detection parameters
        self.conf_threshold = self.model_config.get("conf_threshold", 0.25)
        self.iou_threshold = self.model_config.get("iou_threshold", 0.45)
        self.max_detections = self.model_config.get("max_detections", 100)
        self.num_classes = {num_classes}
        
        # Class names for fire detection
        self.class_names = self.model_config.get("labels", [
            "fire", "smoke", "person", "vehicle"
        ])
        
        logger.info(f"Detection params: conf={{self.conf_threshold}}, iou={{self.iou_threshold}}, max={{self.max_detections}}")
    
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """
        Run detection on input tensor and return results in Frigate format.
        
        Args:
            tensor_input: Input image tensor [1, height, width, 3] as uint8
            
        Returns:
            Detections array with shape [N, 6] containing:
            [ymin, xmin, ymax, xmax, score, class_id]
        """
        try:
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]["index"], tensor_input)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            raw_output = self.interpreter.get_tensor(self.output_details[0]["index"])
            
            # Dequantize if needed
            if self.is_quantized:
                raw_output = (raw_output.astype(np.float32) - self.output_zero_point) * self.output_scale
            
            # Process YOLOv8 output
            detections = self._process_yolov8_output(raw_output)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {{e}}")
            return np.zeros((0, 6), dtype=np.float32)
    
    def _process_yolov8_output(self, output: np.ndarray) -> np.ndarray:
        """
        Process YOLOv8 output format [1, 36, 8400] to Frigate format.
        
        Args:
            output: Raw model output
            
        Returns:
            Processed detections
        """
        # Remove batch dimension if present
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]  # Now [36, 8400]
        
        # Transpose to [8400, 36] for easier processing
        predictions = output.T
        
        # Split into components
        # First 4 values are bbox (x_center, y_center, width, height)
        boxes = predictions[:, :4]
        # Remaining values are class scores
        class_scores = predictions[:, 4:4+self.num_classes]
        
        # Get best class for each prediction
        best_class_ids = np.argmax(class_scores, axis=1)
        best_class_scores = np.max(class_scores, axis=1)
        
        # Filter by confidence threshold
        valid_mask = best_class_scores >= self.conf_threshold
        
        if not np.any(valid_mask):
            return np.zeros((0, 6), dtype=np.float32)
        
        # Get valid predictions
        valid_boxes = boxes[valid_mask]
        valid_scores = best_class_scores[valid_mask]
        valid_classes = best_class_ids[valid_mask]
        
        # Convert YOLO format to corner format
        detections = []
        for i in range(len(valid_boxes)):
            x_center, y_center, width, height = valid_boxes[i]
            
            # Convert from center format to corner format
            # Ensure coordinates are normalized (0-1)
            if x_center > 1.0 or y_center > 1.0:
                # Coordinates are in pixel space, normalize by image size
                # Assuming 640x640 input
                x_center /= 640.0
                y_center /= 640.0
                width /= 640.0
                height /= 640.0
            
            # Calculate corners
            x1 = max(0.0, x_center - width / 2.0)
            y1 = max(0.0, y_center - height / 2.0)
            x2 = min(1.0, x_center + width / 2.0)
            y2 = min(1.0, y_center + height / 2.0)
            
            # Frigate expects: [ymin, xmin, ymax, xmax, score, class_id]
            detections.append([
                y1, x1, y2, x2,
                float(valid_scores[i]),
                float(valid_classes[i])
            ])
        
        # Apply NMS
        if len(detections) > 0:
            detections = self._apply_nms(detections)
        
        # Limit to max detections
        detections = detections[:self.max_detections]
        
        return np.array(detections, dtype=np.float32)
    
    def _apply_nms(self, detections: List[List[float]]) -> List[List[float]]:
        """
        Apply Non-Maximum Suppression to reduce overlapping detections.
        
        Args:
            detections: List of detections [ymin, xmin, ymax, xmax, score, class]
            
        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return detections
        
        # Sort by score (descending)
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        while detections and len(keep) < self.max_detections:
            # Take the detection with highest score
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections with high IoU
            remaining = []
            for det in detections:
                # Only apply NMS to same class
                if det[5] != best[5]:
                    remaining.append(det)
                    continue
                
                iou = self._calculate_iou(best[:4], det[:4])
                if iou < self.iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First box [ymin, xmin, ymax, xmax]
            box2: Second box [ymin, xmin, ymax, xmax]
            
        Returns:
            IoU value between 0 and 1
        """
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection_area = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0


# For testing outside of Frigate
if __name__ == "__main__":
    import sys
    
    # Test the detector
    print("Testing YOLOv8 detector...")
    
    # Create detector instance
    model_config = {{
        "path": sys.argv[1] if len(sys.argv) > 1 else "/models/yolov8.tflite",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 100
    }}
    
    detector = YOLOv8Detector(model_config=model_config)
    
    # Create dummy input
    dummy_input = np.random.randint(0, 255, size=(1, 640, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect_raw(dummy_input)
    
    print(f"Detections shape: {{detections.shape}}")
    print(f"Number of detections: {{len(detections)}}")
    
    if len(detections) > 0:
        print("\\nFirst detection:")
        print(f"  Box: [{{detections[0][0]:.3f}}, {{detections[0][1]:.3f}}, {{detections[0][2]:.3f}}, {{detections[0][3]:.3f}}]")
        print(f"  Score: {{detections[0][4]:.3f}}")
        print(f"  Class: {{int(detections[0][5])}}")
'''
    
    # Save the detector
    detector_path = output_dir / "yolov8_frigate.py"
    with open(detector_path, 'w') as f:
        f.write(detector_code)
    
    logger.info(f"✓ Created YOLOv8 Frigate detector: {detector_path}")
    
    # Create Frigate configuration example
    config_example = f'''# Frigate Configuration for YOLOv8

detectors:
  coral:
    type: yolov8
    device: usb
    model:
      path: /models/{Path(model_path).name}
      conf_threshold: 0.25
      iou_threshold: 0.45
      max_detections: 100
      labels:
        - fire
        - smoke
        - person
        - vehicle

# Model configuration
model:
  width: 640
  height: 640
  input_tensor: nhwc
  input_pixel_format: rgb
  # No need for output tensor meta - handled by custom detector

# Object tracking
objects:
  track:
    - fire
    - smoke
    - person
  filters:
    fire:
      min_score: 0.3
      threshold: 0.35
    smoke:
      min_score: 0.25
      threshold: 0.3

# Camera configuration example
cameras:
  front_yard:
    ffmpeg:
      inputs:
        - path: rtsp://user:pass@camera_ip/stream
          roles:
            - detect
    detect:
      width: 640
      height: 640
      fps: 5
'''
    
    config_path = output_dir / "frigate_config_example.yml"
    with open(config_path, 'w') as f:
        f.write(config_example)
    
    logger.info(f"✓ Created Frigate config example: {config_path}")
    
    # Create setup instructions
    instructions = f'''# YOLOv8 Frigate Setup Instructions

This custom detector enables YOLOv8 models with [1, 36, 8400] output format to work with Frigate.

## Installation Steps

### 1. Copy the Detector

Copy `yolov8_frigate.py` to your Frigate configuration directory:

```bash
# If using Docker
docker cp {detector_path} frigate:/config/custom_detectors/yolov8_frigate.py

# If using direct installation
cp {detector_path} /path/to/frigate/config/custom_detectors/
```

### 2. Copy Your Model

Copy your YOLOv8 TFLite model:

```bash
# If using Docker
docker cp {model_path} frigate:/models/{Path(model_path).name}

# If using direct installation  
cp {model_path} /path/to/frigate/models/
```

### 3. Update Frigate Configuration

Edit your `frigate.yml` to use the custom detector. See `frigate_config_example.yml` for a complete example.

Key configuration:
```yaml
detectors:
  coral:
    type: yolov8  # This matches the type_key in our detector
    device: usb
    model:
      path: /models/{Path(model_path).name}
```

### 4. Restart Frigate

```bash
docker restart frigate
# or
systemctl restart frigate
```

### 5. Verify Operation

Check Frigate logs:
```bash
docker logs frigate | grep YOLOv8
```

You should see:
- "YOLOv8 detector initialized with model: /models/..."
- "Input shape: [1, 640, 640, 3]"
- "Output shape: [1, 36, 8400]"

## Testing the Detector

You can test the detector standalone:

```bash
# Test with your model
python3 yolov8_frigate.py {model_path}
```

## Troubleshooting

1. **Import Error**: If `tflite_runtime` is not found, the detector will fall back to `tensorflow.lite`.

2. **Model Not Found**: Ensure the model path in frigate.yml matches the actual location in the container.

3. **No Detections**: Try lowering the confidence threshold in the model config.

4. **Performance Issues**: 
   - Reduce detection fps in camera config
   - Increase detection interval
   - Use hardware acceleration (Coral TPU)

## Model Requirements

Your YOLOv8 model must:
- Accept input shape [1, 640, 640, 3] as UINT8
- Output shape [1, 36, 8400] where 36 = 4 bbox + 32 classes
- Be in TFLite format (quantized or float32)

## Fire Detection Optimization

For fire detection, consider these settings:
- Lower confidence threshold (0.2-0.3) for early detection
- Higher max_detections to catch multiple fire sources
- Adjust IoU threshold based on fire/smoke overlap patterns
'''
    
    instructions_path = output_dir / "SETUP_INSTRUCTIONS.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    logger.info(f"✓ Created setup instructions: {instructions_path}")
    
    # Create a test script
    test_script = f'''#!/usr/bin/env python3
"""Test the YOLOv8 detector with a real image."""

import sys
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, '.')

# Import the detector
from yolov8_frigate import YOLOv8Detector

def test_detector(model_path, image_path=None):
    """Test the detector with an image."""
    
    # Create detector
    model_config = {{
        "path": model_path,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 100
    }}
    
    print(f"Initializing detector with model: {{model_path}}")
    detector = YOLOv8Detector(model_config=model_config)
    
    # Load or create test image
    if image_path:
        print(f"Loading test image: {{image_path}}")
        img = Image.open(image_path).convert('RGB')
        img = img.resize((640, 640))
        input_tensor = np.array(img, dtype=np.uint8)
        input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dimension
    else:
        print("Using random test image")
        input_tensor = np.random.randint(0, 255, size=(1, 640, 640, 3), dtype=np.uint8)
    
    # Run detection
    print("Running detection...")
    detections = detector.detect_raw(input_tensor)
    
    print(f"\\nResults:")
    print(f"  Total detections: {{len(detections)}}")
    
    if len(detections) > 0:
        print("\\nDetections:")
        for i, det in enumerate(detections):
            y1, x1, y2, x2, score, class_id = det
            class_name = detector.class_names[int(class_id)] if int(class_id) < len(detector.class_names) else f"class{{int(class_id)}}"
            print(f"  {{i+1}}. {{class_name}}: score={{score:.3f}}, box=[{{y1:.3f}}, {{x1:.3f}}, {{y2:.3f}}, {{x2:.3f}}]")
    else:
        print("  No detections found")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <model_path> [image_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_detector(model_path, image_path)
'''
    
    test_path = output_dir / "test_detector.py"
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    logger.info(f"✓ Created test script: {test_path}")
    
    # Make test script executable
    test_path.chmod(0o755)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create YOLOv8 custom detector for Frigate"
    )
    parser.add_argument('--model', 
                       default='/home/seth/wildfire-watch/converted_models/frigate_models/yolo8l_fire_640x640_frigate.tflite',
                       help='Path to YOLOv8 TFLite model')
    parser.add_argument('--output-dir',
                       default='/home/seth/wildfire-watch/converted_models/frigate_yolo_solution',
                       help='Output directory for detector and instructions')
    parser.add_argument('--num-classes', type=int, default=32,
                       help='Number of classes in the model')
    
    args = parser.parse_args()
    
    # Create the custom detector
    output_dir = create_yolov8_frigate_detector(args.model, args.output_dir, args.num_classes)
    
    logger.info("\n✅ YOLOv8 Frigate detector created successfully!")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nFiles created:")
    logger.info("  - yolov8_frigate.py: Custom detector for Frigate")
    logger.info("  - frigate_config_example.yml: Example Frigate configuration")
    logger.info("  - SETUP_INSTRUCTIONS.md: Detailed setup guide")
    logger.info("  - test_detector.py: Test script for verification")
    logger.info("\nNext steps:")
    logger.info("1. Read SETUP_INSTRUCTIONS.md for installation guide")
    logger.info("2. Test the detector: python3 test_detector.py <model_path>")
    logger.info("3. Copy files to Frigate and restart")


if __name__ == "__main__":
    main()