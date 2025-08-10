#!/usr/bin/env python3
"""
Simplified YOLO EdgeTPU detector for Frigate stable.
Works around Frigate's initialization patterns.
"""

import logging
import numpy as np
import os

try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.python.interpreter import load_delegate

from frigate.detectors.detection_api import DetectionApi

logger = logging.getLogger(__name__)


class YoloEdgeTpu(DetectionApi):
    """
    Simplified YOLO detector that works with Frigate stable.
    Lazy loads model on first detection.
    """
    
    type_key = "yolo_edgetpu"
    
    def __init__(self, det_device=None, model_config=None):
        """Initialize detector - model loaded lazily."""
        self.det_device = det_device
        self.model_config = model_config or {}
        
        # Detection parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_detections = 100
        
        # Model state
        self.interpreter = None
        self.model_loaded = False
        
        logger.info(f"YoloEdgeTpu initialized with device: {det_device}")
    
    def _lazy_load_model(self):
        """Load model on first use."""
        if self.model_loaded:
            return
            
        # Find model path - try multiple locations
        model_paths = [
            "/config/model/yolo8l_fire_640x640_frigate_edgetpu.tflite",
            "/config/model/yolo8l_fire_640x640_frigate.tflite",
            "/models/yolo8l_fire_640x640_frigate_edgetpu.tflite",
            "/models/yolo8l_fire_640x640_frigate.tflite",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        if not model_path:
            logger.error(f"No YOLO model found in: {model_paths}")
            return
            
        logger.info(f"Loading YOLO model from: {model_path}")
        
        # Load EdgeTPU delegate
        delegate = None
        if self.det_device:
            try:
                device_config = f":{self.det_device}" if isinstance(self.det_device, int) else self.det_device
                delegate = load_delegate("libedgetpu.so.1.0", {"device": device_config})
                logger.info("EdgeTPU delegate loaded")
            except Exception as e:
                logger.warning(f"EdgeTPU not available: {e}")
        
        # Create interpreter
        try:
            if delegate:
                self.interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate]
                )
            else:
                self.interpreter = Interpreter(model_path=model_path)
                
            self.interpreter.allocate_tensors()
            
            # Get details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Model dimensions
            self.model_height = self.input_details[0]['shape'][1]
            self.model_width = self.input_details[0]['shape'][2]
            
            # Check quantization
            out_detail = self.output_details[0]
            self.is_quantized = out_detail['dtype'] == np.uint8
            if self.is_quantized and out_detail.get('quantization'):
                self.output_scale = out_detail['quantization'][0]
                self.output_zero = out_detail['quantization'][1] if len(out_detail['quantization']) > 1 else 0
            else:
                self.output_scale = 1.0
                self.output_zero = 0
                
            self.model_loaded = True
            logger.info(f"Model loaded successfully - Input: {self.input_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def detect_raw(self, tensor_input):
        """Run detection with lazy model loading."""
        # Ensure model is loaded
        self._lazy_load_model()
        
        if not self.model_loaded or self.interpreter is None:
            return np.zeros((0, 6), dtype=np.float32)
        
        # Add batch dimension if needed
        if tensor_input.ndim == 3:
            tensor_input = np.expand_dims(tensor_input, axis=0)
            
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor_input)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        # Dequantize
        if self.is_quantized:
            output = (output.astype(np.float32) - self.output_zero) * self.output_scale
        
        # Process YOLO output
        return self._process_output(output)
    
    def _process_output(self, output):
        """Convert YOLO [1, 36, 8400] to Frigate format."""
        # Remove batch
        if output.ndim == 3:
            output = output[0]
            
        # Transpose to [8400, 36]
        predictions = output.T
        
        # Extract components
        boxes = predictions[:, :4]
        scores = predictions[:, 4:36]
        
        # Get best class
        best_classes = np.argmax(scores, axis=1)
        best_scores = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = best_scores >= self.conf_threshold
        if not np.any(mask):
            return np.zeros((0, 6), dtype=np.float32)
            
        # Get valid detections
        valid_boxes = boxes[mask]
        valid_scores = best_scores[mask]
        valid_classes = best_classes[mask]
        
        # Convert to Frigate format
        detections = []
        for i in range(len(valid_boxes)):
            x, y, w, h = valid_boxes[i]
            
            # Normalize if needed
            if x > 1.0:
                x /= self.model_width
                y /= self.model_height
                w /= self.model_width
                h /= self.model_height
            
            # Convert to corners
            x1 = max(0, x - w/2)
            y1 = max(0, y - h/2)
            x2 = min(1, x + w/2)
            y2 = min(1, y + h/2)
            
            # [ymin, xmin, ymax, xmax, score, class]
            detections.append([y1, x1, y2, x2, valid_scores[i], valid_classes[i]])
        
        # Simple NMS
        if detections:
            detections = sorted(detections, key=lambda x: x[4], reverse=True)
            detections = detections[:self.max_detections]
            
        return np.array(detections, dtype=np.float32)