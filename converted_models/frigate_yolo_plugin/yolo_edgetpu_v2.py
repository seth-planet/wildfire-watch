#!/usr/bin/env python3
"""
YOLO EdgeTPU detector plugin for Frigate NVR - Version 2.
This version handles model loading during initialization.
"""

import logging
import numpy as np
from typing import Literal, Union, Optional
import os

try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.python.interpreter import load_delegate

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelConfig

logger = logging.getLogger(__name__)


class YoloEdgeTpuDetectorConfig(BaseDetectorConfig):
    """Configuration for YOLO EdgeTPU detector."""
    
    type: Literal["yolo_edgetpu"]
    device: Optional[str] = None
    num_threads: int = 3


class YoloEdgeTpu(DetectionApi):
    """
    YOLO detector optimized for EdgeTPU - Version 2.
    This version loads the model during initialization based on Frigate's pattern.
    """
    
    type_key = "yolo_edgetpu"
    
    def __init__(self, detector_config: YoloEdgeTpuDetectorConfig, model_config: ModelConfig):
        """Initialize YOLO EdgeTPU detector with model."""
        # Store configs
        self.detector_config = detector_config
        self.model_config = model_config
        
        # Detection parameters from model config
        self.conf_threshold = 0.25  # Default, could be from config
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.num_threads = detector_config.num_threads
        
        # Model dimensions from config
        self.model_input_height = model_config.height
        self.model_input_width = model_config.width
        
        # Initialize model immediately
        self._load_model(model_config.path)
    
    def _load_model(self, model_path: str):
        """Load the YOLO TFLite model."""
        logger.info(f"Loading YOLO model from {model_path}")
        
        # Prepare EdgeTPU delegate
        delegate = None
        device_config = ""
        if self.detector_config.device:
            device_config = f":{self.detector_config.device}"
            
        try:
            logger.info(f"Attempting to load EdgeTPU delegate with device {device_config}")
            delegate = load_delegate("libedgetpu.so.1.0", {"device": device_config})
            logger.info("EdgeTPU delegate loaded successfully")
        except (ValueError, OSError) as e:
            logger.warning(f"EdgeTPU delegate not available: {e}, using CPU fallback")
        
        # Create interpreter
        if delegate:
            self.interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate],
                num_threads=self.num_threads
            )
        else:
            self.interpreter = Interpreter(
                model_path=model_path,
                num_threads=self.num_threads
            )
            
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model info
        logger.info(f"Model loaded - Input: {self.input_details[0]['shape']}, "
                   f"Output: {self.output_details[0]['shape']}")
        
        # Check for quantization
        output_detail = self.output_details[0]
        self.is_quantized = output_detail['dtype'] == np.uint8
        if self.is_quantized and output_detail.get('quantization'):
            quant_params = output_detail['quantization']
            self.output_scale = quant_params[0] if len(quant_params) > 0 else 1.0
            self.output_zero_point = quant_params[1] if len(quant_params) > 1 else 0
            logger.info(f"Model is quantized: scale={self.output_scale}, "
                       f"zero={self.output_zero_point}")
        else:
            self.output_scale = 1.0
            self.output_zero_point = 0
            self.is_quantized = False
    
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """
        Run YOLO detection and return results in Frigate format.
        
        Args:
            tensor_input: Input image [height, width, 3] as uint8
            
        Returns:
            Detections [N, 6] as [ymin, xmin, ymax, xmax, score, class_id]
        """
        # Prepare input - add batch dimension if needed
        if tensor_input.ndim == 3:
            tensor_input = np.expand_dims(tensor_input, axis=0)
            
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor_input)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        raw_output = self.interpreter.get_tensor(self.output_details[0]["index"])
        
        # Dequantize if needed
        if self.is_quantized:
            raw_output = (raw_output.astype(np.float32) - self.output_zero_point) * self.output_scale
        
        # Process YOLO output
        return self._process_yolo_output(raw_output)
    
    def _process_yolo_output(self, output: np.ndarray) -> np.ndarray:
        """Convert YOLO output to Frigate format with NMS."""
        # Remove batch dimension
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]  # Now [36, 8400]
        
        # Transpose to [8400, 36]
        predictions = output.T
        
        # Extract components
        boxes = predictions[:, :4]  # [x_center, y_center, width, height]
        class_scores = predictions[:, 4:36]  # 32 classes
        
        # Get best class and score for each prediction
        best_class_ids = np.argmax(class_scores, axis=1)
        best_scores = np.max(class_scores, axis=1)
        
        # Filter by confidence
        valid_mask = best_scores >= self.conf_threshold
        if not np.any(valid_mask):
            return np.zeros((0, 6), dtype=np.float32)
        
        # Get valid predictions
        valid_boxes = boxes[valid_mask]
        valid_scores = best_scores[valid_mask]
        valid_classes = best_class_ids[valid_mask]
        
        # Convert to corner format and normalize
        detections = []
        for i in range(len(valid_boxes)):
            x_center, y_center, width, height = valid_boxes[i]
            
            # Normalize if in pixel coordinates
            if x_center > 1.0:
                x_center /= self.model_input_width
                y_center /= self.model_input_height
                width /= self.model_input_width
                height /= self.model_input_height
            
            # Convert to corners
            x1 = max(0.0, x_center - width / 2.0)
            y1 = max(0.0, y_center - height / 2.0)
            x2 = min(1.0, x_center + width / 2.0)
            y2 = min(1.0, y_center + height / 2.0)
            
            # Frigate format: [ymin, xmin, ymax, xmax, score, class]
            detections.append([y1, x1, y2, x2, 
                             float(valid_scores[i]), 
                             float(valid_classes[i])])
        
        # Apply NMS
        if detections:
            detections = self._nms(detections)
            
        # Limit detections
        detections = detections[:self.max_detections]
        
        return np.array(detections, dtype=np.float32)
    
    def _nms(self, detections: list) -> list:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return detections
        
        # Sort by score
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        
        keep = []
        while detections and len(keep) < self.max_detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove high IoU detections of same class
            remaining = []
            for det in detections:
                # Only apply NMS to same class
                if det[5] != best[5]:
                    remaining.append(det)
                    continue
                    
                # Calculate IoU
                iou = self._calculate_iou(best[:4], det[:4])
                if iou < self.iou_threshold:
                    remaining.append(det)
                    
            detections = remaining
            
        return keep
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """Calculate IoU between two boxes [ymin, xmin, ymax, xmax]."""
        y1_min, x1_min, y1_max, x1_max = box1
        y2_min, x2_min, y2_max, x2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
            
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0