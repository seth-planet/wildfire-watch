#!/usr/bin/env python3
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
        self.model_config = model_config or {}
        
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
            logger.info(f"YOLOv8 detector initialized with model: {model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")
            
            # Get quantization parameters if model is quantized
            output_detail = self.output_details[0]
            if output_detail['quantization'] and len(output_detail['quantization']) >= 2:
                self.output_scale = output_detail['quantization'][0]
                self.output_zero_point = output_detail['quantization'][1]
                self.is_quantized = output_detail['dtype'] == np.uint8
                logger.info(f"Model is quantized: scale={self.output_scale}, zero={self.output_zero_point}")
            else:
                self.output_scale = 1.0
                self.output_zero_point = 0
                self.is_quantized = False
                
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 detector: {e}")
            raise
        
        # Detection parameters
        self.conf_threshold = self.model_config.get("conf_threshold", 0.25)
        self.iou_threshold = self.model_config.get("iou_threshold", 0.45)
        self.max_detections = self.model_config.get("max_detections", 100)
        self.num_classes = 32
        
        # Class names for fire detection
        self.class_names = self.model_config.get("labels", [
            "fire", "smoke", "person", "vehicle"
        ])
        
        logger.info(f"Detection params: conf={self.conf_threshold}, iou={self.iou_threshold}, max={self.max_detections}")
    
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
            logger.error(f"Detection failed: {e}")
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
    model_config = {
        "path": sys.argv[1] if len(sys.argv) > 1 else "/models/yolov8.tflite",
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 100
    }
    
    detector = YOLOv8Detector(model_config=model_config)
    
    # Create dummy input
    dummy_input = np.random.randint(0, 255, size=(1, 640, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect_raw(dummy_input)
    
    print(f"Detections shape: {detections.shape}")
    print(f"Number of detections: {len(detections)}")
    
    if len(detections) > 0:
        print("\nFirst detection:")
        print(f"  Box: [{detections[0][0]:.3f}, {detections[0][1]:.3f}, {detections[0][2]:.3f}, {detections[0][3]:.3f}]")
        print(f"  Score: {detections[0][4]:.3f}")
        print(f"  Class: {int(detections[0][5])}")
