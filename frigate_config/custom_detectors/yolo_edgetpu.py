#!/usr/bin/env python3
"""
Custom YOLO EdgeTPU detector for Frigate.

This detector handles YOLO output format and converts it to Frigate's expected format.
Place this file in Frigate's custom detector directory.
"""

import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from frigate.detectors.detection_api import DetectionApi
import logging

logger = logging.getLogger(__name__)


class YoloEdgeTpu(DetectionApi):
    """Custom YOLO detector for EdgeTPU that handles YOLO output format."""
    
    type_key = "yolo_edgetpu"
    
    def __init__(self, det_device=None, model_config=None):
        self.model_config = model_config or {}
        device_config = {"device": det_device} if det_device else {}
        
        # Initialize EdgeTPU
        edge_tpu_delegate = edgetpu.load_edgetpu_delegate(device_config)
        
        # Load the model
        model_path = self.model_config.get("path", "/edgetpu_model.tflite")
        self.interpreter = edgetpu.make_interpreter(model_path, delegate=edge_tpu_delegate)
        self.interpreter.allocate_tensors()
        
        self.tensor_input_details = self.interpreter.get_input_details()
        self.tensor_output_details = self.interpreter.get_output_details()
        
        logger.info(f"YOLO EdgeTPU detector initialized with device: {det_device}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Input shape: {self.tensor_input_details[0]['shape']}")
        logger.info(f"Output shape: {self.tensor_output_details[0]['shape']}")
        
        # Detection parameters
        self.conf_threshold = self.model_config.get("conf_threshold", 0.25)
        self.iou_threshold = self.model_config.get("iou_threshold", 0.45)
        self.max_detections = self.model_config.get("max_detections", 100)
        self.input_size = self.model_config.get("width", 640)
        
        # Class mapping for fire detection
        self.class_names = ["fire", "smoke", "person", "vehicle"]
    
    def detect_raw(self, tensor_input):
        """Run detection and convert YOLO output to Frigate format."""
        # Set input tensor
        self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get raw YOLO output
        raw_output = self.interpreter.get_tensor(self.tensor_output_details[0]["index"])
        
        # Parse YOLO output to Frigate format
        detections = self._parse_yolo_output(raw_output)
        
        return detections
    
    def _parse_yolo_output(self, raw_output):
        """
        Convert YOLO output to Frigate format.
        
        YOLO output: [1, num_predictions, 4 + 1 + num_classes] or similar
        Frigate expects: [x, y, w, h, confidence, class_id, class_prob]
        """
        detections = []
        
        # Handle different YOLO output formats
        if len(raw_output.shape) == 3:
            # Format: [1, num_predictions, features]
            predictions = raw_output[0]
        elif len(raw_output.shape) == 2:
            # Format: [num_predictions, features]
            predictions = raw_output
        else:
            # Try to reshape based on expected format
            # YOLO typically has 4 (bbox) + 1 (objectness) + num_classes
            num_classes = len(self.class_names)
            features_per_pred = 4 + 1 + num_classes
            
            if raw_output.shape[-1] % features_per_pred == 0:
                num_predictions = raw_output.shape[-1] // features_per_pred
                predictions = raw_output.reshape(-1, num_predictions, features_per_pred)[0]
            else:
                # Fallback: assume it's already in the right shape
                predictions = raw_output.reshape(-1, raw_output.shape[-1])
        
        # Process each prediction
        for pred in predictions:
            # YOLO format typically: [x_center, y_center, width, height, objectness, ...class_scores]
            if len(pred) >= 5:
                x_center = float(pred[0])
                y_center = float(pred[1])
                width = float(pred[2])
                height = float(pred[3])
                objectness = float(pred[4])
                
                # Get class scores
                if len(pred) > 5:
                    class_scores = pred[5:5+len(self.class_names)]
                    class_id = np.argmax(class_scores)
                    class_score = float(class_scores[class_id])
                else:
                    # No class scores, use objectness
                    class_id = 0
                    class_score = objectness
                
                # Calculate confidence
                confidence = objectness * class_score
                
                # Apply confidence threshold
                if confidence >= self.conf_threshold:
                    # Convert to corner format if needed
                    # Check if coordinates are normalized (0-1) or pixel values
                    if x_center > 1.0 or y_center > 1.0:
                        # Pixel coordinates, normalize them
                        x_center /= self.input_size
                        y_center /= self.input_size
                        width /= self.input_size
                        height /= self.input_size
                    
                    # Convert center format to corner format
                    x_min = x_center - width / 2.0
                    y_min = y_center - height / 2.0
                    
                    # Ensure coordinates are within bounds
                    x_min = max(0.0, min(1.0, x_min))
                    y_min = max(0.0, min(1.0, y_min))
                    width = max(0.0, min(1.0 - x_min, width))
                    height = max(0.0, min(1.0 - y_min, height))
                    
                    # Frigate expects: [x, y, w, h, confidence, class_id, class_prob]
                    detections.append([
                        x_min,
                        y_min,
                        width,
                        height,
                        confidence,
                        float(class_id),
                        class_score
                    ])
        
        # Apply NMS if we have too many detections
        if len(detections) > self.max_detections:
            detections = self._apply_nms(detections)
        
        return np.array(detections, dtype=np.float32)
    
    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression to reduce overlapping detections."""
        if not detections:
            return detections
        
        # Convert to numpy array for easier manipulation
        dets = np.array(detections)
        
        # Extract components
        x = dets[:, 0]
        y = dets[:, 1]
        w = dets[:, 2]
        h = dets[:, 3]
        scores = dets[:, 4]
        
        # Calculate areas
        areas = w * h
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if len(keep) >= self.max_detections:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
            
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]


# Register the detector
def register():
    """Register this detector with Frigate."""
    return YoloEdgeTpu