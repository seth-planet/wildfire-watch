#!/usr/bin/env python3.12
"""
YOLO to Frigate Model Wrapper

This script creates a wrapper model that converts YOLO output format
to Frigate's expected format (TensorFlow Object Detection API style).
"""

import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOToFrigateWrapper:
    """Wraps YOLO models to output Frigate-compatible format."""
    
    def __init__(self, model_path: str, num_classes: int = 4):
        """
        Initialize wrapper with YOLO model.
        
        Args:
            model_path: Path to YOLO TFLite model
            num_classes: Number of detection classes
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"Loaded model: {model_path}")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
        logger.info(f"Output shape: {self.output_details[0]['shape']}")
    
    def parse_yolo_output(self, raw_output, conf_threshold=0.25, iou_threshold=0.45):
        """
        Parse raw YOLO output to extract boxes, scores, and classes.
        
        YOLO output format: [1, num_predictions, 4 + 1 + num_classes]
        Where each prediction is: [x, y, w, h, objectness, class_probs...]
        
        Returns:
            boxes: [num_detections, 4] in format [ymin, xmin, ymax, xmax] normalized
            scores: [num_detections] confidence scores
            classes: [num_detections] class indices
            num_detections: scalar count of valid detections
        """
        # Handle dict output format from some YOLO implementations
        if isinstance(raw_output, dict):
            # Try common keys for output tensors
            raw_output = raw_output.get('output', raw_output.get('predictions', 
                        raw_output.get('detection', list(raw_output.values())[0])))
        
        # Handle different YOLO output formats
        if len(raw_output.shape) == 3:
            # Format: [1, num_predictions, features]
            predictions = raw_output[0]
        elif len(raw_output.shape) == 2:
            # Format: [num_predictions, features]
            predictions = raw_output
        else:
            # Try to reshape
            total_features = raw_output.shape[-1]
            num_predictions = raw_output.size // total_features
            predictions = raw_output.reshape(num_predictions, total_features)
        
        # Extract components
        # YOLO format: [x_center, y_center, width, height, objectness, ...class_scores]
        xy = predictions[:, 0:2]  # Center coordinates
        wh = predictions[:, 2:4]  # Width and height
        objectness = predictions[:, 4:5]  # Objectness score
        class_probs = predictions[:, 5:5+self.num_classes]  # Class probabilities
        
        # Calculate confidence scores
        scores = objectness * np.max(class_probs, axis=1, keepdims=True)
        scores = scores.squeeze()
        
        # Get class indices
        classes = np.argmax(class_probs, axis=1).astype(np.float32)
        
        # Filter by confidence threshold
        mask = scores > conf_threshold
        if not np.any(mask):
            # No detections above threshold
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                0
            )
        
        # Apply mask
        xy = xy[mask]
        wh = wh[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        # Convert to corner format
        x1 = xy[:, 0] - wh[:, 0] / 2
        y1 = xy[:, 1] - wh[:, 1] / 2
        x2 = xy[:, 0] + wh[:, 0] / 2
        y2 = xy[:, 1] + wh[:, 1] / 2
        
        # Normalize to [0, 1] if coordinates are in pixel space
        # Assuming 640x640 input
        if np.max(x2) > 1.0:
            x1 /= 640.0
            y1 /= 640.0
            x2 /= 640.0
            y2 /= 640.0
        
        # Format as [ymin, xmin, ymax, xmax] for TF Object Detection API
        boxes = np.stack([y1, x1, y2, x2], axis=1)
        
        # Apply NMS
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=100,
            iou_threshold=iou_threshold
        ).numpy()
        
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        classes = classes[selected_indices]
        num_detections = len(selected_indices)
        
        return boxes, scores, classes, num_detections
    
    def create_wrapped_model(self, output_path: str):
        """
        Create a new TFLite model that wraps the YOLO model with format conversion.
        """
        logger.info("Creating wrapped model with Frigate-compatible output format...")
        
        # Define the wrapper model
        class FrigateWrapper(tf.Module):
            def __init__(self, yolo_model_path, num_classes):
                super().__init__()
                self.yolo_model_path = yolo_model_path
                self.num_classes = num_classes
                # Load the YOLO model
                with open(yolo_model_path, 'rb') as f:
                    self.yolo_model_bytes = f.read()
            
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, 640, 640, 3], dtype=tf.uint8)
            ])
            def __call__(self, images):
                # For TFLite conversion, we need to simulate the YOLO output
                # In practice, this would run the actual YOLO model
                # Here we create placeholder outputs in Frigate format
                
                # Frigate expects 4 outputs:
                # 1. boxes: [1, max_detections, 4] - normalized coordinates
                # 2. classes: [1, max_detections] - class indices as float32
                # 3. scores: [1, max_detections] - confidence scores
                # 4. num_detections: [1] - number of valid detections
                
                max_detections = 100
                
                # Create dummy outputs for now
                # In deployment, these would be computed from YOLO output
                boxes = tf.zeros([1, max_detections, 4], dtype=tf.float32)
                classes = tf.zeros([1, max_detections], dtype=tf.float32)
                scores = tf.zeros([1, max_detections], dtype=tf.float32)
                num_detections = tf.constant([0], dtype=tf.float32)
                
                return {
                    'detection_boxes': boxes,
                    'detection_classes': classes,
                    'detection_scores': scores,
                    'num_detections': num_detections
                }
        
        # Create the wrapper model
        wrapper = FrigateWrapper(self.model_path, self.num_classes)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [wrapper.__call__.get_concrete_function()]
        )
        
        # Apply quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32  # Frigate expects float outputs
        
        # Representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.randint(0, 255, size=(1, 640, 640, 3), dtype=np.uint8)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Convert
        tflite_model = converter.convert()
        
        # Save the wrapped model
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"Wrapped model saved to: {output_path}")
        return output_path
    
    def create_post_processing_script(self, output_dir: str):
        """
        Create a custom detector script for Frigate that handles YOLO output.
        """
        script_content = '''#!/usr/bin/env python3
"""Custom YOLO detector for Frigate."""

import numpy as np
import tensorflow as tf
from frigate.detectors.detection_api import DetectionApi
import logging

logger = logging.getLogger(__name__)


class YoloTfLite(DetectionApi):
    """Custom YOLO detector for Frigate."""
    
    type_key = "yolo_tflite"
    
    def __init__(self, det_device=None, model_config=None):
        self.model_config = model_config or {}
        
        # Load the model
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_config.get("path", "/models/yolo.tflite")
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info(f"YOLO detector initialized with model: {self.model_config.get('path')}")
    
    def detect_raw(self, tensor_input):
        """Run detection and return results in Frigate format."""
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor_input)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get raw YOLO output
        raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Parse YOLO output to Frigate format
        detections = self._parse_yolo_output(raw_output)
        
        return detections
    
    def _parse_yolo_output(self, raw_output, conf_threshold=0.25):
        """Convert YOLO output to Frigate format."""
        # YOLO output processing logic
        # This is a simplified version - adjust based on your model's exact format
        
        detections = []
        
        # Handle dict output format from some YOLO implementations
        if isinstance(raw_output, dict):
            # Try common keys for output tensors
            raw_output = raw_output.get('output', raw_output.get('predictions', 
                        raw_output.get('detection', list(raw_output.values())[0])))
        
        # Assuming YOLO format: [1, num_anchors, 4 + 1 + num_classes]
        if len(raw_output.shape) == 3:
            predictions = raw_output[0]
        else:
            predictions = raw_output
        
        for pred in predictions:
            # Extract components
            x, y, w, h = pred[0:4]
            objectness = pred[4]
            class_scores = pred[5:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # Calculate confidence
            confidence = objectness * class_score
            
            if confidence > conf_threshold:
                # Convert to Frigate format [x, y, w, h, conf, class_id, class_prob]
                # Note: Frigate expects normalized coordinates
                detections.append([
                    x / 640.0,  # Normalize x
                    y / 640.0,  # Normalize y
                    w / 640.0,  # Normalize width
                    h / 640.0,  # Normalize height
                    confidence,  # Detection confidence
                    float(class_id),  # Class ID
                    class_score  # Class probability
                ])
        
        return np.array(detections, dtype=np.float32)
'''
        
        output_path = Path(output_dir) / "yolo_detector.py"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Custom detector script saved to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model for Frigate")
    parser.add_argument('--model', required=True, help='Path to YOLO TFLite model')
    parser.add_argument('--output-dir', default='frigate_models', help='Output directory')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--create-wrapper', action='store_true', help='Create wrapped model')
    parser.add_argument('--create-detector', action='store_true', help='Create custom detector script')
    
    args = parser.parse_args()
    
    wrapper = YOLOToFrigateWrapper(args.model, args.num_classes)
    
    if args.create_wrapper:
        output_path = Path(args.output_dir) / f"{Path(args.model).stem}_frigate_wrapped.tflite"
        wrapper.create_wrapped_model(str(output_path))
    
    if args.create_detector:
        wrapper.create_post_processing_script(args.output_dir)
    
    # Test the parsing logic
    logger.info("\nTesting YOLO output parsing...")
    # Create dummy YOLO output for testing
    dummy_output = np.random.rand(1, 8400, 9).astype(np.float32)  # [batch, predictions, 4+1+4classes]
    boxes, scores, classes, num_det = wrapper.parse_yolo_output(dummy_output)
    logger.info(f"Parsed {num_det} detections")


if __name__ == "__main__":
    main()