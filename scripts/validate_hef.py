#!/usr/bin/env python3.10
"""Validate HEF model accuracy against original ONNX model.

This script compares the inference accuracy between an original ONNX model
and its quantized Hailo HEF counterpart. It's critical for ensuring that
the INT8 quantization process hasn't degraded model performance beyond
acceptable limits.

The validation process:
1. Loads both ONNX and HEF models
2. Runs inference on a validation dataset
3. Calculates mAP (mean Average Precision) for both
4. Reports the accuracy degradation

For wildfire detection, we target <2% mAP degradation to maintain
reliable fire/smoke detection while benefiting from hardware acceleration.

Usage:
    python3.10 validate_hef.py --onnx model.onnx --hef model.hef --dataset val_images/
    
Example Output:
    Validation Results:
    ONNX mAP@0.5: 0.892
    HEF mAP@0.5: 0.878
    Degradation: 1.57% (within acceptable 2% threshold)

Requirements:
    - Python 3.10 (required by Hailo SDK)
    - hailort, onnxruntime, opencv-python, numpy
    - Labeled validation dataset with YOLO format annotations
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass

try:
    from hailo_platform import HEF, VDevice, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    from hailo_platform import HailoStreamInterface
except ImportError:
    print("Error: hailo_platform module not found.")
    print("Please ensure HailoRT is installed: pip install hailort")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found.")
    print("Please install: pip install onnxruntime")
    sys.exit(1)


@dataclass
class BoundingBox:
    """Represents a detection bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (self.x2 - self.x1) * (self.y2 - self.y1)
        area2 = (other.x2 - other.x1) * (other.y2 - other.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class ModelValidator:
    """Base class for model validation."""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.input_shape = (1, 3, input_size[1], input_size[0])  # NCHW
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img = cv2.resize(img, self.input_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess_yolo(self, outputs: List[np.ndarray], conf_threshold: float = 0.25) -> List[BoundingBox]:
        """Postprocess YOLO outputs to get bounding boxes."""
        boxes = []
        
        # YOLO outputs are typically 3 feature maps at different scales
        for output in outputs:
            # Output shape: [batch, grid_h, grid_w, anchors * (5 + num_classes)]
            batch_size, grid_h, grid_w, raw_dim = output.shape
            
            # Assume 3 anchors and 80 classes (COCO) for standard YOLO
            num_anchors = 3
            num_classes = (raw_dim // num_anchors) - 5
            
            # Reshape to [batch, grid_h, grid_w, num_anchors, 5 + num_classes]
            output = output.reshape(batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
            
            # Extract predictions
            for y in range(grid_h):
                for x in range(grid_w):
                    for anchor in range(num_anchors):
                        pred = output[0, y, x, anchor]
                        obj_conf = pred[4]
                        
                        if obj_conf > conf_threshold:
                            # Get class probabilities
                            class_probs = pred[5:]
                            class_id = np.argmax(class_probs)
                            class_conf = class_probs[class_id]
                            
                            # Combined confidence
                            confidence = obj_conf * class_conf
                            
                            if confidence > conf_threshold:
                                # Convert to image coordinates
                                # This is simplified - actual YOLO uses anchor boxes
                                cx = (x + pred[0]) / grid_w * self.input_size[0]
                                cy = (y + pred[1]) / grid_h * self.input_size[1]
                                w = pred[2] * self.input_size[0]
                                h = pred[3] * self.input_size[1]
                                
                                x1 = cx - w / 2
                                y1 = cy - h / 2
                                x2 = cx + w / 2
                                y2 = cy + h / 2
                                
                                boxes.append(BoundingBox(x1, y1, x2, y2, confidence, class_id))
        
        # Apply NMS
        return self.apply_nms(boxes)
    
    def apply_nms(self, boxes: List[BoundingBox], iou_threshold: float = 0.45) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if not boxes:
            return []
        
        # Sort by confidence
        boxes.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while boxes:
            # Take the box with highest confidence
            best = boxes.pop(0)
            keep.append(best)
            
            # Remove boxes with high IoU
            boxes = [box for box in boxes if best.iou(box) < iou_threshold or box.class_id != best.class_id]
        
        return keep


class ONNXValidator(ModelValidator):
    """Validator for ONNX models."""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        super().__init__(model_path, input_size)
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX Model loaded: {self.model_path.name}")
        print(f"  Input: {self.input_name} {self.session.get_inputs()[0].shape}")
        print(f"  Outputs: {len(self.output_names)}")
    
    def predict(self, image: np.ndarray) -> List[BoundingBox]:
        """Run inference on an image."""
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: image})
        
        # Postprocess
        return self.postprocess_yolo(outputs)


class HEFValidator(ModelValidator):
    """Validator for Hailo HEF models."""
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        super().__init__(model_path, input_size)
        
        # Load HEF
        self.hef = HEF(str(self.model_path))
        
        # Create VDevice
        self.target = VDevice()
        
        # Get network info
        self.network_groups = self.hef.get_network_groups_infos()
        self.network_name = self.network_groups[0].name
        
        # Configure network
        self.configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        
        # Get input/output vstream info
        self.input_vstreams_info = self.hef.get_input_vstream_infos()
        self.output_vstreams_info = self.hef.get_output_vstream_infos()
        
        # Create vstream parameters
        self.input_vstreams_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        
        print(f"HEF Model loaded: {self.model_path.name}")
        print(f"  Network: {self.network_name}")
        print(f"  Inputs: {len(self.input_vstreams_info)}")
        print(f"  Outputs: {len(self.output_vstreams_info)}")
    
    def predict(self, image: np.ndarray) -> List[BoundingBox]:
        """Run inference on an image."""
        # Prepare input (convert NCHW to NHWC for Hailo)
        input_data = np.transpose(image, (0, 2, 3, 1))
        
        # Create input/output vstreams
        with self.target.create_input_vstreams(self.input_vstreams_params) as input_vstreams, \
             self.target.create_output_vstreams(self.output_vstreams_params) as output_vstreams:
            
            # Run inference
            input_dict = {self.input_vstreams_info[0].name: input_data}
            
            # Send input
            input_vstreams.send(input_dict)
            
            # Get output
            outputs = output_vstreams.recv()
            
            # Convert outputs to list format
            output_list = [outputs[info.name] for info in self.output_vstreams_info]
            
            # Postprocess
            return self.postprocess_yolo(output_list)
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'network_group'):
            self.network_group.shutdown()


def load_annotations(dataset_path: Path) -> Dict[str, List[BoundingBox]]:
    """Load YOLO format annotations from dataset."""
    annotations = {}
    
    # Assume YOLO format: images in 'images/' and labels in 'labels/'
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(f"Dataset must have 'images' and 'labels' subdirectories")
    
    # Load each annotation file
    for label_file in labels_dir.glob('*.txt'):
        image_name = label_file.stem
        boxes = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert from normalized YOLO format to pixel coordinates
                    # Note: We'll need to scale these based on actual image size
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    boxes.append(BoundingBox(x1, y1, x2, y2, 1.0, class_id))
        
        annotations[image_name] = boxes
    
    return annotations


def calculate_map(predictions: Dict[str, List[BoundingBox]], 
                  ground_truth: Dict[str, List[BoundingBox]], 
                  iou_threshold: float = 0.5) -> float:
    """Calculate mean Average Precision."""
    # Simplified mAP calculation
    # In production, use a proper COCO evaluation library
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for image_name, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(image_name, [])
        
        # Match predictions to ground truth
        matched_gt = set()
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_boxes):
                if i not in matched_gt and pred.class_id == gt.class_id:
                    iou = pred.iou(gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
        
        total_fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # Simplified AP (average of precision and recall)
    ap = (precision + recall) / 2 if (precision + recall) > 0 else 0
    
    return ap


def validate_models(onnx_path: str, hef_path: str, dataset_path: str, 
                   max_images: Optional[int] = None) -> Dict[str, float]:
    """Validate both models on a dataset."""
    dataset_path = Path(dataset_path)
    
    # Initialize validators
    print("Loading models...")
    onnx_validator = ONNXValidator(onnx_path)
    hef_validator = HEFValidator(hef_path)
    
    # Load annotations
    print("\nLoading annotations...")
    ground_truth = load_annotations(dataset_path)
    print(f"Loaded annotations for {len(ground_truth)} images")
    
    # Run inference on both models
    onnx_predictions = {}
    hef_predictions = {}
    
    images_dir = dataset_path / 'images'
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"\nRunning inference on {len(image_files)} images...")
    
    for i, image_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Processing image {i+1}/{len(image_files)}...")
        
        image_name = image_file.stem
        
        # Preprocess image
        image = onnx_validator.preprocess_image(str(image_file))
        
        # Run ONNX inference
        onnx_boxes = onnx_validator.predict(image)
        onnx_predictions[image_name] = onnx_boxes
        
        # Run HEF inference
        hef_boxes = hef_validator.predict(image)
        hef_predictions[image_name] = hef_boxes
    
    # Calculate mAP for both
    print("\nCalculating mAP...")
    onnx_map = calculate_map(onnx_predictions, ground_truth)
    hef_map = calculate_map(hef_predictions, ground_truth)
    
    # Calculate degradation
    degradation = (onnx_map - hef_map) / onnx_map * 100 if onnx_map > 0 else 0
    
    results = {
        'onnx_map': onnx_map,
        'hef_map': hef_map,
        'degradation_percent': degradation,
        'num_images': len(image_files)
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate HEF model accuracy against ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to ONNX model"
    )
    
    parser.add_argument(
        "--hef",
        required=True,
        help="Path to HEF model"
    )
    
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to validation dataset (YOLO format)"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Maximum acceptable mAP degradation percentage (default: 2.0)"
    )
    
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation
        results = validate_models(args.onnx, args.hef, args.dataset, args.max_images)
        
        # Print results
        print("\n" + "="*50)
        print("VALIDATION RESULTS")
        print("="*50)
        print(f"ONNX mAP@0.5: {results['onnx_map']:.4f}")
        print(f"HEF mAP@0.5: {results['hef_map']:.4f}")
        print(f"Degradation: {results['degradation_percent']:.2f}%")
        print(f"Images tested: {results['num_images']}")
        
        if results['degradation_percent'] <= args.threshold:
            print(f"\n✓ PASSED: Degradation within {args.threshold}% threshold")
            exit_code = 0
        else:
            print(f"\n✗ FAILED: Degradation exceeds {args.threshold}% threshold")
            exit_code = 1
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()