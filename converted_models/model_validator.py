#!/usr/bin/env python3.10
"""
Model Validator for YOLO-NAS Detection Models
Validates model accuracy between different model versions and formats
"""
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Represents a single detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2] or [x_center, y_center, width, height]
    confidence: float
    class_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox.tolist(),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id)
        }


class ModelAccuracyValidator:
    """Validates accuracy between different model versions for object detection"""
    
    def __init__(self, confidence_threshold: float = 0.25, iou_threshold: float = 0.5):
        """
        Initialize the validator with thresholds.
        
        Args:
            confidence_threshold: Minimum confidence score for detections
            iou_threshold: Minimum IoU for matching detections
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.results_cache = {}
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Ensure boxes are in corner format [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_detections(
        self, 
        detections1: List[Dict], 
        detections2: List[Dict]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Match detections between two models and calculate agreement.
        
        Args:
            detections1: List of detections from model 1
            detections2: List of detections from model 2
            
        Returns:
            Tuple of (overall_agreement, per_class_agreement)
        """
        # Group detections by class
        dets1_by_class = {}
        dets2_by_class = {}
        
        for det in detections1:
            class_id = det['class_id']
            if class_id not in dets1_by_class:
                dets1_by_class[class_id] = []
            dets1_by_class[class_id].append(det)
        
        for det in detections2:
            class_id = det['class_id']
            if class_id not in dets2_by_class:
                dets2_by_class[class_id] = []
            dets2_by_class[class_id].append(det)
        
        # Calculate matches per class
        all_classes = set(dets1_by_class.keys()) | set(dets2_by_class.keys())
        per_class_agreement = {}
        total_matches = 0
        total_detections = 0
        
        for class_id in all_classes:
            class_dets1 = dets1_by_class.get(class_id, [])
            class_dets2 = dets2_by_class.get(class_id, [])
            
            if not class_dets1 and not class_dets2:
                per_class_agreement[class_id] = 1.0
                continue
            
            if not class_dets1 or not class_dets2:
                per_class_agreement[class_id] = 0.0
                total_detections += len(class_dets1) + len(class_dets2)
                continue
            
            # Match detections using greedy approach
            matched = 0
            used_indices = set()
            
            for det1 in class_dets1:
                best_iou = 0
                best_idx = -1
                
                for idx, det2 in enumerate(class_dets2):
                    if idx in used_indices:
                        continue
                    
                    # Convert bbox to numpy array if needed
                    bbox1 = np.array(det1['bbox']) if not isinstance(det1['bbox'], np.ndarray) else det1['bbox']
                    bbox2 = np.array(det2['bbox']) if not isinstance(det2['bbox'], np.ndarray) else det2['bbox']
                    
                    iou = self.calculate_iou(bbox1, bbox2)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_idx = idx
                
                if best_idx >= 0:
                    used_indices.add(best_idx)
                    matched += 1
                    
                    # Also check confidence agreement
                    conf_diff = abs(det1['confidence'] - class_dets2[best_idx]['confidence'])
                    if conf_diff > 0.1:  # More than 10% confidence difference
                        matched -= 0.5  # Partial match
            
            total_class_dets = max(len(class_dets1), len(class_dets2))
            per_class_agreement[class_id] = matched / total_class_dets if total_class_dets > 0 else 0
            
            total_matches += matched
            total_detections += total_class_dets
        
        overall_agreement = total_matches / total_detections if total_detections > 0 else 1.0
        
        return overall_agreement, per_class_agreement
    
    def validate_model_outputs(
        self,
        model1_outputs: Dict[str, List[Dict]],
        model2_outputs: Dict[str, List[Dict]],
        required_agreement: float = 0.98
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate outputs between two models on same test images.
        
        Args:
            model1_outputs: Dict mapping image_path to detections
            model2_outputs: Dict mapping image_path to detections
            required_agreement: Minimum agreement threshold (default 98%)
            
        Returns:
            Tuple of (passed, metrics_dict)
        """
        agreements = []
        per_class_agreements = {}
        fire_class_agreements = []  # Track fire class specifically
        
        for image_path in model1_outputs:
            if image_path not in model2_outputs:
                logger.warning(f"Image {image_path} missing from model2 outputs")
                continue
            
            agreement, class_agreement = self.match_detections(
                model1_outputs[image_path],
                model2_outputs[image_path]
            )
            
            agreements.append(agreement)
            
            # Track per-class statistics
            for class_id, class_agr in class_agreement.items():
                if class_id not in per_class_agreements:
                    per_class_agreements[class_id] = []
                per_class_agreements[class_id].append(class_agr)
                
                # Special tracking for fire class (ID 26)
                if class_id == 26:
                    fire_class_agreements.append(class_agr)
        
        # Calculate overall metrics
        overall_agreement = np.mean(agreements) if agreements else 0
        
        # Calculate per-class average agreement
        class_metrics = {}
        for class_id, agrs in per_class_agreements.items():
            class_metrics[class_id] = {
                'mean_agreement': np.mean(agrs),
                'min_agreement': np.min(agrs),
                'num_images': len(agrs)
            }
        
        # Fire class specific metrics
        fire_metrics = {
            'mean_agreement': np.mean(fire_class_agreements) if fire_class_agreements else 1.0,
            'min_agreement': np.min(fire_class_agreements) if fire_class_agreements else 1.0,
            'num_detections': len(fire_class_agreements)
        }
        
        metrics = {
            'overall_agreement': overall_agreement,
            'num_images_compared': len(agreements),
            'per_class_metrics': class_metrics,
            'fire_class_metrics': fire_metrics,
            'passed': overall_agreement >= required_agreement
        }
        
        # Log detailed results
        logger.info(f"Model comparison results:")
        logger.info(f"  Overall agreement: {overall_agreement:.2%}")
        logger.info(f"  Fire class agreement: {fire_metrics['mean_agreement']:.2%}")
        logger.info(f"  Images compared: {len(agreements)}")
        
        return metrics['passed'], metrics
    
    def compare_inference_speed(
        self,
        model1_times: List[float],
        model2_times: List[float]
    ) -> Dict[str, float]:
        """
        Compare inference speed between two models.
        
        Args:
            model1_times: List of inference times (ms) for model 1
            model2_times: List of inference times (ms) for model 2
            
        Returns:
            Dictionary with speed comparison metrics
        """
        model1_mean = np.mean(model1_times) if model1_times else 0
        model2_mean = np.mean(model2_times) if model2_times else 0
        
        speedup = model1_mean / model2_mean if model2_mean > 0 else 1.0
        
        return {
            'model1_mean_ms': model1_mean,
            'model1_std_ms': np.std(model1_times) if model1_times else 0,
            'model2_mean_ms': model2_mean,
            'model2_std_ms': np.std(model2_times) if model2_times else 0,
            'speedup': speedup,
            'faster_model': 'model2' if speedup > 1 else 'model1'
        }
    
    def validate_quantization_accuracy(
        self,
        fp32_outputs: Dict[str, List[Dict]],
        int8_outputs: Dict[str, List[Dict]],
        max_degradation: float = 0.02
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate accuracy after quantization (FP32 to INT8).
        
        Args:
            fp32_outputs: Original FP32 model outputs
            int8_outputs: Quantized INT8 model outputs
            max_degradation: Maximum allowed accuracy degradation (default 2%)
            
        Returns:
            Tuple of (passed, metrics_dict)
        """
        passed, metrics = self.validate_model_outputs(
            fp32_outputs,
            int8_outputs,
            required_agreement=1.0 - max_degradation
        )
        
        # Add quantization-specific metrics
        metrics['quantization_info'] = {
            'format': 'INT8',
            'max_allowed_degradation': max_degradation,
            'actual_degradation': 1.0 - metrics['overall_agreement']
        }
        
        return passed, metrics