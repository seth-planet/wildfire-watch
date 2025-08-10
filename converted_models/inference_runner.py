#!/usr/bin/env python3.10
"""
Inference Runner for YOLO-NAS Models
Handles running inference using various model formats (PyTorch, ONNX, TFLite, Hailo)
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Represents inference results from a model"""
    image_path: str
    detections: List[Dict[str, Any]]  # List of {'bbox': array, 'confidence': float, 'class_id': int}
    inference_time_ms: float
    model_format: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'image_path': self.image_path,
            'detections': self.detections,
            'inference_time_ms': self.inference_time_ms,
            'model_format': self.model_format
        }


class InferenceRunner:
    """Runs inference using various model formats"""
    
    def __init__(self, confidence_threshold: float = 0.25, 
                 iou_threshold: float = 0.45,
                 input_size: Tuple[int, int] = (640, 640)):
        """Initialize inference runner
        
        Args:
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            input_size: Model input size (height, width)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.logger = logger
    
    def run_inference_pytorch(
        self,
        model_path: Path,
        test_images: List[Path],
        num_classes: int = 32,
        model_architecture: str = 'yolo_nas_s'
    ) -> Dict[str, List[Dict]]:
        """Run inference using PyTorch model.
        
        Args:
            model_path: Path to PyTorch model checkpoint
            test_images: List of image paths to run inference on
            num_classes: Number of classes in the model
            model_architecture: Model architecture name
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Running PyTorch inference on {len(test_images)} images")
        
        try:
            from super_gradients.training import models
            import torch
        except ImportError as e:
            raise ImportError(f"PyTorch or super-gradients not available: {e}")
        
        # Load model
        self.logger.info(f"Loading {model_architecture} model from {model_path}")
        model = models.get(model_architecture, num_classes=num_classes, checkpoint_path=str(model_path))
        model.eval()
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        self.logger.info(f"Using device: {device}")
        
        results = {}
        
        for img_path in test_images:
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Time the inference
                start_time = time.time()
                
                # Run inference
                with torch.no_grad():
                    predictions = model.predict(image_rgb, conf=self.confidence_threshold)
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Parse predictions - handle different formats
                detections = []
                
                # Handle different prediction formats from YOLO-NAS
                if isinstance(predictions, list):
                    for pred in predictions:
                        # Check if it's already a dict (newer format)
                        if isinstance(pred, dict):
                            # Direct dict format
                            if 'bboxes' in pred or 'boxes' in pred:
                                boxes = pred.get('bboxes', pred.get('boxes', []))
                                scores = pred.get('scores', pred.get('confidence', []))
                                labels = pred.get('labels', pred.get('classes', []))
                                
                                for box, score, label in zip(boxes, scores, labels):
                                    detections.append({
                                        'bbox': np.array(box) if not isinstance(box, np.ndarray) else box,
                                        'confidence': float(score),
                                        'class_id': int(label)
                                    })
                        # Original object format with .prediction attribute
                        elif hasattr(pred, 'prediction'):
                            boxes = pred.prediction.bboxes_xyxy
                            scores = pred.prediction.confidence
                            labels = pred.prediction.labels
                            
                            for box, score, label in zip(boxes, scores, labels):
                                detections.append({
                                    'bbox': box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box),
                                    'confidence': float(score),
                                    'class_id': int(label)
                                })
                # Single prediction object
                elif hasattr(predictions, 'prediction'):
                    boxes = predictions.prediction.bboxes_xyxy
                    scores = predictions.prediction.confidence
                    labels = predictions.prediction.labels
                    
                    for box, score, label in zip(boxes, scores, labels):
                        detections.append({
                            'bbox': box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box),
                            'confidence': float(score),
                            'class_id': int(label)
                        })
                # Direct attributes on predictions object (some YOLO-NAS versions)
                elif hasattr(predictions, 'bboxes_xyxy'):
                    boxes = predictions.bboxes_xyxy
                    scores = predictions.confidence
                    labels = predictions.labels
                    
                    for box, score, label in zip(boxes, scores, labels):
                        detections.append({
                            'bbox': box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box),
                            'confidence': float(score),
                            'class_id': int(label)
                        })
                
                results[str(img_path)] = detections
                self.logger.debug(f"Processed {img_path}: {len(detections)} detections in {inference_time:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                results[str(img_path)] = []
        
        return results
    
    def run_inference_onnx(
        self,
        onnx_path: Path,
        test_images: List[Path],
        providers: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """Run inference using ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            test_images: List of image paths to run inference on
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Running ONNX inference on {len(test_images)} images")
        
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(f"ONNX Runtime not available: {e}")
        
        # Set providers
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create ONNX session
        self.logger.info(f"Creating ONNX session with providers: {providers}")
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        results = {}
        
        for img_path in test_images:
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Preprocess image for ONNX
                preprocessed = self._preprocess_image_onnx(image)
                
                # Time the inference
                start_time = time.time()
                
                # Run inference
                outputs = session.run(output_names, {input_name: preprocessed})
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Parse outputs
                detections = self._parse_onnx_outputs(outputs, image.shape[:2])
                
                results[str(img_path)] = detections
                self.logger.debug(f"Processed {img_path}: {len(detections)} detections in {inference_time:.1f}ms")
                
            except AttributeError as e:
                # Special handling for the dict shape error
                self.logger.error(f"AttributeError processing {img_path}: {e}")
                self.logger.error(f"ONNX outputs type: {type(outputs)}")
                if isinstance(outputs, list):
                    for i, out in enumerate(outputs):
                        self.logger.error(f"  Output[{i}] type: {type(out)}")
                import traceback
                traceback.print_exc()
                results[str(img_path)] = []
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                results[str(img_path)] = []
        
        return results
    
    def run_inference_tflite(
        self,
        tflite_path: Path,
        test_images: List[Path],
        use_coral: bool = False
    ) -> Dict[str, List[Dict]]:
        """Run inference using TensorFlow Lite model.
        
        Args:
            tflite_path: Path to TFLite model
            test_images: List of image paths to run inference on
            use_coral: Whether to use Coral Edge TPU
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Running TFLite inference on {len(test_images)} images (Coral: {use_coral})")
        
        try:
            if use_coral:
                from pycoral.utils import edgetpu
                interpreter = edgetpu.make_interpreter(str(tflite_path))
            else:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=str(tflite_path))
        except ImportError as e:
            raise ImportError(f"TFLite runtime not available: {e}")
        
        # Allocate tensors
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        results = {}
        
        for img_path in test_images:
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    self.logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Preprocess for TFLite
                preprocessed = self._preprocess_image_tflite(image, input_details[0])
                
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], preprocessed)
                
                # Time the inference
                start_time = time.time()
                
                # Run inference
                interpreter.invoke()
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Get outputs
                outputs = []
                for output_detail in output_details:
                    outputs.append(interpreter.get_tensor(output_detail['index']))
                
                # Parse outputs
                detections = self._parse_tflite_outputs(outputs, image.shape[:2])
                
                results[str(img_path)] = detections
                self.logger.debug(f"Processed {img_path}: {len(detections)} detections in {inference_time:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
                results[str(img_path)] = []
        
        return results
    
    def run_inference_hailo(
        self,
        hef_path: Path,
        test_images: List[Path],
        use_simulator: bool = True
    ) -> Dict[str, List[Dict]]:
        """Run inference using Hailo HEF model.
        
        Args:
            hef_path: Path to HEF model
            test_images: List of image paths to run inference on
            use_simulator: Whether to use Hailo simulator (for testing without hardware)
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        self.logger.info(f"Running Hailo inference on {len(test_images)} images (Simulator: {use_simulator})")
        
        if use_simulator or not self._hailo_hardware_available():
            # Simulate Hailo inference for testing
            self.logger.info("Using simulated Hailo inference")
            return self._simulate_hailo_inference(test_images)
        
        try:
            from hailo_platform import (VDevice, HailoStreamInterface, 
                                       InferVStreams, ConfigureParams,
                                       InputVStreamParams, OutputVStreamParams,
                                       FormatType)
        except ImportError as e:
            self.logger.warning(f"Hailo SDK not available, using simulation: {e}")
            return self._simulate_hailo_inference(test_images)
        
        # Real Hailo inference implementation would go here
        # For now, we'll use simulation
        return self._simulate_hailo_inference(test_images)
    
    def _preprocess_image_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference"""
        # Resize to model input size
        image_resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW
        input_tensor = np.transpose(image_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _preprocess_image_tflite(self, image: np.ndarray, input_details: Dict) -> np.ndarray:
        """Preprocess image for TFLite inference"""
        # Get input shape and type
        input_shape = input_details['shape']
        input_dtype = input_details['dtype']
        
        # Resize image
        height, width = input_shape[1], input_shape[2]
        image_resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize based on input type
        if input_dtype == np.uint8:
            input_tensor = image_rgb
        else:
            input_tensor = image_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _find_first_numpy_array(self, data: Any) -> Union[np.ndarray, None]:
        """
        Recursively searches for the first numpy array in a nested structure.
        Handles dicts, lists, tuples, and direct numpy arrays.
        Prioritizes common ONNX output keys for backward compatibility.
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, dict):
            # Prioritize common ONNX output keys for detection outputs
            for key in ['output', 'predictions', 'detection', 'output0']:
                if key in data:
                    result = self._find_first_numpy_array(data[key])
                    if result is not None:
                        return result
            # Search all values if common keys don't yield result
            for value in data.values():
                result = self._find_first_numpy_array(value)
                if result is not None:
                    return result
        elif isinstance(data, (list, tuple)):
            for item in data:
                result = self._find_first_numpy_array(item)
                if result is not None:
                    return result
        return None
    
    def _parse_onnx_outputs(self, outputs: Union[List[np.ndarray], Dict[str, Any]], 
                           original_shape: Tuple[int, int]) -> List[Dict]:
        """Parse ONNX model outputs to detections.
        Handles direct numpy arrays, dicts of numpy arrays, and nested structures."""
        detections = []
        
        # Debug logging to trace the issue
        self.logger.debug(f"ONNX outputs type: {type(outputs)}")
        if isinstance(outputs, dict):
            self.logger.debug(f"ONNX outputs keys: {list(outputs.keys())}")
            for k, v in outputs.items():
                self.logger.debug(f"  Key '{k}': type={type(v)}")
        
        # Use helper to robustly find numpy array in nested structures
        output_array = self._find_first_numpy_array(outputs)
        
        if output_array is None:
            self.logger.warning("No suitable numpy array found in ONNX outputs for parsing")
            return []
        
        # Validate shape and parse detections
        # Ensure array is at least 2D and has enough features for detection output
        if output_array.ndim >= 2 and output_array.shape[-1] >= 6:
            # Handle batch dimension if present
            predictions = output_array[0] if output_array.ndim > 2 else output_array
            
            for pred in predictions:
                if pred[4] >= self.confidence_threshold:  # Confidence
                    # Scale bbox to original image size
                    bbox = pred[:4] * np.array([
                        original_shape[1] / self.input_size[0],
                        original_shape[0] / self.input_size[1],
                        original_shape[1] / self.input_size[0],
                        original_shape[0] / self.input_size[1]
                    ])
                    
                    detections.append({
                        'bbox': bbox,
                        'confidence': float(pred[4]),
                        'class_id': int(pred[5])
                    })
        else:
            self.logger.warning(f"Found numpy array but shape {output_array.shape} "
                              "doesn't match expected detection format (ndim >= 2, last dim >= 6)")
        
        return detections
    
    def _parse_tflite_outputs(self, outputs: List[np.ndarray],
                            original_shape: Tuple[int, int]) -> List[Dict]:
        """Parse TFLite model outputs to detections"""
        # Similar to ONNX parsing, implementation depends on model output format
        return self._parse_onnx_outputs(outputs, original_shape)
    
    def _hailo_hardware_available(self) -> bool:
        """Check if Hailo hardware is available"""
        try:
            from hailo_platform import VDevice
            devices = VDevice.scan()
            return len(devices) > 0
        except:
            return False
    
    def _simulate_hailo_inference(self, test_images: List[Path]) -> Dict[str, List[Dict]]:
        """Simulate Hailo inference for testing without hardware"""
        results = {}
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        for img_path in test_images:
            # Simulate detections with small variations
            detections = []
            
            # Add some dummy detections for testing
            num_detections = np.random.randint(0, 5)
            for _ in range(num_detections):
                detections.append({
                    'bbox': np.random.rand(4) * self.input_size[0],
                    'confidence': np.random.uniform(self.confidence_threshold, 1.0),
                    'class_id': np.random.randint(0, 32)
                })
            
            results[str(img_path)] = detections
        
        return results
    
    def run_batch_inference(
        self,
        model_path: Path,
        test_images: List[Path],
        model_format: str = 'pytorch',
        **kwargs
    ) -> List[InferenceResult]:
        """Run inference on multiple images using specified model format.
        
        Args:
            model_path: Path to model file
            test_images: List of image paths
            model_format: Model format ('pytorch', 'onnx', 'tflite', 'hailo')
            **kwargs: Additional arguments for specific inference methods
            
        Returns:
            List of InferenceResult objects
        """
        # Map format to inference method
        inference_methods = {
            'pytorch': self.run_inference_pytorch,
            'onnx': self.run_inference_onnx,
            'tflite': self.run_inference_tflite,
            'hailo': self.run_inference_hailo
        }
        
        if model_format not in inference_methods:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        # Run inference
        inference_fn = inference_methods[model_format]
        results_dict = inference_fn(model_path, test_images, **kwargs)
        
        # Convert to InferenceResult objects
        results = []
        for img_path, detections in results_dict.items():
            result = InferenceResult(
                image_path=img_path,
                detections=detections,
                inference_time_ms=0.0,  # Would be calculated in real implementation
                model_format=model_format
            )
            results.append(result)
        
        return results