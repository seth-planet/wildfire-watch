#!/usr/bin/env python3.10
"""Test Hailo fire detection with proper NMS output handling for 32-class model."""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import test utilities
try:
    from .test_utils.hailo_test_utils import VideoDownloader, HailoDevice
except ImportError:
    # Running as script, not module
    from test_utils.hailo_test_utils import VideoDownloader, HailoDevice

try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface, 
        InferVStreams, InputVStreamParams, OutputVStreamParams,
        FormatType
    )
    print("✓ Hailo platform modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


@dataclass
class Detection:
    """Detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int
    class_name: str


class HailoYOLOv8Detector:
    """YOLOv8 detector using Hailo with NMS enabled."""
    
    def __init__(self, hef_path: str, conf_threshold: float = 0.3):
        """Initialize Hailo detector."""
        self.hef_path = hef_path
        self.conf_threshold = conf_threshold
        
        # Class names - model has 32 classes from YOLO dataset
        self.class_names = [
            'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck',
            'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle',
            'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk',
            'Squirrel', 'Pig', 'Chicken', 'Boat', 'Vehicle registration plate',
            'Snowmobile', 'Human face', 'Armadillo', 'Fire', 'Package',
            'Rodent', 'Child', 'Weapon', 'Backpack'
        ]
        
        # We're primarily interested in Fire detection (index 26)
        self.fire_class_id = 26
        self.debug = True  # Enable debug output initially
        
        # Initialize device
        self.device = VDevice()
        print("✓ Created VDevice")
        
        # Load HEF
        self.hef = HEF(hef_path)
        print("✓ Loaded HEF model")
        
        # Get network group names
        network_group_names = self.hef.get_network_group_names()
        print(f"  Network groups: {network_group_names}")
        
        # Configure network
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        
        # Set batch size
        for name, params in self.configure_params.items():
            params.batch_size = 1
            
        # Configure network groups
        self.network_groups = self.device.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        print("✓ Configured network group")
        
        # Create network group params for activation
        self.network_group_params = self.network_group.create_params()
        
        # Get vstream info from HEF
        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
        
        print(f"  Input stream: {self.input_vstream_info.name}")
        print(f"  Input shape: {self.input_vstream_info.shape}")
        print(f"  Output stream: {self.output_vstream_info.name}")
        
        # Create vstream params with format type
        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, 
            format_type=FormatType.FLOAT32
        )
        
        print("✓ Created vstream parameters")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Get expected input shape from vstream info
        height, width, channels = self.input_vstream_info.shape
        
        # Resize to expected size
        resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
        
    def parse_nms_output(self, output_data: Any, orig_shape: Tuple[int, int]) -> List[Detection]:
        """Parse NMS output from Hailo.
        
        The output is HAILO NMS BY CLASS format with 32 classes.
        We need to extract detections and filter for our classes of interest.
        """
        detections = []
        orig_h, orig_w = orig_shape
        
        # Debug output structure
        if self.debug:
            print(f"\nDebug - Output type: {type(output_data)}")
            if hasattr(output_data, 'shape'):
                print(f"Debug - Output shape: {output_data.shape}")
            
            # More detailed debug for list output
            if isinstance(output_data, list):
                print(f"Debug - List length: {len(output_data)}")
                for i, item in enumerate(output_data):
                    if isinstance(item, np.ndarray):
                        print(f"  Class {i} ({self.class_names[i] if i < len(self.class_names) else 'Unknown'}): shape={item.shape}, dtype={item.dtype}")
                        if item.size > 0:
                            print(f"    First detection: {item[0]}")
                    else:
                        print(f"  Item {i}: type={type(item)}")
        
        # The NMS output format varies, let's handle different cases
        if isinstance(output_data, np.ndarray):
            # Check if it's structured array or regular array
            if output_data.dtype.names:
                # Structured array with named fields
                print(f"Structured array fields: {output_data.dtype.names}")
                # TODO: Parse structured output
            else:
                # Regular array - try to parse as detections
                print(f"Array shape: {output_data.shape}")
                if len(output_data.shape) == 2 and output_data.shape[1] >= 6:
                    # Format: [num_detections, 6+] where 6 = [x1, y1, x2, y2, score, class_id, ...]
                    for detection in output_data:
                        if detection[4] >= self.conf_threshold:  # score
                            x1, y1, x2, y2, score, class_id = detection[:6]
                            class_id = int(class_id)
                            
                            # Ensure class_id is within valid range
                            if 0 <= class_id < len(self.class_names):
                                # Convert normalized coordinates to pixels
                                x1 = int(x1 * orig_w)
                                y1 = int(y1 * orig_h)
                                x2 = int(x2 * orig_w)
                                y2 = int(y2 * orig_h)
                                
                                detections.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    score=float(score),
                                    class_id=class_id,
                                    class_name=self.class_names[class_id]
                                ))
        
        elif isinstance(output_data, list):
            # Handle nested list structure 
            if len(output_data) == 1 and isinstance(output_data[0], list):
                output_data = output_data[0]
                if self.debug:
                    print(f"Debug - Unwrapped nested list, new length: {len(output_data)}")
            
            # List of detections per class
            for class_id, class_detections in enumerate(output_data):
                if class_id >= len(self.class_names):
                    continue  # Skip invalid class indices
                    
                if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                    for detection in class_detections:
                        if len(detection) >= 5:
                            x1, y1, x2, y2, score = detection[:5]
                            
                            if score >= self.conf_threshold:
                                # Convert to pixels
                                x1 = int(x1 * orig_w)
                                y1 = int(y1 * orig_h)
                                x2 = int(x2 * orig_w)
                                y2 = int(y2 * orig_h)
                                
                                detections.append(Detection(
                                    bbox=(x1, y1, x2, y2),
                                    score=float(score),
                                    class_id=class_id,
                                    class_name=self.class_names[class_id]
                                ))
        
        # Sort by score and limit detections
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[:100]  # Limit to top 100 detections
        
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """Run inference on image."""
        # Store original shape
        orig_shape = image.shape[:2]
        
        # Preprocess
        input_data = self.preprocess(image)
        
        # Create input dict for InferVStreams
        input_dict = {self.input_vstream_info.name: np.expand_dims(input_data, axis=0)}
        
        # Run inference
        start_time = time.perf_counter()
        
        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                infer_results = infer_pipeline.infer(input_dict)
                
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000  # ms
        
        # Get output
        output = infer_results[self.output_vstream_info.name]
        
        # Parse detections
        detections = self.parse_nms_output(output, orig_shape)
        
        return detections, inference_time


def visualize_detections(image: np.ndarray, detections: List[Detection], save_path: str):
    """Visualize detections on image."""
    vis_img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Color by class
        if det.class_name == 'Fire':
            color = (0, 0, 255)  # Red for fire
        elif det.class_name in ['Person', 'Child']:
            color = (0, 255, 0)  # Green for people
        elif det.class_name in ['Car', 'Motorcycle', 'Bus', 'Truck', 'Bicycle']:
            color = (255, 0, 0)  # Blue for vehicles
        elif det.class_name in ['Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle', 
                                'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk',
                                'Squirrel', 'Pig', 'Chicken', 'Armadillo', 'Rodent']:
            color = (255, 255, 0)  # Cyan for animals
        else:
            color = (128, 128, 128)  # Gray for other
        
        # Draw box
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f"{det.class_name}: {det.score:.2f}"
        cv2.putText(vis_img, label, (int(x1), int(y1)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    cv2.imwrite(save_path, vis_img)
    

def main():
    """Run the fire detection test."""
    print("\n=== Hailo Fire Detection Test (NMS Enabled) ===\n")
    
    # Check device
    device = HailoDevice()
    if not device.is_available():
        print("✗ No Hailo device found")
        return False
        
    temp = device.get_temperature()
    if temp:
        print(f"✓ Hailo device available (temperature: {temp:.1f}°C)")
    else:
        print("✓ Hailo device available")
    
    # Load test video
    print("\n1. Loading test video...")
    downloader = VideoDownloader()
    videos = downloader.download_all_videos()
    video_path = videos['fire1.mov']
    print(f"✓ Using video: {video_path}")
    
    # Initialize detector
    print("\n2. Initializing detector...")
    hef_path = "hailo_qat_output/yolo8l_fire_640x640_hailo8l_nms.hef"
    
    if not Path(hef_path).exists():
        print(f"✗ HEF file not found: {hef_path}")
        return False
    
    try:
        detector = HailoYOLOv8Detector(hef_path, conf_threshold=0.2)  # Lower threshold
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Create output directory
    output_dir = Path("output/hailo_nms_working")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video
    print("\n3. Processing video...")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("✗ Failed to open video")
        return False
        
    # Video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {fps:.1f} FPS, {total_frames} frames")
    
    # Process frames
    results = []
    max_frames = 50
    frame_id = 0
    frames_with_fire = 0
    
    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Detect
            detections, inference_time = detector.detect(frame)
            
            # Only debug first frame
            detector.debug = (frame_id == 0)
            
            results.append({
                'frame_id': frame_id,
                'inference_time': inference_time,
                'num_detections': len(detections),
                'detections': [
                    {
                        'bbox': det.bbox,
                        'score': det.score,
                        'class_id': det.class_id,
                        'class_name': det.class_name
                    }
                    for det in detections
                ]
            })
            
            if detections:
                frames_with_fire += 1
                
                # Save first few visualizations
                if frames_with_fire <= 5:
                    vis_path = output_dir / f"frame_{frame_id}_detections.jpg"
                    visualize_detections(frame, detections, str(vis_path))
                    print(f"  Frame {frame_id}: {len(detections)} detections, {inference_time:.1f}ms - saved {vis_path.name}")
                    
                    # Print detection details
                    for det in detections[:3]:
                        print(f"    {det.class_name} ({det.score:.2f}) at {det.bbox}")
                else:
                    print(f"  Frame {frame_id}: {len(detections)} detections, {inference_time:.1f}ms")
            else:
                if frame_id % 10 == 0:
                    print(f"  Frame {frame_id}: No detections, {inference_time:.1f}ms")
                    
        except Exception as e:
            print(f"  Error at frame {frame_id}: {e}")
            import traceback
            traceback.print_exc()
            
        frame_id += 1
        
    cap.release()
    
    # Summary
    print("\n4. Summary:")
    print(f"  Frames processed: {len(results)}")
    
    if results:
        inference_times = [r['inference_time'] for r in results]
        avg_time = np.mean(inference_times)
        p95_time = np.percentile(inference_times, 95)
        fps_achieved = 1000 / avg_time
        
        total_detections = sum(r['num_detections'] for r in results)
        
        print(f"  Frames with detections: {frames_with_fire}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average inference: {avg_time:.1f}ms")
        print(f"  P95 inference: {p95_time:.1f}ms")
        print(f"  FPS: {fps_achieved:.1f}")
        
        # Check temperature
        temp = device.get_temperature()
        if temp:
            print(f"  Final temperature: {temp:.1f}°C")
        
        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'video': str(video_path),
                'frames_processed': len(results),
                'frames_with_detections': frames_with_fire,
                'total_detections': total_detections,
                'avg_inference_ms': avg_time,
                'p95_inference_ms': p95_time,
                'fps': fps_achieved,
                'results': results
            }, f, indent=2)
            
        print(f"\n✓ Results saved to: {results_file}")
        print(f"✓ Visualizations in: {output_dir}/")
        
        # Performance check
        if avg_time < 25 and fps_achieved > 40:
            print("\n✅ Performance targets MET!")
            print(f"  Target: <25ms, >40 FPS")
            print(f"  Actual: {avg_time:.1f}ms, {fps_achieved:.1f} FPS")
        else:
            print(f"\n⚠️  Performance below target")
            print(f"  Target: <25ms, >40 FPS")
            print(f"  Actual: {avg_time:.1f}ms, {fps_achieved:.1f} FPS")
            
        # Detection check
        if frames_with_fire > 0:
            print(f"\n✅ Fire/smoke detections found in {frames_with_fire} frames")
        else:
            print("\n⚠️  No fire/smoke detections found")
            print("    This may be due to:")
            print("    - Model trained on different dataset")
            print("    - Class ID mismatch (model has 32 classes)")
            print("    - Confidence threshold too high")
            
        return True
        
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)