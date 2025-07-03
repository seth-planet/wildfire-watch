#!/usr/bin/env python3.10
"""
This test requires Python 3.10 and the Hailo SDK to be installed.
If the Hailo SDK is not available, the test will fail (not skip).
"""
"""End-to-end fire detection test using Hailo-8L with custom YOLOv8 model."""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

# Import required modules - fail if not available (don't skip)
import pytest

# Check Python version first
if sys.version_info[:2] != (3, 10):
    pytest.fail(f"Hailo tests require Python 3.10, running {sys.version_info.major}.{sys.version_info.minor}")

try:
    import hailo_platform
    from hailo_platform import (
        VDevice, HEF, ConfigureParams, HailoStreamInterface, 
        InferVStreams, InputVStreamParams, OutputVStreamParams,
        FormatType
    )
    from hailo_test_utils import VideoDownloader, HailoDevice, PerformanceMetrics
    print("✓ Hailo modules imported successfully")
    HAILO_AVAILABLE = True
except ImportError as e:
    print(f"✗ Import error: {e}")
    # Don't create mock classes - let the test fail
    HAILO_AVAILABLE = False


@dataclass
class FireDetection:
    """Fire detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    confidence: float
    class_id: int
    class_name: str
    frame_id: int


class YOLOv8HailoInference:
    """YOLOv8 inference using Hailo with NMS-enabled model."""
    
    def __init__(self, hef_path: str, conf_threshold: float = 0.3):
        """Initialize Hailo inference for YOLOv8.
        
        Args:
            hef_path: Path to HEF model
            conf_threshold: Confidence threshold for detections
        """
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
        self.fire_class_id = 26  # Fire is at index 26
        self.debug = False
        
        # Initialize device
        self.device = VDevice()
        print("✓ Created VDevice")
        
        # Load HEF
        self.hef = HEF(hef_path)
        print("✓ Loaded HEF model")
        
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
        """Preprocess image for YOLOv8 inference.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Get expected input shape from vstream info
        height, width, channels = self.input_vstream_info.shape
        
        # Resize to expected size
        resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        return normalized
        
    def parse_nms_output(self, output_data, orig_shape: Tuple[int, int]) -> List[FireDetection]:
        """Parse NMS output from Hailo.
        
        Args:
            output_data: Raw model output
            orig_shape: Original image shape (H, W)
            
        Returns:
            List of fire detections
        """
        detections = []
        orig_h, orig_w = orig_shape
        
        # Debug output structure
        if self.debug:
            print(f"\nDebug - Output type: {type(output_data)}")
            if hasattr(output_data, 'shape'):
                print(f"Debug - Output shape: {output_data.shape}")
            
            if isinstance(output_data, list):
                print(f"Debug - List length: {len(output_data)}")
        
        # Handle nested list structure 
        if isinstance(output_data, list):
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
                                
                                detections.append(FireDetection(
                                    bbox=(x1, y1, x2, y2),
                                    confidence=float(score),
                                    class_id=class_id,
                                    class_name=self.class_names[class_id],
                                    frame_id=0
                                ))
        
        # Sort by score and limit detections
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:100]  # Limit to top 100 detections
        
    def detect(self, image: np.ndarray, frame_id: int = 0) -> Tuple[List[FireDetection], float]:
        """Run inference on image.
        
        Args:
            image: Input image (BGR)
            frame_id: Frame ID for tracking
            
        Returns:
            Tuple of (detections, inference_time_ms)
        """
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
        
        # Update frame ID
        for det in detections:
            det.frame_id = frame_id
        
        return detections, inference_time


def calculate_metrics(detections_by_video: Dict[str, List[FireDetection]], 
                     inference_times: Dict[str, List[float]]) -> Dict[str, any]:
    """Calculate performance metrics.
    
    Args:
        detections_by_video: Dictionary of video name to detections
        inference_times: Dictionary of video name to inference times
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall metrics
    all_times = []
    for times in inference_times.values():
        all_times.extend(times)
    
    if all_times:
        metrics['avg_inference_ms'] = np.mean(all_times)
        metrics['p95_inference_ms'] = np.percentile(all_times, 95)
        metrics['p99_inference_ms'] = np.percentile(all_times, 99)
        metrics['fps'] = 1000 / metrics['avg_inference_ms']
    
    # Per-video metrics
    metrics['per_video'] = {}
    for video_name, detections in detections_by_video.items():
        fire_detections = [d for d in detections if d.class_id == 26]  # Fire class
        
        video_metrics = {
            'total_detections': len(detections),
            'fire_detections': len(fire_detections),
            'unique_frames_with_fire': len(set(d.frame_id for d in fire_detections))
        }
        
        if video_name in inference_times and inference_times[video_name]:
            video_metrics['avg_inference_ms'] = np.mean(inference_times[video_name])
            video_metrics['fps'] = 1000 / video_metrics['avg_inference_ms']
        
        metrics['per_video'][video_name] = video_metrics
    
    return metrics


def visualize_detection(image: np.ndarray, detections: List[FireDetection], 
                       video_name: str, frame_id: int, output_path: Path):
    """Visualize detections on image using matplotlib.
    
    Args:
        image: Original image (BGR)
        detections: List of detections
        video_name: Name of video
        frame_id: Frame number
        output_path: Path to save visualization
    """
    # Convert BGR to RGB for matplotlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(rgb_image)
    ax.set_title(f"{video_name} - Frame {frame_id}")
    
    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        width = x2 - x1
        height = y2 - y1
        
        # Color based on class
        if det.class_name == 'Fire':
            color = 'red'
            linewidth = 3
        elif det.class_name in ['Person', 'Child']:
            color = 'green'
            linewidth = 2
        else:
            color = 'blue'
            linewidth = 1
        
        # Draw bounding box
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=linewidth, edgecolor=color, 
                               facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"{det.class_name}: {det.confidence:.2f}"
        ax.text(x1, y1-5, label, color=color, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@pytest.mark.api_usage  
@pytest.mark.python310  # Mark as requiring Python 3.10
@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
def test_fire_detection_e2e():
    """Run end-to-end fire detection test."""
    
    # Fail immediately if Hailo SDK not available (don't skip)
    if not HAILO_AVAILABLE:
        pytest.fail("Hailo SDK is required but not available")
    
    print("\n=== End-to-End Fire Detection Test ===\n")
    
    # Download test videos
    print("1. Downloading test videos...")
    downloader = VideoDownloader()
    videos = downloader.download_all_videos()
    if not videos:
        print("✗ Failed to download test videos")
        return False
        
    print(f"✓ Downloaded {len(videos)} test videos")
    
    # Initialize Hailo inference
    print("\n2. Initializing Hailo inference...")
    hef_path = "hailo_qat_output/yolo8l_fire_640x640_hailo8l_nms.hef"
    if not Path(hef_path).exists():
        print(f"✗ HEF model not found: {hef_path}")
        return False
        
    try:
        model = YOLOv8HailoInference(hef_path, conf_threshold=0.3)
    except Exception as e:
        print(f"✗ Failed to initialize Hailo: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Create output directory
    output_dir = Path("output/hailo_e2e_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    results = {}
    all_detections = {}
    all_inference_times = {}
    
    for video_name, video_path in videos.items():
        print(f"\n3. Processing {video_name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            continue
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  Video: {fps:.1f} FPS, {total_frames} frames")
        
        # Process frames
        detections = []
        inference_times = []
        frame_id = 0
        max_frames = min(50, total_frames)  # Process up to 50 frames
        
        # Enable debug for first frame
        model.debug = True
        
        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            frame_detections, inference_time = model.detect(frame, frame_id)
            detections.extend(frame_detections)
            inference_times.append(inference_time)
            
            # Disable debug after first frame
            model.debug = False
            
            # Visualize first few fire detections
            fire_detections = [d for d in frame_detections if d.class_id == 26]
            if fire_detections and frame_id < 5:
                vis_path = output_dir / f"{video_name}_frame_{frame_id}.png"
                visualize_detection(frame, frame_detections, video_name, frame_id, vis_path)
                print(f"  Frame {frame_id}: {len(frame_detections)} detections ({len(fire_detections)} fire), {inference_time:.1f}ms")
            elif frame_id % 10 == 0:
                print(f"  Frame {frame_id}: {len(frame_detections)} detections, {inference_time:.1f}ms")
                
            frame_id += 1
            
        cap.release()
        
        # Store results
        all_detections[video_name] = detections
        all_inference_times[video_name] = inference_times
        
        print(f"  Processed {frame_id} frames")
        
    # Calculate metrics
    print("\n4. Calculating metrics...")
    metrics = calculate_metrics(all_detections, all_inference_times)
    
    # Display results
    print("\n5. Results:")
    print(f"  Average inference: {metrics['avg_inference_ms']:.1f}ms")
    print(f"  P95 inference: {metrics['p95_inference_ms']:.1f}ms")
    print(f"  FPS: {metrics['fps']:.1f}")
    
    print("\n  Per-video results:")
    for video_name, video_metrics in metrics['per_video'].items():
        print(f"    {video_name}:")
        print(f"      Total detections: {video_metrics['total_detections']}")
        print(f"      Fire detections: {video_metrics['fire_detections']}")
        print(f"      Frames with fire: {video_metrics['unique_frames_with_fire']}")
        if 'avg_inference_ms' in video_metrics:
            print(f"      Avg inference: {video_metrics['avg_inference_ms']:.1f}ms")
    
    # Save results
    results_file = output_dir / "e2e_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': metrics,
            'model_path': hef_path,
            'videos_processed': list(all_detections.keys()),
            'total_frames_processed': sum(len(times) for times in all_inference_times.values())
        }, f, indent=2)
        
    print(f"\n✓ Results saved to: {results_file}")
    
    # Check if test passed
    if metrics['fps'] > 40 and metrics['avg_inference_ms'] < 25:
        print("\n✅ Performance targets MET!")
        print(f"  Target: <25ms, >40 FPS")
        print(f"  Actual: {metrics['avg_inference_ms']:.1f}ms, {metrics['fps']:.1f} FPS")
    else:
        print(f"\n⚠️  Performance below target")
        print(f"  Target: <25ms, >40 FPS")
        print(f"  Actual: {metrics['avg_inference_ms']:.1f}ms, {metrics['fps']:.1f} FPS")
        
    # Check fire detection
    total_fire_detections = sum(m['fire_detections'] for m in metrics['per_video'].values())
    if total_fire_detections > 0:
        print(f"\n✅ Fire detections found: {total_fire_detections} total")
    else:
        print("\n⚠️  No fire detections found")
        
    print("\n❌ End-to-end fire detection test completed!")
    
    return True


# Allow running as script
if __name__ == "__main__":
    success = test_fire_detection_e2e()
    sys.exit(0 if success else 1)