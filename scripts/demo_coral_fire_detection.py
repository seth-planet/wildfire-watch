#!/usr/bin/env python3.8
"""
Demo: Coral TPU Fire Detection with Real-time Camera Feed
Shows how to use Coral TPU for fire detection in production
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
import argparse

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters import common


class CoralFireDetector:
    """Fire detector using Coral TPU"""
    
    def __init__(self, model_path, device_id=0):
        """Initialize Coral TPU fire detector"""
        self.model_path = model_path
        self.device_id = device_id
        
        # Check available TPUs
        tpus = list_edge_tpus()
        print(f"Found {len(tpus)} Coral TPUs")
        
        if device_id >= len(tpus):
            raise ValueError(f"TPU {device_id} not found. Only {len(tpus)} TPUs available.")
        
        # Initialize interpreter
        # Get actual device path from TPU list
        tpu_info = tpus[device_id]
        device_path = tpu_info['path']
        print(f"Using TPU {device_id}: {device_path}")
        
        # Use pci:<index> format for device specification
        device_spec = f'pci:{device_id}'
        self.interpreter = make_interpreter(model_path, device=device_spec)
        self.interpreter.allocate_tensors()
        
        # Get model info
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.height, self.width = self.input_details[0]['shape'][1:3]
        print(f"Model input size: {self.width}x{self.height}")
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
    def detect(self, frame):
        """Run fire detection on a frame"""
        # Preprocess
        resized = cv2.resize(frame, (self.width, self.height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Inference
        start = time.perf_counter()
        common.set_input(self.interpreter, rgb_frame)
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output, inference_time
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        recent = self.inference_times[-100:]  # Last 100 frames
        return {
            'avg_ms': np.mean(recent),
            'min_ms': np.min(recent),
            'max_ms': np.max(recent),
            'total_frames': self.frame_count,
            'fps': 1000.0 / np.mean(recent) if recent else 0
        }


class MultiTPUDetector:
    """Fire detector using multiple Coral TPUs for load balancing"""
    
    def __init__(self, model_path, num_tpus=None):
        """Initialize multi-TPU detector"""
        tpus = list_edge_tpus()
        self.num_tpus = min(num_tpus or len(tpus), len(tpus))
        
        print(f"Initializing {self.num_tpus} TPUs for parallel detection")
        
        # Create detector for each TPU
        self.detectors = []
        for i in range(self.num_tpus):
            detector = CoralFireDetector(model_path, device_id=i)
            self.detectors.append(detector)
        
        # Round-robin counter
        self.current_tpu = 0
        self.lock = Lock()
        
    def detect(self, frame):
        """Run detection using next available TPU"""
        with self.lock:
            detector = self.detectors[self.current_tpu]
            self.current_tpu = (self.current_tpu + 1) % self.num_tpus
        
        return detector.detect(frame)
    
    def get_all_stats(self):
        """Get stats from all TPUs"""
        stats = {}
        for i, detector in enumerate(self.detectors):
            stats[f'tpu_{i}'] = detector.get_stats()
        
        # Calculate combined stats
        all_times = []
        total_frames = 0
        for detector in self.detectors:
            all_times.extend(detector.inference_times[-100:])
            total_frames += detector.frame_count
        
        if all_times:
            stats['combined'] = {
                'avg_ms': np.mean(all_times),
                'total_frames': total_frames,
                'fps': len(self.detectors) * 1000.0 / np.mean(all_times)
            }
        
        return stats


def demo_camera_detection(detector, camera_source=0):
    """Demo fire detection with camera feed"""
    print(f"\nStarting camera detection demo...")
    print(f"Camera source: {camera_source}")
    print("Press 'q' to quit, 's' for stats")
    
    # Open camera
    if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
        # Network camera
        cap = cv2.VideoCapture(camera_source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # USB camera
        cap = cv2.VideoCapture(int(camera_source))
    
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {width}x{height} @ {fps:.1f} FPS")
    
    # Main loop
    frame_time = time.time()
    show_stats = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Run detection
        output, inference_time = detector.detect(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - frame_time) if current_time != frame_time else 0
        frame_time = current_time
        
        # Draw info on frame
        info_text = f"Coral TPU | Inference: {inference_time:.1f}ms | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show stats if enabled
        if show_stats:
            if isinstance(detector, MultiTPUDetector):
                stats = detector.get_all_stats()
                y_pos = 60
                for tpu_id, tpu_stats in stats.items():
                    if tpu_stats:
                        stat_text = f"{tpu_id}: {tpu_stats.get('avg_ms', 0):.1f}ms, {tpu_stats.get('total_frames', 0)} frames"
                        cv2.putText(frame, stat_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_pos += 25
            else:
                stats = detector.get_stats()
                if stats:
                    stat_text = f"Avg: {stats['avg_ms']:.1f}ms | Min: {stats['min_ms']:.1f}ms | Max: {stats['max_ms']:.1f}ms"
                    cv2.putText(frame, stat_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display
        cv2.imshow('Coral TPU Fire Detection', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    print("\nFinal Statistics:")
    if isinstance(detector, MultiTPUDetector):
        stats = detector.get_all_stats()
        for tpu_id, tpu_stats in stats.items():
            if tpu_stats:
                print(f"{tpu_id}: {tpu_stats}")
    else:
        stats = detector.get_stats()
        print(f"Single TPU: {stats}")


def demo_benchmark(detector, num_frames=1000):
    """Benchmark Coral TPU performance"""
    print(f"\nRunning benchmark with {num_frames} frames...")
    
    # Create test frames of different sizes
    test_sizes = [(640, 480), (1280, 720), (1920, 1080)]
    
    for size in test_sizes:
        print(f"\nTesting {size[0]}x{size[1]} frames:")
        
        # Reset stats
        if isinstance(detector, MultiTPUDetector):
            for d in detector.detectors:
                d.inference_times = []
                d.frame_count = 0
        else:
            detector.inference_times = []
            detector.frame_count = 0
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(10):
            detector.detect(test_frame)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_frames):
            detector.detect(test_frame)
        
        total_time = time.time() - start_time
        
        # Results
        if isinstance(detector, MultiTPUDetector):
            stats = detector.get_all_stats()
            combined = stats.get('combined', {})
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average inference: {combined.get('avg_ms', 0):.2f}ms")
            print(f"  Throughput: {num_frames / total_time:.1f} FPS")
            print(f"  Theoretical max FPS: {combined.get('fps', 0):.1f}")
        else:
            stats = detector.get_stats()
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average inference: {stats['avg_ms']:.2f}ms")
            print(f"  Throughput: {num_frames / total_time:.1f} FPS")
            print(f"  Theoretical max FPS: {stats['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Coral TPU Fire Detection Demo')
    parser.add_argument('--model', default='converted_models/yolov8n_320_edgetpu.tflite',
                        help='Path to Edge TPU model')
    parser.add_argument('--camera', default='0', help='Camera source (0 for USB, or RTSP URL)')
    parser.add_argument('--multi-tpu', action='store_true', help='Use multiple TPUs')
    parser.add_argument('--num-tpus', type=int, help='Number of TPUs to use')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark instead of camera')
    
    args = parser.parse_args()
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return 1
    
    # Create detector
    if args.multi_tpu:
        detector = MultiTPUDetector(args.model, args.num_tpus)
    else:
        detector = CoralFireDetector(args.model)
    
    # Run demo
    if args.benchmark:
        demo_benchmark(detector)
    else:
        # Parse camera source
        camera = args.camera
        if camera.isdigit():
            camera = int(camera)
        
        demo_camera_detection(detector, camera)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())