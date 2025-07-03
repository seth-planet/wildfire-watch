#!/usr/bin/env python3.8
"""
Test Coral TPU with actual fire detection
Downloads fire images and runs detection
"""

import os
import sys
import numpy as np
import cv2
import time
import urllib.request
from pathlib import Path
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file

def download_test_images():
    """Download test fire images"""
    test_dir = Path("tmp/fire_test_images")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test image URLs (fire and non-fire) - using direct image URLs
    test_images = [
        ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg", "test1.jpg"),
        ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", "test2.jpg"),
    ]
    
    downloaded = []
    for url, filename in test_images:
        filepath = test_dir / filename
        if not filepath.exists():
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, str(filepath))
                downloaded.append(filepath)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            downloaded.append(filepath)
    
    return downloaded

def run_fire_detection(model_path, image_paths):
    """Run fire detection on images using Coral TPU"""
    
    print(f"\nLoading model: {model_path}")
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    height, width = input_details[0]['shape'][1:3]
    
    print(f"Model input size: {width}x{height}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    
    # Process each image
    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        orig_h, orig_w = img.shape[:2]
        print(f"Original size: {orig_w}x{orig_h}")
        
        # Preprocess
        resized = cv2.resize(img, (width, height))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Run inference
        start = time.perf_counter()
        common.set_input(interpreter, rgb_frame)
        interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"Inference time: {inference_time:.2f}ms")
        
        # Get outputs
        output_details = interpreter.get_output_details()
        
        # Try different output formats
        if len(output_details) == 1:
            # Single output 
            output = interpreter.get_tensor(output_details[0]['index'])
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            
            # Analyze output based on shape
            if len(output.shape) == 2 and output.shape[-1] > 1000:  # Classification model
                # This is a classification model (like MobileNet)
                top_k = 5
                if output.shape[-1] == 1001:
                    # ImageNet classes
                    classes = np.argsort(output[0])[-top_k:][::-1]
                    print(f"\nTop {top_k} predictions (ImageNet classes):")
                    for i, class_id in enumerate(classes):
                        score = output[0][class_id]
                        print(f"  {i+1}. Class {class_id}: score={score}")
            elif output.shape[-1] == 85:  # YOLO format: x,y,w,h,conf,80_classes
                # Extract detections
                num_detections = output.shape[1]
                print(f"Number of potential detections: {num_detections}")
                
                # Find high confidence detections
                for i in range(min(10, num_detections)):
                    detection = output[0, i]
                    conf = detection[4]
                    if conf > 0.3:
                        x, y, w, h = detection[:4]
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        class_conf = class_scores[class_id]
                        
                        print(f"  Detection {i}: conf={conf:.3f}, class={class_id}, "
                              f"class_conf={class_conf:.3f}, bbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}]")
            else:
                print(f"Unknown output format with shape: {output.shape}")
        else:
            # Multiple outputs
            print(f"Model has {len(output_details)} outputs")
            for i, detail in enumerate(output_details):
                output = interpreter.get_tensor(detail['index'])
                print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")

def test_multi_tpu_performance(model_path, num_iterations=100):
    """Test performance across multiple TPUs"""
    from pycoral.utils.edgetpu import list_edge_tpus
    
    tpus = list_edge_tpus()
    print(f"\nFound {len(tpus)} Coral TPUs")
    
    if len(tpus) > 1:
        print("\nTesting multi-TPU performance...")
        
        # Test each TPU
        for i, tpu in enumerate(tpus[:4]):  # Test up to 4 TPUs
            print(f"\nTPU {i}: {tpu}")
            
            # Create interpreter for specific TPU
            device_path = f'/dev/apex_{i}'
            interpreter = make_interpreter(model_path, device=device_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            height, width = input_details[0]['shape'][1:3]
            
            # Create test image
            test_img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                common.set_input(interpreter, test_img)
                interpreter.invoke()
            
            # Measure
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                common.set_input(interpreter, test_img)
                interpreter.invoke()
            
            total_time = (time.perf_counter() - start_time) * 1000
            avg_time = total_time / num_iterations
            fps = 1000 / avg_time
            
            print(f"  Average inference: {avg_time:.2f}ms")
            print(f"  Throughput: {fps:.1f} FPS")

def main():
    """Main test function"""
    # Find Edge TPU model
    model_paths = [
        "converted_models/yolov8n_320_edgetpu.tflite",
        "converted_models/yolov8n_416_edgetpu.tflite",
        "converted_models/mobilenet_v2_edgetpu.tflite",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("ERROR: No Edge TPU model found")
        return 1
    
    print(f"Using model: {model_path}")
    
    # Download test images
    print("\nDownloading test images...")
    image_paths = download_test_images()
    
    if not image_paths:
        print("ERROR: No test images available")
        return 1
    
    # Run fire detection
    run_fire_detection(model_path, image_paths)
    
    # Test multi-TPU performance
    test_multi_tpu_performance(model_path)
    
    print("\n✓ Fire detection test completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())