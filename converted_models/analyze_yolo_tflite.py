#!/usr/bin/env python3
"""Analyze YOLO TFLite model structure to understand input/output format."""

import tensorflow as tf
import numpy as np

def analyze_tflite_model(model_path):
    """Analyze TFLite model to understand its structure."""
    # Load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("=== YOLO TFLite Model Analysis ===")
    print(f"\nModel: {model_path}")
    
    print("\n--- Input Details ---")
    for i, input_detail in enumerate(input_details):
        print(f"\nInput {i}:")
        print(f"  Name: {input_detail['name']}")
        print(f"  Shape: {input_detail['shape']}")
        print(f"  Type: {input_detail['dtype']}")
        print(f"  Quantization: {input_detail['quantization']}")
        print(f"  Index: {input_detail['index']}")
    
    print("\n--- Output Details ---")
    for i, output_detail in enumerate(output_details):
        print(f"\nOutput {i}:")
        print(f"  Name: {output_detail['name']}")
        print(f"  Shape: {output_detail['shape']}")
        print(f"  Type: {output_detail['dtype']}")
        print(f"  Quantization: {output_detail['quantization']}")
        print(f"  Index: {output_detail['index']}")
    
    # Run inference with dummy data to see actual output shape
    print("\n--- Test Inference ---")
    input_shape = input_details[0]['shape']
    input_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Actual output shape: {output_data.shape}")
    print(f"Output data type: {output_data.dtype}")
    print(f"Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")
    
    # Check if this is YOLO format
    if len(output_data.shape) == 3:
        batch_size, num_predictions, pred_size = output_data.shape
        print(f"\nDetected YOLO format:")
        print(f"  Batch size: {batch_size}")
        print(f"  Number of predictions: {num_predictions}")
        print(f"  Prediction size: {pred_size}")
        
        if pred_size >= 5:
            print(f"  Format appears to be: [x, y, w, h, confidence, class_scores...]")
            print(f"  Number of classes: {pred_size - 5}")

if __name__ == "__main__":
    model_path = "/home/seth/wildfire-watch/converted_models/frigate_models/yolo8l_fire_640x640_frigate.tflite"
    analyze_tflite_model(model_path)