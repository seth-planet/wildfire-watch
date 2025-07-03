#!/usr/bin/env python3.12
"""Debug consensus detection logic"""
import sys
import os
sys.path.insert(0, 'fire_consensus')

# Mock MQTT to prevent connection issues
class MockMQTTClient:
    def __init__(self, *args, **kwargs):
        pass
    
    def on_connect(self, *args):
        pass
    
    def on_disconnect(self, *args):
        pass
    
    def on_message(self, *args):
        pass
    
    def tls_set(self, *args, **kwargs):
        pass
    
    def will_set(self, *args, **kwargs):
        pass
    
    def connect(self, *args, **kwargs):
        pass
    
    def loop_start(self):
        pass
    
    def subscribe(self, *args, **kwargs):
        pass
    
    def publish(self, *args, **kwargs):
        pass

# Patch MQTT before importing consensus
import paho.mqtt.client as mqtt
mqtt.Client = MockMQTTClient

from consensus import FireConsensus, Detection

def test_area_calculation():
    consensus = FireConsensus()
    
    # Test different bounding box formats
    test_cases = [
        ([100, 100, 50, 40], "width/height format"),  # area = 0.002
        ([100, 100, 150, 140], "x1,y1,x2,y2 format"),  # area depends on interpretation
        ([0.1, 0.1, 0.05, 0.04], "normalized format"),  # area = 0.002
    ]
    
    for bbox, description in test_cases:
        area = consensus._calculate_area(bbox)
        valid = consensus._validate_detection(0.8, area)
        print(f"{description}: {bbox} -> area={area:.6f}, valid={valid}")
        print(f"  Min: {consensus.config.MIN_AREA_RATIO}, Max: {consensus.config.MAX_AREA_RATIO}")

def test_detection_processing():
    consensus = FireConsensus()
    
    # Test detection with proper format
    detection_data = {
        'camera_id': 'cam1',
        'object': 'fire',
        'object_id': 'fire1',
        'confidence': 0.8,
        'bounding_box': [0.1, 0.1, 0.05, 0.04],  # Normalized format
        'timestamp': 1234567890
    }
    
    # Process detection manually
    camera_id = detection_data['camera_id']
    confidence = detection_data['confidence']
    bbox = detection_data['bounding_box']
    timestamp = detection_data['timestamp']
    object_id = detection_data['object_id']
    
    area = consensus._calculate_area(bbox)
    valid = consensus._validate_detection(confidence, area)
    
    print(f"\nDetection processing:")
    print(f"Camera: {camera_id}")
    print(f"Confidence: {confidence}")
    print(f"Bbox: {bbox}")
    print(f"Area: {area:.6f}")
    print(f"Valid: {valid}")
    
    if valid:
        detection = Detection(
            camera_id=camera_id,
            timestamp=timestamp,
            confidence=confidence,
            area=area,
            bbox=bbox,
            object_id=object_id
        )
        print(f"Detection object created: {detection.to_dict()}")

if __name__ == "__main__":
    test_area_calculation()
    test_detection_processing()