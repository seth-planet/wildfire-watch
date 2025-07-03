#!/usr/bin/env python3.10
"""
Demo script showing Hailo fire detection via Frigate integration.

This demonstrates the recommended way to use Hailo for fire detection
through the Frigate NVR integration rather than direct API calls.
"""

import time
import json
import argparse
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

class FireDetectionMonitor:
    """Monitor fire detections from Frigate via MQTT."""
    
    def __init__(self, mqtt_host="localhost", mqtt_port=1883):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.detections = defaultdict(list)
        self.camera_stats = defaultdict(lambda: {"fire_count": 0, "smoke_count": 0})
        self.start_time = time.time()
        
        # MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            print(f"✓ Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            
            # Subscribe to relevant topics
            topics = [
                ("frigate/+/fire", 0),
                ("frigate/+/smoke", 0),
                ("frigate/stats", 0),
                ("telemetry/inference_metrics", 0),
                ("trigger/fire_detected", 0)
            ]
            
            for topic, qos in topics:
                client.subscribe(topic, qos)
                print(f"  Subscribed to: {topic}")
        else:
            print(f"✗ Failed to connect to MQTT broker (code: {rc})")
            
    def on_message(self, client, userdata, msg):
        """MQTT message callback."""
        topic = msg.topic
        
        try:
            # Decode payload
            if msg.payload:
                payload = json.loads(msg.payload.decode())
            else:
                payload = {}
                
            # Handle different message types
            if "frigate/" in topic and "/fire" in topic:
                self.handle_fire_detection(topic, payload)
            elif "frigate/" in topic and "/smoke" in topic:
                self.handle_smoke_detection(topic, payload)
            elif topic == "frigate/stats":
                self.handle_stats(payload)
            elif topic == "telemetry/inference_metrics":
                self.handle_metrics(payload)
            elif topic == "trigger/fire_detected":
                self.handle_trigger(payload)
                
        except Exception as e:
            print(f"Error processing message on {topic}: {e}")
            
    def handle_fire_detection(self, topic, payload):
        """Handle fire detection event."""
        camera = topic.split('/')[1]
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n🔥 FIRE DETECTED - Camera: {camera} at {timestamp}")
        print(f"   Confidence: {payload.get('score', 0):.2%}")
        print(f"   Location: {payload.get('box', 'unknown')}")
        
        self.camera_stats[camera]["fire_count"] += 1
        self.detections[camera].append({
            "type": "fire",
            "timestamp": timestamp,
            "confidence": payload.get('score', 0)
        })
        
    def handle_smoke_detection(self, topic, payload):
        """Handle smoke detection event."""
        camera = topic.split('/')[1]
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n💨 SMOKE DETECTED - Camera: {camera} at {timestamp}")
        print(f"   Confidence: {payload.get('score', 0):.2%}")
        
        self.camera_stats[camera]["smoke_count"] += 1
        self.detections[camera].append({
            "type": "smoke",
            "timestamp": timestamp,
            "confidence": payload.get('score', 0)
        })
        
    def handle_stats(self, payload):
        """Handle Frigate stats update."""
        if 'detectors' in payload and 'hailo8l' in payload['detectors']:
            hailo_stats = payload['detectors']['hailo8l']
            
            # Only print if significant
            if hailo_stats.get('detection_fps', 0) > 0:
                print(f"\n📊 Hailo Performance:")
                print(f"   Inference: {hailo_stats.get('inference_speed', 0):.1f}ms")
                print(f"   FPS: {hailo_stats.get('detection_fps', 0):.1f}")
                
    def handle_metrics(self, payload):
        """Handle telemetry metrics."""
        if payload.get('detector') == 'hailo8l':
            print(f"\n📈 Hailo Metrics:")
            print(f"   Temperature: {payload.get('temperature', 0):.1f}°C")
            print(f"   Utilization: {payload.get('utilization', 0):.1f}%")
            
    def handle_trigger(self, payload):
        """Handle fire suppression trigger."""
        print(f"\n🚨 FIRE SUPPRESSION TRIGGERED!")
        print(f"   Cameras: {payload.get('cameras', [])}")
        print(f"   Confidence: {payload.get('confidence', 0):.2%}")
        print(f"   Action: {payload.get('action', 'unknown')}")
        
    def print_summary(self):
        """Print detection summary."""
        runtime = int(time.time() - self.start_time)
        
        print("\n" + "="*50)
        print("Detection Summary")
        print("="*50)
        print(f"Runtime: {runtime} seconds")
        print(f"\nPer Camera:")
        
        total_fire = 0
        total_smoke = 0
        
        for camera, stats in self.camera_stats.items():
            fire_count = stats["fire_count"]
            smoke_count = stats["smoke_count"]
            total_fire += fire_count
            total_smoke += smoke_count
            
            if fire_count > 0 or smoke_count > 0:
                print(f"\n{camera}:")
                print(f"  🔥 Fire detections: {fire_count}")
                print(f"  💨 Smoke detections: {smoke_count}")
                
                # Recent detections
                recent = self.detections[camera][-3:]
                if recent:
                    print(f"  Recent events:")
                    for det in recent:
                        print(f"    {det['timestamp']} - {det['type']} ({det['confidence']:.2%})")
                        
        print(f"\nTotal:")
        print(f"  🔥 Fire: {total_fire}")
        print(f"  💨 Smoke: {total_smoke}")
        
    def run(self, duration=None):
        """Run the monitor."""
        print("\n🔍 Wildfire Detection Monitor (Hailo-8L)")
        print("="*50)
        
        # Connect to MQTT
        try:
            self.client.connect(self.mqtt_host, self.mqtt_port, 60)
        except Exception as e:
            print(f"✗ Failed to connect to MQTT broker: {e}")
            print("\nMake sure MQTT broker is running:")
            print("  docker-compose up mqtt-broker")
            return
            
        # Start monitoring
        print("\nMonitoring for fire/smoke detections...")
        print("Press Ctrl+C to stop\n")
        
        try:
            if duration:
                # Run for specified duration
                self.client.loop_start()
                time.sleep(duration)
                self.client.loop_stop()
            else:
                # Run until interrupted
                self.client.loop_forever()
        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
        finally:
            self.print_summary()
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Hailo-based fire detection via Frigate"
    )
    parser.add_argument(
        "--host", 
        default="localhost",
        help="MQTT broker host (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Run for specified seconds (default: run forever)"
    )
    
    args = parser.parse_args()
    
    # Create and run monitor
    monitor = FireDetectionMonitor(args.host, args.port)
    monitor.run(args.duration)


if __name__ == "__main__":
    main()