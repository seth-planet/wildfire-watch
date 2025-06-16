#!/usr/bin/env python3.12
"""
Docker-based integration tests for Wildfire Watch
Tests services running in Docker containers
"""
import os
import sys
import time
import json
import pytest
import subprocess
import docker
import paho.mqtt.client as mqtt
from typing import Dict, List, Optional

class DockerIntegrationTest:
    """Test integration with Docker containers"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.containers = {}
        self.test_passed = False
        
    def build_docker_images(self):
        """Build required Docker images"""
        print("Building Docker images...")
        
        services = ['mqtt_broker', 'fire_consensus', 'gpio_trigger']
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for service in services:
            dockerfile_path = os.path.join(project_root, service)
            if os.path.exists(os.path.join(dockerfile_path, 'Dockerfile')):
                print(f"Building {service}...")
                try:
                    # Build using docker-compose
                    result = subprocess.run([
                        'docker-compose', 'build', service
                    ], cwd=project_root, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"✓ Built {service}")
                    else:
                        print(f"Failed to build {service}: {result.stderr}")
                        # Try direct docker build as fallback
                        self.docker_client.images.build(
                            path=dockerfile_path,
                            tag=f"wildfire-watch/{service}:test",
                            rm=True
                        )
                        print(f"✓ Built {service} with docker build")
                except Exception as e:
                    print(f"Error building {service}: {e}")
                    
    def start_mqtt_container(self):
        """Start MQTT broker container"""
        print("Starting MQTT broker container...")
        
        # Check if mosquitto image exists
        try:
            self.docker_client.images.get("eclipse-mosquitto:latest")
        except docker.errors.ImageNotFound:
            print("Pulling mosquitto image...")
            self.docker_client.images.pull("eclipse-mosquitto:latest")
        
        # Remove existing test container if any
        try:
            old_container = self.docker_client.containers.get("mqtt-test")
            old_container.stop()
            old_container.remove()
        except docker.errors.NotFound:
            pass
            
        # Create and start container
        container = self.docker_client.containers.run(
            "eclipse-mosquitto:latest",
            name="mqtt-test",
            ports={'1883/tcp': 11883},
            detach=True,
            remove=False
        )
        
        self.containers['mqtt'] = container
        
        # Wait for MQTT to be ready
        time.sleep(3)
        
        # Test connection
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test")
            client.connect("localhost", 11883, 60)
            client.disconnect()
            print("✓ MQTT broker ready")
            return True
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            return False
            
    def start_consensus_container(self):
        """Start fire consensus container"""
        print("Starting fire consensus container...")
        
        # Build custom consensus image
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        consensus_path = os.path.join(project_root, 'fire_consensus')
        
        try:
            # Build image
            image, _ = self.docker_client.images.build(
                path=consensus_path,
                tag="wildfire-consensus:test",
                rm=True
            )
            
            # Remove old container
            try:
                old = self.docker_client.containers.get("consensus-test")
                old.stop()
                old.remove()
            except docker.errors.NotFound:
                pass
            
            # Start container
            container = self.docker_client.containers.run(
                "wildfire-consensus:test",
                name="consensus-test",
                environment={
                    'MQTT_BROKER': 'host.docker.internal',
                    'MQTT_PORT': '11883',
                    'CONSENSUS_THRESHOLD': '1',
                    'SINGLE_CAMERA_TRIGGER': 'true',
                    'MIN_CONFIDENCE': '0.7',
                    'LOG_LEVEL': 'DEBUG'
                },
                extra_hosts={'host.docker.internal': 'host-gateway'},
                detach=True,
                remove=False
            )
            
            self.containers['consensus'] = container
            print("✓ Fire consensus container started")
            return True
            
        except Exception as e:
            print(f"Failed to start consensus container: {e}")
            return False
            
    def test_docker_fire_detection_flow(self):
        """Test fire detection flow with Docker containers"""
        print("\n=== DOCKER INTEGRATION TEST ===")
        
        # 1. Build images
        self.build_docker_images()
        
        # 2. Start MQTT
        if not self.start_mqtt_container():
            return False
            
        # 3. Start consensus
        if not self.start_consensus_container():
            return False
            
        # 4. Wait for services to connect
        print("Waiting for services to initialize...")
        time.sleep(5)
        
        # 5. Monitor for fire trigger
        triggered = False
        def on_message(client, userdata, msg):
            nonlocal triggered
            if msg.topic == "fire/trigger":
                print(f"✓ Fire trigger received!")
                triggered = True
                
        monitor = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "monitor")
        monitor.on_message = on_message
        monitor.connect("localhost", 11883)
        monitor.subscribe("fire/trigger", qos=1)
        monitor.loop_start()
        
        # 6. Inject fire detections
        print("Injecting fire detections...")
        publisher = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "publisher")
        publisher.connect("localhost", 11883)
        
        for i in range(8):
            detection = {
                'camera_id': 'docker_test_cam',
                'object': 'fire',
                'object_id': 'fire_1',
                'confidence': 0.75 + i * 0.02,
                'bounding_box': [0.1, 0.1, 0.04 + i * 0.01, 0.04 + i * 0.008],
                'timestamp': time.time() + i * 0.5
            }
            publisher.publish('fire/detection', json.dumps(detection), qos=1)
            print(f"  Sent detection {i+1}")
            time.sleep(0.5)
            
        # 7. Wait and check
        print("Waiting for consensus...")
        time.sleep(10)
        
        # Check consensus container logs
        if 'consensus' in self.containers:
            logs = self.containers['consensus'].logs(tail=20).decode()
            print("\nConsensus container logs:")
            print(logs)
        
        # Cleanup
        monitor.loop_stop()
        monitor.disconnect()
        publisher.disconnect()
        
        self.test_passed = triggered
        return triggered
        
    def cleanup(self):
        """Clean up containers"""
        print("\nCleaning up containers...")
        for name, container in self.containers.items():
            try:
                container.stop()
                container.remove()
                print(f"✓ Removed {name} container")
            except Exception as e:
                print(f"Error removing {name}: {e}")


def test_docker_integration():
    """Test Docker container integration"""
    test = DockerIntegrationTest()
    
    try:
        success = test.test_docker_fire_detection_flow()
        assert success, "Docker integration test should trigger fire consensus"
        print("\n✅ DOCKER INTEGRATION TEST PASSED")
    finally:
        test.cleanup()
        

if __name__ == "__main__":
    test_docker_integration()