#!/usr/bin/env python3.10
"""End-to-end integration test for Frigate with Hailo detector using Docker.

This test verifies the complete pipeline from video input through
Frigate NVR with Hailo detection to MQTT fire detection events.
"""

import os
import sys
import time
import json
import yaml
import pytest
import docker
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import paho.mqtt.client as mqtt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

try:
    from test_utils.hailo_test_utils import VideoDownloader, HailoDevice
except ImportError:
    print("Warning: hailo_test_utils not found")
    
    
class MQTTTestClient:
    """Test MQTT client for monitoring messages."""
    
    def __init__(self, broker="localhost", port=18833):
        self.broker = broker
        self.port = port
        self.messages = []
        self.client = None
        self.connected = False
        
    def connect(self):
        """Connect to MQTT broker."""
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "test_monitor")
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            start = time.time()
            while not self.connected and time.time() - start < timeout:
                time.sleep(0.1)
                
            return self.connected
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            return False
            
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle connection event."""
        if rc == 0:
            self.connected = True
            print("MQTT test client connected")
        else:
            print(f"MQTT connection failed with code: {rc}")
            
    def _on_message(self, client, userdata, message):
        """Handle incoming message."""
        try:
            payload = json.loads(message.payload.decode())
            self.messages.append({
                'topic': message.topic,
                'payload': payload,
                'timestamp': time.time()
            })
        except:
            self.messages.append({
                'topic': message.topic,
                'payload': message.payload.decode() if message.payload else "",
                'timestamp': time.time()
            })
            
    def subscribe(self, topics):
        """Subscribe to topics."""
        if isinstance(topics, str):
            topics = [topics]
        for topic in topics:
            self.client.subscribe(topic)
            print(f"Subscribed to: {topic}")
            
    def wait_for_message(self, topic_pattern, timeout=30):
        """Wait for a message matching topic pattern."""
        start = time.time()
        while time.time() - start < timeout:
            for msg in self.messages:
                if topic_pattern in msg['topic']:
                    return msg
            time.sleep(0.1)
        return None
        
    def clear_messages(self):
        """Clear message buffer."""
        self.messages = []
        
    def disconnect(self):
        """Disconnect from broker."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            

class DockerE2ETest:
    """Docker-based end-to-end test manager."""
    
    def __init__(self):
        try:
            # Try different ways to connect to Docker
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker connection issue: {e}")
            # Try with explicit socket
            self.docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        self.temp_dir = None
        self.containers = []
        
    def cleanup(self):
        """Clean up containers and resources."""
        print("Cleaning up test resources...")
        
        # Stop and remove test containers
        for container in self.containers:
            try:
                container.stop(timeout=5)
                container.remove()
                print(f"Removed container: {container.name}")
            except:
                pass
                
        # Clean up temp directory
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def start_mqtt_broker(self):
        """Start MQTT broker container."""
        print("Starting MQTT broker...")
        
        # Remove existing container if any
        try:
            existing = self.docker_client.containers.get("test-mqtt")
            existing.stop()
            existing.remove()
        except:
            pass
            
        # Start new container
        container = self.docker_client.containers.run(
            "eclipse-mosquitto:2.0",
            name="test-mqtt",
            ports={'1883/tcp': None},
            detach=True,
            remove=False,
            command=["mosquitto", "-c", "/mosquitto-no-auth.conf"]
        )
        
        self.containers.append(container)
        
        # Wait for broker to be ready
        time.sleep(2)
        
        # Verify it's running
        container.reload()
        if container.status != "running":
            raise RuntimeError(f"MQTT broker failed to start: {container.status}")
            
        print("MQTT broker started successfully")
        return container
        
    def generate_frigate_config(self, rtsp_url: str, hef_path: str) -> Path:
        """Generate Frigate configuration for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        config = {
            'mqtt': {
                'enabled': True,
                'host': 'host.docker.internal',  # Access host from container
                'port': 18833,  # Use test MQTT port
                'topic_prefix': 'frigate',
                'client_id': 'frigate_test',
                'stats_interval': 5
            },
            'detectors': {
                'hailo': {
                    'type': 'hailo8l',
                    'device': 'PCIe',
                    'model': {
                        'path': '/models/yolo8l_fire.hef',
                        'width': 640,
                        'height': 640,
                        'input_tensor': 'nhwc',
                        'input_pixel_format': 'rgb',
                        'model_type': 'yolov8'
                    }
                }
            },
            'model': {
                'width': 640,
                'height': 640,
                'labelmap_path': '/models/labelmap.txt'
            },
            'objects': {
                'track': ['fire', 'person'],
                'filters': {
                    'fire': {
                        'min_score': 0.3,
                        'threshold': 0.5,
                        'min_area': 100
                    }
                }
            },
            'cameras': {
                'test_cam': {
                    'ffmpeg': {
                        'inputs': [{
                            'path': rtsp_url,
                            'roles': ['detect']
                        }]
                    },
                    'detect': {
                        'enabled': True,
                        'width': 640,
                        'height': 640,
                        'fps': 5
                    }
                }
            },
            'record': {
                'enabled': False
            },
            'snapshots': {
                'enabled': False
            },
            'logger': {
                'default': 'info',
                'logs': {
                    'frigate.detector.hailo': 'debug'
                }
            }
        }
        
        # Write config
        config_path = self.temp_dir / 'config.yml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Write labelmap
        labelmap_path = self.temp_dir / 'labelmap.txt'
        class_names = [
            'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck',
            'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle',
            'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk',
            'Squirrel', 'Pig', 'Chicken', 'Boat', 'Vehicle registration plate',
            'Snowmobile', 'Human face', 'Armadillo', 'Fire', 'Package',
            'Rodent', 'Child', 'Weapon', 'Backpack'
        ]
        
        with open(labelmap_path, 'w') as f:
            for i, name in enumerate(class_names):
                f.write(f"{i}\n{name.lower()}\n")
                
        return self.temp_dir
        
    def start_frigate(self, config_dir: Path, hef_path: Path):
        """Start Frigate container with Hailo support."""
        print("Starting Frigate with Hailo support...")
        
        # Remove existing container if any
        try:
            existing = self.docker_client.containers.get("test-frigate")
            existing.stop()
            existing.remove()
        except:
            pass
            
        # Prepare volumes
        volumes = {
            str(config_dir): {'bind': '/config', 'mode': 'ro'},
            str(hef_path.parent): {'bind': '/models', 'mode': 'ro'},
            '/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'},
            '/dev/hailo0': {'bind': '/dev/hailo0', 'mode': 'rw'}
        }
        
        # Start Frigate
        container = self.docker_client.containers.run(
            "ghcr.io/blakeblackshear/frigate:stable-hailo8l",
            name="test-frigate",
            ports={'5000/tcp': None, '8554/tcp': 8554},
            volumes=volumes,
            devices=['/dev/hailo0:/dev/hailo0:rwm'],
            privileged=True,
            shm_size="256m",
            environment={
                'FRIGATE_RTSP_PASSWORD': 'password'
            },
            detach=True,
            remove=False
        )
        
        self.containers.append(container)
        
        # Wait for Frigate to start
        print("Waiting for Frigate to initialize...")
        time.sleep(10)
        
        # Check logs for errors
        logs = container.logs(tail=50).decode('utf-8')
        if "error" in logs.lower() or "failed" in logs.lower():
            print(f"Frigate logs:\n{logs}")
            
        container.reload()
        if container.status != "running":
            raise RuntimeError(f"Frigate failed to start: {container.status}")
            
        print("Frigate started successfully")
        return container
        

@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.timeout(300)  # 5 minutes
def test_frigate_hailo_docker_e2e(parallel_test_context, docker_container_manager):
    """Test Frigate with Hailo detector using Docker containers."""
    
    print("\n=== Frigate + Hailo Docker E2E Test ===\n")
    
    # Check prerequisites
    device = HailoDevice()
    if not device.is_available():
        pytest.skip("Hailo device not available")
        
    # Download test video
    print("1. Downloading test video...")
    downloader = VideoDownloader()
    videos = downloader.download_all_videos()
    if not videos or 'fire1.mov' not in videos:
        pytest.fail("Failed to download test video")
        
    fire_video = videos['fire1.mov']
    
    # Initialize test manager
    test_manager = DockerE2ETest()
    
    try:
        # Use existing MQTT broker
        print("\n2. Using existing MQTT broker on port 18833...")
        
        # Connect MQTT test client
        print("\n3. Connecting MQTT test client...")
        mqtt_client = MQTTTestClient()
        if not mqtt_client.connect():
            pytest.fail("Failed to connect to MQTT broker")
            
        mqtt_client.subscribe([
            'frigate/+/+',
            'frigate/events',
            'frigate/stats',
            'frigate/available'
        ])
        
        # Start simple RTSP server
        print("\n4. Starting RTSP server...")
        rtsp_process = subprocess.Popen([
            'ffmpeg',
            '-re',
            '-stream_loop', '-1',
            '-i', str(fire_video),
            '-c', 'copy',
            '-f', 'rtsp',
            'rtsp://localhost:8554/test'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)  # Let RTSP server start
        
        # Generate Frigate config
        print("\n5. Generating Frigate configuration...")
        hef_path = Path("hailo_qat_output/yolo8l_fire_640x640_hailo8l_nms.hef")
        if not hef_path.exists():
            pytest.fail(f"HEF model not found: {hef_path}")
            
        config_dir = test_manager.generate_frigate_config(
            "rtsp://host.docker.internal:8554/test",
            hef_path
        )
        
        # Start Frigate
        print("\n6. Starting Frigate container...")
        frigate_container = test_manager.start_frigate(config_dir, hef_path)
        
        # Wait for Frigate to be ready
        print("\n7. Waiting for Frigate to be ready...")
        available_msg = mqtt_client.wait_for_message('frigate/available', timeout=30)
        if not available_msg:
            # Check Frigate logs
            logs = frigate_container.logs(tail=100).decode('utf-8')
            print(f"Frigate logs:\n{logs}")
            pytest.fail("Frigate did not become available")
            
        # Clear messages
        mqtt_client.clear_messages()
        
        # Wait for detection events
        print("\n8. Waiting for fire detection events...")
        start_time = time.time()
        fire_events = []
        stats_received = False
        
        while time.time() - start_time < 60:  # Wait up to 60 seconds
            # Check for events
            event_msg = mqtt_client.wait_for_message('frigate/events', timeout=5)
            if event_msg:
                payload = event_msg['payload']
                if payload.get('after', {}).get('label') == 'fire':
                    fire_events.append(event_msg)
                    print(f"Fire detected! Confidence: {payload['after']['score']}")
                    
            # Check for stats
            stats_msg = mqtt_client.wait_for_message('frigate/stats', timeout=5)
            if stats_msg and not stats_received:
                stats_received = True
                stats = stats_msg['payload']
                print(f"Frigate stats received: {json.dumps(stats, indent=2)}")
                
                # Verify Hailo detector is being used
                if 'detectors' in stats and 'hailo' in stats['detectors']:
                    detector_stats = stats['detectors']['hailo']
                    print(f"Hailo detector: {detector_stats['detection_fps']} FPS, "
                          f"{detector_stats['inference_speed']}ms inference")
                          
            mqtt_client.clear_messages()
            
            # Break if we have enough fire events
            if len(fire_events) >= 3:
                break
                
        # Verify results
        print(f"\n9. Test Results:")
        print(f"  - Fire events detected: {len(fire_events)}")
        print(f"  - Stats received: {stats_received}")
        
        assert len(fire_events) > 0, "No fire detection events received"
        assert stats_received, "No Frigate stats received"
        
        print("\n✅ Frigate + Hailo Docker E2E test passed!")
        
    finally:
        # Cleanup
        print("\n10. Cleaning up...")
        
        # Stop RTSP server
        if 'rtsp_process' in locals():
            rtsp_process.terminate()
            
        # Disconnect MQTT
        if 'mqtt_client' in locals():
            mqtt_client.disconnect()
            
        # Clean up Docker resources
        test_manager.cleanup()
        

if __name__ == '__main__':
    # Run the test
    test_frigate_hailo_docker_e2e()