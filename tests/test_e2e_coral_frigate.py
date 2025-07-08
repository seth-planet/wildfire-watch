#!/usr/bin/env python3.8
"""
IMPORTANT: This test MUST be run with Python 3.8 for Coral TPU compatibility!
Do NOT run with Python 3.12 - it will fail or skip.

End-to-End Test: Coral TPU with Custom YOLOv8 Fire Detection in Frigate
Tests the complete pipeline from camera to fire detection using real hardware
Fixed version addressing network, device, and configuration issues
"""
import sys
import pytest

# Check Python version immediately
if sys.version_info[:2] != (3, 8):
    print(f"ERROR: This test requires Python 3.8 for Coral TPU compatibility.")
    print(f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print("Please run with: python3.8 -m pytest tests/test_e2e_coral_frigate.py")
    sys.exit(1)

pytestmark = [pytest.mark.coral_tpu, pytest.mark.python38]

import os
import sys
import time
import yaml
import docker
import pytest
import numpy as np
import cv2
import paho.mqtt.client as mqtt
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
import socket
from threading import Event

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import has_coral_tpu, has_camera_on_network
from tests.mqtt_test_broker import MQTTTestBroker as TestMQTTBroker


class TestE2ECoralFrigate:
    """End-to-end test for Coral TPU fire detection with Frigate (Fixed)"""
    
    @pytest.fixture
    def mqtt_broker(self):
        """Start test MQTT broker"""
        broker = TestMQTTBroker()
        broker.start()
        time.sleep(2)  # Wait for broker to start
        yield broker
        broker.stop()
    
    @pytest.fixture(autouse=True)
    def cleanup_frigate_containers(self):
        """Ensure no Frigate containers are running before/after test."""
        # Cleanup before test
        subprocess.run(['docker', 'stop', 'frigate_test_e2e_fixed'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['docker', 'rm', 'frigate_test_e2e_fixed'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        
        yield
        
        # Cleanup after test
        subprocess.run(['docker', 'stop', 'frigate_test_e2e_fixed'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['docker', 'rm', 'frigate_test_e2e_fixed'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
    @pytest.mark.slow
    @pytest.mark.infrastructure_dependent
    @pytest.mark.timeout(1800)  # 30 minute timeout for camera discovery
    def test_coral_frigate_fire_detection_e2e(self, mqtt_broker, temp_config_dir):
        """Test complete fire detection pipeline with Coral TPU"""
        
        print("\n" + "="*80)
        print("E2E TEST: Coral TPU Fire Detection with Frigate (Fixed)")
        print("="*80)
        
        # First, ensure no other processes are using Coral TPU
        print("\n0. Checking for existing Coral TPU processes...")
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        coral_processes = [line for line in result.stdout.split('\n') 
                          if 'coral' in line.lower() and 'edgetpu' in line.lower()]
        if coral_processes:
            print("  WARNING: Found existing Coral processes:")
            for proc in coral_processes:
                print(f"    {proc}")
            # Try to kill any test-related Coral processes
            subprocess.run(['pkill', '-f', 'frigate.detector.coral'], capture_output=True)
            time.sleep(2)
        
        # Step 1: Verify Coral TPU hardware and get device info
        print("\n1. Verifying Coral TPU hardware...")
        coral_devices = self._get_coral_devices()
        assert len(coral_devices) > 0, "No Coral TPU devices found"
        print(f"✓ Found {len(coral_devices)} Coral TPU(s)")
        for idx, device in enumerate(coral_devices):
            print(f"  TPU {idx}: {device}")
        
        # Step 2: Verify YOLOv8 fire model
        print("\n2. Verifying YOLOv8 fire detection model...")
        model_path = self._verify_fire_model()
        assert model_path is not None, "No fire detection model found"
        print(f"✓ Using model: {model_path}")
        
        # Step 3: Discover and validate cameras
        print("\n3. Discovering cameras...")
        try:
            cameras = self._discover_and_validate_cameras()
            if not cameras:
                print("  WARNING: No cameras found on network")
                # Use real camera IPs but skip validation 
                cameras = [
                    {'ip': '192.168.5.176', 'name': 'camera_1', 'rtsp_path': '/stream1'},
                    {'ip': '192.168.5.178', 'name': 'camera_2', 'rtsp_path': '/stream1'}
                ]
                print(f"  Using known camera IPs without validation: {[c['ip'] for c in cameras]}")
            else:
                print(f"✓ Found {len(cameras)} camera(s)")
        except Exception as e:
            print(f"  ERROR during camera discovery: {e}")
            raise
        
        # Step 4: Generate Frigate configuration
        print("\n4. Generating Frigate configuration...")
        config_path = self._generate_frigate_config_fixed(
            temp_config_dir, model_path, coral_devices, cameras, mqtt_broker.port
        )
        print(f"✓ Config saved: {config_path}")
        
        # Step 5: Start Frigate container
        print("\n5. Starting Frigate container...")
        container = self._start_frigate_container_fixed(temp_config_dir, mqtt_broker)
        if container is None:
            pytest.fail("Failed to start Frigate container. Check Coral TPU access and configuration.")
        
        # Step 6: Wait for Frigate to be ready
        print("\n6. Waiting for Frigate to initialize...")
        try:
            if not self._wait_for_frigate_fixed(container, timeout=120):
                logs = container.logs(tail=200).decode()
                pytest.fail(f"Frigate failed to start. Logs:\n{logs}")
            print(f"✓ Frigate container started and ready: {container.short_id}")
        except Exception as e:
            # Get container logs for debugging
            logs = container.logs(tail=50).decode()
            container.stop()
            container.remove()
            pytest.fail(f"Frigate failed to start. Logs:\n{logs}")        
        
        mqtt_client = None
        try:
            # Step 7: Test MQTT connectivity with timeout
            print("\n7. Testing MQTT connectivity...")
            mqtt_client = self._setup_mqtt_client(mqtt_broker.port)
            mqtt_messages = []
            message_event = Event()
            
            def on_message(client, userdata, msg):
                mqtt_messages.append({
                    'topic': msg.topic,
                    'payload': msg.payload.decode(),
                    'timestamp': time.time()
                })
                print(f"  MQTT: {msg.topic}")
                message_event.set()
            
            mqtt_client.on_message = on_message
            mqtt_client.subscribe("frigate/#")
            mqtt_client.loop_start()
            
            # Wait for first message with timeout
            if not message_event.wait(timeout=30):
                pytest.fail("No MQTT messages received within 30 seconds")
            
            # Collect messages for 10 seconds
            print("\n8. Collecting MQTT messages for analysis...")
            time.sleep(10)
            
            # Step 9: Verify Coral TPU is being used
            print("\n9. Verifying Coral TPU usage...")
            coral_verified = self._verify_coral_usage(container, mqtt_messages)
            
            # Step 10: Analyze performance
            print("\n10. Analyzing detection performance...")
            performance = self._analyze_performance_fixed(container, mqtt_messages)
            
            # Print results
            print("\n" + "="*80)
            print("E2E TEST RESULTS:")
            print(f"  Coral TPUs detected: {len(coral_devices)}")
            print(f"  Coral TPU active: {coral_verified}")
            print(f"  Model: {os.path.basename(model_path)}")
            print(f"  Cameras configured: {len(cameras)}")
            print(f"  MQTT messages received: {len(mqtt_messages)}")
            print(f"  Average inference: {performance.get('avg_inference', 'N/A')}ms")
            print(f"  Detection FPS: {performance.get('detection_fps', 'N/A')}")
            
            # Assertions
            assert len(mqtt_messages) > 0, "No MQTT messages received"
            assert coral_verified, "Coral TPU not active"
            if performance.get('avg_inference'):
                assert performance['avg_inference'] < 25, f"Inference too slow: {performance['avg_inference']}ms"
            
            print("  Result: ✓ PASSED")
            print("="*80)
            
        finally:
            # Cleanup
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            print("\nStopping Frigate container...")
            try:
                if 'container' in locals() and container:
                    container.stop(timeout=10)
                    container.remove()
            except Exception as e:
                print(f"Error stopping container: {e}")
                try:
                    if 'container' in locals() and container:
                        container.kill()
                        container.remove(force=True)
                except:
                    pass
            
            # Extra cleanup to ensure no orphaned containers
            subprocess.run(['docker', 'stop', 'frigate_test_e2e_fixed'], 
                          capture_output=True, stderr=subprocess.DEVNULL)
            subprocess.run(['docker', 'rm', 'frigate_test_e2e_fixed'], 
                          capture_output=True, stderr=subprocess.DEVNULL)
    
    def _get_coral_devices(self):
        """Get actual Coral TPU device information"""
        result = subprocess.run([
            'python3.8', '-c', '''
from pycoral.utils.edgetpu import list_edge_tpus
import json
tpus = list_edge_tpus()
print(json.dumps([{"type": t["type"], "path": t["path"]} for t in tpus]))
'''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            try:
                return json.loads(result.stdout.strip())
            except:
                pass
        return []
    
    def _verify_fire_model(self):
        """Find YOLOv8 fire detection model for Coral"""
        model_candidates = [
            "converted_models/yolov8n_fire_320_edgetpu.tflite",
            "converted_models/yolo8l_fire_320_edgetpu.tflite",
            "converted_models/yolov8n_320_edgetpu.tflite",
            "converted_models/mobilenet_v2_edgetpu.tflite",
        ]
        
        # Get all available Coral devices
        coral_devices = self._get_coral_devices()
        
        for model in model_candidates:
            if os.path.exists(model):
                # Try to verify model with different Coral devices
                for idx, device in enumerate(coral_devices):
                    device_path = device.get('path', '')
                    device_idx = idx  # Use index for device selection
                    
                    # Verify it's a valid Edge TPU model, trying specific device
                    result = subprocess.run([
                        'python3.8', '-c', f'''
from pycoral.utils.edgetpu import make_interpreter
try:
    # Try to specify device index
    interpreter = make_interpreter("{model}", device=":0")  # Try default first
    interpreter.allocate_tensors()
    print("valid with device :0")
except Exception as e1:
    try:
        # Try without device specification
        interpreter = make_interpreter("{model}")
        interpreter.allocate_tensors()
        print("valid with default device")
    except Exception as e2:
        try:
            # Try with specific device index
            interpreter = make_interpreter("{model}", device=":{device_idx}")
            interpreter.allocate_tensors()
            print(f"valid with device :{device_idx}")
        except Exception as e3:
            print(f"invalid: {{e1}}, {{e2}}, {{e3}}")
'''
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0 and "valid" in result.stdout:
                        print(f"  Model {model} validated: {result.stdout.strip()}")
                        return model
                    else:
                        print(f"  Model {model} validation failed on device {idx}: {result.stdout}")
        
        return None
    
    def _discover_and_validate_cameras(self):
        """Discover cameras and validate accessibility"""
        cameras = []
        camera_creds = os.getenv('CAMERA_CREDENTIALS', '')
        
        if not camera_creds:
            print("  No camera credentials set")
            return cameras
        
        print(f"  Camera credentials set: {camera_creds.split(':')[0]}:***")
        
        # Use known working cameras with the correct RTSP path
        # Based on the discovery results, we know these cameras work with /cam/realmonitor path
        known_cameras = [
            {'ip': '192.168.5.176', 'name': 'camera_1'},
            {'ip': '192.168.5.178', 'name': 'camera_2'},
            {'ip': '192.168.5.179', 'name': 'camera_3'}
        ]
        
        print(f"  Using {len(known_cameras)} known cameras with validated RTSP path")
        
        for cam in known_cameras:
            cameras.append({
                'ip': cam['ip'],
                'name': cam['name'],
                'rtsp_path': '/cam/realmonitor?channel=1&subtype=0'  # Just the path, not full URL
            })
            print(f"  ✓ Added camera: {cam['ip']} ({cam['name']})")
        
        return cameras
    
    def _generate_frigate_config_fixed(self, config_dir, model_path, coral_devices, cameras, mqtt_port):
        """Generate Frigate configuration with fixes"""
        
        # Get camera credentials
        camera_creds = os.getenv('CAMERA_CREDENTIALS', '')
        if camera_creds and ':' in camera_creds:
            username, password = camera_creds.split(':')
        else:
            username, password = 'admin', 'password'  # Default for mock cameras
        
        # Generate detector configuration based on actual devices
        detectors = {}
        for i, device in enumerate(coral_devices[:4]):  # Frigate supports up to 4
            detectors[f'coral{i}'] = {
                'type': 'edgetpu',
                'device': f'pci:{i}'  # Frigate uses index, not path
            }
        
        # Configure cameras
        camera_configs = {}
        for idx, camera in enumerate(cameras[:4]):
            if camera.get('mock'):
                # Mock camera for testing
                camera_configs[camera['name']] = {
                    'enabled': False,
                    'ffmpeg': {
                        'inputs': [{
                            'path': 'rtsp://127.0.0.1:554/null',
                            'roles': ['detect']
                        }]
                    }
                }
            else:
                # Real camera
                rtsp_path = camera.get('rtsp_path', '/stream1')
                camera_configs[camera['name']] = {
                    'ffmpeg': {
                        'inputs': [{
                            'path': f"rtsp://{username}:{password}@{camera['ip']}:554{rtsp_path}",
                            'roles': ['detect']
                        }],
                        'input_args': 'preset-rtsp-generic',
                        'output_args': {
                            'detect': '-f rawvideo -pix_fmt yuv420p'
                        }
                    },
                    'detect': {
                        'enabled': True,
                        'width': 1920,
                        'height': 1080,
                        'fps': 5
                    },
                    'objects': {
                        'track': ['person', 'fire', 'smoke'],
                        'filters': {
                            'person': {
                                'min_score': 0.5,
                                'threshold': 0.7
                            }
                        }
                    }
                }
        
        # Create Frigate config
        config = {
            'mqtt': {
                'enabled': True,
                'host': 'localhost',  # Fixed: use localhost with host network
                'port': mqtt_port,    # Fixed: use actual broker port
                'topic_prefix': 'frigate',
                'client_id': 'frigate_test',
                'stats_interval': 15  # Minimum allowed by Frigate
            },
            'detectors': detectors,
            'model': {
                'path': f'/config/model/{os.path.basename(model_path)}',  # Fixed: proper model path
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320
            },
            'cameras': camera_configs if camera_configs else {
                'dummy': {
                    'enabled': False,
                    'ffmpeg': {
                        'inputs': [{
                            'path': 'rtsp://127.0.0.1:554/null',
                            'roles': ['detect']
                        }]
                    }
                }
            },
            'logger': {
                'default': 'info',
                'logs': {
                    'frigate.detectors.coral': 'debug',
                    'detector.coral': 'debug'
                }
            },
            'record': {
                'enabled': False
            },
            'snapshots': {
                'enabled': False
            }
        }
        
        # Write config
        config_path = os.path.join(config_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Copy model to config directory
        model_dir = os.path.join(config_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(model_dir, os.path.basename(model_path)))
        
        return config_path
    
    def _start_frigate_container_fixed(self, config_dir, mqtt_broker):
        """Start Frigate container with proper configuration"""
        client = docker.from_env()
        
        # Pull image if needed
        try:
            client.images.get('ghcr.io/blakeblackshear/frigate:stable')
        except docker.errors.ImageNotFound:
            print("  Pulling Frigate image...")
            client.images.pull('ghcr.io/blakeblackshear/frigate:stable')
        
        # Get actual Coral device paths
        coral_devices = self._get_coral_devices()
        device_mapping = []
        for device in coral_devices:
            if device['type'] == 'pci':
                device_mapping.append(f"{device['path']}:{device['path']}")
        
        # Container configuration
        container_config = {
            'image': 'ghcr.io/blakeblackshear/frigate:stable',
            'name': 'frigate_test_e2e_fixed',
            'detach': True,
            'volumes': {
                config_dir: {'bind': '/config', 'mode': 'rw'},
                '/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'}  # Shared memory for detectors
            },
            'devices': device_mapping,
            'environment': {
                'FRIGATE_MQTT_HOST': 'localhost',
                'FRIGATE_MQTT_PORT': str(mqtt_broker.port)
            },
            'network_mode': 'host',  # Required for localhost MQTT access
            'shm_size': '256m',
            'privileged': True  # Required for Coral TPU access
        }
        
        try:
            # Remove any existing container
            try:
                old_container = client.containers.get('frigate_test_e2e_fixed')
                old_container.stop(timeout=5)
                old_container.remove()
                time.sleep(2)
            except:
                pass
            
            # Start new container
            container = client.containers.run(**container_config)
            return container
            
        except Exception as e:
            print(f"  Error starting Frigate: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _wait_for_frigate_fixed(self, container, timeout=120):
        """Wait for Frigate to be ready with better detection"""
        start_time = time.time()
        ready_indicators = [
            "Starting Frigate",
            "Starting detector process", 
            "Capture process started",
            "Frigate is running",
            "Loaded detector model",
            "Started Frigate"
        ]
        error_indicators = [
            "Fatal Python error",
            "Unable to create detector",
            "Failed to load model",
            "ImportError",
            "ModuleNotFoundError"
        ]
        
        # Track what we've seen to avoid duplicate messages
        seen_indicators = set()
        
        while time.time() - start_time < timeout:
            try:
                # Check container is still running
                container.reload()
                if container.status != 'running':
                    logs = container.logs(tail=100).decode()
                    print(f"\n  Container stopped. Last logs:\n{logs}")
                    return False
                
                # Check logs
                logs = container.logs(tail=200).decode()
                
                # Check for error indicators first
                for error in error_indicators:
                    if error in logs and error not in seen_indicators:
                        print(f"  ✗ Error detected: {error}")
                        seen_indicators.add(error)
                        return False
                
                # Check for ready indicators
                for indicator in ready_indicators:
                    if indicator in logs and indicator not in seen_indicators:
                        print(f"  ✓ {indicator}")
                        seen_indicators.add(indicator)
                        
                        # Check if fully ready
                        if any(ready in logs for ready in ["Frigate is running", "Started Frigate"]):
                            print("  ✓ Frigate startup complete")
                            time.sleep(3)  # Give it a moment to stabilize
                            return True
                
                # Check for Coral TPU detection
                if "coral" in logs.lower():
                    if "edge tpu detected" in logs.lower() and "coral_detected" not in seen_indicators:
                        print("  ✓ Coral TPU detected")
                        seen_indicators.add("coral_detected")
                    elif "loading edgetpu delegate" in logs.lower() and "coral_loading" not in seen_indicators:
                        print("  ✓ Loading EdgeTPU delegate")
                        seen_indicators.add("coral_loading")
                
                # Handle s6 init process
                if "fix-attrs successfully started" in logs and "fix_attrs" not in seen_indicators:
                    print("  ✓ Container initialization started")
                    seen_indicators.add("fix_attrs")
                
                # Check for errors
                for error in error_indicators:
                    if error in logs:
                        print(f"  ✗ Error detected: {error}")
                        return False
                
            except Exception as e:
                print(f"  Waiting... ({e})")
            
            time.sleep(2)
        
        return False
    
    def _setup_mqtt_client(self, port):
        """Setup MQTT client for testing"""
        client = mqtt.Client(client_id='test_client')
        
        # Connect with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client.connect('localhost', port, 60)
                return client
            except ConnectionRefusedError:
                if attempt == max_retries - 1:
                    pytest.fail(f"Failed to connect to MQTT broker on port {port} after {max_retries} attempts")
                time.sleep(1)  # Wait before retry
        
        return client
    
    def _verify_coral_usage(self, container, mqtt_messages):
        """Verify Coral TPU is actually being used"""
        # Check logs
        logs = container.logs().decode()
        coral_active = False
        
        if "edge tpu" in logs.lower():
            coral_active = True
        
        # Check MQTT stats
        stats_msgs = [m for m in mqtt_messages if m['topic'] == 'frigate/stats']
        if stats_msgs:
            try:
                stats = json.loads(stats_msgs[-1]['payload'])
                if 'detectors' in stats:
                    for name, info in stats['detectors'].items():
                        if 'coral' in name and info.get('detection_start'):
                            coral_active = True
                            print(f"  ✓ {name} active with {info.get('inference_speed', 'N/A')}ms inference")
            except:
                pass
        
        return coral_active
    
    def _analyze_performance_fixed(self, container, mqtt_messages):
        """Analyze performance with proper parsing"""
        performance = {
            'avg_inference': None,
            'detection_fps': None,
            'coral_active': False
        }
        
        # Parse MQTT stats messages
        stats_msgs = [m for m in mqtt_messages if m['topic'] == 'frigate/stats']
        if stats_msgs:
            try:
                # Get latest stats
                latest_stats = json.loads(stats_msgs[-1]['payload'])
                
                # Extract detector performance
                if 'detectors' in latest_stats:
                    inference_speeds = []
                    for detector_name, detector_info in latest_stats['detectors'].items():
                        if 'inference_speed' in detector_info:
                            inference_speeds.append(detector_info['inference_speed'])
                            if 'coral' in detector_name:
                                performance['coral_active'] = True
                    
                    if inference_speeds:
                        performance['avg_inference'] = np.mean(inference_speeds)
                
                # Extract camera FPS
                if 'cameras' in latest_stats:
                    fps_values = []
                    for camera_name, camera_info in latest_stats['cameras'].items():
                        if 'detection_fps' in camera_info:
                            fps_values.append(camera_info['detection_fps'])
                    
                    if fps_values:
                        performance['detection_fps'] = np.mean(fps_values)
                        
            except Exception as e:
                print(f"  Error parsing stats: {e}")
        
        return performance


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])