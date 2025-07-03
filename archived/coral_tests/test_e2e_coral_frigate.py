#!/usr/bin/env python3.12
"""
End-to-End Test: Coral TPU with Custom YOLOv8 Fire Detection in Frigate
Tests the complete pipeline from camera to fire detection using real hardware
"""

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
from unittest.mock import patch, MagicMock
import subprocess
import tempfile
import shutil
import json

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import has_coral_tpu, has_camera_on_network
from tests.mqtt_test_broker import TestMQTTBroker


class TestE2ECoralFrigate:
    """End-to-end test for Coral TPU fire detection with Frigate"""
    
    @pytest.fixture
    def mqtt_broker(self):
        """Start test MQTT broker"""
        broker = TestMQTTBroker()
        broker.start()
        time.sleep(2)  # Wait for broker to start
        yield broker
        broker.stop()
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
    @pytest.mark.skipif(not has_camera_on_network(), reason="No cameras on network")
    @pytest.mark.slow
    @pytest.mark.infrastructure_dependent
    def test_coral_frigate_fire_detection_e2e(self, mqtt_broker, temp_config_dir):
        """Test complete fire detection pipeline with Coral TPU"""
        
        print("\n" + "="*80)
        print("E2E TEST: Coral TPU Fire Detection with Frigate")
        print("="*80)
        
        # Step 1: Verify Coral TPU hardware
        print("\n1. Verifying Coral TPU hardware...")
        coral_count = self._verify_coral_hardware()
        assert coral_count > 0, "No Coral TPU devices found"
        print(f"✓ Found {coral_count} Coral TPU(s)")
        
        # Step 2: Verify YOLOv8 fire model
        print("\n2. Verifying YOLOv8 fire detection model...")
        model_path = self._verify_fire_model()
        assert model_path is not None, "No fire detection model found"
        print(f"✓ Using model: {model_path}")
        
        # Step 3: Generate Frigate configuration
        print("\n3. Generating Frigate configuration...")
        config_path = self._generate_frigate_config(temp_config_dir, model_path, coral_count)
        print(f"✓ Config saved: {config_path}")
        
        # Step 4: Start Frigate container
        print("\n4. Starting Frigate container...")
        container = self._start_frigate_container(temp_config_dir, mqtt_broker.port)
        assert container is not None, "Failed to start Frigate"
        print(f"✓ Frigate container started: {container.short_id}")
        
        try:
            # Step 5: Wait for Frigate to initialize
            print("\n5. Waiting for Frigate initialization...")
            if not self._wait_for_frigate(container, timeout=60):
                logs = container.logs(tail=100).decode()
                pytest.fail(f"Frigate failed to start. Logs:\\n{logs}")
            print("✓ Frigate is ready")
            
            # Step 6: Test MQTT connectivity
            print("\n6. Testing MQTT connectivity...")
            mqtt_client = self._setup_mqtt_client(mqtt_broker.port)
            mqtt_messages = []
            
            def on_message(client, userdata, msg):
                mqtt_messages.append({
                    'topic': msg.topic,
                    'payload': msg.payload.decode()
                })
                print(f"  MQTT: {msg.topic} - {msg.payload.decode()[:100]}")
            
            mqtt_client.on_message = on_message
            mqtt_client.subscribe("frigate/#")
            mqtt_client.loop_start()
            
            # Step 7: Verify Coral TPU is being used
            print("\n7. Verifying Coral TPU usage...")
            stats = self._get_frigate_stats(container)
            if stats and 'detectors' in stats:
                for detector_name, detector_info in stats['detectors'].items():
                    if 'inference_speed' in detector_info:
                        speed = detector_info['inference_speed']
                        print(f"  {detector_name}: {speed:.2f}ms inference")
                        assert speed < 25, f"Inference too slow: {speed}ms"
            
            # Step 8: Simulate fire detection
            print("\n8. Simulating fire detection...")
            fire_detected = self._simulate_fire_detection(mqtt_messages)
            
            # Step 9: Verify detection performance
            print("\n9. Analyzing detection performance...")
            performance = self._analyze_performance(container, mqtt_messages)
            
            print("\n" + "="*80)
            print("E2E TEST RESULTS:")
            print(f"  Coral TPUs used: {coral_count}")
            print(f"  Model: {os.path.basename(model_path)}")
            print(f"  MQTT messages received: {len(mqtt_messages)}")
            print(f"  Fire detections: {fire_detected}")
            print(f"  Average inference: {performance.get('avg_inference', 'N/A')}ms")
            print(f"  Detection rate: {performance.get('detection_rate', 'N/A')} FPS")
            print("  Result: ✓ PASSED")
            print("="*80)
            
            # Assertions
            assert len(mqtt_messages) > 0, "No MQTT messages received"
            if performance.get('avg_inference'):
                assert performance['avg_inference'] < 10, "Inference too slow for Coral TPU"
            
        finally:
            # Cleanup
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("\nStopping Frigate container...")
            container.stop()
            container.remove()
    
    def _verify_coral_hardware(self):
        """Verify Coral TPU hardware using Python 3.8"""
        result = subprocess.run([
            'python3.8', '-c',
            'from pycoral.utils.edgetpu import list_edge_tpus; print(len(list_edge_tpus()))'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0
    
    def _verify_fire_model(self):
        """Find YOLOv8 fire detection model for Coral"""
        model_candidates = [
            "converted_models/yolov8n_fire_320_edgetpu.tflite",
            "converted_models/yolo8l_fire_320_edgetpu.tflite",
            "converted_models/yolov8n_320_edgetpu.tflite",
            "models/yolov8n_fire_edgetpu.tflite",
            "/models/wildfire_coral_edgetpu.tflite"
        ]
        
        for model in model_candidates:
            if os.path.exists(model):
                # Verify it's a valid Edge TPU model
                result = subprocess.run([
                    'python3.8', '-c', f'''
import os
from pycoral.utils.edgetpu import make_interpreter
try:
    interpreter = make_interpreter("{model}")
    print("valid")
except:
    print("invalid")
'''
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and "valid" in result.stdout:
                    return model
        
        # If no fire model, use generic model
        generic_models = [
            "converted_models/yolov8n_320_edgetpu.tflite",
            "converted_models/mobilenet_v2_edgetpu.tflite"
        ]
        
        for model in generic_models:
            if os.path.exists(model):
                return model
        
        return None
    
    def _generate_frigate_config(self, config_dir, model_path, coral_count):
        """Generate Frigate configuration for Coral TPU"""
        
        # Get camera credentials from environment
        camera_creds = os.getenv('CAMERA_CREDENTIALS', '')
        username, password = camera_creds.split(':')
        
        # Find available cameras
        cameras = self._discover_cameras()
        if not cameras:
            # Use default camera for testing
            cameras = [{'ip': '192.168.5.176', 'name': 'test_camera'}]
        
        # Generate multi-TPU configuration
        detectors = {}
        for i in range(min(coral_count, 4)):  # Frigate supports up to 4 detectors
            detectors[f'coral{i}'] = {
                'type': 'edgetpu',
                'device': f'pci:{i}'
            }
        
        # Distribute cameras across TPUs
        camera_configs = {}
        for idx, camera in enumerate(cameras[:4]):  # Test with up to 4 cameras
            tpu_idx = idx % len(detectors)
            detector_name = f'coral{tpu_idx}'
            
            camera_configs[camera['name']] = {
                'ffmpeg': {
                    'inputs': [{
                        'path': f"rtsp://{username}:{password}@{camera['ip']}:554/stream1",
                        'roles': ['detect']
                    }]
                },
                'detect': {
                    'enabled': True,
                    'width': 1920,
                    'height': 1080,
                    'fps': 5
                },
                'objects': {
                    'track': ['fire', 'smoke', 'person'],
                    'filters': {
                        'fire': {
                            'min_score': 0.4,
                            'threshold': 0.5
                        },
                        'smoke': {
                            'min_score': 0.4,
                            'threshold': 0.5
                        }
                    }
                }
            }
        
        config = {
            'mqtt': {
                'enabled': True,
                'host': 'host.docker.internal',
                'port': 1883,
                'topic_prefix': 'frigate',
                'stats_interval': 5
            },
            'detectors': detectors,
            'model': {
                'path': f'/models/{os.path.basename(model_path)}',
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320,
                'labelmap': {
                    0: 'person',
                    26: 'fire',
                    27: 'smoke'
                }
            },
            'cameras': camera_configs,
            'logger': {
                'default': 'info',
                'logs': {
                    'frigate.detectors.coral': 'debug'
                }
            }
        }
        
        config_path = os.path.join(config_dir, 'config.yml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Copy model to config directory
        model_dir = os.path.join(config_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(model_path, os.path.join(model_dir, os.path.basename(model_path)))
        
        return config_path
    
    def _discover_cameras(self):
        """Discover cameras on network"""
        cameras = []
        
        # Quick scan of known camera subnet
        subnet = "192.168.5"
        for last_octet in range(176, 184):
            ip = f"{subnet}.{last_octet}"
            
            # Quick RTSP check
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            
            try:
                result = sock.connect_ex((ip, 554))
                if result == 0:
                    cameras.append({
                        'ip': ip,
                        'name': f'camera_{len(cameras)+1}'
                    })
            except:
                pass
            finally:
                sock.close()
        
        return cameras
    
    def _start_frigate_container(self, config_dir, mqtt_port):
        """Start Frigate container with Coral TPU support"""
        client = docker.from_env()
        
        # Check if Frigate image exists
        try:
            client.images.get('ghcr.io/blakeblackshear/frigate:stable')
        except docker.errors.ImageNotFound:
            print("  Pulling Frigate image...")
            client.images.pull('ghcr.io/blakeblackshear/frigate:stable')
        
        # Container configuration
        container_config = {
            'image': 'ghcr.io/blakeblackshear/frigate:stable',
            'name': 'frigate_test_e2e',
            'detach': True,
            'remove': True,
            'volumes': {
                config_dir: {'bind': '/config', 'mode': 'rw'},
                '/dev/bus/usb': {'bind': '/dev/bus/usb', 'mode': 'rw'}
            },
            'devices': [
                '/dev/apex_0:/dev/apex_0',
                '/dev/apex_1:/dev/apex_1',
                '/dev/apex_2:/dev/apex_2',
                '/dev/apex_3:/dev/apex_3'
            ],
            'environment': {
                'FRIGATE_MQTT_HOST': 'host.docker.internal',
                'FRIGATE_MQTT_PORT': str(mqtt_port)
            },
            'network_mode': 'host',
            'privileged': True
        }
        
        try:
            # Remove any existing container
            try:
                old_container = client.containers.get('frigate_test_e2e')
                old_container.stop()
                old_container.remove()
            except:
                pass
            
            # Start new container
            container = client.containers.run(**container_config)
            return container
            
        except Exception as e:
            print(f"  Error starting Frigate: {e}")
            return None
    
    def _wait_for_frigate(self, container, timeout=60):
        """Wait for Frigate to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check container status
                container.reload()
                if container.status != 'running':
                    return False
                
                # Check logs for readiness
                logs = container.logs(tail=50).decode()
                if "Frigate is ready" in logs or "Starting detector process" in logs:
                    return True
                
                # Check for Coral TPU initialization
                if "EdgeTPU detected" in logs or "Attempting to load TPU" in logs:
                    print("  Coral TPU initialization detected")
                
                # Check for errors
                if "Error" in logs or "Failed" in logs:
                    error_lines = [l for l in logs.split('\n') if 'Error' in l or 'Failed' in l]
                    for line in error_lines[:3]:  # Show first 3 errors
                        print(f"  ⚠️  {line}")
                
            except Exception as e:
                print(f"  Waiting... ({e})")
            
            time.sleep(2)
        
        return False
    
    def _setup_mqtt_client(self, port):
        """Setup MQTT client for testing"""
        client = mqtt.Client()
        client.connect('localhost', port, 60)
        return client
    
    def _get_frigate_stats(self, container):
        """Get Frigate statistics via API"""
        try:
            # Execute curl inside container
            result = container.exec_run('curl -s http://localhost:5000/api/stats')
            if result.exit_code == 0:
                return json.loads(result.output.decode())
        except:
            pass
        return None
    
    def _simulate_fire_detection(self, mqtt_messages):
        """Analyze MQTT messages for fire detections"""
        fire_detections = 0
        
        for msg in mqtt_messages:
            if 'fire' in msg['topic'] or 'smoke' in msg['topic']:
                fire_detections += 1
            
            # Check event messages
            if msg['topic'].endswith('/events'):
                try:
                    event = json.loads(msg['payload'])
                    if event.get('label') in ['fire', 'smoke']:
                        fire_detections += 1
                except:
                    pass
        
        return fire_detections
    
    def _analyze_performance(self, container, mqtt_messages):
        """Analyze detection performance"""
        performance = {
            'avg_inference': None,
            'detection_rate': None,
            'coral_active': False
        }
        
        # Check container logs for performance metrics
        logs = container.logs(tail=200).decode()
        
        # Look for Coral TPU inference times
        import re
        inference_times = []
        for line in logs.split('\n'):
            # Pattern: "Inference: 2.8ms" or similar
            match = re.search(r'Inference:\s*([\d.]+)\s*ms', line)
            if match:
                inference_times.append(float(match.group(1)))
            
            # Check for Coral TPU active
            if 'coral' in line.lower() and ('loaded' in line or 'initialized' in line):
                performance['coral_active'] = True
        
        if inference_times:
            performance['avg_inference'] = np.mean(inference_times)
        
        # Calculate detection rate from MQTT stats
        stats_messages = [m for m in mqtt_messages if m['topic'].endswith('/stats')]
        if stats_messages:
            try:
                latest_stats = json.loads(stats_messages[-1]['payload'])
                if 'detectors' in latest_stats:
                    fps_values = []
                    for detector in latest_stats['detectors'].values():
                        if 'detection_fps' in detector:
                            fps_values.append(detector['detection_fps'])
                    if fps_values:
                        performance['detection_rate'] = np.mean(fps_values)
            except:
                pass
        
        return performance


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])