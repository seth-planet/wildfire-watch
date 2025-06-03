#!/usr/bin/env python3
"""
Camera Manager for Frigate NVR
Integrates with camera_detector service and manages Frigate configuration
"""
import os
import sys
import json
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self):
        self.config_path = "/config/frigate.yml"
        self.base_config_path = "/config/frigate_base.yml"
        self.custom_cameras_path = "/config/custom_cameras.yml"
        self.detected_cameras_path = "/config/detected_cameras.json"
        self.cameras = {}
        
        # MQTT settings
        self.mqtt_host = os.environ.get('FRIGATE_MQTT_HOST', 'mqtt_broker')
        self.mqtt_port = int(os.environ.get('FRIGATE_MQTT_PORT', '8883'))
        self.mqtt_tls = os.environ.get('FRIGATE_MQTT_TLS', 'true').lower() == 'true'
        
        # Setup MQTT client
        self._setup_mqtt()
        
    def _setup_mqtt(self):
        """Setup MQTT client for camera discovery"""
        self.mqtt_client = mqtt.Client(client_id=f"frigate-camera-manager")
        
        if self.mqtt_tls:
            import ssl
            self.mqtt_client.tls_set(
                ca_certs="/mnt/data/certs/ca.crt",
                certfile="/mnt/data/certs/frigate.crt",
                keyfile="/mnt/data/certs/frigate.key",
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
            
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        try:
            port = 8883 if self.mqtt_tls else 1883
            self.mqtt_client.connect(self.mqtt_host, port, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            # Subscribe to camera discovery
            client.subscribe("camera/discovery/+")
            client.subscribe("frigate/config/cameras")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle camera discovery messages"""
        try:
            if msg.topic.startswith("camera/discovery/"):
                data = json.loads(msg.payload)
                camera = data.get('camera', {})
                if camera and camera.get('online'):
                    self._add_detected_camera(camera)
                    # Regenerate config when new camera found
                    self.generate_frigate_config()
            elif msg.topic == "frigate/config/cameras":
                # Camera detector service published bulk update
                data = json.loads(msg.payload)
                cameras = data.get('cameras', {})
                if cameras:
                    self._update_detected_cameras(cameras)
                    # Regenerate config with all cameras
                    self.generate_frigate_config()
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
            
    def _add_detected_camera(self, camera_data: Dict):
        """Add a detected camera"""
        camera_id = camera_data.get('id')
        if camera_id and camera_data.get('primary_rtsp_url'):
            logger.info(f"Discovered camera: {camera_id} ({camera_data.get('name')})")
            self.cameras[camera_id] = camera_data
            
    def _update_detected_cameras(self, cameras: Dict):
        """Update all detected cameras"""
        logger.info(f"Received {len(cameras)} cameras from detector service")
        self.cameras.update(cameras)
        
        # Save to file for persistence
        with open(self.detected_cameras_path, 'w') as f:
            json.dump(self.cameras, f, indent=2)
            
    def load_cameras(self):
        """Load cameras from all sources"""
        cameras = {}
        
        # 1. Load detected cameras
        if os.path.exists(self.detected_cameras_path):
            try:
                with open(self.detected_cameras_path, 'r') as f:
                    detected = json.load(f)
                    cameras.update(detected)
                    logger.info(f"Loaded {len(detected)} detected cameras")
            except Exception as e:
                logger.error(f"Error loading detected cameras: {e}")
                
        # 2. Load custom cameras (override detected)
        if os.path.exists(self.custom_cameras_path):
            try:
                with open(self.custom_cameras_path, 'r') as f:
                    custom = yaml.safe_load(f)
                    if custom and 'cameras' in custom:
                        cameras.update(custom['cameras'])
                        logger.info(f"Loaded {len(custom['cameras'])} custom cameras")
            except Exception as e:
                logger.error(f"Error loading custom cameras: {e}")
                
        self.cameras = cameras
        return cameras
        
    def generate_frigate_config(self):
        """Generate complete Frigate configuration"""
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load cameras
        self.load_cameras()
        
        # Convert detected cameras to Frigate format
        frigate_cameras = {}
        for camera_id, camera_data in self.cameras.items():
            if isinstance(camera_data, dict) and 'ffmpeg' in camera_data:
                # Already in Frigate format (custom camera)
                frigate_cameras[camera_id] = camera_data
            else:
                # Convert from detected format
                frigate_config = self._convert_to_frigate_config(camera_data)
                if frigate_config:
                    frigate_cameras[camera_id] = frigate_config
                    
        config['cameras'] = frigate_cameras
        
        # Apply environment variable substitutions
        config_str = yaml.dump(config, default_flow_style=False)
        config_str = self._substitute_env_vars(config_str)
        
        # Parse back to validate
        final_config = yaml.safe_load(config_str)
        
        # Adjust based on assigned cameras (for multi-node)
        final_config = self._filter_assigned_cameras(final_config)
        
        # Write final configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(final_config, f, default_flow_style=False)
            
        logger.info(f"Generated Frigate config with {len(final_config['cameras'])} cameras")
        return final_config
        
    def _convert_to_frigate_config(self, camera_data: Dict) -> Optional[Dict]:
        """Convert detected camera to Frigate configuration"""
        if not camera_data.get('primary_rtsp_url'):
            return None
            
        config = {
            'ffmpeg': {
                'inputs': []
            },
            'detect': {
                'enabled': True,
                'width': int(os.environ.get('CAMERA_DETECT_WIDTH', '1280')),
                'height': int(os.environ.get('CAMERA_DETECT_HEIGHT', '720')),
                'fps': int(os.environ.get('DETECTION_FPS', '5')),
            },
            'record': {
                'enabled': True,
                'retain': {
                    'days': int(os.environ.get('RECORD_RETAIN_DAYS', '180')),
                    'mode': 'all'
                },
                'events': {
                    'retain': {
                        'default': 30,
                        'objects': {
                            'fire': 365,
                            'smoke': 365
                        }
                    }
                }
            },
            'snapshots': {
                'enabled': True,
                'timestamp': True,
                'retain': {
                    'default': 7,
                    'objects': {
                        'fire': 365,
                        'smoke': 365
                    }
                }
            },
            'motion': {},
            'ui': {
                'order': 0,
                'dashboard': True
            }
        }
        
        # Main stream for recording and detection
        config['ffmpeg']['inputs'].append({
            'path': camera_data['primary_rtsp_url'],
            'roles': ['detect', 'record']
        })
        
        # Add sub-stream if available
        rtsp_urls = camera_data.get('rtsp_urls', {})
        for key in ['sub', 'substream', 'low']:
            if key in rtsp_urls and rtsp_urls[key] != camera_data['primary_rtsp_url']:
                config['ffmpeg']['inputs'].append({
                    'path': rtsp_urls[key],
                    'roles': ['detect']
                })
                break
                
        return config
        
    def _substitute_env_vars(self, config_str: str) -> str:
        """Substitute environment variables in config"""
        replacements = {
            '{FRIGATE_MQTT_HOST}': os.environ.get('FRIGATE_MQTT_HOST', 'mqtt_broker'),
            '{FRIGATE_MQTT_PORT}': os.environ.get('FRIGATE_MQTT_PORT', '8883'),
            '{FRIGATE_CLIENT_ID}': f"frigate-{os.environ.get('HOSTNAME', 'nvr')}",
            '{DETECTOR_TYPE}': os.environ.get('DETECTOR_TYPE', 'cpu'),
            '{DETECTOR_DEVICE}': os.environ.get('DETECTOR_DEVICE', '0'),
            '{MODEL_PATH}': os.environ.get('MODEL_PATH', '/models/wildfire/wildfire_cpu.tflite'),
            '{HWACCEL_ARGS}': os.environ.get('HWACCEL_ARGS', '[]'),
            '{RECORD_CODEC}': os.environ.get('RECORD_CODEC', 'copy'),
            '{RECORD_PRESET}': os.environ.get('RECORD_PRESET', 'fast'),
            '{RECORD_QUALITY}': os.environ.get('RECORD_QUALITY', '23'),
            '{DETECTION_THRESHOLD}': os.environ.get('DETECTION_THRESHOLD', '0.7'),
            '{RECORD_RETAIN_DAYS}': os.environ.get('RECORD_RETAIN_DAYS', '180'),
            '{LOG_LEVEL}': os.environ.get('LOG_LEVEL', 'info'),
        }
        
        for key, value in replacements.items():
            config_str = config_str.replace(key, str(value))
            
        return config_str
        
    def _filter_assigned_cameras(self, config: Dict) -> Dict:
        """Filter cameras based on assignment for multi-node setup"""
        assigned_cameras = os.environ.get('ASSIGNED_CAMERAS', '')
        
        if assigned_cameras:
            # Parse assigned camera list
            assigned_list = [c.strip() for c in assigned_cameras.split(',') if c.strip()]
            
            if assigned_list:
                # Filter to only assigned cameras
                filtered_cameras = {
                    cam_id: cam_config
                    for cam_id, cam_config in config['cameras'].items()
                    if cam_id in assigned_list
                }
                
                config['cameras'] = filtered_cameras
                logger.info(f"Filtered to {len(filtered_cameras)} assigned cameras")
                
        return config
        
    def test_camera(self, camera_id: str):
        """Test camera connection"""
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return False
            
        camera = self.cameras[camera_id]
        rtsp_url = camera.get('primary_rtsp_url') or camera.get('ffmpeg', {}).get('inputs', [{}])[0].get('path')
        
        if not rtsp_url:
            logger.error(f"No RTSP URL for camera {camera_id}")
            return False
            
        logger.info(f"Testing camera {camera_id}: {rtsp_url}")
        
        # Test with ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', rtsp_url],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'video' in result.stdout:
                logger.info(f"Camera {camera_id} test successful")
                return True
            else:
                logger.error(f"Camera {camera_id} test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Camera {camera_id} test timed out")
            return False
        except Exception as e:
            logger.error(f"Camera {camera_id} test error: {e}")
            return False

def main():
    manager = CameraManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'generate-config':
            manager.generate_frigate_config()
            print("Frigate configuration generated")
            
        elif command == 'list':
            manager.load_cameras()
            print(f"\nConfigured Cameras ({len(manager.cameras)}):")
            print("-" * 60)
            
            for cam_id, cam_data in manager.cameras.items():
                name = cam_data.get('name', cam_id)
                ip = cam_data.get('ip', 'Unknown')
                online = cam_data.get('online', False)
                status = "Online" if online else "Offline"
                
                print(f"{cam_id}: {name}")
                print(f"  IP: {ip}")
                print(f"  Status: {status}")
                print()
                
        elif command == 'test' and len(sys.argv) > 2:
            camera_id = sys.argv[2]
            manager.load_cameras()
            
            if manager.test_camera(camera_id):
                print(f"Camera {camera_id} test passed")
            else:
                print(f"Camera {camera_id} test failed")
                sys.exit(1)
                
        else:
            print("Usage: camera_manager.py [generate-config|list|test <camera_id>]")
            
    else:
        # Default: generate config
        manager.generate_frigate_config()

if __name__ == '__main__':
    main()
