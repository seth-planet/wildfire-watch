#!/usr/bin/env python3
"""Camera configuration manager for Frigate NVR integration.

This module bridges the gap between dynamic camera discovery (via camera_detector)
and Frigate's static YAML configuration. It subscribes to MQTT camera discovery
events and automatically generates Frigate configurations, allowing the system
to adapt to changing camera environments without manual intervention.

The service solves a fundamental mismatch: camera_detector discovers cameras
dynamically, but Frigate requires a static YAML config file. This manager
continuously synchronizes the two, ensuring Frigate always has an up-to-date
view of available cameras.

Configuration Sources (in priority order):
    1. Custom cameras (user-defined in custom_cameras.yml) - highest priority
    2. Detected cameras (from camera_detector via MQTT)
    3. Base configuration template (frigate_base.yml)

Communication Flow:
    1. Subscribes to camera/discovery/+ for individual camera updates
    2. Subscribes to frigate/config/cameras for bulk camera updates
    3. Merges discovered cameras with custom overrides
    4. Generates frigate.yml with environment variable substitution
    5. Optionally filters cameras for multi-node deployments

MQTT Topics:
    Subscribed:
        - camera/discovery/+: Individual camera discovery events
        - frigate/config/cameras: Bulk camera configuration updates
    
    Published:
        - None (configuration written to filesystem)

Configuration Files:
    - /config/frigate_base.yml: Template with detector and system settings
    - /config/custom_cameras.yml: User-defined camera overrides
    - /config/detected_cameras.json: Persistent cache of discovered cameras
    - /config/frigate.yml: Generated output for Frigate NVR

Environment Variables:
    The service performs custom substitution for flexibility beyond Frigate's
    native {FRIGATE_*} support. Variables are substituted using string.replace()
    on the YAML output. This is brittle but allows non-FRIGATE_ prefixed vars.
    
    WARNING: If environment values contain YAML special characters (: { } [ ] ")
    the generated config may be corrupted. Consider using Frigate's native
    {FRIGATE_*} variables where possible.

Known Issues:
    1. Configuration Churn: Every camera discovery triggers full regeneration,
       potentially causing Frigate restarts. Debouncing is recommended.
    2. State Persistence: Single camera updates don't persist to disk, only
       bulk updates do. This can cause state loss on restart.
    3. YAML Injection: Environment variable substitution via string.replace()
       can corrupt YAML if values contain special characters.
    4. Non-Atomic Writes: Config file writes are not atomic, risking corruption
       if the process is interrupted.

Thread Safety:
    This service is single-threaded with MQTT callbacks. No explicit locking
    is required as all operations occur sequentially in the main thread.

Example:
    Run as standalone service:
        $ python camera_manager.py generate-config
        
    List discovered cameras:
        $ python camera_manager.py list
        
    Test specific camera:
        $ python camera_manager.py test camera_01

Note:
    For production deployments, consider implementing:
    - Debounced configuration generation
    - Atomic file writes with rename
    - Recursive dictionary-based substitution
    - Unified state persistence for all updates
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

# Import centralized command runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.command_runner import run_command, CommandError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera configuration synchronization between detector and Frigate.
    
    This class maintains the camera inventory by listening to MQTT discovery
    events and generating Frigate configurations. It handles merging of multiple
    configuration sources and applies environment variable substitutions.
    
    The manager operates reactively - any camera discovery event triggers a
    full configuration regeneration. While simple, this can cause configuration
    churn in environments with unstable cameras or network issues.
    
    Attributes:
        config_path (str): Output path for generated Frigate config
        base_config_path (str): Template config with system settings
        custom_cameras_path (str): User-defined camera overrides
        detected_cameras_path (str): Persistent cache of discovered cameras
        cameras (Dict[str, Dict]): In-memory camera inventory (camera_id -> config)
        mqtt_client (mqtt.Client): MQTT client for receiving updates
        
    Configuration Precedence:
        1. Custom cameras override everything (user has full control)
        2. Detected cameras fill in the gaps
        3. Base config provides system-wide settings
        
    Side Effects:
        - Writes to filesystem on every camera change
        - May trigger Frigate service restarts via config changes
        - Logs extensively for debugging camera discovery issues
    """
    
    def __init__(self):
        """Initialize camera manager with config paths and MQTT connection.
        
        Sets up file paths, initializes empty camera inventory, configures
        MQTT connection parameters from environment, and establishes the
        MQTT client connection.
        
        Environment Variables:
            FRIGATE_MQTT_HOST: MQTT broker hostname (default: mqtt_broker)
            FRIGATE_MQTT_PORT: MQTT broker port (default: 8883)
            FRIGATE_MQTT_TLS: Enable TLS encryption (default: true)
        """
        self.config_path = "/config/frigate.yml"
        self.base_config_path = "/config/frigate_base.yml"
        self.custom_cameras_path = "/config/custom_cameras.yml"
        self.detected_cameras_path = "/config/detected_cameras.json"
        self.cameras = {}
        
        # MQTT settings
        self.mqtt_host = os.environ.get('FRIGATE_MQTT_HOST', os.environ.get('MQTT_BROKER', 'mqtt_broker'))
        # Check both MQTT_TLS and FRIGATE_MQTT_TLS, with MQTT_TLS taking precedence
        self.mqtt_tls = os.environ.get('MQTT_TLS', os.environ.get('FRIGATE_MQTT_TLS', 'true')).lower() == 'true'
        # Default port based on TLS setting
        default_port = '8883' if self.mqtt_tls else '1883'
        self.mqtt_port = int(os.environ.get('FRIGATE_MQTT_PORT', os.environ.get('MQTT_PORT', default_port)))
        
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
        """Handle camera discovery messages from detector service.
        
        Processes two types of messages:
        1. Individual camera discoveries (camera/discovery/+)
        2. Bulk camera updates (frigate/config/cameras)
        
        Both message types trigger immediate configuration regeneration,
        which can cause excessive I/O and service restarts if cameras
        appear/disappear frequently.
        
        Args:
            client: MQTT client instance
            userdata: User data (unused)
            msg: MQTT message with topic and payload
            
        Side Effects:
            - Updates in-memory camera inventory
            - Triggers full configuration regeneration
            - Writes to detected_cameras.json (bulk updates only)
            - May cause Frigate service restart
            
        Known Issues:
            - No debouncing: rapid discoveries cause config churn
            - Asymmetric persistence: single updates not saved to disk
            - No validation of camera data structure
        """
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
        """Generate complete Frigate configuration from all sources.
        
        This is the core method that merges configuration sources and produces
        the final Frigate YAML. It's called on every camera discovery event,
        potentially causing frequent file I/O and service restarts.
        
        Configuration Flow:
            1. Load base template (system-wide settings)
            2. Load all cameras (detected + custom)
            3. Convert detected cameras to Frigate format
            4. Apply environment variable substitutions
            5. Filter cameras for multi-node assignment
            6. Write final configuration to disk
            
        Returns:
            dict: The final configuration dictionary
            
        Side Effects:
            - Reads from multiple config files
            - Writes to /config/frigate.yml (non-atomically)
            - May trigger Frigate restart if watched by supervisor
            - Logs configuration summary
            
        Known Issues:
            1. No debouncing - called on every camera event
            2. Non-atomic writes - corruption risk if interrupted
            3. String-based env var substitution - YAML injection risk
            4. No validation of generated configuration
            5. No error recovery if base config is missing/invalid
            
        Recommendations for Production:
            - Implement debouncing with configurable delay
            - Use atomic write-and-rename pattern
            - Validate configuration before writing
            - Add try-finally for cleanup on errors
            - Consider configuration versioning/rollback
        """
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
        
        # Apply environment variable substitutions on the dictionary
        final_config = self._substitute_env_vars_safe(config)
        
        # Adjust based on assigned cameras (for multi-node)
        final_config = self._filter_assigned_cameras(final_config)
        
        # Write final configuration atomically
        temp_path = self.config_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                yaml.dump(final_config, f, default_flow_style=False)
            os.rename(temp_path, self.config_path)
        except Exception as e:
            logger.error(f"Failed to write frigate config: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
            
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
        
    def _substitute_env_vars_safe(self, config: Dict) -> Dict:
        """Safely substitute environment variables in configuration dictionary.
        
        Performs variable substitution on the dictionary before YAML serialization
        to avoid YAML injection vulnerabilities.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dict: Configuration with variables substituted
        """
        # Convert to JSON string for safe substitution
        config_str = json.dumps(config)
        
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
            config_str = config_str.replace(key, json.dumps(value).strip('"'))
            
        return json.loads(config_str)
        
        
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
            return_code, stdout, stderr = run_command(
                ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', rtsp_url],
                timeout=10,
                check=False,
                retries=2  # Retry once for transient network issues
            )
            
            if return_code == 0 and 'video' in stdout:
                logger.info(f"Camera {camera_id} test successful")
                return True
            else:
                logger.error(f"Camera {camera_id} test failed: {stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("ffprobe not found. Please install ffmpeg package.")
            return False
        except (CommandError, PermissionError) as e:
            logger.error(f"Camera {camera_id} test error: {e}")
            return False

def main():
    """Command-line interface for camera configuration management.
    
    Provides utility commands for managing the camera inventory and
    generating Frigate configurations. Can be run as a one-shot command
    or as a persistent service (when no command is specified).
    
    Commands:
        generate-config: Generate Frigate configuration and exit
        list: Display all configured cameras with status
        test <camera_id>: Test RTSP connectivity for specific camera
        (no command): Run as service, listening for MQTT updates
        
    Exit Codes:
        0: Success
        1: Camera test failed or invalid command
        
    Examples:
        Generate configuration once:
            $ python camera_manager.py generate-config
            
        List all cameras:
            $ python camera_manager.py list
            
        Test specific camera:
            $ python camera_manager.py test camera_01
            
        Run as persistent service:
            $ python camera_manager.py
            
    Note:
        When run without arguments, the service stays alive listening
        for MQTT camera discovery events. This is the normal production
        mode when deployed in Docker.
    """
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
