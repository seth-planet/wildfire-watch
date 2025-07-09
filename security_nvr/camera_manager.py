#!/usr/bin/env python3
"""Camera configuration manager for Frigate NVR integration - Refactored version.

This module bridges the gap between dynamic camera discovery (via camera_detector)
and Frigate's static YAML configuration. Uses base classes for standardized
MQTT handling, health reporting, and thread management.
"""

import os
import sys
import json
import yaml
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.mqtt_service import MQTTService
from utils.health_reporter import HealthReporter
from utils.thread_manager import ThreadSafeService
from security_nvr.camera_manager_config import CameraManagerConfig


class CameraManagerHealthReporter(HealthReporter):
    """Health reporter for camera manager service."""
    
    def __init__(self, manager):
        self.manager = manager
        super().__init__(manager, manager.config.health_report_interval)
        
    def get_service_health(self) -> Dict[str, Any]:
        """Get camera manager specific health metrics."""
        with self.manager._state_lock:
            camera_count = len(self.manager.cameras)
            last_update = time.time() - self.manager._last_config_update \
                         if self.manager._last_config_update else None
            
        health = {
            'detected_cameras': camera_count,
            'custom_cameras_loaded': self.manager._custom_cameras_loaded,
            'config_file_exists': os.path.exists(self.manager.config.config_path),
            'last_config_update': last_update,
            'pending_update': self.manager._pending_update,
            'filtering_enabled': self.manager.config.is_filtering_enabled,
        }
        
        if self.manager.config.camera_whitelist:
            health['camera_whitelist'] = self.manager.config.camera_whitelist
            
        return health


class CameraManager(MQTTService, ThreadSafeService):
    """Manages camera configuration synchronization between detector and Frigate.
    
    Refactored to use base classes for:
    - MQTTService: Standardized MQTT handling with reconnection
    - ThreadSafeService: Thread management and safety
    - HealthReporter: Consistent health monitoring
    """
    
    def __init__(self):
        # Load configuration
        self.config = CameraManagerConfig()
        
        # Initialize base classes
        ThreadSafeService.__init__(self, "camera_manager", logging.getLogger(__name__))
        MQTTService.__init__(self, "camera_manager", self.config)
        
        # Set logging level
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Camera state
        self.cameras: Dict[str, Dict] = {}  # camera_id -> config
        self._custom_cameras_loaded = False
        self._last_config_update = None
        self._pending_update = False
        self._update_timer = None
        
        # Load persisted cameras if available
        self._load_detected_cameras()
        
        # Load custom cameras
        self._load_custom_cameras()
        
        # Initialize health reporter
        self.health_reporter = CameraManagerHealthReporter(self)
        
        # Setup MQTT
        subscriptions = [
            self.config.camera_discovery_topic,
            self.config.bulk_config_topic
        ]
        
        self.setup_mqtt(
            on_connect=self._on_connect,
            on_message=self._on_message,
            subscriptions=subscriptions
        )
        
        # Connect to MQTT
        self.connect()
        
        # Start health reporting
        self.health_reporter.start()
        
        # Generate initial config
        self._generate_config()
        
    def _load_detected_cameras(self):
        """Load previously detected cameras from persistent storage."""
        if not self.config.persist_detected_cameras:
            return
            
        try:
            if os.path.exists(self.config.detected_cameras_path):
                with open(self.config.detected_cameras_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.cameras = data
                        self.logger.info(f"Loaded {len(self.cameras)} detected cameras")
        except Exception as e:
            self.logger.error(f"Failed to load detected cameras: {e}")
            
    def _save_detected_cameras(self):
        """Save detected cameras to persistent storage."""
        if not self.config.persist_detected_cameras:
            return
            
        try:
            os.makedirs(os.path.dirname(self.config.detected_cameras_path), exist_ok=True)
            with open(self.config.detected_cameras_path, 'w') as f:
                json.dump(self.cameras, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save detected cameras: {e}")
            
    def _load_custom_cameras(self):
        """Load custom camera configurations."""
        if not self.config.merge_custom_cameras:
            return
            
        try:
            if os.path.exists(self.config.custom_cameras_path):
                with open(self.config.custom_cameras_path, 'r') as f:
                    custom_config = yaml.safe_load(f)
                    if custom_config and 'cameras' in custom_config:
                        self._custom_cameras_loaded = True
                        self.logger.info(f"Loaded custom cameras configuration")
        except Exception as e:
            self.logger.error(f"Failed to load custom cameras: {e}")
            
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
            # Resubscribe handled by MQTTService base class
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            if msg.topic.startswith(self.config.camera_discovery_topic.replace('+', '')):
                # Individual camera discovery
                self._handle_camera_discovery(msg)
            elif msg.topic == self.config.bulk_config_topic:
                # Bulk camera update
                self._handle_bulk_update(msg)
        except Exception as e:
            self.logger.error(f"Error handling message on {msg.topic}: {e}")
            
    def _handle_camera_discovery(self, msg):
        """Handle individual camera discovery messages."""
        try:
            if not msg.payload:
                # Empty payload means camera went offline
                camera_id = msg.topic.split('/')[-1]
                self._remove_camera(camera_id)
            else:
                camera_data = json.loads(msg.payload.decode('utf-8'))
                self._add_or_update_camera(camera_data)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in camera discovery: {e}")
            
    def _handle_bulk_update(self, msg):
        """Handle bulk camera configuration updates."""
        try:
            if not msg.payload:
                self.logger.warning("Received empty bulk update")
                return
                
            cameras = json.loads(msg.payload.decode('utf-8'))
            if isinstance(cameras, dict):
                with self._state_lock:
                    # Replace all cameras
                    self.cameras = cameras
                    self.logger.info(f"Bulk update: {len(cameras)} cameras")
                    
                self._save_detected_cameras()
                self._schedule_config_update()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in bulk update: {e}")
            
    def _add_or_update_camera(self, camera_data: Dict):
        """Add or update a camera configuration."""
        camera_id = camera_data.get('id') or camera_data.get('mac')
        if not camera_id:
            self.logger.error("Camera data missing ID")
            return
            
        # Apply filtering if enabled
        if self.config.is_filtering_enabled:
            if camera_id not in self.config.camera_whitelist:
                self.logger.debug(f"Camera {camera_id} not in whitelist, ignoring")
                return
                
        with self._state_lock:
            self.cameras[camera_id] = camera_data
            self.logger.info(f"Updated camera: {camera_id}")
            
        self._save_detected_cameras()
        self._schedule_config_update()
        
    def _remove_camera(self, camera_id: str):
        """Remove a camera from configuration."""
        with self._state_lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                self.logger.info(f"Removed camera: {camera_id}")
                
        self._save_detected_cameras()
        self._schedule_config_update()
        
    def _schedule_config_update(self):
        """Schedule a configuration update with debouncing."""
        with self._state_lock:
            self._pending_update = True
            
            # Cancel existing timer
            if self._update_timer:
                self._update_timer.cancel()
                
            # Schedule new update
            self._update_timer = threading.Timer(
                self.config.config_generation_delay,
                self._generate_config
            )
            self._update_timer.start()
            
    def _generate_config(self):
        """Generate Frigate configuration file."""
        try:
            # Load base configuration
            base_config = self._load_base_config()
            if not base_config:
                return
                
            # Add detected cameras
            if 'cameras' not in base_config:
                base_config['cameras'] = {}
                
            with self._state_lock:
                for camera_id, camera_data in self.cameras.items():
                    # Generate Frigate camera config
                    camera_config = self._generate_camera_config(camera_data)
                    if camera_config:
                        base_config['cameras'][camera_id] = camera_config
                        
            # Merge custom cameras (they override detected ones)
            if self.config.merge_custom_cameras:
                custom_config = self._load_custom_config()
                if custom_config and 'cameras' in custom_config:
                    base_config['cameras'].update(custom_config['cameras'])
                    
            # Convert to YAML
            yaml_content = yaml.dump(base_config, default_flow_style=False, sort_keys=False)
            
            # Perform environment variable substitution
            yaml_content = self._substitute_env_vars(yaml_content)
            
            # Write configuration
            self._write_config(yaml_content)
            
            with self._state_lock:
                self._last_config_update = time.time()
                self._pending_update = False
                
            self.logger.info(f"Generated Frigate config with {len(base_config.get('cameras', {}))} cameras")
            
        except Exception as e:
            self.logger.error(f"Failed to generate config: {e}")
            
    def _load_base_config(self) -> Optional[Dict]:
        """Load base Frigate configuration."""
        try:
            if os.path.exists(self.config.base_config_path):
                with open(self.config.base_config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Base config not found: {self.config.base_config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load base config: {e}")
            return None
            
    def _load_custom_config(self) -> Optional[Dict]:
        """Load custom camera configuration."""
        try:
            if os.path.exists(self.config.custom_cameras_path):
                with open(self.config.custom_cameras_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load custom config: {e}")
        return None
        
    def _generate_camera_config(self, camera_data: Dict) -> Optional[Dict]:
        """Generate Frigate configuration for a camera."""
        rtsp_urls = camera_data.get('rtsp_urls', {})
        if not rtsp_urls:
            self.logger.warning(f"Camera {camera_data.get('id')} has no RTSP URLs")
            return None
            
        # Use main stream for detection/recording, sub stream for detection if available
        main_stream = rtsp_urls.get('main_stream') or rtsp_urls.get('stream1') or next(iter(rtsp_urls.values()))
        sub_stream = rtsp_urls.get('sub_stream') or rtsp_urls.get('stream2')
        
        camera_config = {
            'ffmpeg': {
                'inputs': [
                    {
                        'path': main_stream,
                        'roles': ['detect', 'record']
                    }
                ]
            },
            'detect': {
                'enabled': True,
                'width': 640,
                'height': 360
            },
            'record': {
                'enabled': True,
                'retain': {
                    'days': 7,
                    'mode': 'motion'
                },
                'events': {
                    'retain': {
                        'default': 14,
                        'objects': {
                            'fire': 30,
                            'smoke': 30
                        }
                    }
                }
            },
            'objects': {
                'track': ['fire', 'smoke'],
                'filters': {
                    'fire': {
                        'min_score': 0.5,
                        'threshold': 0.7
                    },
                    'smoke': {
                        'min_score': 0.5,
                        'threshold': 0.7
                    }
                }
            }
        }
        
        # Add sub stream if available for better detection performance
        if sub_stream and sub_stream != main_stream:
            camera_config['ffmpeg']['inputs'].append({
                'path': sub_stream,
                'roles': ['detect']
            })
            # Remove detect role from main stream
            camera_config['ffmpeg']['inputs'][0]['roles'] = ['record']
            
        return camera_config
        
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration."""
        # This is a simple implementation that replaces ${VAR_NAME} patterns
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
            
        # Replace ${VAR_NAME} patterns
        content = re.sub(r'\$\{([^}]+)\}', replace_var, content)
        
        # Also replace {VAR_NAME} patterns for Frigate compatibility
        content = re.sub(r'\{([^}]+)\}', replace_var, content)
        
        return content
        
    def _write_config(self, content: str):
        """Write configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
            
            # Write to temporary file first
            temp_path = f"{self.config.config_path}.tmp"
            with open(temp_path, 'w') as f:
                f.write(content)
                
            # Atomic rename
            os.rename(temp_path, self.config.config_path)
            
            self.logger.info(f"Wrote configuration to {self.config.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write config: {e}")
            
    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Shutting down camera manager")
        
        # Cancel pending timer
        if self._update_timer:
            self._update_timer.cancel()
            
        # Stop health reporter
        if hasattr(self, 'health_reporter'):
            self.health_reporter.stop()
            
        # Disconnect MQTT
        self.disconnect()
        
        # Stop threads
        self.stop_threads()


def main():
    """Main entry point."""
    manager = CameraManager()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.logger.info("Received shutdown signal")
    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()