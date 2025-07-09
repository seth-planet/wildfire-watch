#!/usr/bin/env python3
"""Configuration for Camera Manager Service"""

import os
from typing import Dict, Any, Optional
from utils.config_base import ConfigBase


class CameraManagerConfig(ConfigBase):
    """Configuration for Camera Manager service."""
    
    def __init__(self):
        super().__init__("camera_manager")
        
    def _define_defaults(self) -> Dict[str, Any]:
        """Define default configuration values."""
        return {
            # MQTT Settings
            'mqtt_broker': 'mqtt-broker',
            'mqtt_port': 1883,
            'mqtt_tls': False,
            'mqtt_username': '',
            'mqtt_password': '',
            'mqtt_client_id': 'camera-manager',
            
            # Topic Configuration
            'topic_prefix': '',
            'camera_discovery_topic': 'camera/discovery/+',
            'bulk_config_topic': 'frigate/config/cameras',
            
            # File Paths
            'config_path': '/config/frigate.yml',
            'base_config_path': '/config/frigate_base.yml',
            'custom_cameras_path': '/config/custom_cameras.yml',
            'detected_cameras_path': '/config/detected_cameras.json',
            
            # Health Reporting
            'health_report_interval': 60,
            'health_topic': 'system/camera_manager/health',
            
            # Camera Filtering
            'camera_whitelist': '',  # Comma-separated camera IDs
            'node_id': '',
            
            # Service Behavior
            'merge_custom_cameras': True,
            'persist_detected_cameras': True,
            'config_generation_delay': 2,  # Seconds to wait before regenerating config
            
            # Logging
            'log_level': 'INFO',
        }
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # MQTT Settings
        config['mqtt_broker'] = os.getenv('MQTT_BROKER', self.defaults['mqtt_broker'])
        config['mqtt_port'] = int(os.getenv('MQTT_PORT', str(self.defaults['mqtt_port'])))
        config['mqtt_tls'] = os.getenv('MQTT_TLS', '').lower() == 'true'
        config['mqtt_username'] = os.getenv('MQTT_USERNAME', '')
        config['mqtt_password'] = os.getenv('MQTT_PASSWORD', '')
        config['mqtt_client_id'] = os.getenv('MQTT_CLIENT_ID', self.defaults['mqtt_client_id'])
        
        # Topic Configuration
        config['topic_prefix'] = os.getenv('TOPIC_PREFIX', '')
        config['camera_discovery_topic'] = os.getenv('CAMERA_DISCOVERY_TOPIC', 
                                                     self.defaults['camera_discovery_topic'])
        config['bulk_config_topic'] = os.getenv('BULK_CONFIG_TOPIC', 
                                                self.defaults['bulk_config_topic'])
        
        # File Paths
        config['config_path'] = os.getenv('FRIGATE_CONFIG_PATH', 
                                         self.defaults['config_path'])
        config['base_config_path'] = os.getenv('FRIGATE_BASE_CONFIG_PATH', 
                                               self.defaults['base_config_path'])
        config['custom_cameras_path'] = os.getenv('CUSTOM_CAMERAS_PATH', 
                                                  self.defaults['custom_cameras_path'])
        config['detected_cameras_path'] = os.getenv('DETECTED_CAMERAS_PATH', 
                                                    self.defaults['detected_cameras_path'])
        
        # Health Reporting
        config['health_report_interval'] = int(os.getenv('HEALTH_REPORT_INTERVAL', 
                                                         str(self.defaults['health_report_interval'])))
        config['health_topic'] = os.getenv('HEALTH_TOPIC', self.defaults['health_topic'])
        
        # Camera Filtering
        config['camera_whitelist'] = os.getenv('CAMERA_WHITELIST', '')
        config['node_id'] = os.getenv('NODE_ID', '')
        
        # Service Behavior
        config['merge_custom_cameras'] = os.getenv('MERGE_CUSTOM_CAMERAS', 'true').lower() == 'true'
        config['persist_detected_cameras'] = os.getenv('PERSIST_DETECTED_CAMERAS', 'true').lower() == 'true'
        config['config_generation_delay'] = int(os.getenv('CONFIG_GENERATION_DELAY', '2'))
        
        # Logging
        config['log_level'] = os.getenv('LOG_LEVEL', self.defaults['log_level']).upper()
        
        return config
    
    @property
    def mqtt_broker(self) -> str:
        return self.get('mqtt_broker')
    
    @property
    def mqtt_port(self) -> int:
        return self.get('mqtt_port')
    
    @property
    def mqtt_tls(self) -> bool:
        return self.get('mqtt_tls')
    
    @property
    def mqtt_username(self) -> Optional[str]:
        return self.get('mqtt_username') or None
    
    @property
    def mqtt_password(self) -> Optional[str]:
        return self.get('mqtt_password') or None
    
    @property
    def mqtt_client_id(self) -> str:
        return self.get('mqtt_client_id')
    
    @property
    def camera_discovery_topic(self) -> str:
        prefix = self.get('topic_prefix')
        topic = self.get('camera_discovery_topic')
        return f"{prefix}/{topic}" if prefix else topic
    
    @property
    def bulk_config_topic(self) -> str:
        prefix = self.get('topic_prefix')
        topic = self.get('bulk_config_topic')
        return f"{prefix}/{topic}" if prefix else topic
    
    @property
    def health_topic(self) -> str:
        prefix = self.get('topic_prefix')
        topic = self.get('health_topic')
        return f"{prefix}/{topic}" if prefix else topic
    
    @property
    def camera_whitelist(self) -> list:
        """Return list of whitelisted camera IDs."""
        whitelist = self.get('camera_whitelist')
        if whitelist:
            return [cam.strip() for cam in whitelist.split(',') if cam.strip()]
        return []
    
    @property
    def is_filtering_enabled(self) -> bool:
        """Check if camera filtering is enabled."""
        return bool(self.camera_whitelist)