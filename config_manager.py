#!/usr/bin/env python3.12
"""Centralized Configuration Manager for Wildfire Watch

This module provides centralized configuration management to decouple
services and handle Docker vs bare metal differences gracefully.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """Deployment environment detection"""
    BARE_METAL = "bare_metal"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    BALENA = "balena"

class ServiceDiscovery:
    """Service discovery with environment awareness"""
    
    def __init__(self):
        self.mode = self._detect_deployment_mode()
        self._service_cache = {}
        
    def _detect_deployment_mode(self) -> DeploymentMode:
        """Detect current deployment mode"""
        if os.path.exists('/.dockerenv'):
            return DeploymentMode.DOCKER
        elif os.environ.get('KUBERNETES_SERVICE_HOST'):
            return DeploymentMode.KUBERNETES
        elif os.environ.get('BALENA_DEVICE_UUID'):
            return DeploymentMode.BALENA
        else:
            return DeploymentMode.BARE_METAL
    
    def get_service_host(self, service_name: str) -> str:
        """Get hostname for a service based on deployment mode
        
        Args:
            service_name: Name of the service (e.g., 'mqtt-broker', 'frigate')
            
        Returns:
            Appropriate hostname for the deployment mode
        """
        # Check cache first
        if service_name in self._service_cache:
            return self._service_cache[service_name]
        
        # Environment variable override (highest priority)
        env_key = f"{service_name.upper().replace('-', '_')}_HOST"
        if env_host := os.environ.get(env_key):
            self._service_cache[service_name] = env_host
            return env_host
        
        # Deployment mode specific resolution
        if self.mode == DeploymentMode.DOCKER:
            # In Docker, use service names from docker-compose
            host = service_name
        elif self.mode == DeploymentMode.KUBERNETES:
            # In K8s, use service.namespace pattern
            namespace = os.environ.get('K8S_NAMESPACE', 'default')
            host = f"{service_name}.{namespace}.svc.cluster.local"
        elif self.mode == DeploymentMode.BALENA:
            # In Balena, services can reach each other by name
            host = service_name
        else:
            # Bare metal defaults to localhost
            host = 'localhost'
        
        self._service_cache[service_name] = host
        logger.debug(f"Resolved {service_name} to {host} in {self.mode.value} mode")
        return host
    
    def get_service_port(self, service_name: str, default_port: int) -> int:
        """Get port for a service
        
        Args:
            service_name: Name of the service
            default_port: Default port if not overridden
            
        Returns:
            Port number
        """
        env_key = f"{service_name.upper().replace('-', '_')}_PORT"
        return int(os.environ.get(env_key, default_port))

@dataclass
class MQTTTopicConfig:
    """Centralized MQTT topic configuration"""
    # Core topics
    detection_fire: str = "detection/fire"
    detection_smoke: str = "detection/smoke"
    
    # Camera topics
    camera_discovery: str = "camera/discovery/+"
    camera_status: str = "camera/status/+"
    camera_telemetry: str = "telemetry/camera/+"
    
    # Frigate topics
    frigate_events: str = "frigate/+/+"
    frigate_config: str = "frigate/config/cameras"
    frigate_reload: str = "frigate/config/reload"
    
    # Trigger topics
    trigger_fire: str = "trigger/fire_detected"
    trigger_emergency: str = "fire/emergency"
    
    # GPIO topics
    gpio_status: str = "gpio/status"
    gpio_telemetry: str = "system/trigger_telemetry"
    
    # Health topics
    health_prefix: str = "system"
    
    def get_service_health_topic(self, service_name: str) -> str:
        """Get health topic for a service"""
        return f"{self.health_prefix}/{service_name}_health"
    
    def get_lwt_topic(self, service_name: str, node_id: str) -> str:
        """Get Last Will and Testament topic"""
        return f"{self.health_prefix}/{service_name}_health/{node_id}/lwt"

class PathResolver:
    """Resolve paths for Docker vs bare metal deployments"""
    
    def __init__(self, deployment_mode: DeploymentMode):
        self.mode = deployment_mode
        self.app_root = self._get_app_root()
        
    def _get_app_root(self) -> Path:
        """Get application root directory"""
        if self.mode in [DeploymentMode.DOCKER, DeploymentMode.BALENA]:
            # In containers, app is typically in /app
            return Path('/app')
        else:
            # Bare metal uses script location
            return Path(__file__).parent
    
    def get_config_path(self, config_file: str) -> Path:
        """Get configuration file path
        
        Args:
            config_file: Config filename
            
        Returns:
            Full path to config file
        """
        # Check environment variable override
        env_key = f"{config_file.upper().replace('.', '_')}_PATH"
        if env_path := os.environ.get(env_key):
            return Path(env_path)
        
        # Default paths
        if self.mode == DeploymentMode.DOCKER:
            return self.app_root / 'config' / config_file
        else:
            return self.app_root / 'configs' / config_file
    
    def get_cert_path(self, cert_type: str = 'ca') -> Path:
        """Get certificate path
        
        Args:
            cert_type: Type of cert ('ca', 'server', 'client')
            
        Returns:
            Path to certificate
        """
        cert_dir = os.environ.get('CERT_DIR', '/app/certs')
        
        cert_files = {
            'ca': 'ca.crt',
            'server': 'server.crt',
            'client': 'client.crt',
            'server_key': 'server.key',
            'client_key': 'client.key'
        }
        
        return Path(cert_dir) / cert_files.get(cert_type, f"{cert_type}.crt")
    
    def get_data_path(self, data_type: str) -> Path:
        """Get data directory path
        
        Args:
            data_type: Type of data ('models', 'recordings', 'snapshots')
            
        Returns:
            Path to data directory
        """
        # Check environment override
        env_key = f"{data_type.upper()}_PATH"
        if env_path := os.environ.get(env_key):
            return Path(env_path)
        
        # Default paths
        if self.mode == DeploymentMode.DOCKER:
            base = Path('/data')
        else:
            base = self.app_root / 'data'
        
        return base / data_type

class ConfigManager:
    """Central configuration manager for all services"""
    
    def __init__(self, service_name: str):
        """Initialize config manager for a service
        
        Args:
            service_name: Name of the service using this config
        """
        self.service_name = service_name
        self.discovery = ServiceDiscovery()
        self.paths = PathResolver(self.discovery.mode)
        self.topics = MQTTTopicConfig()
        
        # Load service-specific config
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from files and environment"""
        config = {}
        
        # Try to load from config file
        config_file = self.paths.get_config_path(f"{self.service_name}.yaml")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        
        # Override with environment variables
        self._apply_env_overrides(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides to config
        
        Environment variables follow pattern:
        SERVICE_NAME_CONFIG_KEY (all uppercase)
        """
        prefix = f"{self.service_name.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract config key
                config_key = key[len(prefix):].lower()
                
                # Handle nested keys (e.g., MQTT_BROKER_HOST)
                if '_' in config_key:
                    parts = config_key.split('_')
                    current = config
                    
                    # Navigate to nested dict
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Set final value
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    config[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type"""
        # Try to parse as JSON first (handles arrays, objects)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Default to string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Support dot notation for nested keys
        parts = key.split('.')
        current = self._config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def get_mqtt_config(self) -> Dict[str, Any]:
        """Get MQTT configuration with service discovery"""
        return {
            'broker': self.discovery.get_service_host('mqtt-broker'),
            'port': self.discovery.get_service_port('mqtt-broker', 1883),
            'tls_enabled': self.get('mqtt.tls', False),
            'tls_port': self.discovery.get_service_port('mqtt-broker-tls', 8883),
            'ca_cert': str(self.paths.get_cert_path('ca')),
            'keepalive': self.get('mqtt.keepalive', 60),
            'qos': self.get('mqtt.qos', 1),
        }
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """Get configuration for another service
        
        Args:
            service: Service name
            
        Returns:
            Service connection configuration
        """
        return {
            'host': self.discovery.get_service_host(service),
            'port': self.discovery.get_service_port(
                service, 
                self.get(f'services.{service}.default_port', 80)
            ),
            'tls': self.get(f'services.{service}.tls', False)
        }
    
    def save_runtime_config(self, config: Dict[str, Any], filename: Optional[str] = None):
        """Save runtime configuration for debugging
        
        Args:
            config: Configuration to save
            filename: Optional filename (defaults to service name)
        """
        filename = filename or f"{self.service_name}_runtime.json"
        runtime_path = self.paths.get_data_path('configs') / filename
        
        try:
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            with open(runtime_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.debug(f"Saved runtime config to {runtime_path}")
        except Exception as e:
            logger.error(f"Failed to save runtime config: {e}")

# Service-specific configuration classes
class CameraDetectorConfig(ConfigManager):
    """Configuration for Camera Detector service"""
    
    def __init__(self):
        super().__init__('camera_detector')
        
    @property
    def discovery_interval(self) -> int:
        return self.get('discovery.interval', 300)
    
    @property
    def rtsp_timeout(self) -> int:
        return self.get('rtsp.timeout', 10)
    
    @property
    def camera_credentials(self) -> List[str]:
        creds = self.get('cameras.credentials', [])
        if isinstance(creds, str):
            # Parse comma-separated string
            return [c.strip() for c in creds.split(',')]
        return creds

class FireConsensusConfig(ConfigManager):
    """Configuration for Fire Consensus service"""
    
    def __init__(self):
        super().__init__('fire_consensus')
    
    @property
    def consensus_threshold(self) -> int:
        return self.get('consensus.threshold', 2)
    
    @property
    def min_confidence(self) -> float:
        return self.get('detection.min_confidence', 0.7)
    
    @property
    def time_window(self) -> float:
        return self.get('consensus.time_window', 30.0)

class GPIOTriggerConfig(ConfigManager):
    """Configuration for GPIO Trigger service"""
    
    def __init__(self):
        super().__init__('gpio_trigger')
    
    @property
    def max_runtime(self) -> int:
        return self.get('engine.max_runtime', 1800)
    
    @property 
    def gpio_simulation(self) -> bool:
        return self.get('gpio.simulation', not self._is_raspberry_pi())
    
    def _is_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return 'Raspberry Pi' in f.read()
        except:
            return False
    
    def get_pin_config(self) -> Dict[str, int]:
        """Get GPIO pin configuration"""
        defaults = {
            'ENGINE_START': 17,
            'ENGINE_STOP': 27,
            'MAIN_VALVE': 22,
            'PRIMING_VALVE': 23,
            'REFILL_VALVE': 24,
        }
        
        pins = {}
        for name, default_pin in defaults.items():
            pins[name] = self.get(f'pins.{name.lower()}', default_pin)
        
        return pins

def demo_config_manager():
    """Demonstrate configuration management"""
    print("Configuration Manager Demo")
    print("=" * 50)
    
    # Create config manager
    config = ConfigManager('demo_service')
    
    print(f"Deployment Mode: {config.discovery.mode.value}")
    print(f"App Root: {config.paths.app_root}")
    print()
    
    # Service discovery
    print("Service Discovery:")
    for service in ['mqtt-broker', 'frigate', 'camera-detector']:
        host = config.discovery.get_service_host(service)
        print(f"  {service}: {host}")
    print()
    
    # MQTT configuration
    mqtt_config = config.get_mqtt_config()
    print("MQTT Configuration:")
    for key, value in mqtt_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Topic configuration
    print("MQTT Topics:")
    print(f"  Fire Detection: {config.topics.detection_fire}")
    print(f"  Camera Discovery: {config.topics.camera_discovery}")
    print(f"  Service Health: {config.topics.get_service_health_topic('demo')}")
    
    # Path resolution
    print("\nPath Resolution:")
    print(f"  Config: {config.paths.get_config_path('demo.yaml')}")
    print(f"  Models: {config.paths.get_data_path('models')}")
    print(f"  CA Cert: {config.paths.get_cert_path('ca')}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_config_manager()