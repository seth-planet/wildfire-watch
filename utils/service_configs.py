#!/usr/bin/env python3.12
"""Service-specific configuration classes for Wildfire Watch.

This module defines configuration schemas for each service, building on the
ConfigBase foundation. Each service configuration:
- Inherits shared configuration (like MQTT settings)
- Defines service-specific parameters with validation
- Implements cross-service dependency checks
- Provides sensible defaults for all settings

Thread Safety:
    All configuration classes are immutable after initialization.
"""

import json
from typing import List, Dict, Optional
from .config_base import ConfigBase, ConfigSchema, SharedMQTTConfig, ConfigValidationError


class CameraDetectorConfig(SharedMQTTConfig):
    """Configuration for the Camera Detector service."""
    
    # Extend parent schema with service-specific settings
    SCHEMA = {
        **SharedMQTTConfig.SCHEMA,
        
        # Discovery Settings
        'discovery_interval': ConfigSchema(
            int,
            default=300,
            min=30,
            max=3600,
            description="Seconds between discovery scans"
        ),
        'rtsp_timeout': ConfigSchema(
            int,
            default=10,
            min=1,
            max=60,
            description="RTSP stream validation timeout (seconds)"
        ),
        'onvif_timeout': ConfigSchema(
            int,
            default=5,
            min=1,
            max=30,
            description="ONVIF connection timeout (seconds)"
        ),
        'mac_tracking_enabled': ConfigSchema(
            bool,
            default=True,
            description="Track cameras by MAC address"
        ),
        
        # Smart Discovery
        'smart_discovery_enabled': ConfigSchema(
            bool,
            default=True,
            description="Use resource-efficient discovery mode"
        ),
        'initial_discovery_count': ConfigSchema(
            int,
            default=3,
            min=1,
            max=10,
            description="Aggressive scans at startup"
        ),
        'steady_state_interval': ConfigSchema(
            int,
            default=1800,
            min=300,
            max=7200,
            description="Full scan interval in steady state"
        ),
        'quick_check_interval': ConfigSchema(
            int,
            default=60,
            min=30,
            max=300,
            description="Quick health check interval"
        ),
        
        # Camera Credentials
        'default_username': ConfigSchema(
            str,
            default='admin',
            description="Default camera username"
        ),
        'default_password': ConfigSchema(
            str,
            default='',
            description="Default camera password"
        ),
        'camera_credentials': ConfigSchema(
            str,
            default='admin:,username:password,username:password',
            description="Comma-separated user:pass pairs"
        ),
        
        # Health Monitoring
        'health_check_interval': ConfigSchema(
            int,
            default=60,
            min=10,
            max=300,
            description="Camera health check interval"
        ),
        'offline_threshold': ConfigSchema(
            int,
            default=180,
            min=60,
            max=600,
            description="Seconds before marking camera offline"
        ),
        
        # Frigate Integration
        'frigate_config_path': ConfigSchema(
            str,
            default='/config/frigate/cameras.yml',
            description="Path to write Frigate configuration"
        ),
        'frigate_update_enabled': ConfigSchema(
            bool,
            default=True,
            description="Auto-update Frigate configuration"
        ),
        
        # Camera Resolution Support
        'supported_resolutions': ConfigSchema(
            list,
            default=['1920x1080', '1280x720', '640x480', '3840x2160'],
            description="List of supported camera resolutions"
        ),
        'default_resolution': ConfigSchema(
            str,
            default='1920x1080',
            choices=['1920x1080', '1280x720', '640x480', '3840x2160'],
            description="Default resolution for camera config"
        )
    }
    
    def validate(self):
        """Validate camera detector configuration."""
        super().validate()
        
        # Parse and validate credentials
        try:
            creds = self.camera_credentials.split(',')
            for cred in creds:
                if ':' not in cred:
                    raise ConfigValidationError(
                        f"Invalid credential format '{cred}', expected 'user:pass'"
                    )
        except Exception as e:
            raise ConfigValidationError(f"Failed to parse camera credentials: {e}")
            
        # Validate timing relationships
        if self.quick_check_interval >= self.discovery_interval:
            raise ConfigValidationError(
                "quick_check_interval must be less than discovery_interval"
            )
            
        if self.offline_threshold <= self.health_check_interval:
            raise ConfigValidationError(
                "offline_threshold must be greater than health_check_interval"
            )


class FireConsensusConfig(SharedMQTTConfig):
    """Configuration for the Fire Consensus service."""
    
    SCHEMA = {
        **SharedMQTTConfig.SCHEMA,
        
        # Core Consensus
        'consensus_threshold': ConfigSchema(
            int,
            default=2,
            min=1,
            max=10,
            description="Number of cameras required for consensus"
        ),
        'single_camera_trigger': ConfigSchema(
            bool,
            default=False,
            description="Allow single camera to trigger (overrides threshold)"
        ),
        'detection_window': ConfigSchema(
            float,
            default=30.0,
            min=10.0,
            max=120.0,
            description="Time window for detection history (seconds)"
        ),
        'cooldown_period': ConfigSchema(
            float,
            default=300.0,
            min=60.0,
            max=1800.0,
            description="Minimum time between triggers (seconds)"
        ),
        
        # Detection Filtering
        'min_confidence': ConfigSchema(
            float,
            default=0.7,
            min=0.5,
            max=0.99,
            description="Minimum ML confidence score"
        ),
        'min_area_ratio': ConfigSchema(
            float,
            default=0.001,
            min=0.0001,
            max=0.1,
            description="Minimum fire area as fraction of frame"
        ),
        'max_area_ratio': ConfigSchema(
            float,
            default=0.8,
            min=0.1,
            max=1.0,
            description="Maximum fire area as fraction of frame"
        ),
        
        # Growth Analysis
        'area_increase_ratio': ConfigSchema(
            float,
            default=1.2,
            min=1.0,
            max=2.0,
            description="Required growth ratio (1.2 = 20% growth)"
        ),
        'moving_average_window': ConfigSchema(
            int,
            default=3,
            min=1,
            max=10,
            description="Detections for moving average smoothing"
        ),
        
        # Camera Health
        'camera_timeout': ConfigSchema(
            float,
            default=180.0,
            min=60.0,
            max=600.0,
            description="Seconds before marking camera offline"
        ),
        
        # Zone Control
        'zone_activation': ConfigSchema(
            bool,
            default=False,
            description="Enable zone-based sprinkler control"
        ),
        'zone_mapping': ConfigSchema(
            dict,
            default={},
            description="Camera ID to sprinkler zone mapping"
        ),
        
        # Camera Resolution Handling
        'camera_resolutions': ConfigSchema(
            dict,
            default={},
            description="Camera ID to resolution mapping (auto-populated)"
        ),
        'default_camera_resolution': ConfigSchema(
            str,
            default='1920x1080',
            description="Default resolution if not specified"
        )
    }
    
    def validate(self):
        """Validate fire consensus configuration."""
        super().validate()
        
        # Single camera override warning
        if self.single_camera_trigger:
            import logging
            logging.getLogger(__name__).warning(
                "SINGLE_CAMERA_TRIGGER enabled - consensus threshold ignored!"
            )
            
        # Validate area ratios
        if self.min_area_ratio >= self.max_area_ratio:
            raise ConfigValidationError(
                "min_area_ratio must be less than max_area_ratio"
            )
            
        # Validate detection window vs moving average
        min_window = self.moving_average_window * 2  # Need at least 2 samples per window
        if self.detection_window < min_window:
            raise ConfigValidationError(
                f"detection_window ({self.detection_window}s) too small for "
                f"moving_average_window ({self.moving_average_window})"
            )
            
    def validate_cross_service(self, other_configs: Dict[str, ConfigBase]) -> List[str]:
        """Check dependencies with camera detector."""
        warnings = []
        
        # Check if we have camera detector config
        if 'camera_detector' in other_configs:
            cam_config = other_configs['camera_detector']
            
            # Warn if consensus threshold might be too high
            if self.consensus_threshold > 4 and not self.single_camera_trigger:
                warnings.append(
                    f"High consensus threshold ({self.consensus_threshold}) may prevent triggers "
                    "if not enough cameras are available"
                )
                
        return warnings


class GPIOTriggerConfig(SharedMQTTConfig):
    """Configuration for the GPIO Trigger service (safety critical)."""
    
    SCHEMA = {
        **SharedMQTTConfig.SCHEMA,
        
        # MQTT Topics
        'trigger_topic': ConfigSchema(
            str,
            default='fire/trigger',
            description="Topic to receive fire triggers"
        ),
        'emergency_topic': ConfigSchema(
            str,
            default='fire/emergency',
            description="Topic for emergency manual control"
        ),
        'telemetry_topic': ConfigSchema(
            str,
            default='system/trigger_telemetry',
            description="Topic for status telemetry"
        ),
        
        # GPIO Pins - Control
        'main_valve_pin': ConfigSchema(
            int,
            default=18,
            min=2,
            max=27,
            description="Main water valve control pin"
        ),
        'ign_start_pin': ConfigSchema(
            int,
            default=23,
            min=2,
            max=27,
            description="Engine ignition start pin"
        ),
        'ign_on_pin': ConfigSchema(
            int,
            default=24,
            min=2,
            max=27,
            description="Engine ignition on pin"
        ),
        'ign_off_pin': ConfigSchema(
            int,
            default=25,
            min=2,
            max=27,
            description="Engine ignition off pin"
        ),
        'refill_valve_pin': ConfigSchema(
            int,
            default=22,
            min=2,
            max=27,
            description="Refill valve control pin"
        ),
        
        # Critical Safety Timings
        'max_engine_runtime': ConfigSchema(
            float,
            default=1800.0,
            min=60.0,
            max=7200.0,
            description="Maximum engine runtime (seconds) - CRITICAL SAFETY PARAMETER"
        ),
        'refill_multiplier': ConfigSchema(
            float,
            default=40.0,
            min=1.0,
            max=100.0,
            description="Refill time = runtime * multiplier"
        ),
        'fire_off_delay': ConfigSchema(
            float,
            default=1800.0,
            min=300.0,
            max=3600.0,
            description="Engine shutoff delay after fire signal"
        ),
        
        # Valve Timings
        'pre_open_delay': ConfigSchema(
            float,
            default=2.0,
            min=0.5,
            max=10.0,
            description="Valve open before engine start (seconds)"
        ),
        'valve_close_delay': ConfigSchema(
            float,
            default=600.0,
            min=60.0,
            max=1800.0,
            description="Valve close delay after engine stop"
        ),
        
        # Safety Features
        'max_dry_run_time': ConfigSchema(
            float,
            default=300.0,
            min=30.0,
            max=600.0,
            description="Max time without water flow before stop"
        ),
        'hardware_validation_enabled': ConfigSchema(
            bool,
            default=False,
            description="Enable relay feedback validation"
        ),
        'gpio_simulation': ConfigSchema(
            bool,
            default=False,
            description="Enable GPIO simulation mode (auto-detected)"
        ),
        
        # Water Capacity Configuration
        'tank_capacity_gallons': ConfigSchema(
            float,
            default=500.0,
            min=50.0,
            max=10000.0,
            description="Water tank capacity in gallons"
        ),
        'pump_flow_rate_gpm': ConfigSchema(
            float,
            default=50.0,
            min=5.0,
            max=500.0,
            description="Pump flow rate in gallons per minute"
        )
    }
    
    def validate(self):
        """Validate GPIO trigger configuration with safety checks."""
        super().validate()
        
        # Critical safety validation
        calculated_runtime = (self.tank_capacity_gallons / self.pump_flow_rate_gpm) * 60
        safety_margin = 0.8  # 80% safety margin
        safe_runtime = calculated_runtime * safety_margin
        
        if self.max_engine_runtime > safe_runtime:
            raise ConfigValidationError(
                f"CRITICAL: max_engine_runtime ({self.max_engine_runtime}s) exceeds "
                f"safe runtime ({safe_runtime:.0f}s) based on tank capacity "
                f"({self.tank_capacity_gallons}gal) and flow rate ({self.pump_flow_rate_gpm}gpm)"
            )
            
        # Check for GPIO pin conflicts
        pins = [
            self.main_valve_pin, self.ign_start_pin, self.ign_on_pin,
            self.ign_off_pin, self.refill_valve_pin
        ]
        if len(pins) != len(set(pins)):
            raise ConfigValidationError("GPIO pin conflict - pins must be unique")
            
        # Validate timing relationships
        if self.max_dry_run_time >= self.max_engine_runtime:
            raise ConfigValidationError(
                "max_dry_run_time must be less than max_engine_runtime"
            )
            
    def validate_cross_service(self, other_configs: Dict[str, ConfigBase]) -> List[str]:
        """Check safety-critical cross-service dependencies."""
        warnings = []
        
        # Check consensus configuration
        if 'fire_consensus' in other_configs:
            consensus = other_configs['fire_consensus']
            if consensus.cooldown_period < self.fire_off_delay:
                warnings.append(
                    f"Consensus cooldown ({consensus.cooldown_period}s) is less than "
                    f"fire_off_delay ({self.fire_off_delay}s) - may cause issues"
                )
                
        return warnings


class TelemetryConfig(SharedMQTTConfig):
    """Configuration for the Telemetry service."""
    
    SCHEMA = {
        **SharedMQTTConfig.SCHEMA,
        
        'telemetry_interval': ConfigSchema(
            int,
            default=60,
            min=10,
            max=300,
            description="Telemetry report interval (seconds)"
        ),
        'retention_days': ConfigSchema(
            int,
            default=7,
            min=1,
            max=30,
            description="Days to retain telemetry data"
        ),
        'enable_prometheus': ConfigSchema(
            bool,
            default=False,
            description="Enable Prometheus metrics export"
        ),
        'prometheus_port': ConfigSchema(
            int,
            default=9090,
            min=1024,
            max=65535,
            description="Prometheus metrics port"
        )
    }


# Convenience function to load all configurations
def load_all_configs() -> Dict[str, ConfigBase]:
    """Load all service configurations with validation.
    
    Returns:
        Dict of service_name -> ConfigBase instance
        
    Raises:
        ConfigValidationError: If any configuration is invalid
    """
    configs = {
        'camera_detector': CameraDetectorConfig(),
        'fire_consensus': FireConsensusConfig(),
        'gpio_trigger': GPIOTriggerConfig(),
        'telemetry': TelemetryConfig()
    }
    
    # Run cross-service validation
    from .config_base import validate_all_services
    warnings = validate_all_services(configs)
    
    # Log warnings
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for service, service_warnings in warnings.items():
            for warning in service_warnings:
                logger.warning(f"{service}: {warning}")
                
    return configs