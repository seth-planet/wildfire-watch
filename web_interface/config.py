#!/usr/bin/env python3.12
"""Configuration management for the Web Interface service.

This module handles all configuration for the status panel web interface,
including MQTT settings, web server configuration, and security options.
"""

import socket
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config_base import ConfigBase, ConfigSchema, SharedMQTTConfig, ConfigValidationError
from utils.safe_logging import safe_log

logger = logging.getLogger(__name__)


class WebInterfaceConfig(ConfigBase):
    """Configuration for the Web Interface service.
    
    Manages all settings for the status panel including:
    - HTTP server configuration
    - MQTT buffer settings
    - Debug mode and security
    - UI refresh intervals
    """
    
    SCHEMA = {
        # Service identification
        'service_id': ConfigSchema(
            str,
            default=f"web_interface_{socket.gethostname()}",
            description="Unique identifier for this web interface instance"
        ),
        
        # HTTP server settings
        'http_port': ConfigSchema(
            int,
            default=8080,
            min=1,
            max=65535,
            description="HTTP server port"
        ),
        
        'http_host': ConfigSchema(
            str,
            default='127.0.0.1',  # Secure default: localhost only
            description="HTTP server bind address"
        ),
        
        # MQTT buffer configuration
        'mqtt_buffer_size': ConfigSchema(
            int,
            default=1000,
            min=100,
            max=10000,
            description="Maximum number of MQTT events to store in memory"
        ),
        
        # UI settings
        'refresh_interval': ConfigSchema(
            int,
            default=15,
            min=5,
            max=300,
            description="UI refresh interval in seconds"
        ),
        
        # Debug and security
        'debug_mode': ConfigSchema(
            bool,
            default=False,
            description="Enable debug mode with control features"
        ),
        
        'debug_token': ConfigSchema(
            str,
            default='',
            description="Token required for debug mode access"
        ),
        
        # Performance settings
        'max_concurrent_connections': ConfigSchema(
            int,
            default=10,
            min=1,
            max=100,
            description="Maximum concurrent WebSocket connections"
        ),
        
        'enable_compression': ConfigSchema(
            bool,
            default=True,
            description="Enable gzip compression for responses"
        ),
        
        # Security settings
        'allowed_networks': ConfigSchema(
            list,
            default=['127.0.0.1', 'localhost'],  # Secure default: localhost only
            description="Network prefixes allowed to access the interface"
        ),
        
        'enable_csrf': ConfigSchema(
            bool,
            default=True,
            description="Enable CSRF protection for forms"
        ),
        
        # Logging
        'access_log': ConfigSchema(
            bool,
            default=False,
            description="Enable access logging (disabled for Pi performance)"
        ),
        
        # Feature flags
        'show_debug_info': ConfigSchema(
            bool,
            default=False,
            description="Show debug information in UI"
        ),
        
        'enable_api': ConfigSchema(
            bool,
            default=True,
            description="Enable REST API endpoints"
        ),
        
        # Resource limits
        'max_event_age': ConfigSchema(
            int,
            default=3600,
            min=60,
            max=86400,
            description="Maximum age of events to display (seconds)"
        ),
        
        # Health check
        'health_check_interval': ConfigSchema(
            int,
            default=30,
            min=10,
            max=300,
            description="Health check interval in seconds"
        ),
        
        # Additional security settings
        'require_auth': ConfigSchema(
            bool,
            default=False,  # Default to false for backward compatibility
            description="Require authentication for access"
        ),
        
        'session_timeout': ConfigSchema(
            int,
            default=900,  # 15 minutes
            min=60,
            max=3600,
            description="Session timeout in seconds"
        ),
        
        'rate_limit_enabled': ConfigSchema(
            bool,
            default=True,
            description="Enable rate limiting"
        ),
        
        'rate_limit_requests': ConfigSchema(
            int,
            default=60,
            min=10,
            max=1000,
            description="Maximum requests per minute per IP"
        ),
        
        # Audit logging
        'audit_log_enabled': ConfigSchema(
            bool,
            default=True,
            description="Enable audit logging for all access and actions"
        ),
        
        # Safety features
        'allow_remote_control': ConfigSchema(
            bool,
            default=False,  # NEVER allow by default
            description="Allow remote control of fire suppression (DANGEROUS - NOT RECOMMENDED)"
        )
    }
    
    def __init__(self):
        """Initialize web interface configuration."""
        # Load with STATUS_PANEL_ prefix for environment variables
        super().__init__(env_prefix='STATUS_PANEL_')
        
        # Also load shared MQTT configuration
        self.mqtt_config = SharedMQTTConfig()
        
    def validate(self):
        """Validate web interface configuration with enhanced security checks."""
        super().validate()
        
        # Critical: Never allow debug mode in production
        if self.debug_mode and os.getenv('DEPLOYMENT_ENV') == 'production':
            raise ConfigValidationError(
                "Debug mode cannot be enabled in production environment"
            )
        
        # Validate debug mode settings
        if self.debug_mode and not self.debug_token:
            raise ConfigValidationError(
                "Debug mode enabled but no debug token provided. "
                "Set STATUS_PANEL_DEBUG_TOKEN for security."
            )
            
        # Validate debug token strength
        if self.debug_mode and self.debug_token and len(self.debug_token) < 32:
            raise ConfigValidationError(
                "Debug token too weak. Must be at least 32 characters."
            )
            
        # Warn about performance settings
        if self.mqtt_buffer_size > 5000:
            safe_log(logger, 'warning',
                f"Large MQTT buffer size ({self.mqtt_buffer_size}) may impact "
                "performance on Raspberry Pi"
            )
            
        # Validate allowed networks format and security
        for network in self.allowed_networks:
            if not isinstance(network, str):
                raise ConfigValidationError(
                    f"Invalid network prefix in allowed_networks: {network}"
                )
            # Warn about overly permissive networks
            if network in ['0.0.0.0', '::']:
                raise ConfigValidationError(
                    "Cannot allow connections from all addresses (0.0.0.0 or ::)"
                )
                
        # Ensure localhost is always allowed for health checks
        if '127.0.0.1' not in self.allowed_networks and 'localhost' not in self.allowed_networks:
            self.allowed_networks.append('127.0.0.1')
            safe_log(logger, 'info', "Added localhost to allowed_networks for health checks")
            
        # Validate that dangerous combinations are not enabled
        if self.debug_mode and self.http_host == '0.0.0.0':
            raise ConfigValidationError(
                "Debug mode cannot be enabled when binding to all interfaces (0.0.0.0)"
            )
            
        # Check for required security features
        if not self.enable_csrf:
            safe_log(logger, 'warning', "CSRF protection disabled - this is a security risk")
            
        # Validate WebSocket connection limits for DoS protection
        if self.max_concurrent_connections > 50:
            safe_log(logger, 'warning',
                f"High concurrent connection limit ({self.max_concurrent_connections}) "
                "may enable DoS attacks"
            )
            
        # Critical safety check
        if self.allow_remote_control:
            safe_log(logger, 'error',
                "CRITICAL WARNING: Remote control is enabled. This allows web-based "
                "control of the fire suppression system and should NEVER be used in production!"
            )
                
    def validate_cross_service(self, other_configs):
        """Validate configuration against other services."""
        warnings = []
        
        # Check MQTT broker availability
        if 'mqtt_broker' not in other_configs:
            warnings.append(
                "MQTT broker service not found - web interface will not receive data"
            )
            
        # Check if any services are configured to publish
        expected_services = ['gpio_trigger', 'fire_consensus', 'camera_detector']
        missing_services = [s for s in expected_services if s not in other_configs]
        
        if missing_services:
            warnings.append(
                f"Expected services not configured: {', '.join(missing_services)}. "
                "Web interface may have limited data."
            )
            
        return warnings
        
    def get_mqtt_topics(self):
        """Get list of MQTT topics to subscribe to."""
        return [
            # Service health topics
            'system/+/health',
            'system/+/lwt',
            
            # GPIO trigger telemetry
            'system/trigger_telemetry/+',
            
            # Fire detection and consensus
            'fire/trigger',
            'fire/detection/+',
            'fire/consensus_state',
            'consensus/state',
            
            # Camera system
            'camera/discovery/+',
            'camera/status/+',
            'telemetry/camera/+',
            
            # Frigate events
            'frigate/events',
            'frigate/+/fire',
            'frigate/+/smoke',
            
            # GPIO states
            'gpio/status',
            'gpio/state_change',
            
            # System telemetry
            'telemetry/+/+',
        ]
        
    def get_publish_topics(self):
        """Get list of MQTT topics this service publishes to."""
        base_topics = [
            f'system/{self.service_id}/health',
            f'system/{self.service_id}/lwt',
        ]
        
        # Add debug topics if enabled
        if self.debug_mode:
            base_topics.extend([
                'status_panel/debug/request',
                'status_panel/manual_trigger',
            ])
            
        return base_topics


# Module-level configuration instance
_config = None


def get_config():
    """Get or create the configuration instance.
    
    Returns:
        WebInterfaceConfig: The configuration instance
    """
    global _config
    if _config is None:
        _config = WebInterfaceConfig()
    return _config