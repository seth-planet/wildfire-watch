#!/usr/bin/env python3.12
"""Base configuration management for Wildfire Watch services.

This module provides a base class and utilities for consistent configuration
management across all services. It handles:
- Environment variable loading with type conversion
- Value validation with min/max ranges and allowed values
- Configuration export for debugging and backup
- Cross-service dependency validation
- Thread-safe access to configuration values

Key Features:
    - Schema-based validation with detailed error messages
    - Automatic type conversion from environment strings
    - Support for complex types (lists, dicts) via JSON parsing
    - Configuration compatibility checking between services
    - Export to JSON/YAML for documentation and debugging

Thread Safety:
    All configuration classes are immutable after initialization and thread-safe
    for read access. Configuration should not be modified at runtime.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Type
from abc import ABC, abstractmethod
import yaml

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigSchema:
    """Schema definition for configuration values.
    
    Attributes:
        type: Python type (str, int, float, bool, list, dict)
        required: Whether the value must be provided
        default: Default value if not provided
        min: Minimum value (for numeric types)
        max: Maximum value (for numeric types)
        choices: List of allowed values
        description: Human-readable description
    """
    
    def __init__(self, 
                 type: Type,
                 required: bool = False,
                 default: Any = None,
                 min: Optional[Union[int, float]] = None,
                 max: Optional[Union[int, float]] = None,
                 choices: Optional[List[Any]] = None,
                 description: str = ""):
        self.type = type
        self.required = required
        self.default = default
        self.min = min
        self.max = max
        self.choices = choices
        self.description = description


class ConfigBase(ABC):
    """Base class for service configuration.
    
    Subclasses should:
    1. Define a SCHEMA class variable with ConfigSchema definitions
    2. Call super().__init__() in their __init__ method
    3. Optionally override validate_cross_service() for dependency checks
    
    Example:
        class MyServiceConfig(ConfigBase):
            SCHEMA = {
                'mqtt_broker': ConfigSchema(str, required=True, description="MQTT broker host"),
                'mqtt_port': ConfigSchema(int, default=1883, min=1, max=65535),
                'timeout': ConfigSchema(float, default=30.0, min=1.0, max=300.0)
            }
    """
    
    SCHEMA: Dict[str, ConfigSchema] = {}
    
    def __init__(self, env_prefix: str = ""):
        """Initialize configuration from environment variables.
        
        Args:
            env_prefix: Optional prefix for environment variables (e.g., "CAMERA_")
            
        Raises:
            ConfigValidationError: If required values missing or validation fails
        """
        self.env_prefix = env_prefix
        self._values = {}
        
        # Load and validate all schema-defined values
        for key, schema in self.SCHEMA.items():
            env_key = f"{env_prefix}{key.upper()}"
            value = self._load_value(env_key, schema)
            self._values[key] = value
            # Set as instance attribute for easy access
            setattr(self, key, value)
            
        # Run validation
        self.validate()
        
    def _load_value(self, env_key: str, schema: ConfigSchema) -> Any:
        """Load and convert a single configuration value.
        
        Args:
            env_key: Environment variable name
            schema: Schema definition for this value
            
        Returns:
            Converted and validated value
            
        Raises:
            ConfigValidationError: If value cannot be converted or validated
        """
        raw_value = os.getenv(env_key)
        
        # Handle required values
        if raw_value is None:
            if schema.required:
                raise ConfigValidationError(
                    f"Required configuration '{env_key}' not provided"
                )
            return schema.default
            
        # Type conversion with input sanitization
        try:
            if schema.type == bool:
                value = raw_value.lower() in ('true', '1', 'yes', 'on')
            elif schema.type == list:
                # SECURITY FIX: Sanitize JSON input to prevent injection attacks
                if raw_value:
                    sanitized_input = self._sanitize_json_input(raw_value, env_key)
                    value = json.loads(sanitized_input)
                else:
                    value = []
            elif schema.type == dict:
                # SECURITY FIX: Sanitize JSON input to prevent injection attacks
                if raw_value:
                    sanitized_input = self._sanitize_json_input(raw_value, env_key)
                    value = json.loads(sanitized_input)
                else:
                    value = {}
            else:
                # SECURITY FIX: Sanitize string input to prevent injection
                if schema.type == str:
                    value = self._sanitize_string_input(raw_value, env_key)
                else:
                    value = schema.type(raw_value)
        except (ValueError, json.JSONDecodeError) as e:
            raise ConfigValidationError(
                f"Cannot convert '{env_key}' value '{raw_value}' to {schema.type.__name__}: {e}"
            )
            
        # Validate numeric ranges
        if schema.min is not None and value < schema.min:
            logger.warning(f"{env_key} value {value} below minimum {schema.min}, using minimum")
            value = schema.min
        if schema.max is not None and value > schema.max:
            logger.warning(f"{env_key} value {value} above maximum {schema.max}, using maximum")
            value = schema.max
            
        # Validate choices
        if schema.choices is not None and value not in schema.choices:
            raise ConfigValidationError(
                f"{env_key} value '{value}' not in allowed choices: {schema.choices}"
            )
            
        return value
    
    def _sanitize_json_input(self, raw_value: str, env_key: str) -> str:
        """Sanitize JSON input to prevent injection attacks.
        
        Args:
            raw_value: Raw JSON string from environment
            env_key: Environment variable name for logging
            
        Returns:
            Sanitized JSON string
            
        Raises:
            ConfigValidationError: If input contains suspicious patterns
        """
        # Check for suspicious patterns that could indicate injection attempts
        suspicious_patterns = [
            '__import__',  # Python import injection
            'eval(',       # Code evaluation
            'exec(',       # Code execution
            'open(',       # File operations
            'subprocess',  # System commands
            'os.',         # OS operations
            'sys.',        # System operations
            '\\x',         # Hex escape sequences
            '\\u',         # Unicode escape sequences
            '\\"__',       # Dunder method access
        ]
        
        for pattern in suspicious_patterns:
            if pattern in raw_value:
                raise ConfigValidationError(
                    f"Configuration '{env_key}' contains suspicious pattern '{pattern}' - potential injection attack"
                )
        
        # Limit JSON input size to prevent DoS attacks
        max_json_size = 10240  # 10KB limit
        if len(raw_value) > max_json_size:
            raise ConfigValidationError(
                f"Configuration '{env_key}' JSON input too large ({len(raw_value)} bytes > {max_json_size} bytes)"
            )
        
        # Additional check: ensure it's actually valid JSON structure (no code)
        try:
            # Parse and re-serialize to remove any potential code injection
            parsed = json.loads(raw_value)
            # Only allow basic data types: dict, list, str, int, float, bool, None
            self._validate_json_types(parsed, env_key)
            return json.dumps(parsed)  # Return clean JSON
        except json.JSONDecodeError:
            raise ConfigValidationError(f"Configuration '{env_key}' is not valid JSON")
    
    def _validate_json_types(self, obj: any, env_key: str, path: str = ""):
        """Recursively validate JSON object contains only safe data types."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise ConfigValidationError(f"Configuration '{env_key}' contains non-string key at {path}")
                self._validate_json_types(value, env_key, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._validate_json_types(item, env_key, f"{path}[{i}]")
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            raise ConfigValidationError(
                f"Configuration '{env_key}' contains unsafe type {type(obj).__name__} at {path}"
            )
    
    def _sanitize_string_input(self, raw_value: str, env_key: str) -> str:
        """Sanitize string input to prevent injection attacks.
        
        Args:
            raw_value: Raw string from environment
            env_key: Environment variable name for logging
            
        Returns:
            Sanitized string
            
        Raises:
            ConfigValidationError: If input contains suspicious patterns
        """
        # Check for command injection patterns
        injection_patterns = [
            ';',          # Command separator
            '|',          # Pipe
            '&',          # Background/AND
            '$(',         # Command substitution
            '`',          # Command substitution
            '$()',        # Command substitution
            '\n',         # Newline injection
            '\r',         # Carriage return injection
        ]
        
        for pattern in injection_patterns:
            if pattern in raw_value:
                logger.warning(f"Configuration '{env_key}' contains potentially dangerous character '{pattern}' - sanitizing")
                # Remove the dangerous character instead of failing
                raw_value = raw_value.replace(pattern, '')
        
        # Limit string size to prevent DoS
        max_string_size = 4096  # 4KB limit for strings
        if len(raw_value) > max_string_size:
            logger.warning(f"Configuration '{env_key}' string too long ({len(raw_value)} chars), truncating to {max_string_size}")
            raw_value = raw_value[:max_string_size]
        
        return raw_value
        
    def validate(self):
        """Validate all configuration values.
        
        Subclasses should override this to add custom validation logic.
        This base implementation is called automatically during __init__.
        
        Raises:
            ConfigValidationError: If validation fails
        """
        # Subclasses can override for custom validation
        pass
        
    def validate_cross_service(self, other_configs: Dict[str, 'ConfigBase']) -> List[str]:
        """Validate configuration against other services.
        
        Args:
            other_configs: Dict of service_name -> ConfigBase instance
            
        Returns:
            List of warning messages (empty if all valid)
        """
        return []
        
    def export(self, format: str = 'json', include_defaults: bool = True) -> str:
        """Export configuration in specified format.
        
        Args:
            format: 'json' or 'yaml'
            include_defaults: Include values that match defaults
            
        Returns:
            Serialized configuration string
        """
        export_data = {
            'service': self.__class__.__name__,
            'values': {}
        }
        
        for key, schema in self.SCHEMA.items():
            value = self._values[key]
            if include_defaults or value != schema.default:
                export_data['values'][key] = {
                    'value': value,
                    'type': schema.type.__name__,
                    'description': schema.description
                }
                
        if format == 'yaml':
            return yaml.dump(export_data, default_flow_style=False)
        else:
            return json.dumps(export_data, indent=2)
            
    def get_diff(self, other: 'ConfigBase') -> Dict[str, tuple]:
        """Get differences between this config and another.
        
        Args:
            other: Another ConfigBase instance to compare
            
        Returns:
            Dict of key -> (this_value, other_value) for differing values
        """
        diffs = {}
        for key in self.SCHEMA:
            if key in other.SCHEMA:
                this_val = self._values[key]
                other_val = other._values[key]
                if this_val != other_val:
                    diffs[key] = (this_val, other_val)
        return diffs


class SharedMQTTConfig(ConfigBase):
    """Shared MQTT configuration used by all services.
    
    This ensures all services use consistent MQTT settings.
    """
    
    SCHEMA = {
        'mqtt_broker': ConfigSchema(
            str, 
            required=True, 
            default='mqtt_broker',
            description="MQTT broker hostname or IP"
        ),
        'mqtt_port': ConfigSchema(
            int,
            default=1883,
            min=1,
            max=65535,
            description="MQTT broker port"
        ),
        'mqtt_tls': ConfigSchema(
            bool,
            default=False,
            description="Enable TLS encryption for MQTT"
        ),
        'tls_ca_path': ConfigSchema(
            str,
            default='/mnt/data/certs/ca.crt',
            description="Path to CA certificate for TLS"
        ),
        'mqtt_username': ConfigSchema(
            str,
            default='',
            description="MQTT username (empty for anonymous)"
        ),
        'mqtt_password': ConfigSchema(
            str,
            default='',
            description="MQTT password"
        )
    }
    
    def __init__(self):
        super().__init__()
        
    def validate(self):
        """Validate MQTT configuration."""
        # If TLS enabled, ensure CA path exists
        if self.mqtt_tls and not os.path.exists(self.tls_ca_path):
            logger.warning(f"TLS enabled but CA certificate not found at {self.tls_ca_path}")
            
        # Port 8883 is standard for MQTT over TLS
        if self.mqtt_tls and self.mqtt_port == 1883:
            logger.info("TLS enabled with standard non-TLS port 1883, consider using 8883")


# Configuration validation utilities
def validate_all_services(configs: Dict[str, ConfigBase]) -> Dict[str, List[str]]:
    """Validate all service configurations including cross-service dependencies.
    
    Args:
        configs: Dict of service_name -> ConfigBase instance
        
    Returns:
        Dict of service_name -> list of warning messages
    """
    warnings = {}
    
    # Individual service validation is done during init
    # Here we do cross-service validation
    for service_name, config in configs.items():
        service_warnings = config.validate_cross_service(configs)
        if service_warnings:
            warnings[service_name] = service_warnings
            
    return warnings


def export_all_configs(configs: Dict[str, ConfigBase], format: str = 'yaml') -> str:
    """Export all service configurations to a single document.
    
    Args:
        configs: Dict of service_name -> ConfigBase instance
        format: 'json' or 'yaml'
        
    Returns:
        Serialized configuration string
    """
    all_configs = {}
    for service_name, config in configs.items():
        all_configs[service_name] = json.loads(config.export('json'))
        
    if format == 'yaml':
        return yaml.dump(all_configs, default_flow_style=False)
    else:
        return json.dumps(all_configs, indent=2)