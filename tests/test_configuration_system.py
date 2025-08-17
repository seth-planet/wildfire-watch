#!/usr/bin/env python3.12
"""Tests for the new configuration management system."""

import os
import pytest
import tempfile
import json
from unittest.mock import patch

# Add parent directory to path
import sys

# Test tier markers for organization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.smoke,
]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.config_base import ConfigBase, ConfigSchema, ConfigValidationError, SharedMQTTConfig
from utils.service_configs import (
    CameraDetectorConfig, FireConsensusConfig, GPIOTriggerConfig,
    TelemetryConfig, load_all_configs
)


class TestConfigBase:
    """Test the base configuration functionality."""
    
    def test_config_schema_validation(self):
        """Test ConfigSchema validation."""
        schema = ConfigSchema(
            int,
            required=True,
            min=1,
            max=100,
            description="Test value"
        )
        
        assert schema.type == int
        assert schema.required == True
        assert schema.min == 1
        assert schema.max == 100
        
    def test_shared_mqtt_config(self):
        """Test SharedMQTTConfig loading."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'test-broker',
            'MQTT_PORT': '1883',
            'MQTT_TLS': 'false'
        }):
            config = SharedMQTTConfig()
            assert config.mqtt_broker == 'test-broker'
            assert config.mqtt_port == 1883
            assert config.mqtt_tls == False
            
    def test_mqtt_config_validation(self):
        """Test MQTT configuration validation."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'test-broker',
            'MQTT_PORT': '70000',  # Invalid port
            'MQTT_TLS': 'true'
        }):
            config = SharedMQTTConfig()
            # Should clamp to max valid port
            assert config.mqtt_port == 65535
            
    def test_config_export_json(self):
        """Test configuration export to JSON."""
        with patch.dict(os.environ, {'MQTT_BROKER': 'test-broker'}):
            config = SharedMQTTConfig()
            exported = config.export('json')
            data = json.loads(exported)
            
            assert data['service'] == 'SharedMQTTConfig'
            assert 'mqtt_broker' in data['values']
            assert data['values']['mqtt_broker']['value'] == 'test-broker'


class TestServiceConfigs:
    """Test service-specific configurations."""
    
    def test_camera_detector_config(self):
        """Test CameraDetectorConfig validation."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'DISCOVERY_INTERVAL': '300',
            'CAMERA_CREDENTIALS': 'admin:pass1,user:pass2',
            'QUICK_CHECK_INTERVAL': '60',  # Must be less than discovery_interval
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'  # Must be greater than health_check_interval
        }):
            config = CameraDetectorConfig()
            assert config.discovery_interval == 300
            assert config.camera_credentials == 'admin:pass1,user:pass2'
            
    def test_camera_detector_validation_error(self):
        """Test CameraDetectorConfig validation errors."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'CAMERA_CREDENTIALS': 'invalid-format'  # Missing colon
        }):
            with pytest.raises(ConfigValidationError) as exc:
                CameraDetectorConfig()
            assert "Invalid credential format" in str(exc.value)
            
    def test_fire_consensus_config(self):
        """Test FireConsensusConfig loading."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'CONSENSUS_THRESHOLD': '3',
            'MIN_CONFIDENCE': '0.8',
            'SINGLE_CAMERA_TRIGGER': 'false'
        }):
            config = FireConsensusConfig()
            assert config.consensus_threshold == 3
            assert config.min_confidence == 0.8
            assert config.single_camera_trigger == False
            
    def test_fire_consensus_validation(self):
        """Test FireConsensusConfig validation."""
        # Test that values are clamped to safe ranges
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MIN_CONFIDENCE': '1.5',  # Above max, should be clamped
            'DETECTION_WINDOW': '5'  # Below min, should be clamped
        }):
            config = FireConsensusConfig()
            # Values should be clamped, not raise errors
            assert config.min_confidence == 0.99  # Clamped to max
            assert config.detection_window == 10.0  # Clamped to min
            
    def test_gpio_trigger_config_safety(self):
        """Test GPIOTriggerConfig safety validation."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '3600',  # 60 minutes
            'TANK_CAPACITY_GALLONS': '100',  # Small tank
            'PUMP_FLOW_RATE_GPM': '50'  # High flow rate
        }):
            # Should fail safety validation
            with pytest.raises(ConfigValidationError) as exc:
                GPIOTriggerConfig()
            assert "CRITICAL" in str(exc.value)
            assert "exceeds safe runtime" in str(exc.value)
            
    def test_gpio_trigger_config_safe(self):
        """Test GPIOTriggerConfig with safe values."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '300',  # 5 minutes
            'TANK_CAPACITY_GALLONS': '500',
            'PUMP_FLOW_RATE_GPM': '50',
            'MAX_DRY_RUN_TIME': '180'  # Less than engine runtime
        }):
            config = GPIOTriggerConfig()
            assert config.max_engine_runtime == 300.0
            assert config.tank_capacity_gallons == 500.0
            
    def test_gpio_pin_conflicts(self):
        """Test GPIO pin conflict detection."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAIN_VALVE_PIN': '18',
            'IGN_START_PIN': '18',  # Conflict!
            'MAX_ENGINE_RUNTIME': '300'
        }):
            with pytest.raises(ConfigValidationError) as exc:
                GPIOTriggerConfig()
            assert "GPIO pin conflict" in str(exc.value)


class TestCrossServiceValidation:
    """Test cross-service configuration validation."""
    
    def test_load_all_configs(self):
        """Test loading all service configurations."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '400',  # Safe value
            'MAX_DRY_RUN_TIME': '180',
            'QUICK_CHECK_INTERVAL': '60',
            'DISCOVERY_INTERVAL': '300',
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'
        }):
            configs = load_all_configs()
            assert 'camera_detector' in configs
            assert 'fire_consensus' in configs
            assert 'gpio_trigger' in configs
            assert 'telemetry' in configs
            
    def test_cross_service_warnings(self):
        """Test cross-service validation warnings."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '400',
            'MAX_DRY_RUN_TIME': '180',
            'CONSENSUS_THRESHOLD': '5',  # High threshold
            'SINGLE_CAMERA_TRIGGER': 'false',
            'QUICK_CHECK_INTERVAL': '60',
            'DISCOVERY_INTERVAL': '300',
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'
        }):
            configs = load_all_configs()
            consensus = configs['fire_consensus']
            warnings = consensus.validate_cross_service(configs)
            
            # Should warn about high threshold
            assert any('High consensus threshold' in w for w in warnings)
            
    def test_config_diff(self):
        """Test configuration difference detection."""
        with patch.dict(os.environ, {'MQTT_BROKER': 'localhost'}):
            config1 = SharedMQTTConfig()
            
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'other-broker',
            'MQTT_PORT': '8883'
        }):
            config2 = SharedMQTTConfig()
            
        diffs = config1.get_diff(config2)
        assert 'mqtt_broker' in diffs
        assert diffs['mqtt_broker'] == ('localhost', 'other-broker')
        assert 'mqtt_port' in diffs
        assert diffs['mqtt_port'] == (1883, 8883)


class TestConfigurationCLI:
    """Test configuration CLI functionality."""
    
    def test_cli_validate_success(self):
        """Test CLI validate command success."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '400',
            'MAX_DRY_RUN_TIME': '180',
            'QUICK_CHECK_INTERVAL': '60',
            'DISCOVERY_INTERVAL': '300',
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'
        }):
            from utils.config_cli import validate_command
            from argparse import Namespace
            
            args = Namespace()
            result = validate_command(args)
            assert result == 0
            
    def test_cli_export(self):
        """Test CLI export command."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '400',
            'MAX_DRY_RUN_TIME': '180',
            'QUICK_CHECK_INTERVAL': '60',
            'DISCOVERY_INTERVAL': '300',
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'
        }):
            from utils.config_cli import export_command
            from argparse import Namespace
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                args = Namespace(format='json', output=f.name)
                result = export_command(args)
                assert result == 0
                
            # Check file was created
            with open(f.name, 'r') as f:
                data = json.load(f)
                assert 'camera_detector' in data
                assert 'gpio_trigger' in data
                
            os.unlink(f.name)
            
    def test_cli_compatibility_check(self):
        """Test CLI compatibility check."""
        with patch.dict(os.environ, {
            'MQTT_BROKER': 'localhost',
            'MAX_ENGINE_RUNTIME': '400',
            'MAX_DRY_RUN_TIME': '180',
            'TANK_CAPACITY_GALLONS': '500',
            'PUMP_FLOW_RATE_GPM': '50',
            'QUICK_CHECK_INTERVAL': '60',
            'DISCOVERY_INTERVAL': '300',
            'HEALTH_CHECK_INTERVAL': '60',
            'OFFLINE_THRESHOLD': '180'
        }):
            from utils.config_cli import check_compatibility_command
            from argparse import Namespace
            
            args = Namespace()
            result = check_compatibility_command(args)
            assert result == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])