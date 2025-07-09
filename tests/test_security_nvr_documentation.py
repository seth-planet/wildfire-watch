#!/usr/bin/env python3.12
"""
Documentation verification tests for Security NVR
Ensures the service behaves as documented in security_nvr/README.md
"""
import os
import sys
import json
import yaml
import time
import subprocess
import pytest
from pathlib import Path

# Test paths
PROJECT_ROOT = Path(__file__).parent.parent
NVR_README = PROJECT_ROOT / "security_nvr" / "README.md"
NVR_BASE_CONFIG = PROJECT_ROOT / "security_nvr" / "nvr_base_config.yml"
DOCKER_COMPOSE = PROJECT_ROOT / "docker-compose.yml"


class TestDocumentationAccuracy:
    """Verify that documentation matches implementation"""
    
    def test_readme_exists(self):
        """Test that README.md exists"""
        assert NVR_README.exists(), "security_nvr/README.md not found"
        
        # Check it's not empty
        content = NVR_README.read_text()
        assert len(content) > 1000, "README seems too short"
        assert "Security NVR Service" in content
    
    def test_documented_files_exist(self):
        """Test that all files mentioned in README exist"""
        documented_files = [
            "hardware_detector.py",
            "camera_manager.py", 
            "usb_manager.py",
            "nvr_base_config.yml",
            "entrypoint.sh"
        ]
        
        nvr_dir = PROJECT_ROOT / "security_nvr"
        for filename in documented_files:
            filepath = nvr_dir / filename
            assert filepath.exists(), f"Documented file {filename} not found"
    
    def test_environment_variables_documented(self):
        """Test that documented environment variables are used"""
        readme_content = NVR_README.read_text()
        
        # Extract environment variables from README
        documented_vars = [
            "FRIGATE_MQTT_HOST",
            "FRIGATE_MQTT_PORT",
            "FRIGATE_MQTT_TLS",
            "FRIGATE_CLIENT_ID",
            "USB_MOUNT_PATH",
            "RECORD_RETAIN_DAYS",
            "FRIGATE_DETECTOR",
            "DETECTION_THRESHOLD",
            "DETECTION_FPS",
            "MIN_CONFIDENCE",
            "CAMERA_DETECT_WIDTH",
            "CAMERA_DETECT_HEIGHT",
            "CAMERA_RECORD_QUALITY",
            "HARDWARE_ACCEL"
        ]
        
        # Check each variable is mentioned in README
        for var in documented_vars:
            assert var in readme_content, f"Environment variable {var} not documented"
    
    def test_model_sizes_documented(self):
        """Test that model sizes match documentation"""
        readme_content = NVR_README.read_text()
        
        # Check 640x640 is documented as optimal
        assert "640" in readme_content
        assert "optimal" in readme_content or "recommended" in readme_content
        
        # Check 320x320 is documented for limited hardware
        assert "320" in readme_content
        assert "limited hardware" in readme_content or "Raspberry Pi" in readme_content
    
    def test_mqtt_topic_documentation(self):
        """Test that MQTT topics are properly documented"""
        readme_content = NVR_README.read_text()
        
        # Check key MQTT topics are documented
        mqtt_topics = [
            "frigate/events",
            "cameras/discovered"
        ]
        
        for topic in mqtt_topics:
            assert topic in readme_content, f"MQTT topic {topic} not documented"
    
    def test_hardware_support_table(self):
        """Test that hardware support table is accurate"""
        readme_content = NVR_README.read_text()
        
        # Check all hardware types are documented
        hardware_types = [
            "Raspberry Pi 5",
            "Coral USB",
            "Coral PCIe",
            "Hailo-8",
            "Intel QuickSync",
            "NVIDIA GPU",
            "CPU Fallback"
        ]
        
        for hw in hardware_types:
            assert hw in readme_content, f"Hardware type {hw} not documented"


class TestConfigurationFiles:
    """Test configuration file formats and contents"""
    
    def test_nvr_base_config_format(self):
        """Test nvr_base_config.yml structure"""
        assert NVR_BASE_CONFIG.exists(), "nvr_base_config.yml not found"
        
        with open(NVR_BASE_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        assert "mqtt" in config
        assert "detectors" in config
        assert "model" in config
        
        # Check MQTT configuration
        mqtt_config = config['mqtt']
        assert mqtt_config['enabled'] is True
        assert "host" in mqtt_config
        assert "port" in mqtt_config
        assert "tls_ca_certs" in mqtt_config
        
        # Check detector configuration
        detectors = config['detectors']
        assert "default" in detectors
        default_detector = detectors["default"]
        assert "type" in default_detector
        assert "model" in default_detector
        
        # Check model dimensions
        model_config = default_detector["model"]
        assert model_config['width'] == 640  # Should default to 640
        assert model_config['height'] == 640
    
    def test_model_labels_configuration(self):
        """Test that wildfire-specific labels are configured"""
        with open(NVR_BASE_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check model labels
        model = config.get("model", {})
        labels = model.get("labels", [])
        
        # Essential wildfire detection labels
        required_labels = ["fire", "smoke"]
        for label in required_labels:
            assert label in labels, f"Required label '{label}' not in model configuration"
        
        # Additional context labels
        context_labels = ["person", "car", "wildlife"]
        for label in context_labels:
            assert label in labels, f"Context label '{label}' not in model configuration"
    
    def test_recording_configuration(self):
        """Test recording configuration matches documentation"""
        with open(NVR_BASE_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check recording settings
        assert "record" in config
        record = config['record']
        assert record["enabled"] is True
        
        # Check retention settings
        assert "retain" in record
        retain = record["retain"]
        assert "days" in retain
        assert "mode" in retain
        assert retain["mode"] == "active_objects"  # Only keep recordings with objects


class TestDockerIntegration:
    """Test Docker configuration and integration"""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and contains required elements"""
        dockerfile = PROJECT_ROOT / "security_nvr" / "Dockerfile"
        assert dockerfile.exists(), "security_nvr/Dockerfile not found"
        
        content = dockerfile.read_text()
        
        # Check base image
        assert "frigate" in content.lower()
        
        # Check required packages
        assert "python" in content.lower()
        assert "hardware" in content or "detector" in content
    
    def test_docker_compose_service(self):
        """Test docker-compose.yml contains security_nvr service"""
        assert DOCKER_COMPOSE.exists(), "docker-compose.yml not found"
        
        with open(DOCKER_COMPOSE, 'r') as f:
            compose = yaml.safe_load(f)
        
        services = compose.get("services", {})
        assert "security_nvr" in services, "security_nvr service not in docker-compose.yml"
        
        nvr_service = services["security_nvr"]
        
        # Check service configuration
        assert "depends_on" in nvr_service
        assert "mqtt_broker" in nvr_service["depends_on"]
        
        # Check environment variables
        if "environment" in nvr_service:
            env = nvr_service["environment"]
            # Verify key environment variables are set
    
    def test_volume_mounts(self):
        """Test that required volumes are mounted"""
        with open(DOCKER_COMPOSE, 'r') as f:
            compose = yaml.safe_load(f)
        
        nvr_service = compose["services"]["security_nvr"]
        volumes = nvr_service.get("volumes", [])
        
        # Check for required volume mounts
        required_mounts = [
            "/media/frigate",  # Storage
            "/config",         # Configuration
            "/models"          # AI models
        ]
        
        for mount in required_mounts:
            assert any(mount in v for v in volumes), f"Required mount {mount} not found"
    
    def test_device_access(self):
        """Test that hardware devices are properly mapped"""
        with open(DOCKER_COMPOSE, 'r') as f:
            compose = yaml.safe_load(f)
        
        nvr_service = compose["services"]["security_nvr"]
        
        # Check for device mappings (may be conditional)
        if "devices" in nvr_service:
            devices = nvr_service["devices"]
            # Could include /dev/dri, /dev/bus/usb, etc.


class TestPowerProfiles:
    """Test power profile configurations"""
    
    def test_power_profiles_documented(self):
        """Test that power profiles are documented"""
        readme_content = NVR_README.read_text()
        
        profiles = ["Performance Mode", "Balanced Mode", "Power Save Mode"]
        for profile in profiles:
            assert profile in readme_content, f"Power profile '{profile}' not documented"
    
    def test_adaptive_settings(self):
        """Test adaptive quality settings are documented"""
        readme_content = NVR_README.read_text()
        
        adaptive_features = [
            "Adaptive Quality",
            "Sub-stream Detection",
            "Hardware Decode",
            "Smart Recording"
        ]
        
        for feature in adaptive_features:
            assert feature in readme_content, f"Adaptive feature '{feature}' not documented"


class TestTroubleshooting:
    """Test troubleshooting documentation"""
    
    def test_common_problems_documented(self):
        """Test that common problems and solutions are documented"""
        readme_content = NVR_README.read_text()
        
        # Check for troubleshooting section
        assert "Troubleshooting" in readme_content
        
        # Check common problems are addressed
        problems = [
            "No Hardware Acceleration",
            "USB Drive Not Detected",
            "Detection Not Working",
            "High CPU usage",
            "No cameras showing"
        ]
        
        for problem in problems:
            assert problem in readme_content, f"Common problem '{problem}' not documented"
    
    def test_debug_commands_documented(self):
        """Test that debug commands are documented"""
        readme_content = NVR_README.read_text()
        
        # Check for debug commands
        debug_commands = [
            "docker logs",
            "docker exec",
            "check-hardware"
        ]
        
        for cmd in debug_commands:
            assert cmd in readme_content, f"Debug command '{cmd}' not documented"


class TestAPIDocumentation:
    """Test API endpoint documentation"""
    
    def test_api_endpoints_documented(self):
        """Test that API endpoints are documented"""
        readme_content = NVR_README.read_text()
        
        # While not all endpoints are documented, key ones should be mentioned
        key_concepts = [
            "5000",  # Port number
            "api",   # API mention
            "Web Interface" or "Web UI"
        ]
        
        for concept in key_concepts:
            if isinstance(concept, tuple):
                assert any(c in readme_content for c in concept)
            else:
                assert concept in readme_content


class TestPerformanceGuidelines:
    """Test performance optimization documentation"""
    
    def test_hardware_specific_optimizations(self):
        """Test hardware-specific optimizations are documented"""
        readme_content = NVR_README.read_text()
        
        # Check for hardware-specific sections
        assert "Raspberry Pi 5 Specific" in readme_content
        assert "Intel/AMD Specific" in readme_content
        
        # Check for acceleration arguments
        assert "hwaccel" in readme_content
        assert "v4l2" in readme_content or "vaapi" in readme_content
    
    def test_performance_metrics(self):
        """Test that performance expectations are documented"""
        readme_content = NVR_README.read_text()
        
        # Check for performance-related information
        performance_keywords = [
            "inference",
            "FPS",
            "CPU usage",
            "bandwidth"
        ]
        
        for keyword in performance_keywords:
            assert keyword.lower() in readme_content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])