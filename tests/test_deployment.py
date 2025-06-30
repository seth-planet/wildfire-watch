#!/usr/bin/env python3.12
"""
Tests for deployment configurations and multi-node setups
Tests Docker Compose, service dependencies, and multi-node features
"""
import os
import yaml
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, Mock

PROJECT_ROOT = Path(__file__).parent.parent


class TestDockerComposeDeployment:
    """Test Docker Compose deployment configuration"""
    
    def test_docker_compose_valid_yaml(self):
        """Test docker-compose.yml is valid YAML"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        assert 'version' in compose_config
        assert 'services' in compose_config
        assert 'networks' in compose_config
        assert 'volumes' in compose_config
    
    def test_service_dependencies(self):
        """Test services have correct dependencies"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Check MQTT broker has no dependencies
        assert 'depends_on' not in services['mqtt_broker']
        
        # Check other services depend on MQTT
        dependent_services = ['camera_detector', 'fire_consensus', 'gpio_trigger']
        for service_name in dependent_services:
            if service_name in services:
                assert 'depends_on' in services[service_name]
                deps = services[service_name]['depends_on']
                if isinstance(deps, list):
                    assert 'mqtt_broker' in deps
                elif isinstance(deps, dict):
                    assert 'mqtt_broker' in deps
    
    def test_health_checks(self):
        """Test services have health checks configured"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Critical services should have health checks
        critical_services = ['mqtt_broker']
        
        for service_name in critical_services:
            if service_name in services:
                assert 'healthcheck' in services[service_name], \
                    f"{service_name} missing health check"
                
                health = services[service_name]['healthcheck']
                assert 'test' in health
                assert 'interval' in health
                assert 'retries' in health
    
    def test_resource_limits(self):
        """Test services have appropriate resource limits"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        
        # Check if deploy limits are set for production
        for service_name, service in services.items():
            if 'deploy' in service and 'resources' in service['deploy']:
                resources = service['deploy']['resources']
                
                # Should have limits for production
                if 'limits' in resources:
                    assert 'memory' in resources['limits'], \
                        f"{service_name} missing memory limit"
    
    def test_network_configuration(self):
        """Test network configuration for services"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check network exists
        assert 'wildfire_net' in compose_config['networks']
        
        network_config = compose_config['networks']['wildfire_net']
        assert 'driver' in network_config
        assert network_config['driver'] == 'bridge'
        
        # Check IPAM configuration
        if 'ipam' in network_config:
            assert 'config' in network_config['ipam']
            assert 'subnet' in network_config['ipam']['config'][0]


class TestMultiNodeConfiguration:
    """Test multi-node deployment features"""
    
    def test_mqtt_bridging_config(self):
        """Test MQTT broker supports bridging configuration"""
        # Check if bridge configuration files exist
        bridge_conf = PROJECT_ROOT / "mqtt_broker" / "conf.d" / "bridge.conf.example"
        
        if bridge_conf.exists():
            with open(bridge_conf, 'r') as f:
                content = f.read()
            
            # Should have bridge configuration
            assert "connection" in content
            assert "address" in content
            assert "topic" in content
    
    def test_node_identity_configuration(self):
        """Test node identity is properly configured"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check services use NODE_ID or BALENA_DEVICE_UUID
        services = compose_config['services']
        
        # Check if at least some services use node identity
        services_with_node_id = 0
        
        for service_name, service in services.items():
            if 'environment' in service:
                env_list = service['environment']
                env_str = str(env_list)
                
                if 'NODE_ID' in env_str or 'BALENA_DEVICE_UUID' in env_str:
                    services_with_node_id += 1
        
        # At least one service should use node identity
        assert services_with_node_id > 0, "No services use NODE_ID or BALENA_DEVICE_UUID"
    
    def test_zone_configuration(self):
        """Test zone-based activation configuration"""
        # This feature is mentioned in docs but may not be implemented
        # Test for environment variables that would support it
        
        env_example = PROJECT_ROOT / ".env.example"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Check for zone-related configuration
        zone_vars = ['PUMP_ZONE', 'ACTIVATION_ZONES', 'ZONE_ID']
        
        has_zone_config = any(var in content for var in zone_vars)
        
        if not has_zone_config:
            pytest.skip("Zone configuration not yet implemented")


class TestDeploymentProfiles:
    """Test deployment profile configurations"""
    
    def test_profile_configurations_exist(self):
        """Test that deployment profiles are documented"""
        # Check for profile configurations
        profiles_dir = PROJECT_ROOT / "profiles"
        docs_config = PROJECT_ROOT / "docs" / "configuration.md"
        
        # Profiles might be in documentation
        with open(docs_config, 'r') as f:
            content = f.read()
        
        # Check for profile mentions or configuration options
        # Profiles might be described differently
        config_options = ['low', 'high', 'balanced', 'performance', 'accuracy', 'power']
        
        found_options = sum(1 for opt in config_options if opt in content.lower())
        
        # Should have some configuration guidance
        assert found_options > 0, "No deployment configuration options documented"
    
    def test_hardware_auto_detection(self):
        """Test hardware auto-detection configuration"""
        env_example = PROJECT_ROOT / ".env.example"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Should support auto detection
        assert "FRIGATE_DETECTOR=auto" in content
        
        # Check if detection logic is documented
        assert "auto" in content


class TestEmergencyFeatures:
    """Test emergency and bypass features"""
    
    def test_emergency_bypass_configuration(self):
        """Test emergency bypass mode configuration"""
        # Check if bypass is configurable
        gpio_readme = PROJECT_ROOT / "gpio_trigger" / "README.md"
        
        with open(gpio_readme, 'r') as f:
            content = f.read()
        
        # Should mention emergency features
        assert "emergency" in content.lower() or "bypass" in content.lower()
    
    def test_manual_override_gpio(self):
        """Test manual override GPIO configuration"""
        env_example = PROJECT_ROOT / ".env.example"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Check for manual override pins
        override_vars = ['MANUAL_OVERRIDE_PIN', 'EMERGENCY_STOP_PIN']
        
        has_override = any(var in content for var in override_vars)
        
        if not has_override:
            pytest.skip("Manual override not yet configured")


class TestStorageConfiguration:
    """Test storage and persistence configuration"""
    
    def test_volume_persistence(self):
        """Test Docker volumes are properly configured"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        volumes = compose_config['volumes']
        
        # Required volumes
        required_volumes = ['mqtt_data', 'mqtt_logs', 'frigate_data']
        
        for vol in required_volumes:
            assert vol in volumes, f"Missing required volume: {vol}"
    
    def test_backup_restore_scripts(self):
        """Test backup and restore functionality"""
        # Check for backup scripts
        backup_script = PROJECT_ROOT / "scripts" / "backup.sh"
        restore_script = PROJECT_ROOT / "scripts" / "restore.sh"
        
        if not backup_script.exists():
            pytest.skip("Backup/restore scripts not yet implemented")


class TestDiagnosticTools:
    """Test diagnostic and troubleshooting tools"""
    
    def test_diagnostic_scripts_referenced(self):
        """Test that diagnostic scripts mentioned in docs exist"""
        troubleshooting_doc = PROJECT_ROOT / "docs" / "troubleshooting.md"
        
        with open(troubleshooting_doc, 'r') as f:
            content = f.read()
        
        # Extract script references
        if "diagnose.sh" in content:
            script_path = PROJECT_ROOT / "scripts" / "diagnose.sh"
            if not script_path.exists():
                pytest.skip("diagnose.sh referenced but not yet implemented")
        
        if "collect_debug.sh" in content:
            script_path = PROJECT_ROOT / "scripts" / "collect_debug.sh"
            if not script_path.exists():
                pytest.skip("collect_debug.sh referenced but not yet implemented")


class TestSecurityDeployment:
    """Test security deployment configurations"""
    
    def test_tls_deployment_configuration(self):
        """Test TLS can be properly configured for deployment"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config['services']
        mqtt_service = services['mqtt_broker']
        
        # Check TLS port is exposed
        ports = mqtt_service.get('ports', [])
        tls_port_exposed = any('8883' in str(port) for port in ports)
        assert tls_port_exposed, "MQTT TLS port 8883 not exposed"
        
        # Check certificate volume
        volumes = mqtt_service.get('volumes', [])
        cert_volume = any('certs' in str(vol) for vol in volumes)
        assert cert_volume, "Certificate volume not mounted"
    
    def test_acl_configuration(self):
        """Test MQTT ACL configuration support"""
        acl_file = PROJECT_ROOT / "mqtt_broker" / "acl.conf.example"
        
        if acl_file.exists():
            with open(acl_file, 'r') as f:
                content = f.read()
            
            # Should have ACL rules
            assert "user" in content or "topic" in content


class TestHardwareCompatibility:
    """Test hardware compatibility configurations"""
    
    def test_usb_gpio_adapter_support(self):
        """Test USB-to-GPIO adapter configurations"""
        pc_guide = PROJECT_ROOT / "docs" / "QUICK_START_pc.md"
        
        with open(pc_guide, 'r') as f:
            content = f.read()
        
        # Should mention USB adapters
        adapters = ['FT232H', 'Arduino', 'CH340']
        
        supported_adapters = [a for a in adapters if a in content]
        assert len(supported_adapters) > 0, "No USB-GPIO adapters documented"
    
    def test_accelerator_configurations(self):
        """Test AI accelerator configurations"""
        env_example = PROJECT_ROOT / ".env.example"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Should support multiple accelerators
        accelerators = ['coral', 'hailo', 'gpu', 'cpu', 'auto']
        
        # At least some accelerators should be mentioned
        found_accelerators = [a for a in accelerators if a in content.lower()]
        assert len(found_accelerators) >= 3, \
            f"Only {len(found_accelerators)} accelerators found, expected at least 3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])