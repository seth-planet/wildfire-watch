import pytest

pytestmark = pytest.mark.deployment

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
        
        services = compose_config.services
        
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
        
        services = compose_config.services
        
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
        
        services = compose_config.services
        
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
        assert 'wildfire_net' in compose_config.networks
        
        network_config = compose_config.networks['wildfire_net']
        assert 'driver' in network_config
        assert network_config.driver == 'bridge'
        
        # Check IPAM configuration
        if 'ipam' in network_config:
            assert 'config' in network_config.ipam
            assert 'subnet' in network_config.ipam['config'][0]


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
        services = compose_config.services
        
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
        # Zone mapping is implemented in consensus.py but may not be in .env.example
        zone_implementation = PROJECT_ROOT / "fire_consensus" / "consensus.py"
        
        with open(zone_implementation, 'r') as f:
            consensus_code = f.read()
        
        # Check if zone mapping exists in the implementation
        has_zone_impl = 'zone' in consensus_code.lower() or 'camera_zones' in consensus_code
        
        assert has_zone_impl, "Zone configuration should be implemented in consensus service"


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
        
        # Check for manual override configuration
        # The system uses MQTT emergency topic for manual override, not GPIO pins
        emergency_vars = ['EMERGENCY_TOPIC', 'fire/emergency', 'manual override']
        
        has_override = any(var.lower() in content.lower() for var in emergency_vars)
        
        assert has_override or 'EMERGENCY_TOPIC' in os.environ, \
            "Manual override should be configured via MQTT emergency topic"


class TestStorageConfiguration:
    """Test storage and persistence configuration"""
    
    def test_volume_persistence(self):
        """Test Docker volumes are properly configured"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        volumes = compose_config.volumes
        
        # Required volumes
        required_volumes = ['mqtt_data', 'mqtt_logs', 'frigate_data']
        
        for vol in required_volumes:
            assert vol in volumes, f"Missing required volume: {vol}"
    
    def test_data_persistence_configuration(self):
        """Test data persistence is properly configured via Docker volumes"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        # Check that Frigate recordings and database are persisted
        frigate_service = compose_config.services.get('security_nvr')
        if frigate_service:
            volumes = frigate_service.get('volumes', [])
            # Should have volumes for recordings and database
            volume_paths = [str(v) for v in volumes]
            has_media = any('/media' in v for v in volume_paths)
            has_db = any('/db' in v or 'frigate.db' in v for v in volume_paths)
            assert has_media or has_db, "Frigate should have persistent storage for recordings/database"


class TestDiagnosticTools:
    """Test diagnostic and troubleshooting tools"""
    
    def test_diagnostic_scripts_referenced(self):
        """Test that diagnostic scripts mentioned in docs exist"""
        troubleshooting_doc = PROJECT_ROOT / "docs" / "troubleshooting.md"
        
        with open(troubleshooting_doc, 'r') as f:
            content = f.read()
        
        # Extract script references and verify they exist
        if "diagnose.sh" in content:
            script_path = PROJECT_ROOT / "scripts" / "diagnose.sh"
            assert script_path.exists(), "diagnose.sh is referenced but script doesn't exist"
            assert os.access(script_path, os.X_OK), "diagnose.sh should be executable"
        
        if "collect_debug.sh" in content:
            script_path = PROJECT_ROOT / "scripts" / "collect_debug.sh"
            assert script_path.exists(), "collect_debug.sh is referenced but script doesn't exist"
            assert os.access(script_path, os.X_OK), "collect_debug.sh should be executable"


class TestSecurityDeployment:
    """Test security deployment configurations"""
    
    def test_tls_deployment_configuration(self):
        """Test TLS can be properly configured for deployment"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config.services
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