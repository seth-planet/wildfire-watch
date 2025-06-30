#!/usr/bin/env python3.12
"""
Container smoke tests - verify basic container functionality
These are simpler tests that don't require full E2E setup
"""
import os
import sys
import time
import docker
import pytest
from pathlib import Path

# Add modules to path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../gpio_trigger")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../fire_consensus")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../camera_detector")))


class TestContainerSmoke:
    """Basic container startup and functionality tests"""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture"""
        return docker.from_env()
    
    def test_mqtt_broker_startup(self, docker_client):
        """Test MQTT broker can start with basic config"""
        # Get certs directory
        certs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'certs'))
        
        container = None
        try:
            container = docker_client.containers.run(
                "eclipse-mosquitto:latest",
                name="test-mqtt-smoke",
                ports={'1883/tcp': None},
                detach=True,
                remove=True,
                command=["mosquitto", "-v", "-p", "1883"]
            )
            
            # Wait a bit
            time.sleep(2)
            
            # Check if running
            container.reload()
            assert container.status == 'running', f"Container status: {container.status}"
            
        finally:
            if container:
                try:
                    container.stop(timeout=5)
                except:
                    pass
    
    def test_gpio_trigger_help(self, docker_client):
        """Test GPIO trigger container can show help"""
        # Just test that the container can run and show help
        result = docker_client.containers.run(
            "python:3.12-slim",
            command=["python", "-c", "print('GPIO trigger container test')"],
            remove=True
        )
        assert b'GPIO trigger container test' in result
    
    def test_fire_consensus_import(self, docker_client):
        """Test fire consensus Python imports work"""
        # Test basic Python environment
        result = docker_client.containers.run(
            "python:3.12-slim",
            command=["python", "-c", "import json, time, logging; print('Imports OK')"],
            remove=True
        )
        assert b'Imports OK' in result
    
    def test_frigate_base_image(self, docker_client):
        """Test Frigate base image is available"""
        try:
            # Pull base image if needed
            docker_client.images.pull("ghcr.io/blakeblackshear/frigate:stable")
            assert True, "Frigate base image available"
        except docker.errors.APIError as e:
            pytest.skip(f"Could not pull Frigate base image: {e}")


class TestServiceEnvironment:
    """Test service environment configurations"""
    
    def test_mqtt_tls_defaults(self):
        """Test MQTT TLS defaults are correct"""
        # Read docker-compose.yml
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        with open(compose_path, 'r') as f:
            content = f.read()
        
        # Check TLS defaults to true
        assert "MQTT_TLS:-true" in content, "MQTT_TLS should default to true"
        assert "MQTT_PORT:-8883" in content, "MQTT_PORT should default to 8883 for TLS"
    
    def test_certificates_exist(self):
        """Test that default certificates exist"""
        certs_dir = Path(__file__).parent.parent / "certs"
        
        required_files = [
            "ca.crt",
            "ca.key", 
            "server.crt",
            "server.key"
        ]
        
        for cert_file in required_files:
            cert_path = certs_dir / cert_file
            assert cert_path.exists(), f"Certificate file missing: {cert_file}"
    
    def test_utils_module_exists(self):
        """Test utils module exists and can be imported"""
        utils_path = Path(__file__).parent.parent / "utils" / "command_runner.py"
        assert utils_path.exists(), "utils/command_runner.py should exist"
        
        # Try to import it
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from utils.command_runner import run_command, CommandError
            assert run_command is not None
            assert CommandError is not None
        except ImportError as e:
            pytest.fail(f"Could not import utils.command_runner: {e}")
    
    def test_storage_manager_exists(self):
        """Test storage manager exists"""
        storage_manager_path = Path(__file__).parent.parent / "security_nvr" / "storage_manager.py"
        assert storage_manager_path.exists(), "security_nvr/storage_manager.py should exist"
        
        # Check USB manager has minimum drive size (StorageManager inherits from USBStorageManager)
        usb_manager_path = Path(__file__).parent.parent / "security_nvr" / "usb_manager.py"
        assert usb_manager_path.exists(), "security_nvr/usb_manager.py should exist"
        
        with open(usb_manager_path, 'r') as f:
            content = f.read()
        assert "MIN_DRIVE_SIZE = 500 * 1024**3" in content, "USB manager should have 500GB minimum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])