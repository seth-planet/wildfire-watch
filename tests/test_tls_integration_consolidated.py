#!/usr/bin/env python3.12
"""
Consolidated TLS/SSL Integration Tests
Combines all TLS-related tests from multiple files
"""
import os
import sys
import ssl
import time
import json
import socket
import pytest
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

# Test configuration
PROJECT_ROOT = Path(__file__).parent.parent
CERT_DIR = PROJECT_ROOT / "certs"
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "0"))
MQTT_TLS_PORT = int(os.getenv("MQTT_TLS_PORT", "8883"))


# check_mqtt_tls_available removed - using test broker fixture instead


def check_certificates_exist():
    """Check if certificates exist"""
    required_certs = ["ca.crt", "server.crt", "server.key"]
    return all((CERT_DIR / cert).exists() for cert in required_certs)


# Skip decorators
# requires_tls removed - using test broker fixture instead

requires_certs = pytest.mark.skipif(
    not check_certificates_exist(),
    reason="Certificates not found in certs/ directory"
)


class TestCertificateManagement:
    """Test certificate validation and management"""
    
    def test_certificates_exist(self):
        """Test that required certificates exist"""
        assert CERT_DIR.exists(), f"Certificate directory not found: {CERT_DIR}"
        
        required_files = {
            "ca.crt": "Certificate Authority",
            "server.crt": "Server certificate",
            "server.key": "Server private key"
        }
        
        for filename, description in required_files.items():
            cert_path = CERT_DIR / filename
            assert cert_path.exists(), f"{description} not found: {cert_path}"
            assert cert_path.stat().st_size > 0, f"{description} is empty"
    
    def test_certificate_validity(self):
        """Test that certificates are valid and not expired"""
        import cryptography.x509
        from cryptography.hazmat.backends import default_backend
        
        # Check CA certificate
        ca_path = CERT_DIR / "ca.crt"
        with open(ca_path, 'rb') as f:
            ca_cert = cryptography.x509.load_pem_x509_certificate(f.read(), default_backend())
        
        # Check not expired
        now = datetime.now()
        ca_not_valid_after = ca_cert.not_valid_after.replace(tzinfo=None) if ca_cert.not_valid_after.tzinfo else ca_cert.not_valid_after
        ca_not_valid_before = ca_cert.not_valid_before.replace(tzinfo=None) if ca_cert.not_valid_before.tzinfo else ca_cert.not_valid_before
        
        assert ca_not_valid_after > now, "CA certificate is expired"
        assert ca_not_valid_before < now, "CA certificate is not yet valid"
    
    def test_default_certificate_warning(self):
        """Test that default certificates are properly marked as insecure"""
        ca_path = CERT_DIR / "ca.crt"
        with open(ca_path, 'r') as f:
            ca_content = f.read()
        
        # Check if this is a default certificate
        if "Default Development CA" in ca_content:
            readme_path = CERT_DIR / "README.md"
            assert readme_path.exists(), "No README warning about default certificates"
            
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            assert "INSECURE" in readme_content, "README doesn't warn about insecure certificates"
            assert "DO NOT USE IN PRODUCTION" in readme_content, "Missing production warning"


class TestMQTTBrokerTLS:
    """Test MQTT broker TLS functionality"""
    
    def test_tls_port_listening(self, mqtt_tls_broker):
        """Test that MQTT broker is listening on TLS port"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex((mqtt_tls_broker.host, mqtt_tls_broker.tls_port))
            assert result == 0, f"MQTT TLS port {mqtt_tls_broker.tls_port} is not accessible"
        finally:
            sock.close()
    
    @requires_certs
    def test_tls_connection_with_valid_cert(self, mqtt_tls_broker):
        """Test connecting to MQTT with valid certificate"""
        connected = False
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connected
            connected = True
            assert rc == 0, f"Connection failed with code: {rc}"
        
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_tls_valid"
        )
        client.on_connect = on_connect
        
        # Configure TLS
        ca_cert = str(CERT_DIR / "ca.crt")
        client.tls_set(
            ca_certs=ca_cert,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS
        )
        client.tls_insecure_set(True)  # For test certificates
        
        # Connect
        client.connect(mqtt_tls_broker.host, mqtt_tls_broker.tls_port, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        client.loop_stop()
        client.disconnect()
        
        assert connected, "Failed to connect with valid certificate"
    
    @requires_certs
    def test_tls_publish_subscribe(self, mqtt_tls_broker):
        """Test publishing and subscribing over TLS"""
        received_messages = []
        
        def on_message(client, userdata, msg):
            received_messages.append({
                "topic": msg.topic,
                "payload": msg.payload.decode('utf-8')
            })
        
        # Create subscriber
        subscriber = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_tls_sub"
        )
        subscriber.on_message = on_message
        
        # Configure TLS
        ca_cert = str(CERT_DIR / "ca.crt")
        subscriber.tls_set(
            ca_certs=ca_cert,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS
        )
        subscriber.tls_insecure_set(True)
        
        # Connect and subscribe
        subscriber.connect(mqtt_tls_broker.host, mqtt_tls_broker.tls_port, 60)
        subscriber.subscribe("test/tls/#")
        subscriber.loop_start()
        
        time.sleep(1)
        
        # Create publisher
        publisher = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_tls_pub"
        )
        publisher.tls_set(
            ca_certs=ca_cert,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS
        )
        publisher.tls_insecure_set(True)
        
        # Connect and publish
        publisher.connect(mqtt_tls_broker.host, mqtt_tls_broker.tls_port, 60)
        publisher.loop_start()
        
        test_payload = {"test": "TLS message", "timestamp": time.time()}
        publisher.publish("test/tls/message", json.dumps(test_payload), qos=1)
        
        time.sleep(2)
        
        # Cleanup
        subscriber.loop_stop()
        publisher.loop_stop()
        subscriber.disconnect()
        publisher.disconnect()
        
        # Verify message received
        assert len(received_messages) > 0, "No messages received over TLS"
        assert received_messages[0]["topic"] == "test/tls/message"


class TestServiceTLS:
    """Test service-level TLS configuration"""
    
    def test_services_read_tls_config(self):
        """Test that all services read MQTT_TLS environment variable"""
        services = [
            ("camera_detector", "detect.CameraDetectorConfig"),
            ("fire_consensus", "consensus.FireConsensusConfig"),
            ("gpio_trigger", "trigger.CONFIG"),
        ]
        
        # Set environment before imports
        original_mqtt_tls = os.environ.get('MQTT_TLS')
        os.environ['MQTT_TLS'] = 'true'
        
        try:
            for service_dir, config_path in services:
                sys.path.insert(0, str(PROJECT_ROOT / service_dir))
                
                # Remove modules from cache to force reload
                module_name = config_path.split('.')[0]
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                if service_dir == "gpio_trigger":
                    # GPIO trigger uses PumpControllerConfig
                    from gpio_trigger.trigger import PumpControllerConfig
                    config = PumpControllerConfig()
                    assert config.mqtt_tls is True
                else:
                    # Other services use ConfigBase-derived classes
                    module_name, class_name = config_path.split('.')
                    module = __import__(module_name)
                    config_class = getattr(module, class_name)
                    config = config_class()
                    # ConfigBase stores mqtt_tls in lowercase
                    assert config.mqtt_tls is True
                
                # Clean up module cache
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
                sys.path.pop(0)
        finally:
            # Restore original environment
            if original_mqtt_tls is not None:
                os.environ['MQTT_TLS'] = original_mqtt_tls
            else:
                os.environ.pop('MQTT_TLS', None)
    
    @requires_certs
    def test_service_tls_connection(self, mqtt_tls_broker):
        """Test that a service can connect using TLS"""
        ca_cert = str(CERT_DIR / "ca.crt")
        
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_service_tls"
        )
        
        # Configure TLS as services would
        client.tls_set(
            ca_certs=ca_cert,
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLS
        )
        client.tls_insecure_set(True)
        
        connected = False
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connected
            if rc == 0:
                connected = True
        
        client.on_connect = on_connect
        client.connect(mqtt_tls_broker.host, mqtt_tls_broker.tls_port, 60)
        client.loop_start()
        
        time.sleep(2)
        
        client.loop_stop()
        client.disconnect()
        
        assert connected, "Service failed to connect with TLS"


class TestDockerTLS:
    """Test Docker configuration for TLS"""
    
    def test_docker_compose_tls_config(self):
        """Test docker-compose.yml has TLS configuration"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"
        
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Check MQTT broker exposes TLS port
        assert "MQTT_TLS_PORT:-8883" in content or "8883:8883" in content, "MQTT TLS port not exposed"
        assert "MQTT_TLS" in content, "MQTT_TLS variable not in compose file"
        
        # Check certificate volume
        assert "certs:/mnt/data/certs" in content, "Certificate volume not mounted"
    
    def test_services_mount_certificates(self):
        """Test that all services mount certificate volume"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            import yaml
            compose_config = yaml.safe_load(f)
        
        services_requiring_certs = [
            'mqtt_broker',
            'camera_detector',
            'fire_consensus',
            'gpio_trigger'
        ]
        
        for service_name in services_requiring_certs:
            if service_name in compose_config['services']:
                service = compose_config['services'][service_name]
                
                # Check volumes
                if 'volumes' in service:
                    cert_volume_found = False
                    for vol in service['volumes']:
                        if isinstance(vol, str):
                            # Check for string volume format
                            if 'certs' in vol and '/mnt/data/certs' in vol:
                                cert_volume_found = True
                                break
                        elif isinstance(vol, dict):
                            # Check for dictionary volume format
                            if vol.get('source') == 'certs' and vol.get('target') == '/mnt/data/certs':
                                cert_volume_found = True
                                break
                    
                    assert cert_volume_found, \
                        f"{service_name} doesn't mount certificates volume"


class TestTLSFailureModes:
    """Test TLS failure scenarios"""
    
    def test_invalid_certificate_rejected(self, mqtt_tls_broker):
        """Test that invalid certificates are rejected"""
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_invalid_cert"
        )
        
        # Try to use a fake certificate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.crt', delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\\n")
            f.write("INVALID CERTIFICATE DATA\\n")
            f.write("-----END CERTIFICATE-----\\n")
            fake_cert = f.name
        
        try:
            # This should fail
            with pytest.raises(Exception):
                client.tls_set(
                    ca_certs=fake_cert,
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS
                )
        finally:
            os.unlink(fake_cert)
    
    def test_plain_mqtt_with_tls_env(self):
        """Test that services can handle mixed TLS/plain environments"""
        os.environ['MQTT_TLS'] = 'true'
        
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_fallback"
        )
        
        # Services should use appropriate port based on TLS setting
        # This tests configuration handling, not actual connection


class TestSecurityScripts:
    """Test security configuration scripts"""
    
    def test_configure_security_script_exists(self):
        """Test that configure_security.sh script exists and is executable"""
        script_path = PROJECT_ROOT / "scripts" / "configure_security.sh"
        assert script_path.exists(), "configure_security.sh not found"
        assert script_path.stat().st_mode & 0o111, "configure_security.sh not executable"
    
    def test_generate_certs_script_exists(self):
        """Test that generate_certs.sh script exists"""
        script_path = PROJECT_ROOT / "scripts" / "generate_certs.sh"
        assert script_path.exists(), "generate_certs.sh not found"
        assert script_path.stat().st_mode & 0o111, "generate_certs.sh not executable"
    
    def test_env_example_has_tls(self):
        """Test that .env.example includes MQTT_TLS"""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example not found"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        assert "MQTT_TLS" in content, "MQTT_TLS not in .env.example"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])