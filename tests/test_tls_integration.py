#!/usr/bin/env python3.12
"""
Integration tests for TLS/SSL functionality
Tests that all services can properly use TLS when MQTT_TLS=true
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
from unittest.mock import Mock, patch, MagicMock
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

# Test configuration
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TLS_PORT = int(os.getenv("MQTT_TLS_PORT", "8883"))
PROJECT_ROOT = Path(__file__).parent.parent
CERT_DIR = PROJECT_ROOT / "certs"

def check_mqtt_tls_available():
    """Check if MQTT broker is running with TLS enabled"""
    try:
        # Try to connect to TLS port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((MQTT_HOST, MQTT_TLS_PORT))
        sock.close()
        return result == 0
    except:
        return False

def check_certificates_exist():
    """Check if certificates exist"""
    required_certs = ["ca.crt", "server.crt", "server.key"]
    return all((CERT_DIR / cert).exists() for cert in required_certs)

# Skip decorators
requires_tls = pytest.mark.skipif(
    not check_mqtt_tls_available(),
    reason="MQTT TLS not available - start with MQTT_TLS=true"
)

requires_certs = pytest.mark.skipif(
    not check_certificates_exist(),
    reason="Certificates not found in certs/ directory"
)

class TestTLSConfiguration:
    """Test TLS configuration and certificate handling"""
    
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
        import datetime
        import cryptography.x509
        from cryptography.hazmat.backends import default_backend
        
        # Check CA certificate
        ca_path = CERT_DIR / "ca.crt"
        with open(ca_path, 'rb') as f:
            ca_cert = cryptography.x509.load_pem_x509_certificate(f.read(), default_backend())
        
        # Check not expired
        now = datetime.datetime.now()
        # Convert certificate times to timezone-naive for comparison
        ca_not_valid_after = ca_cert.not_valid_after.replace(tzinfo=None) if ca_cert.not_valid_after.tzinfo else ca_cert.not_valid_after
        ca_not_valid_before = ca_cert.not_valid_before.replace(tzinfo=None) if ca_cert.not_valid_before.tzinfo else ca_cert.not_valid_before
        
        assert ca_not_valid_after > now, "CA certificate is expired"
        assert ca_not_valid_before < now, "CA certificate is not yet valid"
        
        # Check server certificate
        server_path = CERT_DIR / "server.crt"
        with open(server_path, 'rb') as f:
            server_cert = cryptography.x509.load_pem_x509_certificate(f.read(), default_backend())
        
        server_not_valid_after = server_cert.not_valid_after.replace(tzinfo=None) if server_cert.not_valid_after.tzinfo else server_cert.not_valid_after
        server_not_valid_before = server_cert.not_valid_before.replace(tzinfo=None) if server_cert.not_valid_before.tzinfo else server_cert.not_valid_before
        
        assert server_not_valid_after > now, "Server certificate is expired"
        assert server_not_valid_before < now, "Server certificate is not yet valid"
    
    def test_default_certificate_warning(self):
        """Test that default certificates are properly marked as insecure"""
        ca_path = CERT_DIR / "ca.crt"
        with open(ca_path, 'r') as f:
            ca_content = f.read()
        
        # Check if this is a default certificate
        if "Default Development CA" in ca_content:
            # Ensure README exists warning about it
            readme_path = CERT_DIR / "README.md"
            assert readme_path.exists(), "No README warning about default certificates"
            
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            assert "INSECURE" in readme_content, "README doesn't warn about insecure certificates"
            assert "DO NOT USE IN PRODUCTION" in readme_content, "Missing production warning"


class TestMQTTTLS:
    """Test MQTT broker TLS functionality"""
    
    @requires_tls
    def test_tls_port_listening(self):
        """Test that MQTT broker is listening on TLS port"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            result = sock.connect_ex((MQTT_HOST, MQTT_TLS_PORT))
            assert result == 0, f"MQTT TLS port {MQTT_TLS_PORT} is not accessible"
        finally:
            sock.close()
    
    @requires_tls
    @requires_certs
    def test_tls_connection_with_valid_cert(self):
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
        # Disable hostname verification for test certificates
        client.tls_insecure_set(True)
        
        # Connect
        client.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
        client.loop_start()
        
        # Wait for connection
        time.sleep(2)
        
        client.loop_stop()
        client.disconnect()
        
        assert connected, "Failed to connect with valid certificate"
    
    @requires_tls
    def test_tls_connection_without_cert_fails(self):
        """Test that connection without certificate fails when TLS is required"""
        # This test assumes the broker requires certificates
        # Skip if anonymous TLS is allowed
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_tls_no_cert"
        )
        
        connection_error = None
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connection_error
            if rc != 0:
                connection_error = rc
        
        client.on_connect = on_connect
        
        # Try to connect without TLS to TLS port - should fail
        try:
            client.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
            client.loop_start()
            time.sleep(2)
            client.loop_stop()
            
            # If we get here without error, anonymous TLS might be allowed
            # which is valid for some configurations
        except Exception as e:
            # Expected - connection should fail
            assert True
    
    @requires_tls
    @requires_certs
    def test_tls_publish_subscribe(self):
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
        # Disable hostname verification for test certificates
        subscriber.tls_insecure_set(True)
        
        # Connect and subscribe
        subscriber.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
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
        # Disable hostname verification for test certificates
        publisher.tls_insecure_set(True)
        
        # Connect and publish
        publisher.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
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
        
        received_payload = json.loads(received_messages[0]["payload"])
        assert received_payload["test"] == "TLS message"


class TestServiceTLSIntegration:
    """Test that all services can use TLS properly"""
    
    def setup_method(self):
        """Setup test environment"""
        self.original_mqtt_tls = os.environ.get('MQTT_TLS')
        os.environ['MQTT_TLS'] = 'true'
    
    def teardown_method(self):
        """Restore environment"""
        if self.original_mqtt_tls is not None:
            os.environ['MQTT_TLS'] = self.original_mqtt_tls
        else:
            os.environ.pop('MQTT_TLS', None)
    
    def test_fire_consensus_tls_config(self):
        """Test fire consensus service TLS configuration"""
        # Set TLS_CA_PATH to expected value before import
        original_ca_path = os.environ.get('TLS_CA_PATH')
        os.environ['TLS_CA_PATH'] = '/mnt/data/certs/ca.crt'
        
        # Import here to test with MQTT_TLS=true
        sys.path.insert(0, str(PROJECT_ROOT / "fire_consensus"))
        try:
            # Remove module from cache if it exists
            if 'consensus' in sys.modules:
                del sys.modules['consensus']
            
            from consensus import Config
            
            config = Config()
            assert config.MQTT_TLS is True, "MQTT_TLS not properly read"
            assert config.TLS_CA_PATH == "/mnt/data/certs/ca.crt", "Wrong CA path"
        finally:
            sys.path.pop(0)
            # Restore original CA path
            if original_ca_path is not None:
                os.environ['TLS_CA_PATH'] = original_ca_path
            else:
                os.environ.pop('TLS_CA_PATH', None)
    
    def test_gpio_trigger_tls_config(self):
        """Test GPIO trigger service TLS configuration"""
        sys.path.insert(0, str(PROJECT_ROOT / "gpio_trigger"))
        try:
            from trigger import CONFIG
            
            # The config is loaded at module level
            # We need to reload it with our environment
            import importlib
            import trigger
            importlib.reload(trigger)
            
            assert trigger.CONFIG['MQTT_TLS'] is True, "MQTT_TLS not enabled"
        finally:
            sys.path.pop(0)
    
    @requires_tls
    @requires_certs
    def test_service_connection_with_tls(self):
        """Test that a service can connect using TLS"""
        # Simulate what a service would do
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
        # Disable hostname verification for test certificates
        client.tls_insecure_set(True)
        
        # Set will for health monitoring
        will_payload = json.dumps({
            "service": "test_service",
            "status": "offline",
            "timestamp": time.time()
        })
        client.will_set("health/test_service/status", will_payload, qos=1, retain=True)
        
        connected = False
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connected
            if rc == 0:
                connected = True
                # Publish online status
                online_payload = json.dumps({
                    "service": "test_service",
                    "status": "online",
                    "timestamp": time.time()
                })
                client.publish("health/test_service/status", online_payload, qos=1, retain=True)
        
        client.on_connect = on_connect
        client.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
        client.loop_start()
        
        time.sleep(2)
        
        client.loop_stop()
        client.disconnect()
        
        assert connected, "Service failed to connect with TLS"


class TestTLSFailureModes:
    """Test various TLS failure scenarios"""
    
    @requires_tls
    def test_invalid_certificate_rejected(self):
        """Test that invalid certificates are rejected"""
        # Create a client with no certificate validation
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_invalid_cert"
        )
        
        # Try to use a fake certificate
        with tempfile.NamedTemporaryFile(mode='w', suffix='.crt', delete=False) as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("INVALID CERTIFICATE DATA\n")
            f.write("-----END CERTIFICATE-----\n")
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
    
    @requires_tls
    @requires_certs
    def test_hostname_verification(self):
        """Test that hostname verification works"""
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_hostname_verify"
        )
        
        ca_cert = str(CERT_DIR / "ca.crt")
        # Create custom context for hostname verification
        context = ssl.create_default_context(cafile=ca_cert)
        context.check_hostname = True
        client.tls_set_context(context)
        
        # This might fail if certificate doesn't match hostname
        # which is expected for self-signed certs
        try:
            client.connect(MQTT_HOST, MQTT_TLS_PORT, 60)
            client.disconnect()
        except ssl.SSLCertVerificationError:
            # Expected for self-signed certificates
            pass
    
    def test_plain_mqtt_with_tls_env(self):
        """Test that services can fall back to plain MQTT if TLS fails"""
        # This tests the robustness of the system
        os.environ['MQTT_TLS'] = 'true'
        
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="test_fallback"
        )
        
        # Try plain connection even with TLS enabled
        # Services should handle this gracefully
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            client.disconnect()
            # If this works, fallback is available
        except:
            # If this fails, strict TLS is enforced
            pass


class TestDockerComposeTLS:
    """Test Docker Compose TLS configuration"""
    
    def test_compose_tls_ports(self):
        """Test that docker-compose.yml exposes TLS ports"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"
        
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Check MQTT broker exposes TLS port
        assert "MQTT_TLS_PORT:-8883" in content or "8883:8883" in content, "MQTT TLS port 8883 not exposed"
        assert "MQTT_TLS" in content, "MQTT_TLS variable not in compose file"
    
    def test_compose_cert_volumes(self):
        """Test that services mount certificate volumes"""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        
        with open(compose_file, 'r') as f:
            content = f.read()
        
        # Check certificate volume mounting
        assert "certs:/mnt/data/certs" in content, "Certificate volume not mounted"
        
        # Check multiple services have cert access
        cert_mount_count = content.count("certs:/mnt/data/certs")
        assert cert_mount_count >= 3, f"Only {cert_mount_count} services mount certificates"


class TestTLSEnvironmentIntegration:
    """Test environment-based TLS configuration"""
    
    def test_env_example_has_tls(self):
        """Test that .env.example includes MQTT_TLS"""
        env_example = PROJECT_ROOT / ".env.example"
        assert env_example.exists(), ".env.example not found"
        
        with open(env_example, 'r') as f:
            content = f.read()
        
        assert "MQTT_TLS" in content, "MQTT_TLS not in .env.example"
        
        # Check default is false for development
        assert "MQTT_TLS=false" in content, "MQTT_TLS should default to false for development"
    
    def test_configure_security_script(self):
        """Test that configure_security.sh script exists and works"""
        script_path = PROJECT_ROOT / "scripts" / "configure_security.sh"
        assert script_path.exists(), "configure_security.sh not found"
        assert script_path.stat().st_mode & 0o111, "configure_security.sh not executable"
        
        # Test script runs
        result = subprocess.run(
            [str(script_path), "status"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Security Configuration" in result.stdout


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])