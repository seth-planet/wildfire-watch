#!/usr/bin/env python3.12
"""
TLS-enabled Test MQTT Broker Infrastructure
Provides a real MQTT broker with TLS support for testing wildfire-watch services
"""
import os
import sys
import time
import threading
import subprocess
import tempfile
import socket
import ssl
from pathlib import Path
import paho.mqtt.client as mqtt
from tests.mqtt_test_broker import MQTTTestBroker

class MQTTTLSTestBroker(MQTTTestBroker):
    """
    Manages a real mosquitto MQTT broker with TLS support for testing
    """
    
    def __init__(self, port=None, tls_port=None, cert_dir=None):
        super().__init__(port)
        self.tls_port = tls_port or self._find_free_port()
        # Use the project's certs directory by default
        self.cert_dir = cert_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'certs'))
        self.ca_cert = None
        self.server_cert = None
        self.server_key = None
        
    def _start_mosquitto(self):
        """Start mosquitto broker with TLS support"""
        # Create temporary directories
        self.data_dir = tempfile.mkdtemp(prefix="mqtt_tls_test_")
        
        # Check if certificates exist, otherwise create test certificates
        self._setup_test_certificates()
        
        # Create mosquitto config for testing with TLS
        config_content = f"""
# Standard port
port {self.port}
protocol mqtt

# TLS port
listener {self.tls_port}
protocol mqtt
cafile {self.ca_cert}
certfile {self.server_cert}
keyfile {self.server_key}
tls_version tlsv1.2
require_certificate false

# Common settings
allow_anonymous true
persistence false
log_type error
log_dest stdout
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto broker
        self.process = subprocess.Popen([
            'mosquitto', '-c', self.config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for broker to be ready with proper health check
        if not self.wait_for_ready(timeout=30):
            # Get output for debugging
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Mosquitto failed to start within 30s. Stdout: {stdout.decode()[:500]} Stderr: {stderr.decode()[:500]}")
    
    def _setup_test_certificates(self):
        """Set up test certificates for TLS"""
        # Use existing certificates if available
        ca_cert_path = os.path.join(self.cert_dir, 'ca.crt')
        server_cert_path = os.path.join(self.cert_dir, 'server.crt')
        server_key_path = os.path.join(self.cert_dir, 'server.key')
        
        if os.path.exists(ca_cert_path) and os.path.exists(server_cert_path) and os.path.exists(server_key_path):
            self.ca_cert = ca_cert_path
            self.server_cert = server_cert_path
            self.server_key = server_key_path
            return
        
        # Otherwise create temporary test certificates
        temp_cert_dir = os.path.join(self.data_dir, 'certs')
        os.makedirs(temp_cert_dir, exist_ok=True)
        
        # Generate self-signed certificates for testing
        self._generate_test_certificates(temp_cert_dir)
        
        self.ca_cert = os.path.join(temp_cert_dir, 'ca.crt')
        self.server_cert = os.path.join(temp_cert_dir, 'server.crt')
        self.server_key = os.path.join(temp_cert_dir, 'server.key')
    
    def _generate_test_certificates(self, cert_dir):
        """Generate self-signed certificates for testing"""
        try:
            # Generate CA key and certificate
            subprocess.run([
                'openssl', 'req', '-new', '-x509', '-days', '365',
                '-extensions', 'v3_ca', '-keyout', os.path.join(cert_dir, 'ca.key'),
                '-out', os.path.join(cert_dir, 'ca.crt'),
                '-subj', '/CN=Test-CA',
                '-nodes'
            ], check=True, capture_output=True)
            
            # Generate server key
            subprocess.run([
                'openssl', 'genrsa', '-out', os.path.join(cert_dir, 'server.key'), '2048'
            ], check=True, capture_output=True)
            
            # Generate server certificate request
            subprocess.run([
                'openssl', 'req', '-new', '-key', os.path.join(cert_dir, 'server.key'),
                '-out', os.path.join(cert_dir, 'server.csr'),
                '-subj', '/CN=localhost'
            ], check=True, capture_output=True)
            
            # Sign server certificate
            subprocess.run([
                'openssl', 'x509', '-req', '-in', os.path.join(cert_dir, 'server.csr'),
                '-CA', os.path.join(cert_dir, 'ca.crt'),
                '-CAkey', os.path.join(cert_dir, 'ca.key'),
                '-CAcreateserial', '-out', os.path.join(cert_dir, 'server.crt'),
                '-days', '365'
            ], check=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to generate test certificates: {e}")
    
    def wait_for_tls_ready(self, timeout=10):
        """Wait for TLS port to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Create test client with TLS
                test_client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"test_tls_ready_{int(time.time() * 1000)}"
                )
                
                # Configure TLS
                test_client.tls_set(
                    ca_certs=self.ca_cert,
                    cert_reqs=ssl.CERT_REQUIRED,
                    tls_version=ssl.PROTOCOL_TLS
                )
                test_client.tls_insecure_set(True)  # For self-signed certs
                
                # Set up connection callback
                connected = threading.Event()
                
                def on_connect(client, userdata, flags, rc, properties=None):
                    if rc == 0:
                        connected.set()
                
                test_client.on_connect = on_connect
                
                # Try to connect to TLS port
                test_client.connect("localhost", self.tls_port, 60)
                test_client.loop_start()
                
                # Wait for connection with timeout
                if connected.wait(timeout=2.0):
                    test_client.disconnect()
                    test_client.loop_stop()
                    return True
                
                # Cleanup on failure
                test_client.loop_stop()
                
            except Exception as e:
                pass  # Keep trying
                
            time.sleep(0.5)
        
        return False
    
    def get_tls_connection_params(self):
        """Get TLS connection parameters for clients"""
        return {
            'host': 'localhost',
            'port': self.tls_port,
            'ca_certs': self.ca_cert,
            'cert_reqs': ssl.CERT_REQUIRED,
            'tls_version': ssl.PROTOCOL_TLS,
            'tls_insecure': True  # For self-signed certs
        }