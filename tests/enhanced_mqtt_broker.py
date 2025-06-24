#!/usr/bin/env python3.12
"""
Enhanced MQTT Test Broker with Connection Pooling and Isolation
"""
import os
import time
import threading
import subprocess
import tempfile
import socket
import logging
from pathlib import Path
from typing import Dict, Optional, Set
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class ConnectionPool:
    """Manages a pool of MQTT client connections"""
    
    def __init__(self, broker_host: str, broker_port: int):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.connections: Dict[str, mqtt.Client] = {}
        self._lock = threading.Lock()
        
    def get_client(self, client_id: str) -> mqtt.Client:
        """Get or create a client connection"""
        with self._lock:
            if client_id not in self.connections:
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
                client.connect(self.broker_host, self.broker_port, keepalive=60)
                client.loop_start()
                self.connections[client_id] = client
                logger.debug(f"Created new client: {client_id}")
            return self.connections[client_id]
    
    def release_client(self, client_id: str):
        """Release a client (keep it in pool for reuse)"""
        # Clients stay in pool for reuse
        pass
    
    def cleanup(self):
        """Clean up all connections"""
        with self._lock:
            for client_id, client in self.connections.items():
                try:
                    client.loop_stop()
                    client.disconnect()
                    logger.debug(f"Disconnected client: {client_id}")
                except Exception as e:
                    logger.warning(f"Error disconnecting {client_id}: {e}")
            self.connections.clear()

class TestMQTTBroker:
    """
    Enhanced MQTT broker for testing with isolation features
    """
    
    # Class-level broker instance for session reuse
    _session_broker = None
    _session_lock = threading.Lock()
    
    def __init__(self, port=None, session_scope=True):
        self.session_scope = session_scope
        
        if session_scope and TestMQTTBroker._session_broker:
            # Reuse existing session broker
            self._reuse_session_broker()
        else:
            # Create new broker instance
            self.port = port or self._find_free_port()
            self.host = 'localhost'
            self.process = None
            self.config_file = None
            self.data_dir = None
            self.connection_pool = None
            self._subscribers: Set[str] = set()
            self._active_topics: Set[str] = set()
    
    def _reuse_session_broker(self):
        """Reuse the session-scoped broker"""
        broker = TestMQTTBroker._session_broker
        self.port = broker.port
        self.host = broker.host
        self.process = broker.process
        self.config_file = broker.config_file
        self.data_dir = broker.data_dir
        self.connection_pool = broker.connection_pool
        logger.debug(f"Reusing session broker on port {self.port}")
    
    def _find_free_port(self):
        """Find an available port for the test broker"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start(self):
        """Start the test MQTT broker"""
        if self.session_scope and TestMQTTBroker._session_broker:
            # Already started
            return
            
        with TestMQTTBroker._session_lock:
            if self.session_scope and TestMQTTBroker._session_broker:
                # Double-check after acquiring lock
                self._reuse_session_broker()
                return
                
            # Try mosquitto first
            try:
                self._start_mosquitto()
                self.connection_pool = ConnectionPool(self.host, self.port)
                
                if self.session_scope:
                    TestMQTTBroker._session_broker = self
                    
            except (FileNotFoundError, RuntimeError) as e:
                logger.error(f"Failed to start mosquitto: {e}")
                raise
    
    def _start_mosquitto(self):
        """Start mosquitto broker with enhanced configuration"""
        # Create temporary directories
        self.data_dir = tempfile.mkdtemp(prefix="mqtt_test_")
        
        # Create mosquitto config with enhanced settings
        config_content = f"""
# Basic Configuration
port {self.port}
allow_anonymous true

# Performance Settings
max_connections 1000
max_queued_messages 10000
max_inflight_messages 100
max_packet_size 1048576

# Persistence (disabled for tests)
persistence false

# Logging
log_type error
log_type warning

# Connection Settings
retry_interval 10
sys_interval 30

# Protocol Settings
protocol mqtt
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto broker
        self.process = subprocess.Popen([
            'mosquitto', '-c', self.config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for broker to start
        time.sleep(1.0)
        
        # Check if process is running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Failed to start mosquitto: {stderr.decode()}")
        
        # Verify connection
        if not self.wait_for_ready(timeout=10):
            raise RuntimeError("MQTT broker failed to become ready")
            
        logger.info(f"Mosquitto broker started on port {self.port}")
    
    def stop(self):
        """Stop the test MQTT broker"""
        if self.session_scope:
            # Don't stop session broker
            logger.debug("Not stopping session-scoped broker")
            return
            
        self._stop_broker()
    
    def _stop_broker(self):
        """Actually stop the broker process"""
        if self.connection_pool:
            self.connection_pool.cleanup()
            
        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        # Don't wait after kill
                self.process = None
            except Exception as e:
                logger.warning(f"Error stopping mosquitto: {e}")
        
        # Clean up temporary files
        if self.data_dir and os.path.exists(self.data_dir):
            import shutil
            try:
                shutil.rmtree(self.data_dir)
            except Exception:
                pass
    
    def get_connection_params(self):
        """Get connection parameters for clients"""
        return {
            'host': self.host,
            'port': self.port,
            'keepalive': 60
        }
    
    def wait_for_ready(self, timeout=10):
        """Wait for broker to be ready to accept connections"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                test_client.connect(self.host, self.port, 60)
                test_client.disconnect()
                return True
            except:
                time.sleep(0.5)
        return False
    
    def get_pooled_client(self, client_id: str) -> mqtt.Client:
        """Get a pooled client connection"""
        if not self.connection_pool:
            raise RuntimeError("Broker not started")
        return self.connection_pool.get_client(client_id)
    
    def is_running(self):
        """Check if the broker is running"""
        if self.process:
            return self.process.poll() is None
        return False
    
    def reset_state(self):
        """Reset broker state between tests"""
        # Clear tracking sets
        self._subscribers.clear()
        self._active_topics.clear()
        
        # Note: We don't disconnect clients as they may be reused
        logger.debug("Reset broker state for next test")
    
    @classmethod
    def cleanup_session(cls):
        """Clean up session broker"""
        with cls._session_lock:
            if cls._session_broker:
                cls._session_broker._stop_broker()
                cls._session_broker = None