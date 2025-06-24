#!/usr/bin/env python3.12
"""
Enhanced MQTT Test Broker with Better Connection Management
Fixes connection issues and port conflicts in tests
"""
import os
import sys
import time
import socket
import subprocess
import tempfile
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedMQTTTestBroker:
    """Enhanced MQTT test broker with better error handling and connection management"""
    
    # Class variable to track used ports
    _used_ports = set()
    _port_lock = threading.Lock()
    
    def __init__(self, port=None):
        self.port = port or self._find_free_port()
        self.host = 'localhost'
        self.process = None
        self.config_file = None
        self.data_dir = None
        self._running = False
        
    def _find_free_port(self):
        """Find an available port that hasn't been used recently"""
        with self._port_lock:
            max_attempts = 50
            for _ in range(max_attempts):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    s.listen(1)
                    port = s.getsockname()[1]
                    
                # Ensure port is in valid range and not recently used
                if 10000 <= port <= 65000 and port not in self._used_ports:
                    self._used_ports.add(port)
                    # Clean up old ports (keep last 10)
                    if len(self._used_ports) > 10:
                        self._used_ports.pop()
                    return port
                    
            # Fallback to any available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                return s.getsockname()[1]
    
    def _is_port_available(self, port):
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('localhost', port))
                return True
        except:
            return False
    
    def start(self):
        """Start the MQTT broker with better error handling"""
        if self._running:
            logger.warning("Broker already running")
            return
            
        # Ensure port is available
        if not self._is_port_available(self.port):
            logger.warning(f"Port {self.port} not available, finding new port")
            self.port = self._find_free_port()
            
        try:
            self._start_mosquitto()
            self._running = True
        except Exception as e:
            logger.error(f"Failed to start mosquitto: {e}")
            raise RuntimeError(f"Could not start MQTT broker: {e}")
    
    def _start_mosquitto(self):
        """Start mosquitto broker with minimal config"""
        # Create temporary directory
        self.data_dir = tempfile.mkdtemp(prefix="mqtt_test_")
        
        # Create minimal mosquitto config
        config_content = f"""
# Minimal test configuration
listener {self.port}
allow_anonymous true
log_type none
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto
        cmd = ['mosquitto', '-c', self.config_file]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.data_dir
        )
        
        # Wait for broker to start
        if not self._wait_for_broker_ready():
            self.stop()
            raise RuntimeError(f"Mosquitto failed to start on port {self.port}")
            
        logger.info(f"Mosquitto started on port {self.port}")
    
    def _wait_for_broker_ready(self, timeout=10):
        """Wait for broker to be ready to accept connections"""
        import paho.mqtt.client as mqtt
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.process and self.process.poll() is not None:
                return False
                
            # Try to connect
            try:
                test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                test_client.connect("localhost", self.port, 10)
                test_client.disconnect()
                return True
            except:
                time.sleep(0.1)
                
        return False
    
    def wait_for_ready(self, timeout=30):
        """Public method to wait for broker readiness"""
        return self._wait_for_broker_ready(timeout)
    
    def stop(self):
        """Stop the broker with proper cleanup"""
        self._running = False
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except:
                    pass
            except:
                pass
            finally:
                self.process = None
        
        # Clean up files
        if self.data_dir and os.path.exists(self.data_dir):
            try:
                import shutil
                shutil.rmtree(self.data_dir)
            except:
                pass
                
        # Release port
        with self._port_lock:
            self._used_ports.discard(self.port)
    
    def get_connection_params(self):
        """Get connection parameters for clients"""
        return {
            'host': self.host,
            'port': self.port,
            'keepalive': 60
        }
    
    def is_running(self):
        """Check if broker is running"""
        return self._running and (self.process is None or self.process.poll() is None)
    
    def __enter__(self):
        """Context manager support"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.stop()
    
    def publish_and_wait(self, client, topic, payload, qos=1, timeout=5):
        """Publish message and wait for delivery confirmation"""
        import json
        
        # Track delivery
        delivered = False
        
        def on_publish(client, userdata, mid, reason_code=None, properties=None):
            nonlocal delivered
            delivered = True
        
        # Set callback
        old_on_publish = client.on_publish
        client.on_publish = on_publish
        
        try:
            # Publish message
            if isinstance(payload, (dict, list)):
                payload = json.dumps(payload)
            elif not isinstance(payload, (str, bytes)):
                payload = str(payload)
                
            result = client.publish(topic, payload, qos=qos)
            
            if result.rc != 0:
                return False
                
            # Wait for delivery
            start_time = time.time()
            while not delivered and time.time() - start_time < timeout:
                time.sleep(0.01)
                
            return delivered
            
        finally:
            # Restore original callback
            client.on_publish = old_on_publish


# Singleton broker manager to prevent multiple brokers on same port
class MQTTBrokerManager:
    """Manages MQTT brokers to prevent conflicts"""
    _instance = None
    _lock = threading.Lock()
    _brokers = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_broker(self, scope='session'):
        """Get or create a broker for the given scope"""
        with self._lock:
            if scope not in self._brokers or not self._brokers[scope].is_running():
                # Clean up old broker if exists
                if scope in self._brokers:
                    try:
                        self._brokers[scope].stop()
                    except:
                        pass
                        
                # Create new broker
                broker = EnhancedMQTTTestBroker()
                broker.start()
                self._brokers[scope] = broker
                
            return self._brokers[scope]
    
    def cleanup(self):
        """Clean up all brokers"""
        with self._lock:
            for broker in self._brokers.values():
                try:
                    broker.stop()
                except:
                    pass
            self._brokers.clear()


# Export the enhanced broker
MQTTTestBroker = EnhancedMQTTTestBroker