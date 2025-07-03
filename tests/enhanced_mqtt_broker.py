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
import signal
import logging
from pathlib import Path
from typing import Dict, Optional, Set
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

# Configuration toggle for per-worker brokers vs shared broker
USE_PER_WORKER_BROKERS = os.getenv('TEST_PER_WORKER_BROKERS', 'true').lower() == 'true'

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
    
    # Class-level broker instances per worker for parallel isolation
    _worker_brokers: Dict[str, 'TestMQTTBroker'] = {}
    _worker_lock = threading.Lock()
    
    def __init__(self, port=None, session_scope=True, worker_id='master'):
        self.session_scope = session_scope
        self.worker_id = worker_id
        
        # Determine if we should use per-worker brokers or shared
        if USE_PER_WORKER_BROKERS:
            # Per-worker broker mode
            if session_scope and worker_id in TestMQTTBroker._worker_brokers:
                # Reuse existing broker for this worker
                self._reuse_worker_broker()
            else:
                # Create new broker instance with worker-based port allocation
                if port is None:
                    port = self._allocate_worker_port()
                self.port = port
                self.host = 'localhost'
                self.process = None
                self.config_file = None
                self.data_dir = None
                self.connection_pool = None
                self._subscribers: Set[str] = set()
                self._active_topics: Set[str] = set()
        else:
            # Shared broker mode - all workers use same broker
            if session_scope and 'shared' in TestMQTTBroker._worker_brokers:
                # Reuse the shared broker
                self.worker_id = 'shared'
                self._reuse_worker_broker()
            else:
                # Create shared broker on default port
                self.worker_id = 'shared'
                self.port = port or 11883
                self.host = 'localhost'
                self.process = None
                self.config_file = None
                self.data_dir = None
                self.connection_pool = None
                self._subscribers: Set[str] = set()
                self._active_topics: Set[str] = set()
    
    def _reuse_worker_broker(self):
        """Reuse the worker-specific broker"""
        broker = TestMQTTBroker._worker_brokers[self.worker_id]
        self.port = broker.port
        self.host = broker.host
        self.process = broker.process
        self.config_file = broker.config_file
        self.data_dir = broker.data_dir
        self.connection_pool = broker.connection_pool
        # Initialize tracking attributes for this instance
        self._subscribers = set()
        self._active_topics = set()
        logger.debug(f"Reusing broker for worker {self.worker_id} on port {self.port}")
    
    def _allocate_worker_port(self):
        """Allocate a port based on worker ID to prevent conflicts"""
        base_port = 20000  # High port range to avoid conflicts
        
        if self.worker_id == 'master':
            # Non-parallel execution uses base port
            return base_port
        elif self.worker_id.startswith('gw'):
            # Extract worker number from ID (gw0 -> 0, gw1 -> 1, etc.)
            try:
                worker_num = int(self.worker_id[2:])
                # Allocate with 100-port spacing to avoid conflicts
                return base_port + (worker_num * 100)
            except ValueError:
                # Fallback to finding a free port
                return self._find_free_port()
        else:
            # Unknown worker ID format, find a free port
            return self._find_free_port()
    
    def _find_free_port(self):
        """Find an available port for the test broker"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start(self):
        """Start the test MQTT broker"""
        if self.session_scope and self.worker_id in TestMQTTBroker._worker_brokers:
            # Already started for this worker
            return
            
        with TestMQTTBroker._worker_lock:
            if self.session_scope and self.worker_id in TestMQTTBroker._worker_brokers:
                # Double-check after acquiring lock
                self._reuse_worker_broker()
                return
                
            # Try mosquitto first
            try:
                self._start_mosquitto()
                self.connection_pool = ConnectionPool(self.host, self.port)
                
                if self.session_scope:
                    TestMQTTBroker._worker_brokers[self.worker_id] = self
                    
            except (FileNotFoundError, RuntimeError) as e:
                logger.error(f"Failed to start mosquitto for worker {self.worker_id}: {e}")
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
sys_interval 30

# Protocol Settings
protocol mqtt
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto broker with process group for better cleanup
        try:
            self.process = subprocess.Popen([
                'mosquitto', '-c', self.config_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
               preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
        except FileNotFoundError:
            raise RuntimeError("mosquitto binary not found. Install with: sudo apt-get install mosquitto")
        
        # Wait for broker to start
        time.sleep(1.5)
        
        # Check if process is running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            error_msg = stderr.decode() if stderr else "Unknown error"
            # Check for common issues
            if "bind" in error_msg.lower() or "address already in use" in error_msg.lower():
                # Port conflict - try to allocate a different port
                new_port = self._find_free_port()
                logger.warning(f"Port {self.port} in use, retrying with port {new_port}")
                self.port = new_port
                return self._start_mosquitto()  # Retry with new port
            raise RuntimeError(f"Failed to start mosquitto: {error_msg}")
        
        # Verify connection with better error reporting
        if not self.wait_for_ready(timeout=10):
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                error_msg = stderr.decode() if stderr else "Process exited"
                raise RuntimeError(f"MQTT broker failed to become ready: {error_msg}")
            else:
                raise RuntimeError("MQTT broker failed to become ready: timeout")
            
        logger.info(f"Mosquitto broker started for worker {self.worker_id} on port {self.port}")
    
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
            pid = self.process.pid
            try:
                if self.process.poll() is None:
                    # First try graceful termination
                    logger.debug(f"Terminating mosquitto process {pid}")
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                        logger.debug(f"Mosquitto process {pid} terminated gracefully")
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful termination fails
                        logger.warning(f"Force killing mosquitto process {pid}")
                        try:
                            # Kill process group if possible
                            if hasattr(os, 'killpg'):
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            else:
                                self.process.kill()
                        except (ProcessLookupError, OSError):
                            # Process already dead
                            pass
                        try:
                            # Wait a bit for kill to take effect
                            self.process.wait(timeout=2)
                            logger.debug(f"Mosquitto process {pid} killed successfully")
                        except subprocess.TimeoutExpired:
                            logger.error(f"Failed to kill mosquitto process {pid}")
                else:
                    logger.debug(f"Mosquitto process {pid} already terminated with code {self.process.returncode}")
                    
                # Always wait for process cleanup
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                    
                self.process = None
            except Exception as e:
                logger.error(f"Error stopping mosquitto process {pid}: {e}")
                # Force set to None to prevent further attempts
                self.process = None
        
        # Clean up temporary files
        if self.data_dir and os.path.exists(self.data_dir):
            import shutil
            try:
                shutil.rmtree(self.data_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp directory {self.data_dir}: {e}")
    
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
    
    def wait_for_connection(self, timeout=5):
        """Wait for MQTT broker to be ready for connections
        
        This method provides compatibility with tests expecting this method.
        """
        return self.wait_for_ready(timeout)
    
    def wait_for_connection_ready(self, client, timeout=10):
        """Wait for MQTT client to be fully connected and ready
        
        Args:
            client: The MQTT client to check
            timeout: Maximum time to wait
            
        Returns:
            bool: True if client is connected, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if client.is_connected():
                # Give extra time for subscription setup
                time.sleep(0.5)
                return True
            time.sleep(0.1)
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
    
    def publish_and_wait(self, client, topic, payload, qos=1, timeout=5):
        """Publish message and wait for delivery confirmation"""
        message_delivered = False
        
        def on_publish(client, userdata, mid, reason_code=None, properties=None):
            nonlocal message_delivered
            message_delivered = True
            
        # Store original callback
        original_on_publish = client.on_publish
        client.on_publish = on_publish
        
        try:
            info = client.publish(topic, payload, qos=qos)
            
            # Wait for delivery
            start_time = time.time()
            while not message_delivered and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            return message_delivered
        finally:
            # Restore original callback
            client.on_publish = original_on_publish
    
    def reset_state(self):
        """Reset broker state between tests"""
        # Clear tracking sets
        self._subscribers.clear()
        self._active_topics.clear()
        
        # Note: We don't disconnect clients as they may be reused
        logger.debug("Reset broker state for next test")
    
    @classmethod
    def cleanup_session(cls, worker_id=None):
        """Clean up session broker(s)"""
        with cls._worker_lock:
            if worker_id:
                # Clean up specific worker's broker
                if worker_id in cls._worker_brokers:
                    broker = cls._worker_brokers[worker_id]
                    logger.info(f"Cleaning up broker for worker {worker_id}")
                    broker._stop_broker()
                    del cls._worker_brokers[worker_id]
            else:
                # Clean up all worker brokers
                logger.info(f"Cleaning up {len(cls._worker_brokers)} worker brokers")
                for worker_id, broker in list(cls._worker_brokers.items()):
                    logger.debug(f"Stopping broker for worker {worker_id}")
                    broker._stop_broker()
                cls._worker_brokers.clear()
                
        # Additional cleanup: kill any stray mosquitto processes from tests
        try:
            # Find mosquitto processes using test temp directories
            result = subprocess.run(['pgrep', '-f', 'mqtt_test_'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                stray_count = 0
                for pid in pids:
                    if pid.strip():
                        try:
                            pid_num = int(pid.strip())
                            # First try graceful termination
                            os.kill(pid_num, signal.SIGTERM)
                            time.sleep(0.5)
                            # Check if still running
                            try:
                                os.kill(pid_num, 0)  # Check if process exists
                                # Still running, force kill
                                os.kill(pid_num, signal.SIGKILL)
                                logger.warning(f"Force killed stray mosquitto process {pid_num}")
                            except ProcessLookupError:
                                # Process already terminated
                                logger.debug(f"Terminated stray mosquitto process {pid_num}")
                            stray_count += 1
                        except (ValueError, ProcessLookupError, OSError):
                            # Invalid PID or process already gone
                            pass
                if stray_count > 0:
                    logger.info(f"Cleaned up {stray_count} stray mosquitto processes")
        except Exception as e:
            logger.warning(f"Error cleaning up stray processes: {e}")