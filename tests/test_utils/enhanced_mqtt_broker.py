#!/usr/bin/env python3
"""
Enhanced MQTT Test Broker with Connection Pooling and Isolation
"""
import os
import sys
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

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from test_utils.safe_logging import safe_log
except ImportError:
    # Fallback if safe_logging is not available
    def safe_log(logger_obj, level: str, message: str, exc_info: bool = False) -> None:
        """Fallback safe logging function."""
        if hasattr(logger_obj, level.lower()):
            log_func = getattr(logger_obj, level.lower())
            log_func(message, exc_info=exc_info)

logger = logging.getLogger(__name__)

def _safe_log(level: str, message: str, exc_info: bool = False) -> None:
    """Wrapper for module-level safe logging."""
    safe_log(logger, level, message, exc_info=exc_info)

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
                # Handle different paho-mqtt versions for Python 3.8 compatibility
                try:
                    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
                except AttributeError:
                    # Older paho-mqtt version without CallbackAPIVersion
                    client = mqtt.Client(client_id=client_id)
                client.connect(self.broker_host, self.broker_port, keepalive=60)
                client.loop_start()
                self.connections[client_id] = client
                _safe_log('debug', f"Created new client: {client_id}")
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
                    _safe_log('debug', f"Disconnected client: {client_id}")
                except Exception as e:
                    _safe_log('warning', f"Error disconnecting {client_id}: {e}")
            self.connections.clear()

class TestMQTTBroker:
    """
    Enhanced MQTT broker for testing with isolation features
    """
    __test__ = False  # Prevent pytest from collecting this as a test class
    
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
        _safe_log('debug', f"Reusing broker for worker {self.worker_id} on port {self.port}")
    
    def _allocate_worker_port(self):
        """Allocate a port based on worker ID to prevent conflicts"""
        base_port = 30000  # Higher port range to avoid conflicts with system services
        
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
        """Find an available port for the test broker with retry logic"""
        import random
        
        # Try specific port ranges first to avoid conflicts
        port_ranges = [
            (30000, 40000),  # Primary range
            (40000, 50000),  # Secondary range
            (50000, 60000),  # Tertiary range
        ]
        
        for min_port, max_port in port_ranges:
            # Try 10 random ports in each range
            for _ in range(10):
                port = random.randint(min_port, max_port)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind(('', port))
                        s.listen(1)
                        return port
                except OSError:
                    continue
        
        # Fallback to system-allocated port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _wait_for_port(self, timeout=15, interval=0.5):
        """Wait for the MQTT broker port to become available.
        
        Args:
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds
            
        Returns:
            True if port is available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the port
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    s.connect((self.host, self.port))
                    _safe_log('debug', f"MQTT broker is ready on port {self.port}")
                    return True
            except (socket.error, ConnectionRefusedError):
                # Port not ready yet
                time.sleep(interval)
            except Exception as e:
                _safe_log('warning', f"Unexpected error checking port {self.port}: {e}")
                time.sleep(interval)
        
        _safe_log('error', f"MQTT broker port {self.port} not available after {timeout}s")
        return False
    
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
                _safe_log('error', f"Failed to start mosquitto for worker {self.worker_id}: {e}")
                raise
    
    def _start_mosquitto(self):
        """Start mosquitto broker with enhanced configuration"""
        # Set environment variable to disable DLT logging which causes FIFO warnings
        env = os.environ.copy()
        env['DLT_LOG'] = '0'
        
        # Clean up DLT FIFO to prevent startup errors
        dlt_fifo = '/tmp/dlt'
        try:
            if os.path.exists(dlt_fifo) and os.path.isfifo(dlt_fifo):
                os.remove(dlt_fifo)
        except (OSError, PermissionError):
            pass  # Ignore errors, mosquitto will handle it
            
        # Create temporary directories with worker-specific prefix
        self.data_dir = tempfile.mkdtemp(prefix=f"mqtt_test_{self.worker_id}_")
        
        # Create mosquitto config with enhanced settings
        pid_file = os.path.join(self.data_dir, "mosquitto.pid")
        config_content = f"""
# Basic Configuration
port {self.port}
allow_anonymous true

# PID file (unique for this instance)
pid_file {pid_file}

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
log_dest file /dev/null

# Connection Settings
sys_interval 30

# Protocol Settings
protocol mqtt
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto broker with process group for better cleanup
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                self.process = subprocess.Popen([
                    'mosquitto', '-c', self.config_file
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   env=env,  # Pass the environment with DLT_LOG=0
                   preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
                break  # Success, exit retry loop
            except FileNotFoundError:
                raise RuntimeError("mosquitto binary not found. Install with: sudo apt-get install mosquitto")
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    _safe_log('warning', f"Failed to start mosquitto (attempt {retry_count}/{max_retries}): {e}")
                    # Allocate a new port and try again
                    self.port = self._find_free_port()
                    # Update config file with new port
                    config_content = f"""
# Basic Configuration
port {self.port}
allow_anonymous true

# PID file (unique for this instance)
pid_file {pid_file}

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
log_dest file /dev/null

# Connection Settings
sys_interval 30

# Protocol Settings
protocol mqtt
"""
                    with open(self.config_file, 'w') as f:
                        f.write(config_content)
        
        if self.process is None and last_error:
            raise RuntimeError(f"Failed to start mosquitto after {max_retries} attempts: {last_error}")
        
        # Wait for broker to be ready using proper health check
        if not self._wait_for_port(timeout=15):
            if self.process:
                self.process.terminate()
                stdout, stderr = self.process.communicate()
                error_msg = f"stdout: {stdout.decode() if stdout else 'No stdout'}, stderr: {stderr.decode() if stderr else 'No stderr'}"
                raise RuntimeError(f"Mosquitto failed to start on port {self.port}: {error_msg}")
            else:
                raise RuntimeError(f"Mosquitto process failed to start")
        
        # Check if process is still running after health check</        
        # Check if process is running
        if self.process and self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            stdout_msg = stdout.decode() if stdout else "No stdout"
            stderr_msg = stderr.decode() if stderr else "No stderr"
            error_msg = f"stdout: {stdout_msg}, stderr: {stderr_msg}"
            _safe_log('error', f"Mosquitto process exited with code {self.process.returncode}")
            _safe_log('error', f"Mosquitto error output: {error_msg}")
            _safe_log('error', f"Python version: {sys.version}")
            _safe_log('error', f"Command: mosquitto -c {self.config_file}")
            _safe_log('error', f"Config file exists: {os.path.exists(self.config_file)}")
            _safe_log('error', f"Port: {self.port}")
            # Check for common issues
            if "bind" in error_msg.lower() or "address already in use" in error_msg.lower():
                # Port conflict - try to allocate a different port
                if retry_count < max_retries - 1:
                    retry_count += 1
                    new_port = self._find_free_port()
                    _safe_log('warning', f"Port {self.port} in use, retrying with port {new_port} (attempt {retry_count}/{max_retries})")
                    self.port = new_port
                    # Update the port in config
                    config_content = f"""
# Basic Configuration
port {self.port}
allow_anonymous true

# PID file (unique for this instance)
pid_file {pid_file}

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
log_dest file /dev/null

# Connection Settings
sys_interval 30

# Protocol Settings
protocol mqtt
"""
                    with open(self.config_file, 'w') as f:
                        f.write(config_content)
                    return self._start_mosquitto()  # Retry with new port
            # Ignore DLT FIFO warnings
            if "dlt" in error_msg.lower() and "fifo" in error_msg.lower():
                _safe_log('debug', "Ignoring DLT FIFO warning")
            else:
                raise RuntimeError(f"Failed to start mosquitto: {error_msg}")
        
        # Verify connection with better error reporting
        if not self.wait_for_ready(timeout=10):
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                stdout_msg = stdout.decode() if stdout else ""
                stderr_msg = stderr.decode() if stderr else ""
                error_msg = f"Process exited with code {self.process.returncode}. "
                if stderr_msg:
                    error_msg += f"stderr: {stderr_msg} "
                if stdout_msg:
                    error_msg += f"stdout: {stdout_msg}"
                error_msg += f" (Python {sys.version.split()[0]}, Port {self.port})"
                raise RuntimeError(f"MQTT broker failed to become ready: {error_msg}")
            else:
                raise RuntimeError(f"MQTT broker failed to become ready: timeout (Python {sys.version.split()[0]}, Port {self.port})")
            
        _safe_log('info', f"Mosquitto broker started for worker {self.worker_id} on port {self.port}")
    
    def stop(self):
        """Stop the test MQTT broker"""
        if self.session_scope:
            # Don't stop session broker
            _safe_log('debug', "Not stopping session-scoped broker")
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
                    _safe_log('debug', f"Terminating mosquitto process {pid}")
                    self.process.terminate()
                    
                    # Use a shorter timeout and non-blocking approach
                    start_time = time.time()
                    while time.time() - start_time < 0.5:  # Very short timeout for termination
                        if self.process.poll() is not None:
                            _safe_log('debug', f"Mosquitto process {pid} terminated gracefully")
                            break
                        time.sleep(0.05)  # Shorter sleep for faster detection
                    else:
                        # Force kill if graceful termination fails
                        # No warning log for force kill - this is expected behavior for mosquitto
                        try:
                            # Kill process group if possible
                            if hasattr(os, 'killpg'):
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            else:
                                self.process.kill()
                        except (ProcessLookupError, OSError):
                            # Process already dead
                            pass
                        
                        # Wait for kill to take effect - much shorter timeout
                        start_time = time.time()
                        while time.time() - start_time < 0.5:  # Only wait 0.5 seconds for kill
                            if self.process.poll() is not None:
                                _safe_log('debug', f"Mosquitto process {pid} killed successfully")
                                break
                            time.sleep(0.05)
                        else:
                            _safe_log('error', f"Failed to kill mosquitto process {pid}, giving up")
                else:
                    _safe_log('debug', f"Mosquitto process {pid} already terminated with code {self.process.returncode}")
                    
                # Don't wait again, just clean up
                pass
                    
                self.process = None
            except Exception as e:
                _safe_log('error', f"Error stopping mosquitto process {pid}: {e}")
                # Force set to None to prevent further attempts
                self.process = None
        
        # Clean up temporary files
        if self.data_dir and os.path.exists(self.data_dir):
            import shutil
            try:
                shutil.rmtree(self.data_dir)
            except Exception as e:
                _safe_log('warning', f"Error cleaning up temp directory {self.data_dir}: {e}")
    
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
        last_error = None
        attempts = 0
        
        while time.time() - start_time < timeout:
            attempts += 1
            try:
                # Create test client with unique ID to avoid conflicts
                # Handle different paho-mqtt versions for Python 3.8 compatibility
                client_id = f"test_ready_{self.worker_id}_{int(time.time() * 1000)}"
                try:
                    test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
                except AttributeError:
                    # Older paho-mqtt version without CallbackAPIVersion
                    test_client = mqtt.Client(client_id=client_id)
                test_client.connect(self.host, self.port, 60)
                test_client.disconnect()
                _safe_log('debug', f"Broker ready on port {self.port} after {attempts} attempts")
                return True
            except Exception as e:
                last_error = e
                # Check if broker process is still running
                if self.process and self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    _safe_log('error', f"Broker process died: stdout={stdout.decode()[:200]}, stderr={stderr.decode()[:200]}")
                    return False
                time.sleep(0.5)
        
        _safe_log('error', f"Broker not ready after {timeout}s and {attempts} attempts. Last error: {last_error}")
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
        _safe_log('debug', "Reset broker state for next test")
    
    @classmethod
    def cleanup_session(cls, worker_id=None):
        """Clean up session broker(s)"""
        with cls._worker_lock:
            if worker_id:
                # Clean up specific worker's broker
                if worker_id in cls._worker_brokers:
                    broker = cls._worker_brokers[worker_id]
                    _safe_log('info', f"Cleaning up broker for worker {worker_id}")
                    broker._stop_broker()
                    del cls._worker_brokers[worker_id]
            else:
                # Clean up all worker brokers
                _safe_log('info', f"Cleaning up {len(cls._worker_brokers)} worker brokers")
                for worker_id, broker in list(cls._worker_brokers.items()):
                    _safe_log('debug', f"Stopping broker for worker {worker_id}")
                    broker._stop_broker()
                cls._worker_brokers.clear()
                
        # Additional cleanup: kill any stray mosquitto processes from tests
        try:
            # Find mosquitto processes using test temp directories
            # If worker_id is specified, only clean up that worker's processes
            if worker_id:
                pattern = f'mqtt_test_{worker_id}_'
            else:
                # When cleaning up all workers, target each known worker's pattern
                pattern = 'mqtt_test_[^_]+_'  # Match any worker pattern
            
            result = subprocess.run(['pgrep', '-f', pattern], 
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
                                _safe_log('warning', f"Force killed stray mosquitto process {pid_num}")
                            except ProcessLookupError:
                                # Process already terminated
                                _safe_log('debug', f"Terminated stray mosquitto process {pid_num}")
                            stray_count += 1
                        except (ValueError, ProcessLookupError, OSError):
                            # Invalid PID or process already gone
                            pass
                if stray_count > 0:
                    _safe_log('info', f"Cleaned up {stray_count} stray mosquitto processes")
        except Exception as e:
            _safe_log('warning', f"Error cleaning up stray processes: {e}")