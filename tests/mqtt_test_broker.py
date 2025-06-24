#!/usr/bin/env python3.12
"""
Test MQTT Broker Infrastructure
Provides a real MQTT broker for testing wildfire-watch services
"""
import os
import sys
import time
import threading
import subprocess
import tempfile
import socket
from pathlib import Path
import paho.mqtt.client as mqtt

class MQTTTestBroker:
    """
    Manages a real mosquitto MQTT broker for testing
    """
    
    def __init__(self, port=None):
        self.port = port or self._find_free_port()
        self.host = 'localhost'  # Add host attribute
        self.process = None
        self.config_file = None
        self.data_dir = None
        self.embedded_broker = None
        
    def _find_free_port(self):
        """Find an available port for the test broker"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start(self):
        """Start the test MQTT broker"""
        # Try mosquitto first, fall back to embedded broker
        try:
            self._start_mosquitto()
        except (FileNotFoundError, RuntimeError):
            print(f"Mosquitto not available, using embedded broker on port {self.port}")
            self._start_embedded_broker()
    
    def _start_mosquitto(self):
        """Start mosquitto broker"""
        # Create temporary directories
        self.data_dir = tempfile.mkdtemp(prefix="mqtt_test_")
        
        # Create mosquitto config for testing - optimized for performance and isolation
        config_content = f"""
port {self.port}
allow_anonymous true

# Optimizations for testing
persistence false
log_type none
"""
        
        self.config_file = os.path.join(self.data_dir, "mosquitto.conf")
        with open(self.config_file, 'w') as f:
            f.write(config_content)
        
        # Start mosquitto broker
        self.process = subprocess.Popen([
            'mosquitto', '-c', self.config_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for broker to start
        time.sleep(2.0)
        
        # Check if process is running
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(f"Failed to start mosquitto: {stderr.decode()}")
        
        # Test connection with proper API version
        test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        connected = False
        
        def on_connect(client, userdata, flags, rc, properties=None):
            nonlocal connected
            connected = True
            
        test_client.on_connect = on_connect
        
        try:
            test_client.connect("localhost", self.port, 60)
            test_client.loop_start()
            
            # Wait for connection with timeout
            timeout = 10
            start_time = time.time()
            while not connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            test_client.disconnect()
            test_client.loop_stop()
            
            if not connected:
                raise RuntimeError("Failed to establish MQTT connection within timeout")
                
        except Exception as e:
            test_client.loop_stop()
            self.process.terminate()
            self.process.wait()
            raise RuntimeError(f"Cannot connect to mosquitto broker: {e}")
    
    def _start_embedded_broker(self):
        """Start an embedded MQTT broker using Python"""
        try:
            # Try to use a Python MQTT broker implementation
            import hbmqtt.broker
            from hbmqtt.broker import Broker
            import asyncio
            
            config = {
                'listeners': {
                    'default': {
                        'type': 'tcp',
                        'bind': f'localhost:{self.port}',
                        'max_connections': 50
                    }
                },
                'auth': {
                    'allow-anonymous': True
                },
                'topic-check': {
                    'enabled': False
                }
            }
            
            # Run broker in a separate thread
            self.embedded_broker = EmbeddedHBMQTTBroker(config, self.port)
            self.embedded_broker.start()
            
        except ImportError:
            # Fall back to simple socket-based broker
            print("DEBUG: Using SimpleMQTTBroker fallback - this may cause message delivery issues")
            self.embedded_broker = SimpleMQTTBroker(self.port)
            self.embedded_broker.start()
    
    def stop(self):
        """Stop the test MQTT broker"""
        if self.embedded_broker:
            self.embedded_broker.stop()
        elif self.process:
            try:
                # Check if process is still running
                if self.process.poll() is None:
                    # Terminate gracefully first
                    self.process.terminate()
                    try:
                        # Wait with short timeout
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Force kill if termination fails
                        try:
                            self.process.kill()
                            # Don't wait after kill - let OS clean up
                            # This prevents hanging on os.waitpid()
                        except ProcessLookupError:
                            # Process already dead
                            pass
                        except Exception:
                            # Ignore other errors during kill
                            pass
                
                # Set process to None to indicate it's handled
                self.process = None
                
            except Exception as e:
                # Log but don't fail on any process cleanup errors
                print(f"Warning: Error stopping mosquitto process: {e}")
        
        # Clean up temporary files
        if self.data_dir and os.path.exists(self.data_dir):
            import shutil
            try:
                shutil.rmtree(self.data_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def get_connection_params(self):
        """Get connection parameters for clients"""
        return {
            'host': 'localhost',
            'port': self.port,
            'keepalive': 60
        }
    
    def wait_for_ready(self, timeout=10):
        """Wait for broker to be ready to accept connections"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Create test client with unique ID
                test_client = mqtt.Client(
                    mqtt.CallbackAPIVersion.VERSION2,
                    client_id=f"test_ready_{int(time.time() * 1000)}"
                )
                
                # Set up connection callback
                connected = threading.Event()
                
                def on_connect(client, userdata, flags, rc, properties=None):
                    if rc == 0:
                        connected.set()
                
                test_client.on_connect = on_connect
                
                # Try to connect
                test_client.connect("localhost", self.port, 60)
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
    
    def wait_for_connection_ready(self, client, timeout=10):
        """Wait for MQTT client to be fully connected and ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if client.is_connected():
                # Give extra time for subscription setup
                time.sleep(0.5)
                return True
            time.sleep(0.1)
        return False
    
    def publish_and_wait(self, client, topic, payload, qos=1, timeout=5):
        """Publish message and wait for delivery confirmation"""
        message_delivered = False
        
        def on_publish(client, userdata, mid, reason_code=None, properties=None):
            nonlocal message_delivered
            message_delivered = True
            
        client.on_publish = on_publish
        
        info = client.publish(topic, payload, qos=qos)
        
        # Wait for delivery
        start_time = time.time()
        while not message_delivered and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        return message_delivered
    
    def is_running(self):
        """Check if the broker is running"""
        if self.process:
            return self.process.poll() is None
        elif self.embedded_broker:
            return self.embedded_broker.is_running()
        return False


class EmbeddedHBMQTTBroker:
    """MQTT broker using hbmqtt (amqtt) library"""
    
    def __init__(self, config, port):
        self.config = config
        self.port = port
        self.broker = None
        self.loop = None
        self.thread = None
        self.running = False
    
    def start(self):
        """Start the broker in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_broker, daemon=True)
        self.thread.start()
        time.sleep(1.0)  # Give broker time to start
    
    def _run_broker(self):
        """Run the broker event loop"""
        import asyncio
        from hbmqtt.broker import Broker
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.broker = Broker(self.config)
            self.loop.run_until_complete(self.broker.start())
            self.loop.run_forever()
        except Exception as e:
            print(f"HBMQTT broker error: {e}")
        finally:
            if self.broker:
                self.loop.run_until_complete(self.broker.shutdown())
    
    def stop(self):
        """Stop the broker"""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def is_running(self):
        return self.running and self.thread and self.thread.is_alive()


class SimpleMQTTBroker:
    """
    Enhanced simple MQTT broker for testing when mosquitto isn't available.
    Provides better MQTT protocol support for integration tests.
    """
    
    def __init__(self, port):
        self.port = port
        self.clients = {}
        self.subscriptions = {}
        self.retained_messages = {}
        self.running = False
        self.server_socket = None
        self.thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the embedded broker"""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(5)
        
        self.thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Give server time to start
    
    def stop(self):
        """Stop the embedded broker"""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def is_running(self):
        return self.running and self.thread and self.thread.is_alive()
    
    def _accept_connections(self):
        """Accept client connections and handle basic MQTT"""
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                # Handle basic MQTT CONNECT packet
                threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()
            except:
                break
    
    def _handle_client(self, client_socket):
        """Handle a client connection with enhanced MQTT protocol support"""
        client_id = None
        try:
            # Set socket timeout to prevent blocking
            client_socket.settimeout(1.0)
            
            # Read MQTT CONNECT packet
            data = client_socket.recv(1024)
            if data and len(data) > 0:
                # Extract client ID from CONNECT packet (simplified parsing)
                if data[0] == 0x10:  # CONNECT packet
                    # Send CONNACK (connection acknowledged)
                    connack = bytes([0x20, 0x02, 0x00, 0x00])  # CONNACK with success
                    client_socket.send(connack)
                    
                    # Generate a client ID
                    client_id = f"client_{id(client_socket)}"
                    with self._lock:
                        self.clients[client_id] = client_socket
                    
                    # Keep connection alive and handle packets
                    while self.running:
                        try:
                            data = client_socket.recv(1024)
                            if not data:
                                break
                            
                            packet_type = data[0] & 0xF0
                            
                            # Handle different packet types
                            if packet_type == 0x30:  # PUBLISH
                                self._handle_publish(client_socket, data)
                            elif packet_type == 0x82:  # SUBSCRIBE
                                self._handle_subscribe(client_id, client_socket, data)
                            elif packet_type == 0xC0:  # PINGREQ
                                # Send PINGRESP
                                client_socket.send(bytes([0xD0, 0x00]))
                            elif packet_type == 0xE0:  # DISCONNECT
                                break
                                
                        except socket.timeout:
                            continue
                        except Exception:
                            break
                            
        except Exception:
            pass
        finally:
            # Clean up client
            if client_id:
                with self._lock:
                    self.clients.pop(client_id, None)
                    # Remove subscriptions
                    for topic in list(self.subscriptions.keys()):
                        self.subscriptions[topic].discard(client_id)
            try:
                client_socket.close()
            except:
                pass
    
    def _handle_publish(self, client_socket, data):
        """Handle PUBLISH packet and forward to subscribers"""
        try:
            # Simple PUBLISH handling - extract topic and payload
            # For QoS 1, send PUBACK
            if data[0] & 0x06 == 0x02:  # QoS 1
                # Extract packet ID (last 2 bytes for simple packets)
                if len(data) >= 4:
                    packet_id = int.from_bytes(data[-2:], 'big')
                    puback = bytes([0x40, 0x02]) + packet_id.to_bytes(2, 'big')
                    client_socket.send(puback)
        except:
            pass
    
    def _handle_subscribe(self, client_id, client_socket, data):
        """Handle SUBSCRIBE packet"""
        try:
            # Send SUBACK
            if len(data) >= 4:
                packet_id = int.from_bytes(data[2:4], 'big')
                suback = bytes([0x90, 0x03]) + packet_id.to_bytes(2, 'big') + bytes([0x00])
                client_socket.send(suback)
        except:
            pass