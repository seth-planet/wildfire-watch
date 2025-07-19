#!/usr/bin/env python3.12
"""Base MQTT service class for Wildfire Watch services.

This module provides a base class for all MQTT-connected services, handling:
- Connection management with exponential backoff
- Thread-safe message publishing
- Last Will Testament (LWT) setup
- Topic prefix support for test isolation
- Automatic reconnection handling
"""

import os
import json
import time
import logging
import threading
from typing import Optional, Callable, Dict, Any, List
import paho.mqtt.client as mqtt

from .config_base import SharedMQTTConfig
from .safe_logging import SafeLoggingMixin, register_logger_for_cleanup


class MQTTService(SafeLoggingMixin):
    """Base class for all MQTT-connected services in Wildfire Watch.
    
    Provides standardized MQTT functionality including connection management,
    reconnection with exponential backoff, thread-safe publishing, and 
    Last Will Testament support.
    
    Attributes:
        service_name: Name of the service for identification
        config: Service configuration dictionary
        logger: Logger instance for this service
    """
    
    def __init__(self, service_name: str, config: Any):
        """Initialize MQTT service base.
        
        Args:
            service_name: Unique name for this service
            config: Configuration object with MQTT settings
        """
        self.service_name = service_name
        self.config = config
        self.logger = logging.getLogger(service_name)
        
        # Register logger for cleanup
        register_logger_for_cleanup(self.logger)
        
        # MQTT state
        self._mqtt_client: Optional[mqtt.Client] = None
        self._mqtt_connected = False
        self._mqtt_lock = threading.Lock()
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._shutdown = False
        
        # Topic prefix for test isolation
        topic_prefix = getattr(config, 'topic_prefix', config.get('TOPIC_PREFIX', '') if hasattr(config, 'get') else '')
        
        # Validate prefix to prevent invalid characters
        if topic_prefix:
            invalid_chars = ['#', '+', '\n', '\r', '\0']
            if any(char in topic_prefix for char in invalid_chars):
                raise ValueError(f"Invalid characters in TOPIC_PREFIX: {topic_prefix}")
        
        self._topic_prefix = topic_prefix
        if self._topic_prefix and not self._topic_prefix.endswith('/'):
            self._topic_prefix += '/'
        
        # Log the effective prefix for debugging
        if self._topic_prefix:
            self._safe_log('info', f"MQTT topic prefix configured: '{self._topic_prefix}'")
        
        # Callbacks
        self._on_connect_callback: Optional[Callable] = None
        self._on_message_callback: Optional[Callable] = None
        self._subscriptions: List[str] = []
        
        # Offline message queue (optional)
        self._offline_queue_enabled = False
        self._offline_queue: List[tuple] = []
        self._max_offline_queue_size = 100
        
    def setup_mqtt(self, 
                   on_connect: Optional[Callable] = None,
                   on_message: Optional[Callable] = None,
                   subscriptions: Optional[List[str]] = None) -> None:
        """Setup MQTT client with callbacks and subscriptions.
        
        Args:
            on_connect: Optional callback for connection events
            on_message: Optional callback for received messages
            subscriptions: List of topics to subscribe to
        """
        self._on_connect_callback = on_connect
        self._on_message_callback = on_message
        self._subscriptions = subscriptions or []
        
        # Create client with unique ID
        # Check if MQTT_CLIENT_ID is set in environment (for testing)
        env_client_id = os.environ.get('MQTT_CLIENT_ID')
        if env_client_id:
            client_id = env_client_id
            self._safe_log('debug', f"Using MQTT_CLIENT_ID from environment: {client_id}")
        else:
            client_id = f"{self.service_name}_{os.getpid()}"
        
        self._mqtt_client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=True,
            protocol=mqtt.MQTTv311
        )
        
        # Set callbacks
        self._mqtt_client.on_connect = self._on_mqtt_connect
        self._mqtt_client.on_disconnect = self._on_mqtt_disconnect
        if on_message:
            self._mqtt_client.on_message = self._on_mqtt_message
        
        # Configure authentication if provided
        mqtt_username = getattr(self.config, 'mqtt_username', self.config.get('MQTT_USERNAME', '') if hasattr(self.config, 'get') else '')
        if mqtt_username:
            mqtt_password = getattr(self.config, 'mqtt_password', self.config.get('MQTT_PASSWORD', '') if hasattr(self.config, 'get') else '')
            self._mqtt_client.username_pw_set(
                mqtt_username,
                mqtt_password
            )
        
        # Configure TLS if enabled
        mqtt_tls = getattr(self.config, 'mqtt_tls', self.config.get('MQTT_TLS', False) if hasattr(self.config, 'get') else False)
        if mqtt_tls:
            self._setup_tls()
        
        # Set Last Will Testament
        lwt_topic = self._format_topic(f"system/{self.service_name}/lwt")
        self._mqtt_client.will_set(
            lwt_topic,
            payload="offline",
            qos=1,
            retain=True
        )
        
        # Don't connect yet - let subclass decide when to connect
        # This prevents race conditions during initialization
        # self._connect_with_retry()
        
    def _setup_tls(self) -> None:
        """Configure TLS for MQTT connection."""
        ca_cert = getattr(self.config, 'tls_ca_path', self.config.get('MQTT_CA_CERT', 'certs/ca.crt') if hasattr(self.config, 'get') else 'certs/ca.crt')
        client_cert = getattr(self.config, 'tls_cert_path', self.config.get('MQTT_CLIENT_CERT', None) if hasattr(self.config, 'get') else None)
        client_key = getattr(self.config, 'tls_key_path', self.config.get('MQTT_CLIENT_KEY', None) if hasattr(self.config, 'get') else None)
        
        # Configure TLS
        # Only use client certs if both are provided and not empty
        if client_cert and client_key:
            self._mqtt_client.tls_set(
                ca_certs=ca_cert,
                certfile=client_cert,
                keyfile=client_key,
                cert_reqs=mqtt.ssl.CERT_REQUIRED,
                tls_version=mqtt.ssl.PROTOCOL_TLSv1_2
            )
        else:
            # CA cert only (no client authentication)
            self._mqtt_client.tls_set(
                ca_certs=ca_cert,
                certfile=None,
                keyfile=None,
                cert_reqs=mqtt.ssl.CERT_REQUIRED,
                tls_version=mqtt.ssl.PROTOCOL_TLSv1_2
            )
        
        # Optionally disable hostname verification for self-signed certs
        tls_insecure = getattr(self.config, 'tls_insecure', self.config.get('MQTT_TLS_INSECURE', False) if hasattr(self.config, 'get') else False)
        if tls_insecure:
            self._mqtt_client.tls_insecure_set(True)
    
    def connect(self) -> None:
        """Connect to the MQTT broker and start the network loop."""
        if self._mqtt_client and not self._shutdown:
            mqtt_broker = getattr(self.config, 'mqtt_broker', self.config.get('MQTT_BROKER', 'localhost') if hasattr(self.config, 'get') else 'localhost')
            mqtt_port = getattr(self.config, 'mqtt_port', self.config.get('MQTT_PORT', 1883) if hasattr(self.config, 'get') else 1883)
            self._safe_log('info', f"Starting MQTT connection process for {self.service_name}...")
            self._safe_log('info', f"Target broker: {mqtt_broker}:{mqtt_port}")
            
            # Resolve host.docker.internal if needed
            if mqtt_broker == 'host.docker.internal':
                self._safe_log('info', "Detected host.docker.internal, checking resolution...")
                try:
                    import socket
                    resolved_ip = socket.gethostbyname(mqtt_broker)
                    self._safe_log('info', f"host.docker.internal resolved to: {resolved_ip}")
                except Exception as e:
                    self._safe_log('error', f"Failed to resolve host.docker.internal: {e}")
            
            # Start connection in background thread to avoid blocking
            threading.Thread(target=self._connect_with_retry, daemon=True).start()
    
    def _connect_with_retry(self) -> None:
        """Connect to MQTT broker with exponential backoff."""
        self._reconnect_delay = 1.0  # Reset delay
        
        while not self._shutdown:
            try:
                mqtt_broker = getattr(self.config, 'mqtt_broker', self.config.get('MQTT_BROKER', 'localhost') if hasattr(self.config, 'get') else 'localhost')
                mqtt_port = getattr(self.config, 'mqtt_port', self.config.get('MQTT_PORT', 1883) if hasattr(self.config, 'get') else 1883)
                self._safe_log('info', f"Connecting to MQTT broker at {mqtt_broker}:{mqtt_port}")
                self._mqtt_client.connect(
                    mqtt_broker,
                    mqtt_port,
                    60
                )
                self._mqtt_client.loop_start()
                break
            except Exception as e:
                self._safe_log('error', f"MQTT connection failed: {e}")
                self._safe_log('error', f"Exception type: {type(e).__name__}")
                self._safe_log('error', f"Broker: {mqtt_broker}, Port: {mqtt_port}")
                
                # Check if it's a network connectivity issue
                import errno
                if hasattr(e, 'errno'):
                    if e.errno == errno.ECONNREFUSED:
                        self._safe_log('error', "Connection refused - is MQTT broker running?")
                        self._safe_log('error', f"Attempted connection to {mqtt_broker}:{mqtt_port}")
                        # Try to ping the host
                        try:
                            import socket
                            test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_sock.settimeout(2)
                            result = test_sock.connect_ex((mqtt_broker, mqtt_port))
                            test_sock.close()
                            if result == 0:
                                self._safe_log('error', "Port is open but MQTT connection refused")
                            else:
                                self._safe_log('error', f"Port {mqtt_port} is not reachable (error code: {result})")
                        except Exception as test_e:
                            self._safe_log('error', f"Network test failed: {test_e}")
                    elif e.errno == errno.EHOSTUNREACH:
                        self._safe_log('error', "Host unreachable - check Docker networking")
                    elif e.errno == errno.ETIMEDOUT:
                        self._safe_log('error', "Connection timed out")
                
                self._safe_log('info', f"Retrying in {self._reconnect_delay}s...")
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )
    
    def _on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        """Handle MQTT connection events."""
        print(f"[MQTT BASE] _on_mqtt_connect called with rc={rc}", flush=True)
        if rc == 0:
            self._safe_log('info', "Connected to MQTT broker")
            print(f"[MQTT BASE] Setting _mqtt_connected = True", flush=True)
            with self._mqtt_lock:
                self._mqtt_connected = True
                self._reconnect_delay = 1.0  # Reset delay on success
            
            # Publish online status
            lwt_topic = self._format_topic(f"system/{self.service_name}/lwt")
            client.publish(lwt_topic, "online", qos=1, retain=True)
            
            # Subscribe to topics
            for topic in self._subscriptions:
                formatted_topic = self._format_topic(topic)
                client.subscribe(formatted_topic)
                self._safe_log('debug', f"Subscribed to {formatted_topic}")
            
            # Process offline queue if enabled
            if self._offline_queue_enabled:
                self._process_offline_queue()
            
            # Call user callback
            self._safe_log('debug', f"About to call user callback: {self._on_connect_callback}")
            if self._on_connect_callback:
                # Pass properties if the callback accepts it
                try:
                    # Try calling with properties parameter
                    self._on_connect_callback(client, userdata, flags, rc, properties)
                    self._safe_log('debug', "User callback called successfully with properties")
                except TypeError as e:
                    self._safe_log('debug', f"TypeError calling with properties: {e}, trying without")
                    # Fall back to 4 parameters if callback doesn't accept properties
                    self._on_connect_callback(client, userdata, flags, rc)
                    self._safe_log('debug', "User callback called successfully without properties")
                except Exception as e:
                    self._safe_log('error', f"Error calling user on_connect callback: {e}")
        else:
            self._safe_log('error', f"Failed to connect to MQTT broker: {mqtt.error_string(rc)}")
            with self._mqtt_lock:
                self._mqtt_connected = False
    

    def _on_mqtt_disconnect(self, client, userdata, rc, properties=None, reasoncode=None):
        """Handle MQTT disconnection events."""
        with self._mqtt_lock:
            self._mqtt_connected = False
        
        if rc != 0:
            self._safe_log('warning', f"Unexpected MQTT disconnection: {mqtt.error_string(rc)}")
            if not self._shutdown:
                self._safe_log('info', "Attempting to reconnect...")
                threading.Thread(target=self._connect_with_retry, daemon=True).start()
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle received MQTT messages."""
        try:
            # Remove topic prefix if present
            topic = msg.topic
            if self._topic_prefix and topic.startswith(self._topic_prefix):
                topic = topic[len(self._topic_prefix):]
            
            # Parse JSON payload if possible
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = msg.payload
            
            # Call user callback with cleaned topic
            if self._on_message_callback:
                self._on_message_callback(topic, payload)
                
        except Exception as e:
            self._safe_log('error', f"Error processing message on {msg.topic}: {e}")
    
    def _format_topic(self, topic: str) -> str:
        """Format topic with prefix if configured.
        
        Args:
            topic: Base topic name
            
        Returns:
            Formatted topic with prefix
        """
        if self._topic_prefix:
            # Ensure proper separator between prefix and topic
            if not self._topic_prefix.endswith('/'):
                return f"{self._topic_prefix}/{topic}"
            return f"{self._topic_prefix}{topic}"
        return topic
    
    def publish_message(self, topic: str, payload: Any, 
                       retain: bool = False, qos: int = 0,
                       queue_if_offline: bool = False) -> bool:
        """Thread-safe message publishing.
        
        Args:
            topic: Topic to publish to (will be prefixed if configured)
            payload: Message payload (will be JSON encoded if dict)
            retain: Whether to retain the message
            qos: Quality of Service level (0, 1, or 2)
            queue_if_offline: Queue message if not connected
            
        Returns:
            True if published successfully, False otherwise
        """
        with self._mqtt_lock:
            if not self._mqtt_connected:
                if queue_if_offline and self._offline_queue_enabled:
                    if len(self._offline_queue) < self._max_offline_queue_size:
                        self._offline_queue.append((topic, payload, retain, qos))
                        self._safe_log('debug', f"Queued message for {topic} (queue size: {len(self._offline_queue)})")
                        return True
                    else:
                        self._safe_log('warning', f"Offline queue full, dropping message for {topic}")
                else:
                    self._safe_log('warning', f"Cannot publish to {topic} - not connected")
                return False
            
            full_topic = self._format_topic(topic)
            
            # Encode payload
            if isinstance(payload, dict):
                encoded_payload = json.dumps(payload)
            elif isinstance(payload, str):
                encoded_payload = payload
            else:
                encoded_payload = str(payload)
            
            try:
                result = self._mqtt_client.publish(
                    full_topic,
                    encoded_payload,
                    qos=qos,
                    retain=retain
                )
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self._safe_log('debug', f"Published to {full_topic}")
                    self._safe_log('info', f"[MQTT DEBUG] Successfully published to full topic: '{full_topic}' (prefix: '{self._topic_prefix}')")
                    return True
                else:
                    self._safe_log('error', f"Failed to publish to {full_topic}: {mqtt.error_string(result.rc)}")
                    return False
                    
            except Exception as e:
                self._safe_log('error', f"Exception publishing to {full_topic}: {e}")
                return False
    
    def enable_offline_queue(self, max_size: int = 100) -> None:
        """Enable offline message queuing.
        
        Args:
            max_size: Maximum number of messages to queue
        """
        self._offline_queue_enabled = True
        self._max_offline_queue_size = max_size
        self._safe_log('info', f"Offline message queuing enabled (max size: {max_size})")
    
    def _process_offline_queue(self) -> None:
        """Process queued messages after reconnection."""
        if not self._offline_queue:
            return
        
        self._safe_log('info', f"Processing {len(self._offline_queue)} queued messages")
        
        # Copy and clear queue
        with self._mqtt_lock:
            messages = self._offline_queue.copy()
            self._offline_queue.clear()
        
        # Publish queued messages
        for topic, payload, retain, qos in messages:
            self.publish_message(topic, payload, retain, qos)
    
    def shutdown(self) -> None:
        """Gracefully shutdown MQTT connection."""
        self._safe_log('info', f"Shutting down {self.service_name} MQTT service")
        self._shutdown = True
        
        if self._mqtt_client:
            try:
                # Publish offline status
                lwt_topic = self._format_topic(f"system/{self.service_name}/lwt")
                self._mqtt_client.publish(lwt_topic, "offline", qos=1, retain=True)
                
                # Give a brief moment for any pending callbacks to complete
                import time
                time.sleep(0.1)
                
                # Stop loop and disconnect
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
                
                # Give another brief moment for disconnection to complete
                time.sleep(0.1)
                
            except Exception as e:
                self._safe_log('error', f"Error during MQTT shutdown: {e}")
        
        self._mqtt_connected = False
        self._safe_log('info', f"{self.service_name} MQTT service shutdown complete")
    
    @property
    def is_connected(self) -> bool:
        """Check if MQTT is connected."""
        return self._mqtt_connected
    
    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for MQTT connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._mqtt_connected:
                return True
            time.sleep(0.1)
        return False