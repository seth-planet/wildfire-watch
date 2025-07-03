#!/usr/bin/env python3.12
"""Resilient MQTT Client for Safety-Critical Systems

This module provides a thread-safe, timeout-aware MQTT client wrapper that
prevents hanging connections and ensures graceful degradation when the broker
is unavailable. Designed for the Wildfire Watch system where reliability is
critical for safety.

Key Features:
    - Socket timeouts on initial connection
    - Exponential backoff with jitter
    - Maximum retry limits to prevent infinite loops
    - Thread-safe message queuing
    - Graceful degradation to FAILED state
    - Health monitoring integration
    - Automatic recovery attempts

Thread Model:
    - Dedicated MQTT thread handles all network I/O
    - Application threads use queues for message passing
    - No direct MQTT operations from application threads
"""

import paho.mqtt.client as mqtt
import queue
import threading
import time
import random
import socket
import logging
import json
from datetime import datetime, timezone
from typing import Optional, Tuple, Callable, Dict, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """MQTT connection states"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    FAILED = auto()
    SHUTDOWN = auto()

class ResilientMQTTClient:
    """Thread-safe MQTT client with robust error handling and timeout support.
    
    This class wraps the Paho MQTT client to provide:
    - Connection timeouts to prevent hanging
    - Retry limits to prevent infinite loops
    - Thread-safe message queuing
    - Graceful degradation when broker is unavailable
    - Automatic recovery from transient failures
    
    The client uses a dedicated thread for all MQTT operations, with
    application communication via thread-safe queues.
    
    Attributes:
        state (ConnectionState): Current connection state
        outgoing_queue (queue.Queue): Queue for messages to publish
        incoming_queue (queue.Queue): Queue for received messages
        health_callback (Callable): Optional callback for health updates
        
    Example:
        # Create queues for communication
        outgoing = queue.Queue(maxsize=1000)
        incoming = queue.Queue(maxsize=1000)
        
        # Create client
        client = ResilientMQTTClient(
            host="localhost",
            port=1883,
            client_id="detector-1",
            outgoing_queue=outgoing,
            incoming_queue=incoming
        )
        
        # Start the client
        client.start()
        
        # Send a message
        outgoing.put(("fire/detection", json.dumps({"confidence": 0.95})))
        
        # Process incoming messages
        while True:
            try:
                msg = incoming.get(timeout=1.0)
                process_message(msg)
            except queue.Empty:
                continue
    """
    
    def __init__(self,
                 host: str,
                 port: int,
                 client_id: str,
                 outgoing_queue: queue.Queue,
                 incoming_queue: queue.Queue,
                 keepalive: int = 60,
                 socket_timeout: int = 10,
                 max_retries: int = 5,
                 initial_retry_delay: int = 5,
                 max_retry_delay: int = 120,
                 tls_config: Optional[Dict[str, str]] = None,
                 lwt_config: Optional[Dict[str, Any]] = None,
                 health_callback: Optional[Callable[[ConnectionState], None]] = None):
        """Initialize the resilient MQTT client.
        
        Args:
            host: MQTT broker hostname
            port: MQTT broker port
            client_id: Unique client identifier
            outgoing_queue: Queue for messages to publish
            incoming_queue: Queue for received messages
            keepalive: MQTT keepalive interval in seconds
            socket_timeout: Socket timeout for connection attempts
            max_retries: Maximum connection retry attempts
            initial_retry_delay: Initial retry delay in seconds
            max_retry_delay: Maximum retry delay in seconds
            tls_config: Optional TLS configuration dict with 'ca_certs' key
            lwt_config: Optional LWT config with 'topic', 'payload', 'qos', 'retain'
            health_callback: Optional callback for state changes
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.outgoing_queue = outgoing_queue
        self.incoming_queue = incoming_queue
        self.keepalive = keepalive
        self.socket_timeout = socket_timeout
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.health_callback = health_callback
        
        # State management
        self._state_lock = threading.RLock()
        self._state = ConnectionState.DISCONNECTED
        self._shutdown = False
        self._subscriptions = []
        self._subscription_lock = threading.Lock()
        
        # Statistics for monitoring
        self._stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'last_connected': None,
            'last_disconnected': None,
            'current_retry_count': 0
        }
        
        # Create MQTT client
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
            clean_session=True
        )
        
        # Set callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        
        # Configure TLS if provided
        if tls_config and 'ca_certs' in tls_config:
            import ssl
            self._client.tls_set(
                ca_certs=tls_config['ca_certs'],
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
        
        # Set LWT if provided
        if lwt_config:
            self._client.will_set(
                lwt_config['topic'],
                lwt_config['payload'],
                qos=lwt_config.get('qos', 1),
                retain=lwt_config.get('retain', True)
            )
        
        # Worker thread
        self._thread = threading.Thread(
            target=self._run,
            name=f"MQTT-{client_id}",
            daemon=True
        )
    
    @property
    def state(self) -> ConnectionState:
        """Get current connection state atomically"""
        with self._state_lock:
            return self._state
    
    def _set_state(self, new_state: ConnectionState):
        """Set state and notify health callback"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            
            if old_state != new_state:
                logger.info(f"MQTT state transition: {old_state.name} -> {new_state.name}")
                
                # Update statistics
                if new_state == ConnectionState.CONNECTED:
                    self._stats['last_connected'] = datetime.now(timezone.utc)
                elif new_state == ConnectionState.DISCONNECTED:
                    self._stats['last_disconnected'] = datetime.now(timezone.utc)
                
                # Notify health callback
                if self.health_callback:
                    try:
                        self.health_callback(new_state)
                    except Exception as e:
                        logger.error(f"Health callback error: {e}")
    
    def start(self):
        """Start the MQTT client thread"""
        logger.info(f"Starting MQTT client: {self.client_id}")
        self._thread.start()
    
    def stop(self, timeout: float = 5.0):
        """Stop the MQTT client gracefully
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        logger.info(f"Stopping MQTT client: {self.client_id}")
        self._shutdown = True
        self._set_state(ConnectionState.SHUTDOWN)
        
        # Disconnect if connected
        if self._state == ConnectionState.CONNECTED:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        # Wait for thread to finish
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("MQTT thread did not stop within timeout")
    
    def subscribe(self, topic: str, qos: int = 1):
        """Subscribe to a topic
        
        Args:
            topic: MQTT topic to subscribe to
            qos: Quality of Service level
        """
        with self._subscription_lock:
            self._subscriptions.append((topic, qos))
            
            # If already connected, subscribe immediately
            if self._state == ConnectionState.CONNECTED:
                try:
                    self._client.subscribe(topic, qos)
                    logger.info(f"Subscribed to {topic}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {topic}: {e}")
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("MQTT connected successfully")
            self._set_state(ConnectionState.CONNECTED)
            self._stats['successful_connections'] += 1
            self._stats['current_retry_count'] = 0
            
            # Resubscribe to all topics
            with self._subscription_lock:
                for topic, qos in self._subscriptions:
                    try:
                        client.subscribe(topic, qos)
                        logger.info(f"Subscribed to {topic}")
                    except Exception as e:
                        logger.error(f"Failed to subscribe to {topic}: {e}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self._stats['failed_connections'] += 1
    
    def _on_disconnect(self, client, userdata, rc, properties=None, reasoncode=None):
        """MQTT disconnection callback"""
        if self._state != ConnectionState.SHUTDOWN:
            logger.warning(f"MQTT disconnected with code {rc}")
            self._set_state(ConnectionState.DISCONNECTED)
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback - minimal work, just queue"""
        try:
            self.incoming_queue.put_nowait(msg)
            self._stats['messages_received'] += 1
        except queue.Full:
            logger.warning(f"Incoming queue full, dropping message on {msg.topic}")
            self._stats['messages_dropped'] += 1
    
    def _run(self):
        """Main MQTT client thread loop"""
        logger.info(f"MQTT thread started for {self.client_id}")
        
        # Set socket timeout for initial connections
        socket.setdefaulttimeout(self.socket_timeout)
        
        recovery_wait_time = 300  # 5 minutes between recovery attempts in FAILED state
        last_recovery_attempt = 0
        
        while not self._shutdown:
            try:
                current_state = self.state
                
                # Handle connection states
                if current_state == ConnectionState.DISCONNECTED:
                    self._attempt_connection()
                    
                elif current_state == ConnectionState.FAILED:
                    # In FAILED state, wait longer between attempts
                    current_time = time.time()
                    if current_time - last_recovery_attempt > recovery_wait_time:
                        logger.info("Attempting recovery from FAILED state")
                        last_recovery_attempt = current_time
                        self._set_state(ConnectionState.DISCONNECTED)
                        self._stats['current_retry_count'] = 0
                
                # Process outgoing messages (only when connected)
                try:
                    topic, payload = self.outgoing_queue.get(timeout=0.1)
                    if current_state == ConnectionState.CONNECTED:
                        try:
                            self._client.publish(topic, payload)
                            self._stats['messages_sent'] += 1
                            logger.debug(f"Published to {topic}")
                        except Exception as e:
                            logger.error(f"Failed to publish to {topic}: {e}")
                            # Put message back on queue if publish fails
                            try:
                                self.outgoing_queue.put_nowait((topic, payload))
                            except queue.Full:
                                logger.error("Outgoing queue full, message lost")
                                self._stats['messages_dropped'] += 1
                    else:
                        # Not connected, put message back
                        try:
                            self.outgoing_queue.put_nowait((topic, payload))
                        except queue.Full:
                            logger.warning("Outgoing queue full while disconnected")
                            self._stats['messages_dropped'] += 1
                except queue.Empty:
                    pass  # Normal, no messages to send
                
                # Maintain MQTT network loop
                if current_state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
                    self._client.loop(timeout=0.1)
                else:
                    # Not connected, just sleep
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Unexpected error in MQTT thread: {e}", exc_info=True)
                time.sleep(1)
        
        # Restore default socket timeout
        socket.setdefaulttimeout(None)
        logger.info(f"MQTT thread stopped for {self.client_id}")
    
    def _attempt_connection(self):
        """Attempt to connect with exponential backoff and jitter"""
        self._set_state(ConnectionState.CONNECTING)
        self._stats['connection_attempts'] += 1
        
        retry_count = self._stats['current_retry_count']
        
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) reached. Entering FAILED state.")
            self._set_state(ConnectionState.FAILED)
            return
        
        try:
            logger.info(f"Attempting MQTT connection (attempt {retry_count + 1}/{self.max_retries})")
            
            # Set socket timeout for this connection attempt
            self._client._sock = None  # Reset socket
            
            # Connect with timeout
            self._client.connect(self.host, self.port, keepalive=self.keepalive)
            
            # Give time for on_connect callback
            for _ in range(30):  # Wait up to 3 seconds
                if self.state == ConnectionState.CONNECTED:
                    return
                self._client.loop(timeout=0.1)
                
            # If still not connected, it failed
            raise ConnectionError("Connection callback not received within timeout")
            
        except Exception as e:
            logger.error(f"Connection attempt failed: {e}")
            self._stats['current_retry_count'] += 1
            self._set_state(ConnectionState.DISCONNECTED)
            
            if self._stats['current_retry_count'] < self.max_retries:
                # Calculate backoff with jitter
                delay = min(
                    self.initial_retry_delay * (2 ** retry_count),
                    self.max_retry_delay
                )
                jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
                total_delay = delay + jitter
                
                logger.info(f"Retrying in {total_delay:.1f}s...")
                time.sleep(total_delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring"""
        with self._state_lock:
            stats = self._stats.copy()
            stats['current_state'] = self._state.name
            stats['queue_sizes'] = {
                'outgoing': self.outgoing_queue.qsize(),
                'incoming': self.incoming_queue.qsize()
            }
            return stats
    
    def is_healthy(self) -> bool:
        """Check if client is in a healthy state"""
        return self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]