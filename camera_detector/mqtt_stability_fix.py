#!/usr/bin/env python3.12
"""MQTT Stability Fix for Camera Detector

This module provides a stable MQTT implementation that prevents disconnections
caused by blocking network operations and improves thread safety.
"""

import threading
import queue
import time
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

@dataclass
class MQTTMessage:
    """Message to be published via MQTT"""
    topic: str
    payload: str
    qos: int = 0
    retain: bool = False
    
class StableMQTTHandler:
    """Thread-safe MQTT handler with dedicated publishing thread
    
    This handler solves stability issues by:
    1. Running MQTT client loop in a dedicated thread
    2. Using a queue for all outgoing messages
    3. Implementing proper reconnection logic
    4. Using clean_session=True to prevent message buildup
    5. Providing non-blocking publish operations
    """
    
    def __init__(self, 
                 broker: str,
                 port: int = 1883,
                 client_id: str = None,
                 keepalive: int = 30,  # Reduced from 60 for faster detection
                 tls_enabled: bool = False,
                 ca_cert_path: str = None):
        """Initialize MQTT handler
        
        Args:
            broker: MQTT broker hostname
            port: MQTT broker port
            client_id: Unique client ID
            keepalive: Keepalive interval in seconds (default: 30)
            tls_enabled: Whether to use TLS
            ca_cert_path: Path to CA certificate for TLS
        """
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.tls_enabled = tls_enabled
        self.ca_cert_path = ca_cert_path
        
        # State management
        self.connected = threading.Event()
        self.running = True
        self._lock = threading.RLock()
        
        # Message queue for outgoing messages
        self.message_queue = queue.Queue(maxsize=1000)
        
        # Callbacks
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        
        # MQTT client - use clean_session=True for stability
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=True  # Important: prevents message buildup
        )
        
        # Set callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        # Configure TLS if enabled
        if tls_enabled and ca_cert_path:
            import ssl
            self.client.tls_set(
                ca_certs=ca_cert_path,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
        
        # Start threads
        self._mqtt_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
        self._publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
        
    def start(self):
        """Start MQTT handler threads"""
        logger.info(f"Starting MQTT handler for {self.broker}:{self.port}")
        self._mqtt_thread.start()
        self._publisher_thread.start()
        
    def stop(self):
        """Stop MQTT handler"""
        logger.info("Stopping MQTT handler")
        self.running = False
        
        # Clear the queue
        try:
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
        except:
            pass
            
        # Disconnect client
        try:
            self.client.disconnect()
        except:
            pass
            
        # Wait for threads
        self._mqtt_thread.join(timeout=2.0)
        self._publisher_thread.join(timeout=2.0)
        
    def set_will(self, topic: str, payload: str, qos: int = 1, retain: bool = True):
        """Set Last Will and Testament"""
        self.client.will_set(topic, payload, qos, retain)
        
    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> bool:
        """Non-blocking publish to MQTT
        
        Args:
            topic: MQTT topic
            payload: Message payload
            qos: Quality of Service level
            retain: Whether to retain message
            
        Returns:
            True if queued successfully, False if queue full
        """
        try:
            message = MQTTMessage(topic, payload, qos, retain)
            self.message_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning(f"MQTT message queue full, dropping message for topic: {topic}")
            return False
            
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Wait for MQTT connection
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout
        """
        return self.connected.wait(timeout)
        
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self.connected.is_set()
        
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Handle MQTT connection"""
        if rc == 0:
            logger.info("MQTT connected successfully")
            self.connected.set()
            if self.on_connect_callback:
                try:
                    self.on_connect_callback()
                except Exception as e:
                    logger.error(f"Error in connect callback: {e}")
        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self.connected.clear()
            
    def _on_disconnect(self, client, userdata, rc, properties=None, reasoncode=None):
        """Handle MQTT disconnection"""
        was_connected = self.connected.is_set()
        self.connected.clear()
        
        if rc == 0:
            logger.info("MQTT disconnected cleanly")
        else:
            logger.warning(f"MQTT unexpected disconnect with code {rc}")
            
        if was_connected and self.on_disconnect_callback:
            try:
                self.on_disconnect_callback()
            except Exception as e:
                logger.error(f"Error in disconnect callback: {e}")
                
    def _mqtt_loop(self):
        """Dedicated thread for MQTT client loop"""
        reconnect_delay = 1.0
        max_reconnect_delay = 60.0
        
        while self.running:
            try:
                # Connect if not connected
                if not self.client.is_connected():
                    logger.info(f"Attempting MQTT connection to {self.broker}:{self.port}")
                    self.client.connect(self.broker, self.port, self.keepalive)
                    
                # Run the loop
                self.client.loop_forever(retry_first_connection=False)
                
            except Exception as e:
                logger.error(f"MQTT loop error: {e}")
                self.connected.clear()
                
                # Exponential backoff for reconnection
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                
    def _publisher_loop(self):
        """Dedicated thread for publishing messages"""
        while self.running:
            try:
                # Wait for message with timeout
                message = self.message_queue.get(timeout=1.0)
                
                # Only publish if connected
                if self.connected.is_set():
                    try:
                        info = self.client.publish(
                            message.topic,
                            message.payload,
                            qos=message.qos,
                            retain=message.retain
                        )
                        
                        # For QoS > 0, wait for publish to complete
                        if message.qos > 0:
                            info.wait_for_publish(timeout=5.0)
                            
                    except Exception as e:
                        logger.error(f"Failed to publish to {message.topic}: {e}")
                        # Re-queue message if it's important (QoS > 0)
                        if message.qos > 0:
                            try:
                                self.message_queue.put_nowait(message)
                            except queue.Full:
                                pass
                else:
                    # Re-queue message if not connected and it's important
                    if message.qos > 0:
                        try:
                            self.message_queue.put_nowait(message)
                        except queue.Full:
                            logger.warning(f"Dropping message for {message.topic} - queue full")
                            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Publisher loop error: {e}")
                

def create_mqtt_handler_mixin():
    """Create a mixin class for camera detector to use stable MQTT"""
    
    class MQTTHandlerMixin:
        """Mixin to replace MQTT functionality in CameraDetector"""
        
        def _setup_mqtt(self):
            """Setup stable MQTT handler"""
            # Create handler
            self.mqtt_handler = StableMQTTHandler(
                broker=self.config.MQTT_BROKER,
                port=self.config.MQTT_PORT,
                client_id=self.config.SERVICE_ID,
                keepalive=30,  # Reduced for faster detection
                tls_enabled=self.config.MQTT_TLS,
                ca_cert_path=self.config.TLS_CA_PATH if self.config.MQTT_TLS else None
            )
            
            # Set callbacks
            self.mqtt_handler.on_connect_callback = self._on_mqtt_connect_stable
            self.mqtt_handler.on_disconnect_callback = self._on_mqtt_disconnect_stable
            
            # Set LWT
            lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
            lwt_payload = json.dumps({
                'node_id': self.config.NODE_ID,
                'service': 'camera_detector',
                'status': 'offline',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            })
            self.mqtt_handler.set_will(lwt_topic, lwt_payload, qos=1, retain=True)
            
            # Start handler
            self.mqtt_handler.start()
            
            # Wait for initial connection
            if not self.mqtt_handler.wait_for_connection(timeout=10.0):
                logger.warning("Initial MQTT connection timeout - running in degraded mode")
                
        def _on_mqtt_connect_stable(self):
            """Handle MQTT connection"""
            self.mqtt_connected = True
            logger.info("MQTT connected via stable handler")
            self._publish_health()
            
        def _on_mqtt_disconnect_stable(self):
            """Handle MQTT disconnection"""
            self.mqtt_connected = False
            logger.warning("MQTT disconnected")
            
        def mqtt_publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False):
            """Publish to MQTT using stable handler"""
            if not hasattr(self, 'mqtt_handler'):
                logger.error("MQTT handler not initialized")
                return False
                
            # Convert payload to JSON if needed
            if not isinstance(payload, str):
                payload = json.dumps(payload)
                
            return self.mqtt_handler.publish(topic, payload, qos, retain)
            
        def cleanup_mqtt(self):
            """Clean up MQTT handler"""
            if hasattr(self, 'mqtt_handler'):
                try:
                    # Publish offline status
                    lwt_payload = json.dumps({
                        'node_id': self.config.NODE_ID,
                        'service': 'camera_detector',
                        'status': 'offline',
                        'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
                    })
                    self.mqtt_publish(
                        f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt",
                        lwt_payload,
                        qos=1,
                        retain=True
                    )
                    
                    # Give time for final message
                    time.sleep(0.5)
                    
                    # Stop handler
                    self.mqtt_handler.stop()
                except Exception as e:
                    logger.error(f"Error during MQTT cleanup: {e}")
                    
    return MQTTHandlerMixin