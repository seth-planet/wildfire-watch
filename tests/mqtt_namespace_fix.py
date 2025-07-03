#!/usr/bin/env python3.12
"""
MQTT Namespace Fix for E2E Health Monitoring Tests

This module provides utilities to fix the namespace mismatch between
Docker containers and test clients.
"""

import paho.mqtt.client as mqtt
from typing import Optional, List, Dict, Any, Callable
import logging
import time
from threading import Event, Lock

logger = logging.getLogger(__name__)


class NamespaceBridgeMQTTClient:
    """
    MQTT client that bridges between namespaced and non-namespaced topics.
    
    This client subscribes to both namespaced and non-namespaced versions
    of topics to handle messages from Docker containers that don't know
    about the test namespace.
    """
    
    def __init__(self, client_id: str, namespace: Optional[str] = None):
        """
        Initialize the bridge client.
        
        Args:
            client_id: MQTT client ID
            namespace: Optional namespace prefix (e.g., "test/gw0")
        """
        self.client = mqtt.Client(client_id=client_id)
        self.namespace = namespace
        self.callbacks = {}
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """Set up MQTT callbacks."""
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection."""
        if rc == 0:
            logger.info(f"Bridge client connected (namespace: {self.namespace})")
        else:
            logger.error(f"Bridge client connection failed: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """
        Handle incoming messages and translate namespaces if needed.
        
        This bridges the gap by:
        1. Accepting messages on non-namespaced topics from Docker containers
        2. Translating them to namespaced topics for test expectations
        """
        topic = msg.topic
        
        # If we have a namespace and the message is not namespaced, add it
        if self.namespace and not topic.startswith(self.namespace):
            # This is a non-namespaced message from Docker container
            # Create a namespaced version for test callbacks
            namespaced_topic = f"{self.namespace}/{topic}"
            
            # Call callbacks for both original and namespaced topics
            for callback_topic, callback in self.callbacks.items():
                if self._topic_matches(topic, callback_topic):
                    callback(client, userdata, msg)
                if self._topic_matches(namespaced_topic, callback_topic):
                    # Create a modified message with namespaced topic
                    class NamespacedMessage:
                        def __init__(self, original_msg, new_topic):
                            self.topic = new_topic
                            self.payload = original_msg.payload
                            self.qos = original_msg.qos
                            self.retain = original_msg.retain
                    
                    namespaced_msg = NamespacedMessage(msg, namespaced_topic)
                    callback(client, userdata, namespaced_msg)
        else:
            # Regular message handling
            for callback_topic, callback in self.callbacks.items():
                if self._topic_matches(topic, callback_topic):
                    callback(client, userdata, msg)
                    
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection."""
        logger.warning(f"Bridge client disconnected: {rc}")
        
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """
        Check if a topic matches a subscription pattern.
        
        Handles MQTT wildcards:
        - # matches any number of levels
        - + matches exactly one level
        """
        if pattern == '#':
            return True
        if pattern == topic:
            return True
            
        # Handle wildcards
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if '+' not in pattern and '#' not in pattern:
            return pattern == topic
            
        for i, (pattern_part, topic_part) in enumerate(zip(pattern_parts, topic_parts)):
            if pattern_part == '#':
                return True
            elif pattern_part == '+':
                continue
            elif pattern_part != topic_part:
                return False
                
        return len(pattern_parts) == len(topic_parts)
        
    def subscribe(self, topic: str, qos: int = 0, callback: Optional[Callable] = None):
        """
        Subscribe to a topic with automatic namespace bridging.
        
        If a namespace is set, this subscribes to both:
        1. The namespaced version (for test isolation)
        2. The non-namespaced version (for Docker containers)
        """
        if callback:
            self.callbacks[topic] = callback
            
        # Subscribe to the requested topic
        self.client.subscribe(topic, qos)
        
        # If using namespace, also subscribe to non-namespaced version
        if self.namespace and topic.startswith(self.namespace):
            # Extract the non-namespaced topic
            non_namespaced = topic[len(self.namespace) + 1:]  # +1 for the '/'
            if non_namespaced:
                logger.debug(f"Also subscribing to non-namespaced: {non_namespaced}")
                self.client.subscribe(non_namespaced, qos)
                
    def publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False):
        """Publish a message."""
        return self.client.publish(topic, payload, qos, retain)
        
    def connect(self, host: str, port: int, keepalive: int = 60):
        """Connect to MQTT broker."""
        return self.client.connect(host, port, keepalive)
        
    def loop_start(self):
        """Start the MQTT loop."""
        return self.client.loop_start()
        
    def loop_stop(self):
        """Stop the MQTT loop."""
        return self.client.loop_stop()
        
    def disconnect(self):
        """Disconnect from broker."""
        return self.client.disconnect()


def create_health_monitoring_client(port: int, namespace: Optional[str] = None,
                                  timeout: int = 30) -> tuple:
    """
    Create an MQTT client specifically for health monitoring tests.
    
    This client handles the namespace mismatch issue by subscribing to
    both namespaced and non-namespaced topics.
    
    Args:
        port: MQTT broker port
        namespace: Optional namespace (e.g., "test/gw0")
        timeout: Timeout for receiving messages
        
    Returns:
        Tuple of (client, messages_list, message_event)
    """
    client = NamespaceBridgeMQTTClient(
        client_id=f'health_monitor_{int(time.time())}',
        namespace=namespace
    )
    
    messages = []
    message_lock = Lock()
    message_event = Event()
    
    def on_health_message(client, userdata, msg):
        """Capture health messages."""
        with message_lock:
            messages.append({
                'topic': msg.topic,
                'payload': msg.payload.decode() if isinstance(msg.payload, bytes) else msg.payload,
                'timestamp': time.time()
            })
        message_event.set()
        logger.debug(f"Received health message on {msg.topic}")
    
    # Subscribe to health topics with callback
    health_topics = [
        'system/+/health',  # Matches system/camera_detector_health, etc.
        'system/+_health',  # Alternative pattern
        'health/+',         # General health topics
        '+/health'          # Service-specific health
    ]
    
    # Connect first
    client.connect('localhost', port, 60)
    client.loop_start()
    time.sleep(0.5)  # Give connection time to establish
    
    # Subscribe to health topics
    for topic in health_topics:
        # If namespace is set, subscribe to namespaced version
        if namespace:
            namespaced_topic = f"{namespace}/{topic}"
            client.subscribe(namespaced_topic, callback=on_health_message)
        # Always subscribe to non-namespaced version for Docker containers
        client.subscribe(topic, callback=on_health_message)
    
    # Also subscribe to all topics for debugging
    client.subscribe('#', callback=on_health_message)
    
    return client, messages, message_event


def wait_for_health_messages(messages: List[Dict], expected_services: List[str],
                           timeout: int = 30) -> Dict[str, bool]:
    """
    Wait for health messages from expected services.
    
    Args:
        messages: List to collect messages in
        expected_services: List of service names to expect
        timeout: Maximum time to wait
        
    Returns:
        Dict mapping service name to whether health was received
    """
    start_time = time.time()
    received = {service: False for service in expected_services}
    
    while time.time() - start_time < timeout:
        # Check current messages
        for msg in messages:
            topic = msg['topic']
            # Handle both namespaced and non-namespaced topics
            if '/health' in topic or '_health' in topic:
                for service in expected_services:
                    if service in topic:
                        received[service] = True
                        logger.info(f"Received health from {service}")
        
        # Check if all received
        if all(received.values()):
            logger.info("All health messages received")
            break
            
        time.sleep(0.5)
    
    # Log summary
    for service, got_health in received.items():
        if not got_health:
            logger.warning(f"No health message from {service}")
            
    return received


# Example usage for fixing the test:
def test_e2e_health_monitoring_fixed(mqtt_broker, docker_containers):
    """
    Fixed version of health monitoring test that handles namespace mismatch.
    """
    # Get namespace if using pytest-xdist
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', 'master')
    namespace = f"test/{worker_id}" if worker_id != 'master' else None
    
    # Create health monitoring client
    client, messages, event = create_health_monitoring_client(
        port=mqtt_broker.port,
        namespace=namespace
    )
    
    try:
        # Start Docker containers (they publish to non-namespaced topics)
        containers = start_test_containers(docker_containers, mqtt_broker.port)
        
        # Wait for health messages
        expected_services = ['camera_detector', 'fire_consensus', 'gpio_trigger']
        health_status = wait_for_health_messages(messages, expected_services, timeout=30)
        
        # Assertions
        assert all(health_status.values()), f"Missing health from: {[s for s, h in health_status.items() if not h]}"
        assert len(messages) > 0, "No messages received at all"
        
        # Additional validation
        for msg in messages:
            if 'health' in msg['topic']:
                print(f"Health message: {msg['topic']} = {msg['payload']}")
                
    finally:
        client.loop_stop()
        client.disconnect()