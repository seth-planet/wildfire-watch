#!/usr/bin/env python3.12
"""
MQTT Namespace Bypass for E2E Tests

A simpler approach that bypasses namespace isolation for health monitoring tests
where Docker containers don't support namespacing.
"""

import paho.mqtt.client as mqtt
from typing import List, Dict, Any, Optional, Callable
import time
import json
import logging
from threading import Event, Lock

logger = logging.getLogger(__name__)


class BypassNamespaceClient:
    """
    MQTT client that can bypass namespace isolation for specific topics.
    
    This is useful when testing with Docker containers that publish to
    standard topics while the test framework expects namespaced topics.
    """
    
    def __init__(self, client_id: str):
        """Initialize bypass client."""
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        self.messages = []
        self.message_lock = Lock()
        self.message_event = Event()
        
        # Callbacks for specific topics
        self.topic_callbacks = {}
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle connection."""
        if rc == 0:
            logger.info(f"Bypass client connected successfully")
        else:
            logger.error(f"Connection failed with code: {rc}")
            
    def _on_message(self, client, userdata, msg):
        """Handle all messages."""
        # Store message
        with self.message_lock:
            self.messages.append({
                'topic': msg.topic,
                'payload': msg.payload.decode() if isinstance(msg.payload, bytes) else str(msg.payload),
                'timestamp': time.time()
            })
        self.message_event.set()
        
        # Log for debugging
        logger.debug(f"Received: {msg.topic} = {msg.payload.decode() if isinstance(msg.payload, bytes) else msg.payload}")
        
        # Call topic-specific callbacks
        for pattern, callback in self.topic_callbacks.items():
            if self._matches_pattern(msg.topic, pattern):
                callback(client, userdata, msg)
                
    def _on_disconnect(self, client, userdata, rc):
        """Handle disconnection."""
        if rc != 0:
            logger.warning(f"Unexpected disconnection: {rc}")
            
    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports + and # wildcards)."""
        if pattern == '#':
            return True
        if pattern == topic:
            return True
            
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if '#' in pattern:
            # # must be last
            hash_index = pattern_parts.index('#')
            if hash_index != len(pattern_parts) - 1:
                return False
            # Match up to #
            pattern_parts = pattern_parts[:hash_index]
            topic_parts = topic_parts[:len(pattern_parts)]
            
        if len(pattern_parts) != len(topic_parts):
            return False
            
        for p, t in zip(pattern_parts, topic_parts):
            if p == '+':
                continue
            if p != t:
                return False
                
        return True
        
    def connect(self, host: str, port: int, keepalive: int = 60):
        """Connect to broker."""
        return self.client.connect(host, port, keepalive)
        
    def subscribe(self, topic: str, qos: int = 0, callback: Optional[Callable] = None):
        """Subscribe to topic."""
        if callback:
            self.topic_callbacks[topic] = callback
        return self.client.subscribe(topic, qos)
        
    def publish(self, topic: str, payload: Any, qos: int = 0, retain: bool = False):
        """Publish message."""
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        return self.client.publish(topic, payload, qos, retain)
        
    def loop_start(self):
        """Start network loop."""
        return self.client.loop_start()
        
    def loop_stop(self):
        """Stop network loop."""
        return self.client.loop_stop()
        
    def disconnect(self):
        """Disconnect from broker."""
        return self.client.disconnect()
        
    def wait_for_messages(self, count: int = 1, timeout: int = 10) -> bool:
        """Wait for a specific number of messages."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.message_lock:
                if len(self.messages) >= count:
                    return True
            time.sleep(0.1)
        return False
        
    def get_messages(self, topic_filter: Optional[str] = None) -> List[Dict]:
        """Get messages, optionally filtered by topic pattern."""
        with self.message_lock:
            if topic_filter is None:
                return self.messages.copy()
            return [m for m in self.messages if self._matches_pattern(m['topic'], topic_filter)]


def setup_health_monitoring_test(mqtt_port: int) -> BypassNamespaceClient:
    """
    Set up a test client for health monitoring that bypasses namespace issues.
    
    Args:
        mqtt_port: MQTT broker port
        
    Returns:
        Configured client ready for health monitoring
    """
    client = BypassNamespaceClient(f'health_test_{int(time.time())}')
    
    # Connect to broker
    client.connect('localhost', mqtt_port, 60)
    client.loop_start()
    time.sleep(0.5)  # Allow connection to establish
    
    # Subscribe to all health-related topics
    # Use direct topics that Docker containers publish to
    health_topics = [
        'system/camera_detector_health',
        'system/fire_consensus_health', 
        'system/gpio_trigger_health',
        'health/+',
        'system/+/health',
        '#'  # Subscribe to everything for debugging
    ]
    
    for topic in health_topics:
        client.subscribe(topic)
        logger.info(f"Subscribed to: {topic}")
        
    return client


def verify_service_health(client: BypassNamespaceClient, 
                         expected_services: List[str],
                         timeout: int = 30) -> Dict[str, Dict]:
    """
    Verify health messages from services.
    
    Args:
        client: The bypass client
        expected_services: Service names to check
        timeout: Maximum wait time
        
    Returns:
        Dict with service health status and messages
    """
    start_time = time.time()
    health_status = {service: None for service in expected_services}
    
    # Map service names to expected topic patterns
    service_topics = {
        'camera_detector': ['system/camera_detector_health', 'camera_detector/health'],
        'fire_consensus': ['system/fire_consensus_health', 'fire_consensus/health', 'consensus/health'],
        'gpio_trigger': ['system/gpio_trigger_health', 'gpio_trigger/health', 'gpio/health']
    }
    
    while time.time() - start_time < timeout:
        messages = client.get_messages()
        
        for service in expected_services:
            if health_status[service] is not None:
                continue  # Already found
                
            # Check for health messages from this service
            possible_topics = service_topics.get(service, [f'system/{service}_health'])
            
            for msg in messages:
                topic = msg['topic']
                # Check if this message is from the service
                if any(pt in topic or topic.endswith(pt) for pt in possible_topics):
                    health_status[service] = msg
                    logger.info(f"Found health from {service}: {topic}")
                    break
                # Also check payload for service identification
                elif service in topic or service in msg.get('payload', ''):
                    health_status[service] = msg
                    logger.info(f"Found health from {service} (by content): {topic}")
                    break
                    
        # Check if all found
        if all(v is not None for v in health_status.values()):
            logger.info("All services reported health")
            break
            
        time.sleep(0.5)
        
    # Report missing
    for service, status in health_status.items():
        if status is None:
            logger.warning(f"No health message from {service} after {timeout}s")
            # Show what messages we did receive
            all_topics = [m['topic'] for m in client.get_messages()]
            logger.debug(f"Received topics: {all_topics}")
            
    return health_status


# Test helper function
def run_health_monitoring_test(mqtt_port: int, docker_containers: Dict) -> None:
    """
    Run a health monitoring test with proper message handling.
    
    Args:
        mqtt_port: MQTT broker port
        docker_containers: Dict of service_name -> container objects
    """
    # Set up bypass client
    client = setup_health_monitoring_test(mqtt_port)
    
    try:
        # Give services time to start and connect
        logger.info("Waiting for services to initialize...")
        time.sleep(5)
        
        # Force health publishes if needed (some services might publish on demand)
        # This could involve sending a health request message
        client.publish('health/request', json.dumps({'request': 'status'}))
        
        # Verify health from all services
        expected_services = list(docker_containers.keys())
        health_results = verify_service_health(client, expected_services, timeout=30)
        
        # Validate results
        assert all(v is not None for v in health_results.values()), \
            f"Missing health from: {[k for k, v in health_results.items() if v is None]}"
            
        # Additional validation
        for service, health_msg in health_results.items():
            if health_msg:
                logger.info(f"{service} health: {health_msg['topic']} = {health_msg['payload']}")
                # Could parse payload and check specific fields
                
        logger.info("Health monitoring test passed!")
        
    finally:
        client.loop_stop()
        client.disconnect()


# Example of how to use in actual test:
"""
def test_e2e_health_monitoring(mqtt_broker, docker_client):
    # Start services
    containers = {
        'camera_detector': start_camera_detector(docker_client, mqtt_broker.port),
        'fire_consensus': start_consensus(docker_client, mqtt_broker.port),
        'gpio_trigger': start_gpio_trigger(docker_client, mqtt_broker.port)
    }
    
    try:
        # Run health monitoring test
        run_health_monitoring_test(mqtt_broker.port, containers)
    finally:
        # Cleanup containers
        for container in containers.values():
            container.stop()
            container.remove()
"""