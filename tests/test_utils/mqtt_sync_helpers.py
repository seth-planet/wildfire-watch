#!/usr/bin/env python3
"""
MQTT Synchronization Helpers for Testing

Provides event-based synchronization utilities to replace time.sleep() calls
in tests, making them more reliable and faster.
"""
import time
import threading
import logging
from typing import Callable, Optional, Any, Dict, List
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTSubscriptionWaiter:
    """Wait for MQTT subscription to be ready before proceeding."""
    
    def __init__(self, client: mqtt.Client, timeout: float = 5.0):
        self.client = client
        self.timeout = timeout
        self.ready_event = threading.Event()
        self.subscribed_topics = set()
        self._original_on_subscribe = None
        
    def __enter__(self):
        """Set up subscription monitoring."""
        self._original_on_subscribe = self.client.on_subscribe
        self.client.on_subscribe = self._on_subscribe
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original callback."""
        if self._original_on_subscribe:
            self.client.on_subscribe = self._original_on_subscribe
            
    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None):
        """Callback when subscription is confirmed."""
        self.ready_event.set()
        if self._original_on_subscribe:
            # Call original callback if it exists
            if properties is not None:
                self._original_on_subscribe(client, userdata, mid, granted_qos, properties)
            else:
                # For older paho-mqtt versions
                self._original_on_subscribe(client, userdata, mid, granted_qos)
                
    def wait_for_subscription(self, topic: str) -> bool:
        """Wait for subscription to be ready."""
        result = self.client.subscribe(topic)
        if result[0] == mqtt.MQTT_ERR_SUCCESS:
            return self.ready_event.wait(timeout=self.timeout)
        return False
        

class MQTTMessageWaiter:
    """Wait for specific MQTT messages with event-based synchronization."""
    
    def __init__(self, client: mqtt.Client, timeout: float = 10.0):
        self.client = client
        self.timeout = timeout
        self.message_events: Dict[str, threading.Event] = {}
        self.received_messages: Dict[str, List[Any]] = {}
        self._original_on_message = None
        self.lock = threading.Lock()
        
    def __enter__(self):
        """Set up message monitoring."""
        self._original_on_message = self.client.on_message
        self.client.on_message = self._on_message
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original callback."""
        if self._original_on_message:
            self.client.on_message = self._original_on_message
            
    def _on_message(self, client, userdata, msg):
        """Callback for all messages."""
        with self.lock:
            topic = msg.topic
            
            # Store message
            if topic not in self.received_messages:
                self.received_messages[topic] = []
            self.received_messages[topic].append(msg)
            
            # Set event for this topic
            if topic in self.message_events:
                self.message_events[topic].set()
                
            # Also set events for wildcard patterns
            for pattern in list(self.message_events.keys()):
                if '+' in pattern or '#' in pattern:
                    if self._topic_matches(topic, pattern):
                        self.message_events[pattern].set()
                        
        # Call original callback if it exists
        if self._original_on_message:
            self._original_on_message(client, userdata, msg)
            
    def wait_for_message(self, topic_pattern: str, 
                        condition: Optional[Callable[[Any], bool]] = None,
                        timeout: Optional[float] = None) -> bool:
        """
        Wait for a message on a specific topic pattern.
        
        Args:
            topic_pattern: MQTT topic or pattern (supports + and #)
            condition: Optional function to check message content
            timeout: Override default timeout
            
        Returns:
            True if message received matching condition, False on timeout
        """
        if timeout is None:
            timeout = self.timeout
            
        with self.lock:
            # Check if we already have a matching message
            for topic, messages in self.received_messages.items():
                if self._topic_matches(topic, topic_pattern):
                    if condition is None:
                        return True
                    for msg in messages:
                        if condition(msg):
                            return True
                            
            # Set up event for this pattern
            if topic_pattern not in self.message_events:
                self.message_events[topic_pattern] = threading.Event()
                
        # Wait for message
        start_time = time.time()
        remaining_timeout = timeout
        
        while remaining_timeout > 0:
            if self.message_events[topic_pattern].wait(timeout=min(0.1, remaining_timeout)):
                # Event was set, check if condition is met
                with self.lock:
                    for topic, messages in self.received_messages.items():
                        if self._topic_matches(topic, topic_pattern):
                            if condition is None:
                                return True
                            for msg in messages:
                                if condition(msg):
                                    return True
                    # Reset event if condition not met
                    self.message_events[topic_pattern].clear()
                    
            remaining_timeout = timeout - (time.time() - start_time)
            
        return False
        
    def get_messages(self, topic_pattern: str) -> List[Any]:
        """Get all received messages matching a topic pattern."""
        with self.lock:
            matching_messages = []
            for topic, messages in self.received_messages.items():
                if self._topic_matches(topic, topic_pattern):
                    matching_messages.extend(messages)
            return matching_messages
            
    def clear_messages(self, topic_pattern: Optional[str] = None):
        """Clear received messages."""
        with self.lock:
            if topic_pattern is None:
                self.received_messages.clear()
            else:
                topics_to_clear = []
                for topic in self.received_messages:
                    if self._topic_matches(topic, topic_pattern):
                        topics_to_clear.append(topic)
                for topic in topics_to_clear:
                    del self.received_messages[topic]
                    
    @staticmethod
    def _topic_matches(topic: str, pattern: str) -> bool:
        """Check if a topic matches a pattern with + and # wildcards."""
        if pattern == topic:
            return True
            
        pattern_parts = pattern.split('/')
        topic_parts = topic.split('/')
        
        if '#' not in pattern and len(pattern_parts) != len(topic_parts):
            return False
            
        for i, (pattern_part, topic_part) in enumerate(zip(pattern_parts, topic_parts)):
            if pattern_part == '#':
                return True  # # matches everything after this level
            elif pattern_part == '+':
                continue  # + matches any single level
            elif pattern_part != topic_part:
                return False
                
        return True


def wait_for_service_ready(service, attribute: str = 'health_reporter', 
                          timeout: float = 10.0) -> bool:
    """
    Wait for a service to be fully initialized.
    
    Args:
        service: The service instance to check
        attribute: Attribute that indicates service is ready
        timeout: Maximum time to wait
        
    Returns:
        True if service is ready, False on timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if hasattr(service, attribute) and getattr(service, attribute) is not None:
            # For health_reporter, also check if it's started
            if attribute == 'health_reporter':
                reporter = getattr(service, attribute)
                if hasattr(reporter, '_timer') and reporter._timer is not None:
                    return True
            else:
                return True
        time.sleep(0.1)
        
    return False


def wait_for_consensus_evaluation(consensus_service, timeout: float = 10.0,
                                check_interval: float = 0.1) -> bool:
    """
    Wait for consensus service to evaluate detections.
    
    Args:
        consensus_service: FireConsensus instance
        timeout: Maximum time to wait
        check_interval: How often to check
        
    Returns:
        True if consensus was checked, False on timeout
    """
    start_time = time.time()
    initial_event_count = len(consensus_service.consensus_events)
    
    while time.time() - start_time < timeout:
        current_count = len(consensus_service.consensus_events)
        if current_count > initial_event_count:
            return True
        
        # Also check if _check_consensus was called by monitoring detection processing
        with consensus_service.lock:
            # Check if any camera has recent detections
            for camera in consensus_service.cameras.values():
                if camera.last_detection_time > start_time:
                    # Detection was processed, consensus should have been checked
                    time.sleep(check_interval * 2)  # Give a bit more time
                    return True
                    
        time.sleep(check_interval)
        
    return False


def create_mqtt_event_monitor(client: mqtt.Client) -> Dict[str, threading.Event]:
    """
    Create a dictionary of events for monitoring MQTT activity.
    
    Returns:
        Dictionary with events for 'connected', 'disconnected', 'subscribed'
    """
    events = {
        'connected': threading.Event(),
        'disconnected': threading.Event(),
        'subscribed': threading.Event(),
    }
    
    original_on_connect = client.on_connect
    original_on_disconnect = client.on_disconnect
    original_on_subscribe = client.on_subscribe
    
    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            events['connected'].set()
        if original_on_connect:
            if properties is not None:
                original_on_connect(client, userdata, flags, rc, properties)
            else:
                original_on_connect(client, userdata, flags, rc)
                
    def on_disconnect(client, userdata, flags, rc, properties=None):
        events['disconnected'].set()
        events['connected'].clear()
        if original_on_disconnect:
            if properties is not None:
                original_on_disconnect(client, userdata, flags, rc, properties)
            else:
                original_on_disconnect(client, userdata, flags, rc)
                
    def on_subscribe(client, userdata, mid, granted_qos, properties=None):
        events['subscribed'].set()
        if original_on_subscribe:
            if properties is not None:
                original_on_subscribe(client, userdata, mid, granted_qos, properties)
            else:
                original_on_subscribe(client, userdata, mid, granted_qos)
                
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    
    return events