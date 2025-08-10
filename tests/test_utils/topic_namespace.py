#!/usr/bin/env python3.12
"""
Topic namespace utilities for parallel test isolation.

Provides topic isolation to prevent message interference between parallel test workers.
"""
import re
from typing import Optional, Dict, Any, Callable
import json
import logging

logger = logging.getLogger(__name__)


class TopicNamespace:
    """
    Manages topic namespacing for parallel test isolation.
    
    Each test worker gets its own topic namespace to prevent interference.
    """
    
    def __init__(self, worker_id: str = 'master', prefix: str = 'test'):
        """
        Initialize topic namespace.
        
        Args:
            worker_id: pytest-xdist worker ID (e.g., 'gw0', 'gw1', or 'master')
            prefix: Base prefix for test topics
        """
        self.worker_id = worker_id
        self.prefix = prefix
        self.namespace = f"{prefix}/{worker_id}"
        
    def topic(self, original_topic: str) -> str:
        """
        Convert a topic to its namespaced version.
        
        Args:
            original_topic: Original topic name
            
        Returns:
            Namespaced topic
        """
        # Don't double-namespace
        if original_topic.startswith(self.namespace):
            return original_topic
            
        return f"{self.namespace}/{original_topic}"
    
    def strip(self, namespaced_topic: str) -> str:
        """
        Strip namespace from a topic.
        
        Args:
            namespaced_topic: Namespaced topic
            
        Returns:
            Original topic without namespace
        """
        if namespaced_topic.startswith(f"{self.namespace}/"):
            return namespaced_topic[len(self.namespace) + 1:]
        return namespaced_topic
    
    def is_namespaced(self, topic: str) -> bool:
        """Check if a topic is already namespaced."""
        return topic.startswith(f"{self.namespace}/")
    
    def translate_pattern(self, pattern: str) -> str:
        """
        Translate a topic pattern (with wildcards) to namespaced version.
        
        Args:
            pattern: MQTT topic pattern (e.g., 'camera/+/status')
            
        Returns:
            Namespaced pattern
        """
        # Handle special patterns
        if pattern == '#':
            return f"{self.namespace}/#"
        elif pattern.startswith('#'):
            return f"{self.namespace}/{pattern}"
            
        return self.topic(pattern)
    
    def create_translator(self) -> 'TopicTranslator':
        """Create a topic translator for this namespace."""
        return TopicTranslator(self)


class TopicTranslator:
    """
    Translates MQTT messages between namespaced and original topics.
    
    This is useful for services that need to communicate across namespaces.
    """
    
    def __init__(self, namespace: TopicNamespace):
        """
        Initialize translator.
        
        Args:
            namespace: Topic namespace to use
        """
        self.namespace = namespace
        
    def wrap_callback(self, callback: Callable) -> Callable:
        """
        Wrap an MQTT callback to strip namespace from topics.
        
        Args:
            callback: Original callback function
            
        Returns:
            Wrapped callback that strips namespace
        """
        def wrapped_callback(client, userdata, msg):
            # Create a new message object with stripped topic
            class TranslatedMessage:
                def __init__(self, original_msg, stripped_topic):
                    self.topic = stripped_topic
                    self.payload = original_msg.payload
                    self.qos = original_msg.qos
                    self.retain = original_msg.retain
                    self.mid = original_msg.mid
                    self.timestamp = getattr(original_msg, 'timestamp', None)
                    # Keep reference to original
                    self._original_topic = original_msg.topic
            
            stripped_topic = self.namespace.strip(msg.topic)
            translated_msg = TranslatedMessage(msg, stripped_topic)
            
            # Call original callback with translated message
            return callback(client, userdata, translated_msg)
            
        return wrapped_callback
    
    def translate_publish(self, topic: str, payload: Any, **kwargs) -> tuple:
        """
        Translate a publish call to use namespaced topic.
        
        Args:
            topic: Original topic
            payload: Message payload
            **kwargs: Additional publish arguments
            
        Returns:
            Tuple of (namespaced_topic, payload, kwargs)
        """
        namespaced_topic = self.namespace.topic(topic)
        return namespaced_topic, payload, kwargs
    
    def translate_subscribe(self, topic_or_topics) -> Any:
        """
        Translate subscribe topics to namespaced versions.
        
        Args:
            topic_or_topics: Single topic string or list of topics
            
        Returns:
            Namespaced topic(s)
        """
        if isinstance(topic_or_topics, str):
            return self.namespace.translate_pattern(topic_or_topics)
        elif isinstance(topic_or_topics, (list, tuple)):
            return [self.namespace.translate_pattern(t) if isinstance(t, str) else 
                   (self.namespace.translate_pattern(t[0]), t[1]) for t in topic_or_topics]
        else:
            return topic_or_topics


class NamespacedMQTTClient:
    """
    MQTT client wrapper that automatically handles topic namespacing.
    """
    
    def __init__(self, client, namespace: TopicNamespace):
        """
        Wrap an MQTT client with namespace support.
        
        Args:
            client: Original MQTT client
            namespace: Topic namespace to use
        """
        self.client = client
        self.namespace = namespace
        self.translator = namespace.create_translator()
        
        # Wrap callbacks
        self._wrap_callbacks()
        
    def _wrap_callbacks(self):
        """Wrap client callbacks to handle namespace translation."""
        original_on_message = self.client.on_message
        if original_on_message:
            self.client.on_message = self.translator.wrap_callback(original_on_message)
    
    def publish(self, topic: str, payload: Any = None, qos: int = 0, 
                retain: bool = False, properties: Any = None):
        """Publish with automatic namespace translation."""
        namespaced_topic = self.namespace.topic(topic)
        return self.client.publish(namespaced_topic, payload, qos, retain, properties)
    
    def subscribe(self, topic, qos: int = 0, options: Any = None, properties: Any = None):
        """Subscribe with automatic namespace translation."""
        namespaced_topic = self.translator.translate_subscribe(topic)
        return self.client.subscribe(namespaced_topic, qos, options, properties)
    
    def unsubscribe(self, topic, properties: Any = None):
        """Unsubscribe with automatic namespace translation."""
        namespaced_topic = self.translator.translate_subscribe(topic)
        return self.client.unsubscribe(namespaced_topic, properties)
    
    # Delegate other methods to wrapped client
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped client."""
        return getattr(self.client, name)


def create_namespaced_client(client, worker_id: str) -> NamespacedMQTTClient:
    """
    Convenience function to create a namespaced MQTT client.
    
    Args:
        client: Original MQTT client
        worker_id: Worker ID for namespace
        
    Returns:
        Namespaced MQTT client
    """
    namespace = TopicNamespace(worker_id)
    return NamespacedMQTTClient(client, namespace)