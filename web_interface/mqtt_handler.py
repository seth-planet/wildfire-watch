#!/usr/bin/env python3.12
"""MQTT handler for the Web Interface service.

This module manages MQTT subscriptions and maintains a circular buffer
of events for display in the web interface.
"""

import time
import json
import logging
import threading
from collections import deque
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from utils.mqtt_service import MQTTService
from utils.health_reporter import ServiceHealthReporter

from .config import get_config
from .models import MQTTEvent, SystemStatus, ServiceHealth, GPIOState, EventType
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MQTTHandler(MQTTService):
    """Handles MQTT subscriptions and event buffering for the web interface.
    
    Maintains a thread-safe circular buffer of MQTT events and provides
    methods to retrieve current system status.
    """
    
    def __init__(self):
        """Initialize MQTT handler with configuration."""
        # Keep reference to web interface config
        self.web_config = get_config()
        # Pass MQTT config to base class
        super().__init__("web_interface", self.web_config.mqtt_config)
        
        # Event buffer with thread safety
        self._event_buffer_lock = threading.Lock()
        self._event_buffer = deque(maxlen=self.web_config.mqtt_buffer_size)
        
        # Current state caches
        self._service_states: Dict[str, ServiceHealth] = {}
        self._gpio_states: Dict[str, bool] = {}
        self._fire_state = {
            'active': False,
            'last_trigger': None,
            'consensus_count': 0,
            'triggering_cameras': []
        }
        self._camera_states: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting for event processing
        self._event_counts: Dict[str, int] = {}
        self._event_window_start = time.time()
        
        # Health reporter
        self.health_reporter = ServiceHealthReporter(
            self,
            self._get_health_state,
            interval=self.web_config.health_check_interval
        )
        
        # Event callbacks for UI updates
        self._event_callbacks: List[Callable] = []
        
    def initialize(self):
        """Initialize MQTT connection and subscriptions."""
        # Setup MQTT with callbacks
        self.setup_mqtt(
            on_connect=self._on_connect,
            on_message=self._on_message,
            subscriptions=self.web_config.get_mqtt_topics()
        )
        
        # Enable offline queue for reliability
        self.enable_offline_queue(max_size=100)
        
        # Connect to broker (non-blocking)
        self.connect()
        
        # Start health reporting (also non-blocking)
        self.health_reporter.start_health_reporting()
        
        self._safe_log('info', "MQTT handler initialized, connection in progress")
        
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection and resubscribe to topics."""
        if rc == 0:
            self._safe_log('info', "Connected to MQTT broker, subscribing to topics")
            # Subscriptions are handled by base class
        else:
            self._safe_log('error', f"Failed to connect to MQTT: {rc}")
            
    def _on_message(self, topic: str, payload: Any):
        """Process incoming MQTT messages.
        
        Args:
            topic: MQTT topic (already cleaned by base class)
            payload: Message payload (already parsed if JSON)
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit(topic):
                self._safe_log('debug', f"Rate limit exceeded for topic: {topic}")
                return
                
            # Create event object - convert string to EventType enum
            event_type_str = self._categorize_event(topic)
            event = MQTTEvent(
                timestamp=datetime.utcnow(),
                topic=topic,
                payload=payload,
                event_type=EventType(event_type_str)
            )
            
            # Add to buffer
            with self._event_buffer_lock:
                self._event_buffer.append(event)
                
            # Update state caches based on topic
            self._update_state_caches(topic, payload)
            
            # Notify callbacks
            for callback in self._event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self._safe_log('error', f"Error in event callback: {e}")
                    
        except Exception as e:
            self._safe_log('error', f"Error processing MQTT message on {topic}: {e}")
            
    def _check_rate_limit(self, topic: str) -> bool:
        """Check if topic is within rate limits.
        
        Args:
            topic: MQTT topic
            
        Returns:
            True if within limits, False if rate limited
        """
        # Reset window if needed
        current_time = time.time()
        if current_time - self._event_window_start > 60:  # 1 minute window
            self._event_counts.clear()
            self._event_window_start = current_time
            
        # Extract base topic for counting
        base_topic = topic.split('/')[0]
        
        # Increment count
        self._event_counts[base_topic] = self._event_counts.get(base_topic, 0) + 1
        
        # Check limits (allow more for critical topics)
        limits = {
            'fire': 100,      # Fire events are critical
            'gpio': 100,      # GPIO state changes are important
            'system': 50,     # Health updates
            'camera': 30,     # Camera updates
            'frigate': 50,    # Detection events
            'telemetry': 20   # General telemetry
        }
        
        limit = limits.get(base_topic, 10)  # Default limit
        return self._event_counts[base_topic] <= limit
        
    def _categorize_event(self, topic: str) -> str:
        """Categorize event based on topic.
        
        Args:
            topic: MQTT topic
            
        Returns:
            Event category
        """
        # Store original topic for exact matching
        original_topic = topic
        
        # Remove topic prefix if present
        if hasattr(self, '_topic_prefix') and self._topic_prefix and topic.startswith(self._topic_prefix):
            topic = topic[len(self._topic_prefix):].lstrip('/')
        
        if topic.startswith('fire/'):
            return 'fire'
        elif topic.startswith('gpio/'):
            return 'gpio'
        elif topic.startswith('system/') and '/health' in topic:
            return 'health'
        elif topic.startswith('camera/'):
            return 'camera'
        elif topic.startswith('frigate/'):
            return 'detection'
        elif topic.startswith('telemetry/'):
            return 'telemetry'
        else:
            return 'other'
            
    def _update_state_caches(self, topic: str, payload: Any):
        """Update internal state caches based on message.
        
        Args:
            topic: MQTT topic (with prefix already removed)
            payload: Message payload
        """
        try:
            # Remove topic prefix if present
            if hasattr(self, '_topic_prefix') and self._topic_prefix and topic.startswith(self._topic_prefix):
                topic = topic[len(self._topic_prefix):].lstrip('/')
                
            # Service health updates
            if ('/health' in topic or topic == 'system/trigger_telemetry') and isinstance(payload, dict):
                parts = topic.split('/')
                if len(parts) >= 2:
                    # Special case for trigger_telemetry - it's the gpio_trigger service
                    if topic == 'system/trigger_telemetry':
                        service_name = 'gpio_trigger'
                        # Check if it's a health report from telemetry
                        if payload.get('action') != 'health_report':
                            return  # Skip non-health telemetry messages
                        # Convert uptime from seconds to hours for consistency
                        if 'uptime' in payload:
                            payload['uptime_hours'] = payload['uptime'] / 3600.0
                    else:
                        service_name = parts[1]
                    
                    self._service_states[service_name] = ServiceHealth(
                        name=service_name,
                        status='healthy' if payload.get('mqtt_connected', False) or payload.get('state') == 'IDLE' else 'unhealthy',
                        last_seen=datetime.utcnow(),
                        uptime=payload.get('uptime_hours', 0),
                        details=payload
                    )
                    
            # GPIO state updates
            elif topic == 'gpio/status' and isinstance(payload, dict):
                for pin_name, state in payload.items():
                    self._gpio_states[pin_name] = bool(state)
                    
            # Fire trigger updates
            elif topic == 'fire/trigger':
                self._fire_state['active'] = True
                self._fire_state['last_trigger'] = datetime.utcnow()
                if isinstance(payload, dict):
                    self._fire_state['triggering_cameras'] = payload.get('cameras', [])
                    # Update consensus count from fire trigger payload
                    consensus_cameras = payload.get('consensus_cameras', [])
                    self._fire_state['consensus_count'] = len(consensus_cameras)
                    
            # Fire consensus state
            elif topic == 'fire/consensus_state' and isinstance(payload, dict):
                self._fire_state['consensus_count'] = payload.get('consensus_count', 0)
                self._fire_state['active'] = payload.get('fire_detected', False)
                
            # Camera discovery
            elif topic.startswith('camera/discovery/') and isinstance(payload, dict):
                camera_id = topic.split('/')[-1]
                self._camera_states[camera_id] = {
                    'name': payload.get('name', camera_id),
                    'ip': payload.get('ip'),
                    'last_seen': datetime.utcnow(),
                    'detection_count': 0
                }
                
            # Frigate detections
            elif topic.startswith('frigate/') and '/fire' in topic:
                camera_id = topic.split('/')[1]
                if camera_id in self._camera_states:
                    self._camera_states[camera_id]['detection_count'] += 1
                    
        except Exception as e:
            self._safe_log('error', f"Error updating state cache for {topic}: {e}")
            
    def get_recent_events(self, 
                         limit: Optional[int] = None,
                         event_type: Optional[str] = None,
                         max_age_seconds: Optional[int] = None) -> List[MQTTEvent]:
        """Get recent events from the buffer.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            max_age_seconds: Maximum age of events in seconds
            
        Returns:
            List of recent events
        """
        with self._event_buffer_lock:
            events = list(self._event_buffer)
            
        # Apply filters
        if max_age_seconds:
            cutoff_time = datetime.utcnow().timestamp() - max_age_seconds
            events = [e for e in events if e.timestamp.timestamp() > cutoff_time]
            
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
            
        return events
        
    def get_system_status(self) -> SystemStatus:
        """Get current system status.
        
        Returns:
            Current system status
        """
        return SystemStatus(
            fire_active=self._fire_state['active'],
            last_fire_trigger=self._fire_state['last_trigger'],
            consensus_count=self._fire_state['consensus_count'],
            service_count=len(self._service_states),
            healthy_services=sum(1 for s in self._service_states.values() 
                               if s.status == 'healthy'),
            camera_count=len(self._camera_states),
            active_cameras=sum(1 for c in self._camera_states.values() 
                             if c.get('last_seen') and 
                             (datetime.utcnow() - c['last_seen']).seconds < 300),
            gpio_states=self._gpio_states.copy(),
            mqtt_connected=self.is_connected,
            buffer_size=len(self._event_buffer)
        )
        
    def get_service_health(self) -> List[ServiceHealth]:
        """Get health status of all services.
        
        Returns:
            List of service health states
        """
        return list(self._service_states.values())
        
    def get_gpio_states(self) -> Dict[str, GPIOState]:
        """Get current GPIO pin states.
        
        Returns:
            Dictionary of GPIO states
        """
        gpio_mapping = {
            'main_valve': 'Main Water Valve',
            'ignition_on': 'Engine Ignition',
            'rpm_reduction': 'RPM Reduction',
            'refill_valve': 'Refill Valve',
            'pump_running': 'Pump Status'
        }
        
        result = {}
        for pin_id, pin_name in gpio_mapping.items():
            result[pin_id] = GPIOState(
                pin_id=pin_id,
                name=pin_name,
                state=self._gpio_states.get(pin_id, False),
                last_change=datetime.utcnow()  # TODO: Track actual change times
            )
            
        return result
        
    def add_event_callback(self, callback: Callable):
        """Add a callback for new events.
        
        Args:
            callback: Function to call with new events
        """
        self._event_callbacks.append(callback)
        
    def remove_event_callback(self, callback: Callable):
        """Remove an event callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
            
    def _get_health_state(self) -> Dict[str, Any]:
        """Get health state for health reporter.
        
        Returns:
            Dictionary of health metrics
        """
        return {
            'buffer_size': len(self._event_buffer),
            'buffer_capacity': self.web_config.mqtt_buffer_size,
            'connected_services': len(self._service_states),
            'active_cameras': sum(1 for c in self._camera_states.values() 
                                if c.get('last_seen') and 
                                (datetime.utcnow() - c['last_seen']).seconds < 300),
            'fire_active': self._fire_state['active'],
            'event_rate': sum(self._event_counts.values()) / 60.0  # Events per second
        }
        
    def shutdown(self):
        """Shutdown MQTT handler gracefully."""
        self._safe_log('info', "Shutting down MQTT handler")
        
        # Stop health reporting
        self.health_reporter.stop_health_reporting()
        
        # Clear callbacks
        self._event_callbacks.clear()
        
        # Shutdown MQTT connection
        super().shutdown()
        
        self._safe_log('info', "MQTT handler shutdown complete")