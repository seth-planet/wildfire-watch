#!/usr/bin/env python3.12
"""Data models for the Web Interface service.

This module defines Pydantic models for data structures used throughout
the web interface, including MQTT events, system status, and service health.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class EventType(str, Enum):
    """Types of events in the system."""
    FIRE = "fire"
    GPIO = "gpio"
    HEALTH = "health"
    CAMERA = "camera"
    DETECTION = "detection"
    TELEMETRY = "telemetry"
    OTHER = "other"


class ServiceStatus(str, Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MQTTEvent(BaseModel):
    """Represents a single MQTT event."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime = Field(..., description="When the event occurred")
    topic: str = Field(..., description="MQTT topic")
    payload: Any = Field(..., description="Event payload")
    event_type: EventType = Field(..., description="Categorized event type")
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display in UI.
        
        Returns:
            Dictionary with formatted values for display
        """
        return {
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'topic': self.topic,
            'type': self.event_type.value,
            'payload': self._format_payload()
        }
        
    def _format_payload(self) -> Any:
        """Format payload for display."""
        if isinstance(self.payload, dict):
            # Truncate large dictionaries
            if len(str(self.payload)) > 500:
                return {k: v for k, v in list(self.payload.items())[:5]}
        elif isinstance(self.payload, bytes):
            # Convert bytes to string for JSON serialization
            try:
                return self.payload.decode('utf-8')
            except UnicodeDecodeError:
                return str(self.payload)
        return self.payload


class ServiceHealth(BaseModel):
    """Health status of a single service."""
    name: str = Field(..., description="Service name")
    status: ServiceStatus = Field(..., description="Current status")
    last_seen: datetime = Field(..., description="Last health report time")
    uptime: float = Field(0.0, description="Uptime in hours")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")
    
    @property
    def is_stale(self) -> bool:
        """Check if health data is stale (>2 minutes old)."""
        return (datetime.utcnow() - self.last_seen).seconds > 120
        
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'name': self.name,
            'status': self.status.value,
            'last_seen': self.last_seen.strftime('%H:%M:%S'),
            'uptime_hours': round(self.uptime, 1),
            'stale': self.is_stale,
            'cpu_percent': self.details.get('resources', {}).get('cpu_percent', 0),
            'memory_mb': round(self.details.get('resources', {}).get('memory_mb', 0), 1)
        }


class GPIOState(BaseModel):
    """State of a GPIO pin."""
    pin_id: str = Field(..., description="Pin identifier")
    name: str = Field(..., description="Human-readable pin name")
    state: bool = Field(..., description="Current state (HIGH/LOW)")
    last_change: datetime = Field(..., description="Last state change time")
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'id': self.pin_id,
            'name': self.name,
            'state': 'ON' if self.state else 'OFF',
            'state_bool': self.state,
            'last_change': self.last_change.strftime('%H:%M:%S')
        }


class CameraInfo(BaseModel):
    """Information about a discovered camera."""
    camera_id: str = Field(..., description="Camera identifier")
    name: str = Field(..., description="Camera name")
    ip_address: Optional[str] = Field(None, description="Camera IP address")
    last_seen: datetime = Field(..., description="Last discovery time")
    detection_count: int = Field(0, description="Number of fire detections")
    is_active: bool = Field(True, description="Whether camera is currently active")
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'id': self.camera_id,
            'name': self.name,
            'ip': self.ip_address or 'Unknown',
            'last_seen': self.last_seen.strftime('%H:%M:%S'),
            'detections': self.detection_count,
            'active': self.is_active
        }


class SystemStatus(BaseModel):
    """Overall system status."""
    fire_active: bool = Field(..., description="Whether fire suppression is active")
    last_fire_trigger: Optional[datetime] = Field(None, description="Last fire trigger time")
    consensus_count: int = Field(0, description="Number of cameras in consensus")
    service_count: int = Field(0, description="Total number of services")
    healthy_services: int = Field(0, description="Number of healthy services")
    camera_count: int = Field(0, description="Total number of cameras")
    active_cameras: int = Field(0, description="Number of active cameras")
    gpio_states: Dict[str, bool] = Field(default_factory=dict, description="Current GPIO states")
    mqtt_connected: bool = Field(False, description="MQTT connection status")
    buffer_size: int = Field(0, description="Current event buffer size")
    
    @property
    def system_health_percentage(self) -> int:
        """Calculate overall system health percentage."""
        if self.service_count == 0:
            return 0
        return int((self.healthy_services / self.service_count) * 100)
        
    def to_display_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'fire_active': self.fire_active,
            'fire_status': 'ACTIVE' if self.fire_active else 'IDLE',
            'last_trigger': self.last_fire_trigger.strftime('%Y-%m-%d %H:%M:%S UTC') 
                          if self.last_fire_trigger else 'Never',
            'consensus_count': self.consensus_count,
            'services': {
                'total': self.service_count,
                'healthy': self.healthy_services,
                'percentage': self.system_health_percentage
            },
            'cameras': {
                'total': self.camera_count,
                'active': self.active_cameras
            },
            'mqtt_connected': self.mqtt_connected,
            'buffer_usage': self.buffer_size
        }


class DebugAction(BaseModel):
    """Debug action request."""
    action: str = Field(..., description="Action to perform")
    token: str = Field(..., description="Debug token for authentication")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")


class APIResponse(BaseModel):
    """Standard API response."""
    success: bool = Field(..., description="Whether the request succeeded")
    message: str = Field(..., description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class EventFilter(BaseModel):
    """Filter criteria for events."""
    event_type: Optional[EventType] = Field(None, description="Filter by event type")
    topic_pattern: Optional[str] = Field(None, description="Filter by topic pattern")
    start_time: Optional[datetime] = Field(None, description="Events after this time")
    end_time: Optional[datetime] = Field(None, description="Events before this time")
    limit: int = Field(100, ge=1, le=1000, description="Maximum events to return")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field("ok", description="Health status")
    service: str = Field("web_interface", description="Service name")
    version: str = Field(..., description="Service version")
    mqtt_connected: bool = Field(..., description="MQTT connection status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")