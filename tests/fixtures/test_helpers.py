#!/usr/bin/env python3.12
"""Test helpers for working with refactored services."""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class MockCamera:
    """Mock camera for testing."""
    ip: str
    mac: str
    name: str = "Test Camera"
    manufacturer: str = "TestCam"
    model: str = "TC-1000"
    online: bool = True
    error_count: int = 0
    last_seen: float = field(default_factory=time.time)
    rtsp_urls: Dict[str, str] = field(default_factory=dict)
    discovery_method: str = "mock"
    
    def __post_init__(self):
        if not self.rtsp_urls:
            self.rtsp_urls = {
                'main': f'rtsp://{self.ip}:554/stream1',
                'sub': f'rtsp://{self.ip}:554/stream2'
            }


@dataclass  
class MockDetection:
    """Mock fire/smoke detection for testing."""
    camera_id: str
    confidence: float
    object_type: str = "fire"
    timestamp: float = field(default_factory=time.time)
    bounding_box: List[int] = field(default_factory=lambda: [100, 100, 50, 50])
    object_id: Optional[str] = None
    
    def to_mqtt_payload(self) -> str:
        """Convert to MQTT JSON payload."""
        return json.dumps({
            'camera_id': self.camera_id,
            'confidence': self.confidence,
            'object_type': self.object_type,
            'timestamp': self.timestamp,
            'bounding_box': self.bounding_box,
            'object_id': self.object_id or f"{self.object_type}_{int(self.timestamp)}"
        })


class ServiceTestHelper:
    """Helper for testing refactored services."""
    
    @staticmethod
    def wait_for_mqtt_connection(service, timeout=10):
        """Wait for service MQTT connection."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if hasattr(service, '_mqtt_connected') and service._mqtt_connected:
                return True
            time.sleep(0.1)
        return False
    
    @staticmethod
    def get_published_messages(mock_mqtt_client, topic_filter=None):
        """Get messages published by the service."""
        messages = mock_mqtt_client.published_messages
        if topic_filter:
            messages = [m for m in messages if topic_filter in m['topic']]
        return messages
    
    @staticmethod
    def trigger_health_report(service):
        """Trigger a health report from the service."""
        if hasattr(service, 'health_reporter'):
            service.health_reporter.report_health()
            return True
        return False
    
    @staticmethod
    def simulate_shutdown(service):
        """Simulate service shutdown."""
        if hasattr(service, 'shutdown'):
            service.shutdown()
        elif hasattr(service, '_shutdown'):
            service._shutdown = True
        
        # Stop any background tasks
        for attr in ['discovery_task', 'health_check_task', 'mac_tracking_task']:
            if hasattr(service, attr):
                task = getattr(service, attr)
                if hasattr(task, 'stop'):
                    task.stop()
