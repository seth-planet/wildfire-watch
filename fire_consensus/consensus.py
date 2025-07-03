#!/usr/bin/env python3
"""Fire Consensus Service (Refactored) - Multi-camera fire detection validation.

This is a refactored version of the fire consensus service that uses the new
base classes for reduced code duplication and improved maintainability.

Key Improvements:
1. Uses MQTTService base class for connection management
2. Uses HealthReporter base class for health monitoring
3. Uses ThreadSafeService for thread management
4. Configuration management via ConfigBase
5. Cleaner separation of concerns
"""

import os
import sys
import time
import json
import socket
import threading
import logging
import hashlib
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from dotenv import load_dotenv

# Import base classes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.mqtt_service import MQTTService
from utils.health_reporter import HealthReporter
from utils.thread_manager import ThreadSafeService, SafeTimerManager
from utils.config_base import ConfigBase, ConfigSchema

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration using ConfigBase
# ─────────────────────────────────────────────────────────────
class FireConsensusConfig(ConfigBase):
    """Configuration for Fire Consensus service."""
    
    SCHEMA = {
        # Service identification
        'service_id': ConfigSchema(
            str,
            default=f"fire_consensus_{socket.gethostname()}",
            description="Unique service identifier"
        ),
        
        # Core consensus parameters
        'consensus_threshold': ConfigSchema(
            int,
            default=2,
            min=1,
            max=10,
            description="Number of cameras required for consensus"
        ),
        'single_camera_trigger': ConfigSchema(
            bool,
            default=False,
            description="Allow single camera to trigger (overrides threshold)"
        ),
        'detection_window': ConfigSchema(
            float,
            default=30.0,
            min=10.0,
            max=300.0,
            description="Time window for detection history (seconds)"
        ),
        'cooldown_period': ConfigSchema(
            float,
            default=300.0,
            min=0.0,
            max=3600.0,
            description="Minimum time between triggers (seconds)"
        ),
        
        # Detection filtering
        'min_confidence': ConfigSchema(
            float,
            default=0.7,
            min=0.0,
            max=1.0,
            description="Minimum ML confidence score"
        ),
        'min_area_ratio': ConfigSchema(
            float,
            default=0.001,
            min=0.0,
            max=1.0,
            description="Minimum fire area as fraction of frame"
        ),
        'max_area_ratio': ConfigSchema(
            float,
            default=0.8,
            min=0.0,
            max=1.0,
            description="Maximum fire area as fraction of frame"
        ),
        
        # Growth analysis
        'area_increase_ratio': ConfigSchema(
            float,
            default=1.2,
            min=1.0,
            max=5.0,
            description="Required growth ratio (1.2 = 20% growth)"
        ),
        'moving_average_window': ConfigSchema(
            int,
            default=3,
            min=1,
            max=10,
            description="Detections for moving average smoothing"
        ),
        
        # System health
        'camera_timeout': ConfigSchema(
            float,
            default=60.0,
            min=10.0,
            max=600.0,
            description="Seconds before marking camera offline"
        ),
        'health_interval': ConfigSchema(
            int,
            default=30,
            min=10,
            max=300,
            description="Health report frequency (seconds)"
        ),
        'memory_cleanup_interval': ConfigSchema(
            int,
            default=300,
            min=60,
            max=3600,
            description="State cleanup frequency (seconds)"
        ),
        
        # Zone mapping
        'zone_activation': ConfigSchema(
            bool,
            default=False,
            description="Enable zone-based sprinkler control"
        ),
        'zone_mapping': ConfigSchema(
            dict,
            default={},
            description="Camera ID to zone list mapping"
        ),
        
        # MQTT settings
        'mqtt_broker': ConfigSchema(str, required=True, default='mqtt_broker'),
        'mqtt_port': ConfigSchema(int, default=1883, min=1, max=65535),
        'mqtt_tls': ConfigSchema(bool, default=False),
        'mqtt_username': ConfigSchema(str, default=''),
        'mqtt_password': ConfigSchema(str, default=''),
        'topic_prefix': ConfigSchema(str, default='', description="MQTT topic prefix"),
        
        # TLS settings
        'tls_ca_path': ConfigSchema(str, default='/mnt/data/certs/ca.crt'),
    }
    
    def __init__(self):
        super().__init__()
        
    def validate(self):
        """Validate fire consensus configuration."""
        # If single camera trigger enabled, consensus threshold doesn't matter
        if self.single_camera_trigger and self.consensus_threshold > 1:
            logging.warning("single_camera_trigger=true overrides consensus_threshold")
            
        # Ensure min area < max area
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("min_area_ratio must be less than max_area_ratio")
            
        # Detection window must be long enough for moving average
        min_window = self.moving_average_window * 2
        if self.detection_window < min_window:
            logging.warning(f"detection_window ({self.detection_window}s) may be too short "
                          f"for moving_average_window ({self.moving_average_window})")


# ─────────────────────────────────────────────────────────────
# Data Models (reuse from original)
# ─────────────────────────────────────────────────────────────
class Detection:
    """Single fire detection from a camera."""
    def __init__(self, confidence: float, area: float, object_id: str = "0"):
        self.confidence = confidence
        self.area = area
        self.object_id = object_id
        self.timestamp = time.time()

class CameraState:
    """State tracking for a single camera."""
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.detections: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_seen = time.time()
        self.is_online = True
        self.total_detections = 0
        self.last_detection_time = 0


# ─────────────────────────────────────────────────────────────
# Consensus Health Reporter
# ─────────────────────────────────────────────────────────────
class ConsensusHealthReporter(HealthReporter):
    """Health reporter for fire consensus service."""
    
    def __init__(self, consensus):
        self.consensus = consensus
        super().__init__(consensus, consensus.config.health_interval)
        
    def get_service_health(self) -> Dict[str, any]:
        """Get consensus-specific health metrics."""
        with self.consensus.lock:
            cameras = list(self.consensus.cameras.values())
            online_cameras = [c for c in cameras if c.is_online]
            
            # Calculate detection rates
            now = time.time()
            recent_detections = sum(
                1 for c in cameras 
                if now - c.last_detection_time < 60
            )
            
            # Get consensus state
            consensus_state = "idle"
            if self.consensus.last_trigger_time > 0:
                time_since_trigger = now - self.consensus.last_trigger_time
                if time_since_trigger < self.consensus.config.cooldown_period:
                    consensus_state = "cooldown"
                    
            health = {
                'consensus_state': consensus_state,
                'trigger_count': self.consensus.trigger_count,
                'last_trigger_ago': int(now - self.consensus.last_trigger_time) if self.consensus.last_trigger_time > 0 else -1,
                'cameras_total': len(cameras),
                'cameras_online': len(online_cameras),
                'cameras_detecting': recent_detections,
                'consensus_threshold': self.consensus.config.consensus_threshold,
                'single_camera_mode': self.consensus.config.single_camera_trigger,
                'recent_consensus_events': len(self.consensus.consensus_events),
            }
            
            # Add per-camera stats
            camera_stats = {}
            for cam in cameras:
                total_objects = sum(len(detections) for detections in cam.detections.values())
                camera_stats[cam.camera_id] = {
                    'online': cam.is_online,
                    'total_detections': cam.total_detections,
                    'active_objects': total_objects,
                    'last_seen_ago': int(now - cam.last_seen)
                }
            health['cameras'] = camera_stats
            
        return health


# ─────────────────────────────────────────────────────────────
# Refactored Fire Consensus Service
# ─────────────────────────────────────────────────────────────
class FireConsensus(MQTTService, ThreadSafeService):
    """Refactored fire consensus service using base classes.
    
    This implementation reduces code duplication by:
    1. Using MQTTService for all MQTT handling
    2. Using ThreadSafeService for thread management
    3. Using HealthReporter for health monitoring
    4. Using SafeTimerManager for timer management
    """
    
    def __init__(self):
        # Load configuration
        self.config = FireConsensusConfig()
        
        # Initialize base classes
        MQTTService.__init__(self, "fire_consensus", self.config)
        ThreadSafeService.__init__(self, "fire_consensus", logging.getLogger(__name__))
        
        # Core state
        self.cameras: Dict[str, CameraState] = {}
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.consensus_events = deque(maxlen=1000)
        self.lock = threading.RLock()
        
        # Setup health reporter
        self.health_reporter = ConsensusHealthReporter(self)
        
        # Setup MQTT with subscriptions
        subscriptions = [
            "fire/detection",
            "fire/detection/+",
            "frigate/events",
            "system/camera_telemetry"
        ]
        
        self.setup_mqtt(
            on_connect=self._on_connect,
            on_message=self._on_message,
            subscriptions=subscriptions
        )
        
        # Enable offline queue for resilience
        self.enable_offline_queue(max_size=100)
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info(f"Fire Consensus initialized: {self.config.service_id}")
        
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        self.logger.info("MQTT connected, ready for fire detection messages")
        
    def _on_message(self, topic: str, payload: any):
        """Handle incoming MQTT messages."""
        try:
            # Route messages based on topic
            if topic.startswith("fire/detection"):
                self._handle_fire_detection(topic, payload)
            elif topic == "frigate/events":
                self._handle_frigate_event(payload)
            elif topic == "system/camera_telemetry":
                self._handle_camera_telemetry(payload)
                
        except Exception as e:
            self.logger.error(f"Error processing message on {topic}: {e}", exc_info=True)
            
    def _start_background_tasks(self):
        """Start background tasks using timer manager."""
        # Memory cleanup task
        self.timer_manager.schedule(
            "memory_cleanup",
            self._cleanup_old_data,
            self.config.memory_cleanup_interval
        )
        
        # Start health reporting
        self.health_reporter.start_health_reporting()
        
    def _handle_fire_detection(self, topic: str, payload: Dict):
        """Process fire detection from cameras."""
        try:
            # Extract camera ID
            parts = topic.split('/')
            camera_id = parts[2] if len(parts) > 2 else payload.get('camera_id', 'unknown')
            
            # Validate detection
            confidence = float(payload.get('confidence', 0))
            if confidence < self.config.min_confidence:
                return
                
            # Calculate area
            bbox = payload.get('bbox', [])
            if len(bbox) == 4:
                area = self._calculate_area(bbox)
                if not (self.config.min_area_ratio <= area <= self.config.max_area_ratio):
                    return
            else:
                area = 0.01  # Default area
                
            # Create detection
            object_id = str(payload.get('object_id', '0'))
            detection = Detection(confidence, area, object_id)
            
            # Add detection and check consensus
            self._add_detection(camera_id, detection)
            
        except Exception as e:
            self.logger.error(f"Error handling fire detection: {e}")
            
    def _handle_frigate_event(self, payload: Dict):
        """Process Frigate NVR events."""
        try:
            # Filter for fire/smoke events
            label = payload.get('after', {}).get('label', '')
            if label not in ['fire', 'smoke']:
                return
                
            # Extract data
            camera_id = payload.get('after', {}).get('camera', 'unknown')
            confidence = float(payload.get('after', {}).get('top_score', 0))
            
            if confidence < self.config.min_confidence:
                return
                
            # Calculate area from box
            box = payload.get('after', {}).get('box', [])
            if len(box) == 4:
                area = (box[2] - box[0]) * (box[3] - box[1])
            else:
                area = 0.01
                
            # Create detection
            object_id = str(payload.get('after', {}).get('id', '0'))
            detection = Detection(confidence, area, object_id)
            
            # Add detection
            self._add_detection(camera_id, detection)
            
        except Exception as e:
            self.logger.error(f"Error handling Frigate event: {e}")
            
    def _handle_camera_telemetry(self, payload: Dict):
        """Update camera online status from telemetry."""
        try:
            camera_id = payload.get('camera_id')
            if not camera_id:
                return
                
            with self.lock:
                if camera_id not in self.cameras:
                    self.cameras[camera_id] = CameraState(camera_id)
                    
                camera = self.cameras[camera_id]
                camera.last_seen = time.time()
                camera.is_online = True
                
        except Exception as e:
            self.logger.error(f"Error handling camera telemetry: {e}")
            
    def _add_detection(self, camera_id: str, detection: Detection):
        """Add detection and check for consensus."""
        with self.lock:
            # Ensure camera exists
            if camera_id not in self.cameras:
                self.cameras[camera_id] = CameraState(camera_id)
                
            camera = self.cameras[camera_id]
            camera.detections[detection.object_id].append(detection)
            camera.total_detections += 1
            camera.last_detection_time = time.time()
            camera.last_seen = time.time()
            camera.is_online = True
            
            self.logger.debug(f"Added detection from {camera_id}: "
                            f"confidence={detection.confidence:.2f}, "
                            f"area={detection.area:.4f}, "
                            f"object_id={detection.object_id}")
            
        # Check consensus
        self._check_consensus()
        
    def _check_consensus(self):
        """Check if consensus conditions are met."""
        with self.lock:
            # Check cooldown
            if time.time() - self.last_trigger_time < self.config.cooldown_period:
                return
                
            # Get growing fires from all cameras
            cameras_with_fires = []
            fire_locations = []
            
            for camera_id, camera in self.cameras.items():
                if not camera.is_online:
                    continue
                    
                growing_fires = self.get_growing_fires(camera_id)
                if growing_fires:
                    cameras_with_fires.append(camera_id)
                    fire_locations.extend([
                        (camera_id, obj_id) for obj_id in growing_fires
                    ])
                    
            # Check consensus threshold
            consensus_met = False
            if self.config.single_camera_trigger and len(cameras_with_fires) >= 1:
                consensus_met = True
            elif len(cameras_with_fires) >= self.config.consensus_threshold:
                consensus_met = True
                
            if consensus_met:
                self.logger.warning(f"CONSENSUS REACHED! {len(cameras_with_fires)} cameras "
                                  f"detecting growing fires: {cameras_with_fires}")
                self._trigger_fire_response(cameras_with_fires, fire_locations)
                
                # Record consensus event
                self.consensus_events.append({
                    'timestamp': time.time(),
                    'cameras': cameras_with_fires,
                    'fire_count': len(fire_locations)
                })
                
    def get_growing_fires(self, camera_id: str) -> List[str]:
        """Get list of growing fire object IDs for a camera."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return []
            
        growing_fires = []
        current_time = time.time()
        
        for object_id, detections in camera.detections.items():
            # Filter recent detections
            recent = [d for d in detections 
                     if current_time - d.timestamp <= self.config.detection_window]
            
            if len(recent) < self.config.moving_average_window * 2:
                continue
                
            # Calculate moving averages
            areas = [d.area for d in recent]
            early_avg = np.mean(areas[:self.config.moving_average_window])
            recent_avg = np.mean(areas[-self.config.moving_average_window:])
            
            # Check growth
            if recent_avg >= early_avg * self.config.area_increase_ratio:
                growing_fires.append(object_id)
                
        return growing_fires
        
    def _trigger_fire_response(self, cameras: List[str], fire_locations: List[Tuple[str, str]]):
        """Send fire suppression trigger command."""
        self.last_trigger_time = time.time()
        self.trigger_count += 1
        
        # Build trigger payload
        payload = {
            'action': 'trigger',
            'confidence': 'high',
            'timestamp': self.last_trigger_time,
            'trigger_count': self.trigger_count,
            'consensus_cameras': cameras,
            'fire_locations': fire_locations,
            'service_id': self.config.service_id
        }
        
        # Add zone information if enabled
        if self.config.zone_activation:
            zones = set()
            for camera_id in cameras:
                camera_zones = self.config.zone_mapping.get(camera_id, [])
                zones.update(camera_zones)
            payload['zones'] = list(zones)
            
        # Publish with QoS 2 (exactly once)
        self.publish_message("fire/trigger", payload, qos=2)
        
        self.logger.critical(f"FIRE TRIGGER SENT! Cameras: {cameras}")
        
    def _cleanup_old_data(self):
        """Periodic cleanup of old detection data."""
        with self.lock:
            current_time = time.time()
            
            for camera in self.cameras.values():
                # Clean old detections
                for object_id, detections in list(camera.detections.items()):
                    # Remove old detections
                    while detections and current_time - detections[0].timestamp > self.config.detection_window * 2:
                        detections.popleft()
                        
                    # Remove empty deques
                    if not detections:
                        del camera.detections[object_id]
                        
                # Update online status
                if current_time - camera.last_seen > self.config.camera_timeout:
                    camera.is_online = False
                    
        # Reschedule
        self.timer_manager.schedule(
            "memory_cleanup",
            self._cleanup_old_data,
            self.config.memory_cleanup_interval
        )
        
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate normalized area from bounding box."""
        # Assuming 1920x1080 resolution (should be configurable)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (width * height) / (1920 * 1080)
        
    def cleanup(self):
        """Clean shutdown of service."""
        self.logger.info("Shutting down Fire Consensus")
        
        # Stop health reporting
        if hasattr(self, 'health_reporter'):
            self.health_reporter.stop_health_reporting()
            
        # Cancel all timers
        self.timer_manager.cancel_all()
        
        # Shutdown base services
        ThreadSafeService.shutdown(self)
        MQTTService.shutdown(self)
        
        self.logger.info("Fire Consensus shutdown complete")
        

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────


def main():
    """Main entry point for fire consensus service."""
    # Setup logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    try:
        consensus = FireConsensus()
        
        # Wait for shutdown
        consensus.wait_for_shutdown()
        
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'consensus' in locals():
            consensus.cleanup()
            

if __name__ == "__main__":
    main()