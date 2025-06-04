#!/usr/bin/env python3
"""
Fire Consensus Service - Wildfire Watch
Robust multi-camera consensus algorithm with false positive reduction
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

import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
class Config:
    # MQTT Settings
    MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt_broker")
    MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
    MQTT_TLS = os.getenv("MQTT_TLS", "false").lower() == "true"
    TLS_CA_PATH = os.getenv("TLS_CA_PATH", "/mnt/data/certs/ca.crt")
    
    # Consensus Parameters
    CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", "2"))
    DETECTION_WINDOW = float(os.getenv("CAMERA_WINDOW", "10"))
    INCREASE_COUNT = int(os.getenv("INCREASE_COUNT", "3"))
    COOLDOWN_PERIOD = float(os.getenv("DETECTION_COOLDOWN", "30"))
    
    # Advanced Detection Parameters
    MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.7"))
    MIN_AREA_RATIO = float(os.getenv("MIN_AREA_RATIO", "0.001"))
    MAX_AREA_RATIO = float(os.getenv("MAX_AREA_RATIO", "0.5"))
    AREA_INCREASE_RATIO = float(os.getenv("AREA_INCREASE_RATIO", "1.2"))
    MOVING_AVERAGE_WINDOW = int(os.getenv("MOVING_AVERAGE_WINDOW", "3"))
    
    # Timing and Health
    HEALTH_INTERVAL = int(os.getenv("TELEMETRY_INTERVAL", "60"))
    MEMORY_CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "300"))
    CAMERA_TIMEOUT = float(os.getenv("CAMERA_TIMEOUT", "120"))
    
    # Node Identity
    NODE_ID = os.getenv("BALENA_DEVICE_UUID", socket.gethostname())
    SERVICE_ID = f"consensus-{NODE_ID}"
    
    # Topics
    TOPIC_DETECTION = os.getenv("DETECTION_TOPIC", "fire/detection")
    TOPIC_TRIGGER = os.getenv("TRIGGER_TOPIC", "fire/trigger")
    TOPIC_HEALTH = os.getenv("CONSENSUS_HEALTH_TOPIC", "system/consensus_telemetry")
    TOPIC_FRIGATE = os.getenv("FRIGATE_EVENTS_TOPIC", "frigate/events")
    TOPIC_CAMERA_TELEMETRY = os.getenv("CAMERA_TELEMETRY_TOPIC", "system/camera_telemetry")
    
    # Resilience Settings
    MQTT_RECONNECT_DELAY = 5
    MQTT_KEEPALIVE = 60
    MAX_RECONNECT_ATTEMPTS = -1  # Infinite

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────
class Detection:
    """Represents a single fire detection event"""
    def __init__(self, camera_id: str, timestamp: float, confidence: float,
                 area: float, bbox: List[float], object_id: Optional[str] = None):
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.confidence = confidence
        self.area = area
        self.bbox = bbox
        self.object_id = object_id or self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for detection"""
        data = f"{self.camera_id}-{self.timestamp}-{self.bbox}"
        return hashlib.md5(data.encode()).hexdigest()[:8]
    
    def to_dict(self) -> dict:
        return {
            'camera_id': self.camera_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'area': self.area,
            'bbox': self.bbox,
            'object_id': self.object_id
        }

class CameraState:
    """Tracks state for a single camera"""
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.last_seen = time.time()
        self.last_telemetry = time.time()
        self.detections = deque(maxlen=100)  # Recent detections
        self.fire_objects = {}  # object_id -> list of detections
        self.last_trigger = 0
        self.false_positive_count = 0
        self.true_positive_count = 0
        
    def add_detection(self, detection: Detection):
        """Add a detection and update object tracking"""
        self.detections.append(detection)
        self.last_seen = detection.timestamp
        
        # Track by object ID
        if detection.object_id not in self.fire_objects:
            self.fire_objects[detection.object_id] = []
        self.fire_objects[detection.object_id].append(detection)
        
        # Cleanup old objects
        self._cleanup_old_objects(detection.timestamp)
    
    def _cleanup_old_objects(self, current_time: float):
        """Remove stale object tracks"""
        stale_objects = []
        for obj_id, detections in self.fire_objects.items():
            if detections and current_time - detections[-1].timestamp > Config.DETECTION_WINDOW * 2:
                stale_objects.append(obj_id)
        
        for obj_id in stale_objects:
            del self.fire_objects[obj_id]
    
    def get_growing_fires(self, current_time: float) -> List[str]:
        """Get object IDs that show fire growth pattern using moving averages"""
        growing_fires = []
        
        for obj_id, detections in self.fire_objects.items():
            # Need enough detections for meaningful moving average comparison
            min_detections = Config.MOVING_AVERAGE_WINDOW * 2
            if len(detections) < min_detections:
                continue
            
            # Get recent detections within window
            recent = [d for d in detections
                     if current_time - d.timestamp <= Config.DETECTION_WINDOW]
            
            if len(recent) >= min_detections:
                # Calculate moving averages to reduce noise
                areas = [d.area for d in recent]
                moving_averages = self._calculate_moving_averages(areas, Config.MOVING_AVERAGE_WINDOW)
                
                # Check if moving averages show growth trend
                if self._check_growth_trend(moving_averages, Config.AREA_INCREASE_RATIO):
                    growing_fires.append(obj_id)
        
        return growing_fires
    
    def _calculate_moving_averages(self, areas: List[float], window_size: int) -> List[float]:
        """Calculate moving averages for area values"""
        if len(areas) < window_size:
            return []
        
        moving_averages = []
        for i in range(window_size - 1, len(areas)):
            # Calculate average of current window
            window_values = areas[i - window_size + 1:i + 1]
            avg = sum(window_values) / len(window_values)
            moving_averages.append(avg)
        
        return moving_averages
    
    def _check_growth_trend(self, moving_averages: List[float], growth_ratio: float) -> bool:
        """Check if moving averages show consistent growth pattern"""
        if len(moving_averages) < 2:
            return False
        
        # Check for overall growth trend between first and last averages
        first_avg = moving_averages[0]
        last_avg = moving_averages[-1]
        
        # Require significant overall growth
        if last_avg < first_avg * growth_ratio:
            return False
        
        # Check that trend is generally upward (allow some fluctuation)
        # Count how many moving average transitions show growth
        growth_transitions = 0
        total_transitions = len(moving_averages) - 1
        
        for i in range(1, len(moving_averages)):
            if moving_averages[i] >= moving_averages[i-1] * 0.95:  # Allow 5% tolerance for noise
                growth_transitions += 1
        
        # Require at least 70% of transitions to show growth (or minimal shrinkage)
        growth_percentage = growth_transitions / total_transitions if total_transitions > 0 else 0
        return growth_percentage >= 0.7
    
    def is_online(self, current_time: float) -> bool:
        """Check if camera is online based on telemetry"""
        return current_time - self.last_telemetry < Config.CAMERA_TIMEOUT

# ─────────────────────────────────────────────────────────────
# Fire Consensus Engine
# ─────────────────────────────────────────────────────────────
class FireConsensus:
    def __init__(self):
        self.config = Config()
        self.cameras: Dict[str, CameraState] = {}
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.consensus_events = deque(maxlen=1000)
        self.lock = threading.RLock()
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        self._setup_mqtt()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Fire Consensus initialized: {self.config.SERVICE_ID}")
    
    def _setup_mqtt(self):
        """Setup MQTT client with resilient connection"""
        self.mqtt_client = mqtt.Client(
            client_id=self.config.SERVICE_ID,
            clean_session=False  # Preserve subscriptions across reconnects
        )
        
        # Set callbacks
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        
        # Configure TLS if enabled
        if self.config.MQTT_TLS:
            import ssl
            self.mqtt_client.tls_set(
                ca_certs=self.config.TLS_CA_PATH,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS
            )
        
        # Set LWT
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'fire_consensus',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        self.mqtt_client.will_set(lwt_topic, lwt_payload, qos=1, retain=True)
        
        # Connect with retry
        self._mqtt_connect_with_retry()
    
    def _mqtt_connect_with_retry(self):
        """Connect to MQTT with exponential backoff retry"""
        attempt = 0
        while True:
            try:
                port = 8883 if self.config.MQTT_TLS else 1883
                self.mqtt_client.connect(
                    self.config.MQTT_BROKER,
                    port,
                    keepalive=self.config.MQTT_KEEPALIVE
                )
                self.mqtt_client.loop_start()
                logger.info(f"MQTT connection initiated to {self.config.MQTT_BROKER}:{port}")
                break
            except Exception as e:
                attempt += 1
                delay = min(self.config.MQTT_RECONNECT_DELAY * (2 ** attempt), 300)
                logger.error(f"MQTT connection failed (attempt {attempt}): {e}")
                logger.info(f"Retrying in {delay}s...")
                time.sleep(delay)
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
            logger.info("MQTT connected successfully")
            
            # Subscribe to topics
            topics = [
                (self.config.TOPIC_DETECTION, 1),
                (self.config.TOPIC_FRIGATE, 1),
                (self.config.TOPIC_CAMERA_TELEMETRY, 0),
                (f"{self.config.TOPIC_DETECTION}/+", 1),  # Wildcard for camera-specific
            ]
            
            for topic, qos in topics:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to {topic}")
            
            # Publish online status
            self._publish_health()
        else:
            self.mqtt_connected = False
            logger.error(f"MQTT connection failed with code {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.mqtt_connected = False
        logger.warning(f"MQTT disconnected with code {rc}")
        
        if rc != 0:
            # Unexpected disconnect, will auto-reconnect
            logger.info("Unexpected disconnect, auto-reconnect enabled")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        try:
            # Route message to appropriate handler
            if msg.topic.startswith(self.config.TOPIC_DETECTION):
                self._process_detection(msg)
            elif msg.topic == self.config.TOPIC_FRIGATE:
                self._process_frigate_event(msg)
            elif msg.topic == self.config.TOPIC_CAMERA_TELEMETRY:
                self._process_camera_telemetry(msg)
        except Exception as e:
            logger.error(f"Error processing message on {msg.topic}: {e}")
    
    def _process_detection(self, msg):
        """Process fire detection messages"""
        try:
            data = json.loads(msg.payload)
            
            # Extract detection info
            camera_id = data.get('camera_id')
            confidence = float(data.get('confidence', 0))
            bbox = data.get('bounding_box', [])
            timestamp = data.get('timestamp', time.time())
            object_id = data.get('object_id')  # Optional object ID for tracking
            
            if not camera_id or not bbox or len(bbox) != 4:
                return
            
            # Calculate area
            area = self._calculate_area(bbox)
            
            # Validate detection
            if not self._validate_detection(confidence, area):
                logger.debug(f"Detection from {camera_id} failed validation")
                return
            
            # Create detection object
            detection = Detection(
                camera_id=camera_id,
                timestamp=timestamp,
                confidence=confidence,
                area=area,
                bbox=bbox,
                object_id=object_id
            )
            
            # Process detection
            self._add_detection(detection)
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
    
    def _process_frigate_event(self, msg):
        """Process Frigate NVR events"""
        try:
            event = json.loads(msg.payload)
            
            # Only process fire events
            if event.get('after', {}).get('label') != 'fire':
                return
            
            event_type = event.get('type')
            if event_type not in ('new', 'update'):
                return
            
            # Extract event data
            after = event.get('after', {})
            camera_id = after.get('camera')
            object_id = after.get('id')
            confidence = after.get('current_score', 0)
            bbox = after.get('box', [])
            
            if not camera_id or not bbox:
                return
            
            # Calculate area
            area = self._calculate_area(bbox)
            
            # Validate
            if not self._validate_detection(confidence, area):
                return
            
            # Create detection
            detection = Detection(
                camera_id=camera_id,
                timestamp=time.time(),
                confidence=confidence,
                area=area,
                bbox=bbox,
                object_id=object_id
            )
            
            # Process detection
            self._add_detection(detection)
            
        except Exception as e:
            logger.error(f"Error processing Frigate event: {e}")
    
    def _process_camera_telemetry(self, msg):
        """Process camera telemetry/heartbeat messages"""
        try:
            data = json.loads(msg.payload)
            camera_id = data.get('camera_id')
            
            if camera_id:
                with self.lock:
                    if camera_id not in self.cameras:
                        self.cameras[camera_id] = CameraState(camera_id)
                    self.cameras[camera_id].last_telemetry = time.time()
                    
        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
    
    def _calculate_area(self, bbox: List[float]) -> float:
        """Calculate normalized area from bounding box
        
        Handles two formats:
        1. Direct detections: [x, y, width, height] where values are normalized (0-1)
        2. Frigate events: [x1, y1, x2, y2] where values are pixel coordinates
        """
        if len(bbox) != 4:
            return 0
        
        # Check for invalid values (NaN, inf, negative)
        try:
            if any(not isinstance(coord, (int, float)) or 
                   math.isnan(coord) or math.isinf(coord) or coord < 0 
                   for coord in bbox):
                return 0
        except (TypeError, ValueError):
            return 0
        
        # Detect format by checking if values look like pixel coordinates (>1)
        if any(coord > 1.0 for coord in bbox):
            # Frigate format: [x1, y1, x2, y2] pixel coordinates
            x1, y1, x2, y2 = bbox
            width_pixels = abs(x2 - x1)
            height_pixels = abs(y2 - y1)
            
            # Normalize by assuming reasonable image size (this is imperfect but necessary)
            # Most IP cameras are at least 1920x1080, we'll use conservative estimate
            estimated_image_area = 1920 * 1080
            pixel_area = width_pixels * height_pixels
            area = pixel_area / estimated_image_area
            
            # Check for invalid result
            if math.isnan(area) or math.isinf(area):
                return 0
            return area
        else:
            # Direct detection format: [x, y, width, height] normalized coordinates
            width = bbox[2]
            height = bbox[3]
            area = width * height
            
            # Check for invalid result
            if math.isnan(area) or math.isinf(area):
                return 0
            return area
    
    def _validate_detection(self, confidence: float, area: float) -> bool:
        """Validate detection meets minimum criteria"""
        # Check confidence threshold
        if confidence < self.config.MIN_CONFIDENCE:
            return False
        
        # Check area bounds (filter out too small or too large)
        if area < self.config.MIN_AREA_RATIO or area > self.config.MAX_AREA_RATIO:
            return False
        
        return True
    
    def _add_detection(self, detection: Detection):
        """Add detection and check for consensus"""
        with self.lock:
            # Ensure camera state exists
            if detection.camera_id not in self.cameras:
                self.cameras[detection.camera_id] = CameraState(detection.camera_id)
            
            # Add detection to camera
            camera = self.cameras[detection.camera_id]
            camera.add_detection(detection)
            
            # Check for consensus
            self._check_consensus()
    
    def _check_consensus(self):
        """Check if fire consensus criteria are met"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_trigger_time < self.config.COOLDOWN_PERIOD:
            return
        
        # Get cameras with growing fires
        cameras_with_fire = []
        fire_details = {}
        
        with self.lock:
            for camera_id, camera in self.cameras.items():
                # Skip offline cameras
                if not camera.is_online(current_time):
                    continue
                
                # Get growing fires for this camera
                growing_fires = camera.get_growing_fires(current_time)
                
                if growing_fires:
                    cameras_with_fire.append(camera_id)
                    fire_details[camera_id] = {
                        'objects': growing_fires,
                        'detections': len(camera.detections),
                        'confidence': np.mean([d.confidence for d in camera.detections
                                              if current_time - d.timestamp < self.config.DETECTION_WINDOW])
                    }
        
        # Check if consensus threshold is met
        if len(cameras_with_fire) >= self.config.CONSENSUS_THRESHOLD:
            self._trigger_fire_response(cameras_with_fire, fire_details)
    
    def _trigger_fire_response(self, cameras: List[str], details: dict):
        """Trigger fire response system"""
        with self.lock:
            current_time = time.time()
            self.last_trigger_time = current_time
            self.trigger_count += 1
            
            # Create trigger payload
            payload = {
                'node_id': self.config.NODE_ID,
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'trigger_number': self.trigger_count,
                'consensus_cameras': cameras,
                'camera_count': len(cameras),
                'details': details,
                'confidence': np.mean([d['confidence'] for d in details.values()])
            }
            
            # Log consensus event
            self.consensus_events.append({
                'timestamp': current_time,
                'cameras': cameras,
                'triggered': True
            })
            
            # Publish trigger
            try:
                self.mqtt_client.publish(
                    self.config.TOPIC_TRIGGER,
                    json.dumps(payload),
                    qos=2,  # Exactly once delivery
                    retain=False
                )
                logger.warning(f"FIRE CONSENSUS REACHED - Triggered response! Cameras: {cameras}")
            except Exception as e:
                logger.error(f"Failed to publish fire trigger: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Health reporting
        threading.Timer(
            self.config.HEALTH_INTERVAL,
            self._periodic_health_report
        ).start()
        
        # Memory cleanup
        threading.Timer(
            self.config.MEMORY_CLEANUP_INTERVAL,
            self._periodic_cleanup
        ).start()
    
    def _periodic_health_report(self):
        """Periodically publish health status"""
        try:
            self._publish_health()
        except Exception as e:
            logger.error(f"Error in health report: {e}")
        
        # Reschedule
        threading.Timer(
            self.config.HEALTH_INTERVAL,
            self._periodic_health_report
        ).start()
    
    def _periodic_cleanup(self):
        """Periodically clean up old data"""
        try:
            current_time = time.time()
            removed_cameras = []
            
            with self.lock:
                # Remove very stale cameras
                for camera_id, camera in list(self.cameras.items()):
                    if current_time - camera.last_seen > self.config.CAMERA_TIMEOUT * 2:
                        removed_cameras.append(camera_id)
                        del self.cameras[camera_id]
            
            if removed_cameras:
                logger.info(f"Cleaned up stale cameras: {removed_cameras}")
                
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
        
        # Reschedule
        threading.Timer(
            self.config.MEMORY_CLEANUP_INTERVAL,
            self._periodic_cleanup
        ).start()
    
    def _publish_health(self):
        """Publish health status"""
        current_time = time.time()
        
        with self.lock:
            # Count camera states
            online_cameras = [
                cid for cid, cam in self.cameras.items()
                if cam.is_online(current_time)
            ]
            
            cameras_with_detections = [
                cid for cid, cam in self.cameras.items()
                if cam.detections and current_time - cam.detections[-1].timestamp < self.config.DETECTION_WINDOW
            ]
            
            # Calculate consensus stats
            recent_events = [
                e for e in self.consensus_events
                if current_time - e['timestamp'] < 3600  # Last hour
            ]
            
            payload = {
                'node_id': self.config.NODE_ID,
                'service': 'fire_consensus',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'status': 'online' if self.mqtt_connected else 'degraded',
                'config': {
                    'threshold': self.config.CONSENSUS_THRESHOLD,
                    'window_seconds': self.config.DETECTION_WINDOW,
                    'cooldown_seconds': self.config.COOLDOWN_PERIOD,
                    'min_confidence': self.config.MIN_CONFIDENCE,
                },
                'stats': {
                    'total_cameras': len(self.cameras),
                    'online_cameras': len(online_cameras),
                    'active_detections': len(cameras_with_detections),
                    'total_triggers': self.trigger_count,
                    'last_trigger': self.last_trigger_time,
                    'recent_consensus_events': len(recent_events),
                },
                'cameras': {
                    'online': online_cameras,
                    'detecting': cameras_with_detections,
                }
            }
        
        try:
            self.mqtt_client.publish(
                self.config.TOPIC_HEALTH,
                json.dumps(payload),
                qos=1,
                retain=True
            )
        except Exception as e:
            logger.error(f"Failed to publish health: {e}")
    
    def run(self):
        """Main run loop"""
        logger.info("Fire Consensus Service started")
        
        try:
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        logger.info("Cleaning up Fire Consensus Service")
        
        # Publish offline status
        try:
            lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
            lwt_payload = json.dumps({
                'node_id': self.config.NODE_ID,
                'service': 'fire_consensus',
                'status': 'offline',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            })
            self.mqtt_client.publish(lwt_topic, lwt_payload, qos=1, retain=True)
        except:
            pass
        
        # Disconnect MQTT
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    consensus = FireConsensus()
    consensus.run()

if __name__ == "__main__":
    main()
