#!/usr/bin/env python3.12
"""Fire Consensus with Resilient Network Operations

This is an updated version of consensus.py that uses the ResilientMQTTClient
to prevent hanging and ensure graceful degradation.
"""

import os
import sys
import queue
import logging
import json
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mqtt_resilient import ResilientMQTTClient, ConnectionState

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Fire/smoke detection data"""
    camera_id: str
    confidence: float
    object_type: str  # 'fire' or 'smoke'
    timestamp: float
    bbox: Optional[Dict[str, float]] = None
    area: Optional[float] = None
    id: Optional[str] = None

@dataclass 
class CameraState:
    """Track camera online/offline state"""
    camera_id: str
    mac: str
    ip: str
    last_seen: float = 0.0
    online: bool = True

@dataclass
class ObjectTracker:
    """Track individual fire/smoke objects with growth analysis"""
    object_id: str
    object_type: str
    first_seen: float
    last_seen: float
    area_history: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def add_detection(self, area: float, confidence: float, timestamp: float):
        """Add new detection data"""
        self.area_history.append(area)
        self.confidence_history.append(confidence)
        self.last_seen = timestamp
    
    def get_growth_rate(self) -> float:
        """Calculate area growth rate"""
        if len(self.area_history) < 3:
            return 0.0
        
        # Simple linear regression on recent areas
        x = list(range(len(self.area_history)))
        y = list(self.area_history)
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Normalize by average area
        avg_area = sum_y / n
        if avg_area > 0:
            return slope / avg_area
        return 0.0
    
    def is_growing(self, threshold: float = 0.05) -> bool:
        """Check if object is growing"""
        return self.get_growth_rate() > threshold

class Config:
    """Configuration from environment variables"""
    def __init__(self):
        # Service identity
        self.NODE_ID = os.getenv('NODE_ID', 'consensus-1')
        self.SERVICE_ID = f"fire_consensus_{self.NODE_ID}"
        
        # MQTT settings
        self.MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
        self.MQTT_PORT = int(os.getenv('MQTT_PORT', '1883'))
        self.MQTT_TLS = os.getenv('MQTT_TLS', 'false').lower() == 'true'
        self.TLS_CA_PATH = os.getenv('TLS_CA_PATH', '/app/certs/ca.crt')
        
        # Topics
        self.TOPIC_DETECTION = os.getenv('TOPIC_DETECTION', 'detection/fire')
        self.TOPIC_FRIGATE = os.getenv('TOPIC_FRIGATE', 'frigate/+/+')
        self.TOPIC_TRIGGER = os.getenv('TOPIC_TRIGGER', 'trigger/fire_detected')
        self.TOPIC_CAMERA_DISCOVERY = os.getenv('TOPIC_CAMERA_DISCOVERY', 'camera/discovery/+')
        self.TOPIC_CAMERA_STATUS = os.getenv('TOPIC_CAMERA_STATUS', 'camera/status/+')
        self.TOPIC_CAMERA_TELEMETRY = os.getenv('TOPIC_CAMERA_TELEMETRY', 'telemetry/camera/+')
        self.TOPIC_HEALTH = os.getenv('TOPIC_HEALTH', 'system/fire_consensus_health')
        
        # Consensus parameters
        self.CONSENSUS_THRESHOLD = int(os.getenv('CONSENSUS_THRESHOLD', '2'))
        self.MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', '0.7'))
        self.TIME_WINDOW = float(os.getenv('CONSENSUS_TIME_WINDOW', '30.0'))
        self.COOLDOWN_PERIOD = float(os.getenv('CONSENSUS_COOLDOWN', '300.0'))
        self.MIN_AREA = float(os.getenv('MIN_DETECTION_AREA', '0.001'))
        self.MAX_AREA = float(os.getenv('MAX_DETECTION_AREA', '0.5'))
        self.GROWTH_ANALYSIS_ENABLED = os.getenv('GROWTH_ANALYSIS_ENABLED', 'true').lower() == 'true'
        self.MIN_GROWTH_RATE = float(os.getenv('MIN_GROWTH_RATE', '0.05'))
        
        # Camera health
        self.CAMERA_TIMEOUT = float(os.getenv('CAMERA_TIMEOUT', '120.0'))

class ResilientFireConsensus:
    """Fire Consensus with resilient MQTT operations"""
    
    def __init__(self):
        self.config = Config()
        
        # State management
        self.detection_history: Dict[str, List[Detection]] = defaultdict(list)
        self.object_trackers: Dict[str, Dict[str, ObjectTracker]] = defaultdict(dict)
        self.camera_states: Dict[str, CameraState] = {}
        self.last_trigger_time = 0.0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'detections_processed': 0,
            'consensus_events': 0,
            'triggers_sent': 0,
            'cameras_tracked': 0
        }
        
        # MQTT queues
        self.mqtt_outgoing = queue.Queue(maxsize=1000)
        self.mqtt_incoming = queue.Queue(maxsize=1000)
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_healthy = False
        
        # Service state
        self._running = True
        
        # Setup
        self._setup_mqtt()
        
        logger.info(f"Resilient Fire Consensus initialized: {self.config.SERVICE_ID}")
    
    def _setup_mqtt(self):
        """Setup resilient MQTT client"""
        # LWT configuration
        lwt_topic = f"{self.config.TOPIC_HEALTH}/{self.config.NODE_ID}/lwt"
        lwt_payload = json.dumps({
            'node_id': self.config.NODE_ID,
            'service': 'fire_consensus',
            'status': 'offline',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        })
        
        lwt_config = {
            'topic': lwt_topic,
            'payload': lwt_payload,
            'qos': 1,
            'retain': True
        }
        
        # TLS configuration if enabled
        tls_config = None
        if self.config.MQTT_TLS:
            tls_config = {'ca_certs': self.config.TLS_CA_PATH}
        
        # Create resilient MQTT client
        self.mqtt_client = ResilientMQTTClient(
            host=self.config.MQTT_BROKER,
            port=8883 if self.config.MQTT_TLS else self.config.MQTT_PORT,
            client_id=self.config.SERVICE_ID,
            outgoing_queue=self.mqtt_outgoing,
            incoming_queue=self.mqtt_incoming,
            keepalive=60,
            socket_timeout=10,
            max_retries=5,
            initial_retry_delay=5,
            max_retry_delay=120,
            tls_config=tls_config,
            lwt_config=lwt_config,
            health_callback=self._mqtt_health_callback
        )
        
        # Subscribe to topics
        self.mqtt_client.subscribe(self.config.TOPIC_DETECTION, 1)
        self.mqtt_client.subscribe(self.config.TOPIC_FRIGATE, 1)
        self.mqtt_client.subscribe(self.config.TOPIC_CAMERA_DISCOVERY, 1)
        self.mqtt_client.subscribe(self.config.TOPIC_CAMERA_STATUS, 0)
        self.mqtt_client.subscribe(self.config.TOPIC_CAMERA_TELEMETRY, 0)
        self.mqtt_client.subscribe(f"{self.config.TOPIC_DETECTION}/+", 1)
        
        # Start MQTT client
        self.mqtt_client.start()
        
        logger.info("Resilient MQTT client started with subscriptions")
    
    def _mqtt_health_callback(self, state: ConnectionState):
        """Handle MQTT state changes"""
        self.mqtt_healthy = state == ConnectionState.CONNECTED
        
        if state == ConnectionState.CONNECTED:
            logger.info("MQTT connection healthy")
            self._publish_health()
        elif state == ConnectionState.FAILED:
            logger.error("MQTT connection FAILED - consensus service degraded")
            # In degraded mode, we can't receive detections or send triggers
            # This is a critical failure for a consensus service
    
    def _publish_mqtt(self, topic: str, payload: str, retain: bool = False):
        """Publish MQTT message through resilient client"""
        try:
            self.mqtt_outgoing.put_nowait((topic, payload))
        except queue.Full:
            logger.error(f"MQTT outgoing queue full, dropping message for {topic}")
    
    def run(self):
        """Main service loop"""
        logger.info("Starting Fire Consensus main loop")
        
        # Start background tasks
        threading.Thread(target=self._message_processor, daemon=True).start()
        threading.Thread(target=self._cleanup_loop, daemon=True).start()
        threading.Thread(target=self._health_reporter, daemon=True).start()
        
        # Main loop
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down Fire Consensus")
            self.shutdown()
    
    def _message_processor(self):
        """Process incoming MQTT messages"""
        while self._running:
            try:
                # Get message from queue with timeout
                msg = self.mqtt_incoming.get(timeout=1.0)
                
                # Process based on topic
                topic = msg.topic
                
                try:
                    if topic.startswith(self.config.TOPIC_DETECTION):
                        self._handle_detection_message(msg)
                    elif 'frigate' in topic:
                        self._handle_frigate_message(msg)
                    elif 'camera/discovery' in topic:
                        self._handle_camera_discovery(msg)
                    elif 'camera/status' in topic:
                        self._handle_camera_status(msg)
                    elif 'telemetry/camera' in topic:
                        self._handle_camera_telemetry(msg)
                        
                except Exception as e:
                    logger.error(f"Error processing message on {topic}: {e}", exc_info=True)
                    
            except queue.Empty:
                # Normal timeout, no messages
                pass
            except Exception as e:
                logger.error(f"Message processor error: {e}", exc_info=True)
                time.sleep(1)
    
    def _handle_detection_message(self, msg):
        """Handle fire/smoke detection messages"""
        try:
            data = json.loads(msg.payload.decode())
            
            # Create detection object
            detection = Detection(
                camera_id=data.get('camera_id', 'unknown'),
                confidence=float(data.get('confidence', 0.0)),
                object_type=data.get('object', 'fire'),
                timestamp=float(data.get('timestamp', time.time())),
                bbox=data.get('bbox'),
                area=float(data.get('area', 0.0)) if 'area' in data else None,
                id=data.get('id')
            )
            
            # Process detection
            if self._process_detection(detection):
                logger.info(f"Consensus reached! Triggering fire suppression")
                self._trigger_fire_suppression()
                
            self.stats['detections_processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to handle detection message: {e}")
    
    def _handle_frigate_message(self, msg):
        """Handle Frigate event messages"""
        try:
            # Parse topic: frigate/{camera_name}/{event_type}
            parts = msg.topic.split('/')
            if len(parts) >= 3:
                camera_name = parts[1]
                event_type = parts[2]
                
                if event_type in ['fire', 'smoke']:
                    data = json.loads(msg.payload.decode())
                    
                    # Convert Frigate format to detection
                    detection = Detection(
                        camera_id=camera_name,
                        confidence=float(data.get('score', 0.0)),
                        object_type=event_type,
                        timestamp=float(data.get('timestamp', time.time())),
                        bbox=data.get('box'),
                        id=data.get('id')
                    )
                    
                    # Calculate area from bbox if provided
                    if detection.bbox:
                        detection.area = (
                            detection.bbox.get('width', 0) * 
                            detection.bbox.get('height', 0)
                        )
                    
                    # Process detection
                    if self._process_detection(detection):
                        logger.info(f"Consensus reached via Frigate! Triggering fire suppression")
                        self._trigger_fire_suppression()
                        
        except Exception as e:
            logger.error(f"Failed to handle Frigate message: {e}")
    
    def _handle_camera_discovery(self, msg):
        """Handle camera discovery messages"""
        try:
            data = json.loads(msg.payload.decode())
            camera_id = data.get('id', 'unknown')
            
            with self.lock:
                if camera_id not in self.camera_states:
                    self.camera_states[camera_id] = CameraState(
                        camera_id=camera_id,
                        mac=data.get('mac', ''),
                        ip=data.get('ip', ''),
                        last_seen=time.time(),
                        online=True
                    )
                    self.stats['cameras_tracked'] = len(self.camera_states)
                    logger.info(f"Discovered new camera: {camera_id}")
                    
        except Exception as e:
            logger.error(f"Failed to handle camera discovery: {e}")
    
    def _handle_camera_status(self, msg):
        """Handle camera status updates"""
        try:
            data = json.loads(msg.payload.decode())
            camera_id = data.get('camera_id')
            status = data.get('status')
            
            with self.lock:
                if camera_id in self.camera_states:
                    self.camera_states[camera_id].online = (status == 'online')
                    self.camera_states[camera_id].last_seen = time.time()
                    
        except Exception as e:
            logger.error(f"Failed to handle camera status: {e}")
    
    def _handle_camera_telemetry(self, msg):
        """Handle camera telemetry updates"""
        try:
            data = json.loads(msg.payload.decode())
            camera_id = data.get('camera_id')
            
            with self.lock:
                if camera_id in self.camera_states:
                    self.camera_states[camera_id].last_seen = time.time()
                    self.camera_states[camera_id].online = True
                    
        except Exception as e:
            logger.error(f"Failed to handle camera telemetry: {e}")
    
    def _process_detection(self, detection: Detection) -> bool:
        """Process detection and check for consensus"""
        # Validate detection
        if not self._validate_detection(detection):
            return False
        
        current_time = time.time()
        
        with self.lock:
            # Update camera last seen
            if detection.camera_id in self.camera_states:
                self.camera_states[detection.camera_id].last_seen = current_time
            
            # Add to history
            self.detection_history[detection.camera_id].append(detection)
            
            # Track objects if ID provided
            if detection.id and self.config.GROWTH_ANALYSIS_ENABLED:
                tracker = self.object_trackers[detection.camera_id].get(detection.id)
                if not tracker:
                    tracker = ObjectTracker(
                        object_id=detection.id,
                        object_type=detection.object_type,
                        first_seen=current_time,
                        last_seen=current_time
                    )
                    self.object_trackers[detection.camera_id][detection.id] = tracker
                
                if detection.area:
                    tracker.add_detection(detection.area, detection.confidence, current_time)
            
            # Clean old detections
            self._cleanup_old_detections(current_time)
            
            # Check cooldown
            if current_time - self.last_trigger_time < self.config.COOLDOWN_PERIOD:
                logger.debug("In cooldown period, skipping consensus check")
                return False
            
            # Check for consensus
            return self._check_consensus(current_time)
    
    def _validate_detection(self, detection: Detection) -> bool:
        """Validate detection data"""
        # Check confidence
        if detection.confidence < self.config.MIN_CONFIDENCE:
            logger.debug(f"Detection confidence {detection.confidence} below threshold")
            return False
        
        # Check area if provided
        if detection.area is not None:
            if detection.area < self.config.MIN_AREA:
                logger.debug(f"Detection area {detection.area} too small")
                return False
            if detection.area > self.config.MAX_AREA:
                logger.debug(f"Detection area {detection.area} too large")
                return False
        
        # Check camera is online
        with self.lock:
            if detection.camera_id in self.camera_states:
                if not self.camera_states[detection.camera_id].online:
                    logger.debug(f"Camera {detection.camera_id} is offline")
                    return False
        
        return True
    
    def _check_consensus(self, current_time: float) -> bool:
        """Check if consensus threshold is met"""
        active_cameras = set()
        growing_fires = 0
        
        # Check each camera's recent detections
        for camera_id, detections in self.detection_history.items():
            # Get recent detections
            recent = [d for d in detections 
                     if current_time - d.timestamp < self.config.TIME_WINDOW]
            
            if recent:
                # Camera has recent detections
                active_cameras.add(camera_id)
                
                # Check if any tracked objects are growing
                if self.config.GROWTH_ANALYSIS_ENABLED:
                    for tracker in self.object_trackers[camera_id].values():
                        if tracker.is_growing(self.config.MIN_GROWTH_RATE):
                            growing_fires += 1
                            break
        
        # Consensus logic
        num_cameras = len(active_cameras)
        
        # If growth analysis is enabled, require at least one growing fire
        if self.config.GROWTH_ANALYSIS_ENABLED and growing_fires == 0:
            logger.debug(f"No growing fires detected across {num_cameras} cameras")
            return False
        
        # Check if enough cameras agree
        if num_cameras >= self.config.CONSENSUS_THRESHOLD:
            logger.info(f"Consensus threshold met: {num_cameras}/{self.config.CONSENSUS_THRESHOLD} cameras")
            self.stats['consensus_events'] += 1
            return True
        
        logger.debug(f"Consensus not met: {num_cameras}/{self.config.CONSENSUS_THRESHOLD} cameras")
        return False
    
    def _trigger_fire_suppression(self):
        """Send fire suppression trigger command"""
        self.last_trigger_time = time.time()
        self.stats['triggers_sent'] += 1
        
        # Create trigger message
        trigger_msg = {
            'command': 'activate',
            'source': 'consensus',
            'confidence': 0.95,  # High confidence due to consensus
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'cameras': [],
            'consensus_count': 0
        }
        
        # Add camera details
        with self.lock:
            for camera_id, detections in self.detection_history.items():
                recent = [d for d in detections 
                         if self.last_trigger_time - d.timestamp < self.config.TIME_WINDOW]
                if recent:
                    trigger_msg['cameras'].append({
                        'camera_id': camera_id,
                        'detections': len(recent),
                        'max_confidence': max(d.confidence for d in recent)
                    })
            
            trigger_msg['consensus_count'] = len(trigger_msg['cameras'])
        
        # Publish trigger
        self._publish_mqtt(
            self.config.TOPIC_TRIGGER,
            json.dumps(trigger_msg),
            retain=False
        )
        
        logger.warning(f"FIRE SUPPRESSION TRIGGERED! {trigger_msg['consensus_count']} cameras in consensus")
    
    def _cleanup_old_detections(self, current_time: float):
        """Remove old detections outside time window"""
        with self.lock:
            for camera_id in list(self.detection_history.keys()):
                # Filter recent detections
                self.detection_history[camera_id] = [
                    d for d in self.detection_history[camera_id]
                    if current_time - d.timestamp < self.config.TIME_WINDOW * 2
                ]
                
                # Remove empty entries
                if not self.detection_history[camera_id]:
                    del self.detection_history[camera_id]
                
                # Clean old object trackers
                if camera_id in self.object_trackers:
                    for obj_id in list(self.object_trackers[camera_id].keys()):
                        tracker = self.object_trackers[camera_id][obj_id]
                        if current_time - tracker.last_seen > self.config.TIME_WINDOW:
                            del self.object_trackers[camera_id][obj_id]
                    
                    if not self.object_trackers[camera_id]:
                        del self.object_trackers[camera_id]
    
    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self._running:
            try:
                current_time = time.time()
                
                # Clean old detections
                self._cleanup_old_detections(current_time)
                
                # Update camera states
                with self.lock:
                    for camera in self.camera_states.values():
                        if current_time - camera.last_seen > self.config.CAMERA_TIMEOUT:
                            camera.online = False
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)
    
    def _health_reporter(self):
        """Report health status periodically"""
        while self._running:
            try:
                self._publish_health()
                time.sleep(30)  # Every 30 seconds
            except Exception as e:
                logger.error(f"Health reporter error: {e}")
                time.sleep(30)
    
    def _publish_health(self):
        """Publish health report"""
        try:
            with self.lock:
                online_cameras = sum(1 for c in self.camera_states.values() if c.online)
                total_cameras = len(self.camera_states)
                active_detections = sum(len(d) for d in self.detection_history.values())
            
            health = {
                'node_id': self.config.NODE_ID,
                'service': 'fire_consensus',
                'status': 'online' if self.mqtt_healthy else 'degraded',
                'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'consensus': {
                    'threshold': self.config.CONSENSUS_THRESHOLD,
                    'cooldown_remaining': max(0, self.config.COOLDOWN_PERIOD - (time.time() - self.last_trigger_time))
                },
                'cameras': {
                    'total': total_cameras,
                    'online': online_cameras,
                    'active_detections': active_detections
                },
                'statistics': self.stats,
                'mqtt': {
                    'connected': self.mqtt_healthy,
                    'stats': self.mqtt_client.get_stats() if self.mqtt_client else {}
                }
            }
            
            self._publish_mqtt(
                self.config.TOPIC_HEALTH,
                json.dumps(health),
                retain=True
            )
            
        except Exception as e:
            logger.error(f"Failed to publish health: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Fire Consensus")
        self._running = False
        
        # Stop MQTT client
        if self.mqtt_client:
            self.mqtt_client.stop(timeout=5.0)
        
        logger.info("Fire Consensus shutdown complete")

def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    consensus = ResilientFireConsensus()
    consensus.run()

if __name__ == "__main__":
    main()