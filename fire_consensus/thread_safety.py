#!/usr/bin/env python3.12
"""Thread Safety Enhancements for Fire Consensus Service

This module provides thread-safe wrappers and utilities for the fire consensus
service to ensure proper synchronization of detection history and state.
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Set, Any, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class ThreadSafeDetectionHistory:
    """Thread-safe detection history manager
    
    Manages detection history per camera with proper locking to prevent
    race conditions during concurrent access.
    """
    
    _history: Dict[str, Deque[Any]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _time_window: float = 30.0  # seconds
    
    def add_detection(self, camera_id: str, detection: Any):
        """Add detection atomically"""
        with self._lock:
            self._history[camera_id].append(detection)
    
    def get_recent_detections(self, camera_id: str, current_time: float) -> List[Any]:
        """Get recent detections within time window"""
        with self._lock:
            detections = self._history.get(camera_id, [])
            # Return only detections within time window
            return [d for d in detections 
                   if current_time - d.timestamp < self._time_window]
    
    def get_all_recent_detections(self, current_time: float) -> Dict[str, List[Any]]:
        """Get all recent detections for all cameras"""
        with self._lock:
            result = {}
            for camera_id, detections in self._history.items():
                recent = [d for d in detections 
                         if current_time - d.timestamp < self._time_window]
                if recent:
                    result[camera_id] = recent
            return result
    
    def cleanup_old_detections(self, current_time: float):
        """Remove detections outside time window"""
        with self._lock:
            # Clean up old detections
            for camera_id in list(self._history.keys()):
                # Filter to keep only recent
                recent = deque(
                    (d for d in self._history[camera_id] 
                     if current_time - d.timestamp < self._time_window * 2),
                    maxlen=100
                )
                
                if recent:
                    self._history[camera_id] = recent
                else:
                    # Remove empty entries
                    del self._history[camera_id]
    
    def clear_camera(self, camera_id: str):
        """Clear all detections for a camera"""
        with self._lock:
            self._history.pop(camera_id, None)
    
    def get_camera_count(self) -> int:
        """Get number of cameras with detections"""
        with self._lock:
            return len(self._history)
    
    def get_total_detections(self) -> int:
        """Get total detection count across all cameras"""
        with self._lock:
            return sum(len(detections) for detections in self._history.values())

@dataclass
class ThreadSafeObjectTracker:
    """Thread-safe object tracker for growth analysis
    
    Tracks fire/smoke objects across frames with thread-safe operations.
    """
    
    _trackers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _max_age: float = 60.0  # seconds
    
    def update_object(self, camera_id: str, object_id: str, detection: Any):
        """Update object tracking data atomically"""
        with self._lock:
            if object_id not in self._trackers[camera_id]:
                # Create new tracker
                from fire_consensus.consensus import ObjectTracker
                self._trackers[camera_id][object_id] = ObjectTracker(
                    object_id=object_id,
                    object_type=detection.object_type,
                    first_seen=detection.timestamp,
                    last_seen=detection.timestamp
                )
            
            tracker = self._trackers[camera_id][object_id]
            if detection.area:
                tracker.add_detection(detection.area, detection.confidence, detection.timestamp)
    
    def get_growing_objects(self, camera_id: str, growth_threshold: float = 0.05) -> List[str]:
        """Get IDs of objects that are growing"""
        with self._lock:
            growing = []
            for obj_id, tracker in self._trackers.get(camera_id, {}).items():
                if tracker.is_growing(growth_threshold):
                    growing.append(obj_id)
            return growing
    
    def count_growing_fires(self, growth_threshold: float = 0.05) -> int:
        """Count total growing fires across all cameras"""
        with self._lock:
            count = 0
            for camera_trackers in self._trackers.values():
                for tracker in camera_trackers.values():
                    if tracker.object_type == 'fire' and tracker.is_growing(growth_threshold):
                        count += 1
            return count
    
    def cleanup_old_objects(self, current_time: float):
        """Remove stale object trackers"""
        with self._lock:
            for camera_id in list(self._trackers.keys()):
                # Remove old trackers
                active_trackers = {
                    obj_id: tracker
                    for obj_id, tracker in self._trackers[camera_id].items()
                    if current_time - tracker.last_seen < self._max_age
                }
                
                if active_trackers:
                    self._trackers[camera_id] = active_trackers
                else:
                    del self._trackers[camera_id]
    
    def clear_camera(self, camera_id: str):
        """Clear all trackers for a camera"""
        with self._lock:
            self._trackers.pop(camera_id, None)

@dataclass
class ThreadSafeCameraStates:
    """Thread-safe camera state management"""
    
    _states: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _timeout: float = 120.0  # seconds
    
    def update_camera(self, camera_id: str, camera_state: Any):
        """Update camera state atomically"""
        with self._lock:
            self._states[camera_id] = camera_state
    
    def get_camera(self, camera_id: str) -> Optional[Any]:
        """Get camera state"""
        with self._lock:
            return self._states.get(camera_id)
    
    def is_camera_online(self, camera_id: str) -> bool:
        """Check if camera is online"""
        with self._lock:
            camera = self._states.get(camera_id)
            return camera.online if camera else False
    
    def update_camera_seen(self, camera_id: str, timestamp: float):
        """Update camera last seen time"""
        with self._lock:
            camera = self._states.get(camera_id)
            if camera:
                camera.last_seen = timestamp
    
    def get_online_cameras(self) -> List[str]:
        """Get list of online camera IDs"""
        with self._lock:
            return [cam_id for cam_id, cam in self._states.items() if cam.online]
    
    def update_offline_cameras(self, current_time: float) -> List[str]:
        """Mark timed-out cameras as offline, return affected IDs"""
        with self._lock:
            offline_cameras = []
            for cam_id, camera in self._states.items():
                if camera.online and current_time - camera.last_seen > self._timeout:
                    camera.online = False
                    offline_cameras.append(cam_id)
            return offline_cameras

class ThreadSafeConsensusState:
    """Thread-safe consensus state management"""
    
    def __init__(self, threshold: int = 2, cooldown: float = 300.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._last_trigger_time = 0.0
        self._consensus_events = 0
        self._triggers_sent = 0
        self._lock = threading.RLock()
    
    def can_trigger(self, current_time: float) -> bool:
        """Check if we can trigger (not in cooldown)"""
        with self._lock:
            return current_time - self._last_trigger_time >= self.cooldown
    
    def record_trigger(self, current_time: float):
        """Record that a trigger was sent"""
        with self._lock:
            self._last_trigger_time = current_time
            self._triggers_sent += 1
    
    def record_consensus_event(self):
        """Record that consensus was reached"""
        with self._lock:
            self._consensus_events += 1
    
    def get_cooldown_remaining(self, current_time: float) -> float:
        """Get seconds remaining in cooldown"""
        with self._lock:
            elapsed = current_time - self._last_trigger_time
            return max(0, self.cooldown - elapsed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        with self._lock:
            return {
                'consensus_events': self._consensus_events,
                'triggers_sent': self._triggers_sent,
                'last_trigger_time': self._last_trigger_time
            }

def synchronized_method(lock_name: str = '_lock'):
    """Decorator to synchronize a method with a lock
    
    Args:
        lock_name: Name of the lock attribute
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            lock = getattr(self, lock_name, None)
            if lock:
                with lock:
                    return func(self, *args, **kwargs)
            else:
                # No lock available, run unsynchronized
                logger.warning(f"No lock '{lock_name}' found for {func.__name__}")
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class ThreadSafeFireConsensusMixin:
    """Mixin to add thread safety to FireConsensus
    
    This mixin provides thread-safe implementations of methods that
    access shared state in the FireConsensus service.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize thread safety components"""
        super().__init__(*args, **kwargs)
        
        # Replace unsafe collections with thread-safe versions
        self._init_thread_safe_state()
        
        logger.info("Thread safety mixin initialized for Fire Consensus")
    
    def _init_thread_safe_state(self):
        """Initialize thread-safe state managers"""
        # Replace detection history
        if hasattr(self, 'detection_history'):
            # Migrate existing data if any
            old_history = getattr(self, 'detection_history', {})
            self.detection_history = ThreadSafeDetectionHistory(
                _time_window=getattr(self.config, 'TIME_WINDOW', 30.0)
            )
            # Migrate data
            for camera_id, detections in old_history.items():
                for detection in detections:
                    self.detection_history.add_detection(camera_id, detection)
        
        # Replace object trackers
        if hasattr(self, 'object_trackers'):
            self.object_trackers = ThreadSafeObjectTracker()
        
        # Replace camera states
        if hasattr(self, 'camera_states'):
            old_states = getattr(self, 'camera_states', {})
            self.camera_states = ThreadSafeCameraStates(
                _timeout=getattr(self.config, 'CAMERA_TIMEOUT', 120.0)
            )
            # Migrate data
            for cam_id, state in old_states.items():
                self.camera_states.update_camera(cam_id, state)
        
        # Create consensus state manager
        if not hasattr(self, 'consensus_state'):
            self.consensus_state = ThreadSafeConsensusState(
                threshold=getattr(self.config, 'CONSENSUS_THRESHOLD', 2),
                cooldown=getattr(self.config, 'COOLDOWN_PERIOD', 300.0)
            )
        
        # Ensure we have a lock
        if not hasattr(self, 'lock'):
            self.lock = threading.RLock()
    
    def process_detection(self, detection: Any) -> bool:
        """Thread-safe detection processing
        
        This overrides the base implementation to ensure thread safety.
        """
        # Validate detection
        if not self._validate_detection(detection):
            return False
        
        current_time = time.time()
        
        # Update camera last seen
        if detection.camera_id in self.camera_states._states:
            self.camera_states.update_camera_seen(detection.camera_id, current_time)
        
        # Add to history
        self.detection_history.add_detection(detection.camera_id, detection)
        
        # Update object tracking if enabled
        if detection.id and getattr(self.config, 'GROWTH_ANALYSIS_ENABLED', True):
            self.object_trackers.update_object(
                detection.camera_id, 
                detection.id, 
                detection
            )
        
        # Check cooldown
        if not self.consensus_state.can_trigger(current_time):
            logger.debug("In cooldown period, skipping consensus check")
            return False
        
        # Check for consensus
        return self._check_consensus_thread_safe(current_time)
    
    def _check_consensus_thread_safe(self, current_time: float) -> bool:
        """Thread-safe consensus checking"""
        # Get all recent detections
        all_detections = self.detection_history.get_all_recent_detections(current_time)
        
        # Count cameras with recent detections
        active_cameras = set(all_detections.keys())
        num_cameras = len(active_cameras)
        
        # Check for growing fires if enabled
        growing_fires = 0
        if getattr(self.config, 'GROWTH_ANALYSIS_ENABLED', True):
            growing_fires = self.object_trackers.count_growing_fires(
                getattr(self.config, 'MIN_GROWTH_RATE', 0.05)
            )
            
            # Require at least one growing fire
            if growing_fires == 0:
                logger.debug(f"No growing fires detected across {num_cameras} cameras")
                return False
        
        # Check if consensus threshold is met
        threshold = getattr(self.config, 'CONSENSUS_THRESHOLD', 2)
        if num_cameras >= threshold:
            logger.info(f"Consensus threshold met: {num_cameras}/{threshold} cameras")
            self.consensus_state.record_consensus_event()
            return True
        
        logger.debug(f"Consensus not met: {num_cameras}/{threshold} cameras")
        return False
    
    def trigger_fire_suppression(self):
        """Thread-safe fire suppression trigger"""
        current_time = time.time()
        self.consensus_state.record_trigger(current_time)
        
        # Call parent implementation
        if hasattr(super(), 'trigger_fire_suppression'):
            super().trigger_fire_suppression()
        else:
            logger.warning("No parent trigger_fire_suppression method found")
    
    def cleanup_old_data(self):
        """Thread-safe cleanup of old data"""
        current_time = time.time()
        
        # Clean detection history
        self.detection_history.cleanup_old_detections(current_time)
        
        # Clean object trackers
        self.object_trackers.cleanup_old_objects(current_time)
        
        # Update offline cameras
        offline_cameras = self.camera_states.update_offline_cameras(current_time)
        for cam_id in offline_cameras:
            logger.warning(f"Camera {cam_id} marked offline due to timeout")