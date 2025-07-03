# Thread Safety Guide

This guide documents the thread safety improvements made to the Wildfire Watch system to prevent race conditions and ensure reliable operation under concurrent access.

## Overview

The Wildfire Watch system uses multiple background threads for various operations:
- Camera discovery and health monitoring
- Fire detection consensus evaluation  
- Telemetry collection and reporting
- GPIO pump control with timers

Thread safety is critical because multiple threads access shared state like camera registries, detection histories, and hardware status.

## Thread Safety Principles

### 1. Single Lock Strategy
We use a single `RLock` (reentrant lock) per service to protect all shared state. This approach:
- **Simplifies reasoning** about thread safety
- **Prevents deadlocks** that can occur with multiple locks
- **Allows re-entrant calls** within the same thread

### 2. Minimize Lock Duration
Locks are held only for the minimum time necessary:
```python
# Good: Quick state update
with self.lock:
    self.cameras[camera_id] = camera
    
# Bad: I/O while holding lock
with self.lock:
    self.cameras[camera_id] = camera
    camera.ping()  # Network I/O - DON'T DO THIS!
```

### 3. Separate I/O from State Updates
Network operations and other I/O are performed outside locks:
```python
# Step 1: Get data (with lock)
with self.lock:
    camera_ip = self.cameras[camera_id].ip
    
# Step 2: Perform I/O (no lock)
is_alive = ping_camera(camera_ip)

# Step 3: Update state (with lock)
with self.lock:
    if camera_id in self.cameras:  # Re-validate
        self.cameras[camera_id].online = is_alive
```

## Implementation Details

### Camera Detector Thread Safety

The camera detector has multiple threads accessing shared state:

#### Shared State
- `self.cameras: Dict[str, Camera]` - Camera registry
- `self.known_camera_ips: Set[str]` - Known IP addresses
- `self.discovery_count` - Discovery cycle counter
- `self.mac_tracker` - MAC-to-IP mappings

#### Thread-Safe Implementation
```python
from camera_detector.thread_safety import ThreadSafeCameraRegistry

class CameraDetector:
    def __init__(self):
        self.camera_registry = ThreadSafeCameraRegistry()
        self.lock = threading.RLock()
        
    def add_camera(self, camera):
        """Thread-safe camera addition"""
        return self.camera_registry.add_camera(camera)
        
    def get_camera_by_ip(self, ip):
        """Thread-safe lookup"""
        return self.camera_registry.get_by_ip(ip)
```

#### Using the Thread-Safe Mixin
```python
from camera_detector.detect import CameraDetector
from camera_detector.detect_thread_safe_mixin import ThreadSafeCameraDetectorMixin

class ThreadSafeCameraDetector(ThreadSafeCameraDetectorMixin, CameraDetector):
    """Camera detector with thread safety"""
    pass
```

### Fire Consensus Thread Safety

The consensus service manages detection history from multiple cameras:

#### Shared State
- `self.detection_history` - Per-camera detection queues
- `self.object_trackers` - Fire growth tracking
- `self.camera_states` - Camera online/offline status
- `self.last_trigger_time` - Cooldown management

#### Thread-Safe Components
```python
from fire_consensus.thread_safety import (
    ThreadSafeDetectionHistory,
    ThreadSafeObjectTracker,
    ThreadSafeCameraStates,
    ThreadSafeConsensusState
)

class FireConsensus:
    def __init__(self):
        self.detection_history = ThreadSafeDetectionHistory()
        self.object_trackers = ThreadSafeObjectTracker()
        self.camera_states = ThreadSafeCameraStates()
        self.consensus_state = ThreadSafeConsensusState()
```

### Telemetry Service Thread Safety

The telemetry service already implements basic thread safety:

```python
# Global state protection
_state_lock = threading.Lock()
active_timer = None

def publish_telemetry():
    global active_timer
    with _state_lock:
        # Cancel previous timer
        if active_timer:
            active_timer.cancel()
        # Schedule next publication
        active_timer = threading.Timer(INTERVAL, publish_telemetry)
        active_timer.start()
```

### GPIO Trigger Thread Safety

Already implemented in Phase 1 with:
- `ThreadSafeStateMachine` for atomic state transitions
- `SafeTimerManager` for timer operations
- Read-after-write verification for GPIO operations

## Testing Thread Safety

### 1. Stress Testing
Create many threads that concurrently access shared state:

```python
def test_concurrent_camera_updates():
    detector = ThreadSafeCameraDetector()
    
    def add_cameras(start_id, count):
        for i in range(count):
            camera = Camera(ip=f"192.168.1.{start_id + i}")
            detector.add_camera(camera)
            time.sleep(random.uniform(0, 0.01))  # Random delay
    
    # Create multiple threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=add_cameras, args=(i*10, 10))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Verify all cameras added
    assert detector.camera_registry.count() == 100
```

### 2. Race Condition Detection
Use threading synchronization primitives to force specific interleavings:

```python
def test_race_condition_detection():
    detector = ThreadSafeCameraDetector()
    barrier = threading.Barrier(2)
    
    def update_camera_status(camera_id, status):
        barrier.wait()  # Synchronize threads
        detector.update_camera_status(camera_id, status)
    
    # Two threads try to update same camera
    t1 = threading.Thread(target=update_camera_status, args=("cam1", True))
    t2 = threading.Thread(target=update_camera_status, args=("cam1", False))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # One update should win (no corruption)
    camera = detector.get_camera("cam1")
    assert camera.online in [True, False]
```

### 3. Deadlock Prevention
Verify that the single-lock strategy prevents deadlocks:

```python
def test_no_deadlock():
    detector = ThreadSafeCameraDetector()
    
    def recursive_operation():
        with detector.lock:
            # First lock acquisition
            detector.add_camera(Camera(ip="192.168.1.1"))
            
            with detector.lock:
                # Reentrant lock allows this
                detector.update_camera_status("cam1", True)
    
    # Should complete without deadlock
    recursive_operation()
```

## Migration Guide

### Updating Existing Code

1. **Import thread-safe components**:
```python
from camera_detector.thread_safety import ThreadSafeCameraRegistry
from fire_consensus.thread_safety import ThreadSafeDetectionHistory
```

2. **Replace unsafe collections**:
```python
# Before
self.cameras = {}

# After  
self.cameras = ThreadSafeCameraRegistry()
```

3. **Use thread-safe methods**:
```python
# Before
self.cameras[camera_id] = camera

# After
self.cameras.add_camera(camera)
```

### Backward Compatibility

The thread-safe implementations maintain the same external API:
- Existing tests continue to work
- MQTT message formats unchanged
- Service interfaces remain the same

## Performance Considerations

### Lock Contention
Monitor for lock contention using profiling:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run your service

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats()
```

Look for high cumulative time in lock acquisition.

### Optimization Strategies

1. **Batch Operations**:
```python
# Instead of multiple lock acquisitions
for camera in cameras:
    detector.add_camera(camera)
    
# Use batch operation
detector.add_cameras_batch(cameras)
```

2. **Read-Only Snapshots**:
```python
def get_all_cameras_snapshot(self):
    """Get snapshot for read-only operations"""
    with self.lock:
        return list(self.cameras.values())
```

3. **Separate Read/Write Locks** (future optimization):
If profiling shows high read contention, consider using `threading.RWLock` (when available) or implementing a readers-writer lock.

## Troubleshooting

### Common Issues

1. **AttributeError: object has no attribute 'lock'**
   - Ensure the thread safety mixin is initialized before use
   - Check inheritance order (mixin should come first)

2. **Deadlock symptoms** (service hangs):
   - Check for I/O operations inside locks
   - Verify no circular dependencies between services

3. **Race condition symptoms** (inconsistent state):
   - Verify all shared state access uses locks
   - Check for state modifications outside lock protection

### Debug Logging

Enable thread safety debug logging:
```python
import logging
logging.getLogger('camera_detector.thread_safety').setLevel(logging.DEBUG)
logging.getLogger('fire_consensus.thread_safety').setLevel(logging.DEBUG)
```

## Best Practices Summary

1. **Always use context managers** (`with lock:`) for lock acquisition
2. **Keep critical sections small** - only protect state changes
3. **Never perform I/O while holding a lock**
4. **Document thread safety guarantees** in docstrings
5. **Test concurrent access patterns** in unit tests
6. **Profile for lock contention** in production-like loads
7. **Use thread-safe collections** where available
8. **Avoid global state** - encapsulate in classes