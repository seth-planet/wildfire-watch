# 🔥 Fire Consensus Service

## What Does This Do?

The Fire Consensus Service is your "smart fire alarm" that prevents false alarms while ensuring real fires are never missed. It works like having multiple witnesses confirm a fire before calling the fire department.

### Key Benefits:

- ✅ **Prevents false alarms** - Requires multiple cameras to agree
- 📈 **Detects growing fires** - Only triggers on fires that are spreading
- 🚨 **Fast response** - Triggers within seconds when consensus is reached
- 🔄 **Self-healing** - Continues working even if some cameras fail
- 📊 **Smart filtering** - Ignores shadows, reflections, and other false positives

## Why This Matters

Traditional fire detection often triggers on:
- Sunlight reflections
- Car headlights
- Shadows moving
- Single camera glitches

Our consensus system requires **multiple cameras** to see a **growing fire** before triggering sprinklers. This virtually eliminates false activations while maintaining rapid response to real fires.

## Quick Start

### Default Behavior

Out of the box, the consensus service:
1. Waits for fire detections from cameras
2. Requires 2+ cameras to see fire within 10 seconds
3. Checks that fires are growing (not shrinking)
4. Triggers sprinkler system when consensus is reached
5. Waits 30 seconds before allowing another trigger

### What You'll See

Normal operation logs:
```
[INFO] Fire Consensus Service started
[INFO] Camera north_cam detected fire (confidence: 0.85)
[INFO] Camera south_cam detected fire (confidence: 0.82)
[WARNING] FIRE CONSENSUS REACHED - Triggered response! Cameras: ['north_cam', 'south_cam']
```

## Configuration Options

### Basic Settings (Most Important)

```bash
# How many cameras must agree there's a fire
CONSENSUS_THRESHOLD=2        # Default: 2 cameras

# Time window for consensus (seconds)
CAMERA_WINDOW=10            # Cameras must agree within 10 seconds

# Cooldown between triggers (seconds)
DETECTION_COOLDOWN=30       # Wait 30 seconds before retriggering
```

**💡 Tip**: For high-risk areas, set `CONSENSUS_THRESHOLD=1` for faster response. For low-risk areas with many false positives, increase to 3 or 4.

### Fire Detection Settings

```bash
# Minimum confidence level (0.0 to 1.0)
MIN_CONFIDENCE=0.7          # 70% confidence required

# Fire size limits (portion of camera view)
MIN_AREA_RATIO=0.001        # Ignore tiny fires (0.1% of view)
MAX_AREA_RATIO=0.5          # Ignore huge detections (probably errors)

# Growth detection
INCREASE_COUNT=3            # Minimum detections needed (legacy - now uses MOVING_AVERAGE_WINDOW)
AREA_INCREASE_RATIO=1.2     # Fire must grow 20% overall from first to last average
MOVING_AVERAGE_WINDOW=3     # Window size for moving average calculation
```

### Advanced Settings

```bash
# System health
TELEMETRY_INTERVAL=60       # Report health every minute
CLEANUP_INTERVAL=300        # Clean old data every 5 minutes
CAMERA_TIMEOUT=120          # Mark camera offline after 2 minutes

# MQTT Connection
MQTT_BROKER=mqtt_broker     # MQTT broker hostname
MQTT_PORT=1883              # MQTT broker port
MQTT_TLS=false              # Enable TLS encryption
TLS_CA_PATH=/mnt/data/certs/ca.crt  # CA certificate path

# Node Identity
NODE_ID=${BALENA_DEVICE_UUID:-hostname}  # Unique node identifier

# Logging
LOG_LEVEL=INFO              # Set to DEBUG for troubleshooting
```

## Understanding Consensus

### Moving Average Fire Growth Detection

The system uses moving averages to detect fire growth, which provides several key benefits:

**📊 Noise Reduction**: Object detection can be jittery, with fire areas fluctuating ±10% between frames. Moving averages smooth out these variations to reveal the true growth trend.

**🎯 Robust Detection**: Unlike frame-by-frame comparison that can fail if any single detection is noisy, moving averages look at overall trends across multiple detections.

**⚡ Flexible Sensitivity**: You can tune the moving average window size:
- **Smaller window (2-3)**: More responsive but potentially noisier
- **Larger window (4-6)**: More stable but slower to respond

**🔍 Real vs False Fire Discrimination**: 
- Real fires show consistent growth trends in their moving averages
- False positives (reflections, shadows) show random fluctuations with no clear trend

### How It Works

1. **Detection Phase**
   - Cameras report fire detections with confidence scores
   - Each detection includes size and location

2. **Growth Analysis**
   - System tracks fire size over time
   - Only growing fires trigger consensus
   - Prevents false alarms from static hot spots

3. **Consensus Decision**
   - Counts cameras seeing growing fires
   - Triggers when threshold is met
   - Publishes trigger command to sprinkler system

### Example Scenario

```
Time 0s: Camera-1 sees fire (0.010 area)
Time 1s: Camera-1 sees fire (0.009 area) - temporary dip
Time 2s: Camera-1 sees fire (0.012 area)
Time 3s: Camera-1 sees fire (0.011 area) - slight dip  
Time 4s: Camera-1 sees fire (0.014 area)
Time 5s: Camera-1 sees fire (0.016 area)
        Moving averages: [0.0103, 0.0107, 0.0123, 0.0137] → Growing trend ✓

Time 6s: Camera-2 also shows growing trend ✓
Time 7s: CONSENSUS! Both cameras show growing fires → Trigger sprinklers
```

## Common Issues and Solutions

### Problem: Too Many False Alarms

**Symptoms**: Sprinklers triggering on non-fires

**Solutions**:
1. **Increase threshold**: Require more cameras
   ```bash
   CONSENSUS_THRESHOLD=3
   ```
2. **Increase confidence**: Require higher certainty
   ```bash
   MIN_CONFIDENCE=0.8
   ```
3. **Require more growth**: Make fires prove they're real
   ```bash
   AREA_INCREASE_RATIO=1.3
   MOVING_AVERAGE_WINDOW=5  # Larger window for more smoothing
   ```
4. **Require more detections**: Need more data points
   ```bash
   CAMERA_WINDOW=15  # Longer time window for more detections
   ```

### Problem: Slow Response to Real Fires

**Symptoms**: Takes too long to trigger on actual fires

**Solutions**:
1. **Decrease threshold**: Require fewer cameras
   ```bash
   CONSENSUS_THRESHOLD=1
   ```
2. **Increase window**: Give cameras more time
   ```bash
   CAMERA_WINDOW=20
   ```
3. **Lower confidence**: Accept lower certainty
   ```bash
   MIN_CONFIDENCE=0.6
   ```
4. **Use smaller moving average window**: Faster detection
   ```bash
   MOVING_AVERAGE_WINDOW=2  # Smaller window for quicker response
   ```

### Problem: System Not Triggering at All

**Symptoms**: No triggers even with obvious fires

**Solutions**:
1. **Check camera status**: Ensure cameras are online
2. **Verify MQTT connection**: Check broker connectivity
3. **Lower all thresholds**: Test with minimal requirements
4. **Enable debug logging**: 
   ```bash
   LOG_LEVEL=DEBUG
   ```

## Monitoring System Health

### Health Status

The service publishes health reports showing:
- Number of online cameras
- Recent detection activity
- System configuration
- Trigger history

### MQTT Health Topic
```
Topic: system/consensus_telemetry
```

**Note**: The service also publishes Last Will Testament (LWT) messages to `system/consensus_telemetry/{NODE_ID}/lwt` for automatic offline detection.

Example health report:
```json
{
  "status": "online",
  "stats": {
    "total_cameras": 4,
    "online_cameras": 4,
    "active_detections": 0,
    "total_triggers": 2,
    "last_trigger": 1634567890
  }
}
```

## Technical Details

### Detection Algorithm

1. **Object Tracking**
   - Each fire gets unique ID for growth analysis
   - Tracks movement and growth across detections
   - Handles multiple fires per camera simultaneously
   - Supports both direct detections and Frigate NVR events
   - Automatically converts pixel coordinates to normalized areas

2. **Growth Calculation**
   - Uses moving averages to reduce detection noise
   - Compares smoothed fire size trends over time
   - Requires overall growth from first to last average (default 20%)
   - Tolerates up to 30% of transitions showing temporary shrinkage
   - Validates growth across multiple detection frames
   - Filters out shrinking/static objects automatically

3. **Consensus Logic**
   - Time-windowed voting system
   - Cameras must agree within window
   - Prevents split-second false positives
   - Handles mixed detection sources (direct + Frigate)

4. **Coordinate Handling**
   - **Direct detections**: `[x, y, width, height]` normalized (0-1)
   - **Frigate events**: `[x1, y1, x2, y2]` pixel coordinates
   - Automatic format detection and area calculation
   - Robust validation for invalid/extreme values

### MQTT Topics

**Subscribes to:**
- `fire/detection` - Individual camera detections
- `fire/detection/+` - Camera-specific detection topics (wildcard)
- `frigate/events` - Frigate NVR events
- `system/camera_telemetry` - Camera health status

**Publishes to:**
- `fire/trigger` - Sprinkler activation command
- `system/consensus_telemetry` - Service health
- `system/consensus_telemetry/{NODE_ID}/lwt` - Last Will Testament for offline detection

### State Management

The service maintains:
- **Camera states** (online/offline based on telemetry)
- **Recent detections** per camera (up to 100 per camera)
- **Fire object tracking** by unique object ID across detections
- **Growing fire detection** using area increase ratios
- **Consensus event history** (last 1000 events)
- **Thread-safe operation** with comprehensive locking
- **Automatic cleanup** of stale cameras and objects

## Fine-Tuning for Your Environment

### High-Risk Areas (Dry Forests)
```bash
CONSENSUS_THRESHOLD=1       # Single camera trigger
MIN_CONFIDENCE=0.6          # Lower confidence OK
CAMERA_WINDOW=15            # Longer window
DETECTION_COOLDOWN=20       # Shorter cooldown
```

### Low-Risk Areas (Near Water)
```bash
CONSENSUS_THRESHOLD=3       # Three cameras required
MIN_CONFIDENCE=0.8          # High confidence needed
INCREASE_COUNT=5            # More growth proof
AREA_INCREASE_RATIO=1.4     # Significant growth required
```

### Windy Locations
```bash
# Reduce sensitivity to moving shadows
MIN_AREA_RATIO=0.005        # Ignore smaller detections
INCREASE_COUNT=4            # Require sustained growth
```

### Camera-Dense Deployments
```bash
# With many cameras, require more consensus
CONSENSUS_THRESHOLD=4       # 4+ cameras must agree
CAMERA_WINDOW=5             # Tighter time window
```

## Security and Reliability

### Fail-Safe Design

- **No single point of failure**: Works with some cameras offline
- **Network resilient**: Handles intermittent connections with exponential backoff
- **Stateless operation**: Can restart without losing protection
- **Tamper evident**: Logs all configuration changes
- **Thread-safe**: Concurrent detection processing with proper locking
- **Input validation**: Rejects malformed data (NaN, infinity, negative values)
- **Automatic cleanup**: Removes stale cameras and old object tracks

### Security Features

- **Encrypted MQTT**: TLS encryption for all messages
- **Input validation**: Rejects malformed detections and extreme values
- **Rate limiting**: Cooldown periods prevent trigger spam
- **Authenticated cameras**: Only trusted sources accepted
- **Last Will Testament**: Automatic offline detection via MQTT LWT
- **Error isolation**: Processing errors don't crash the service

## Advanced Customization

### Custom Detection Logic

Edit `consensus.py` to modify detection criteria:

```python
# Add weather-based adjustments
if weather == "windy":
    self.config.MIN_AREA_RATIO *= 2
    
# Time-of-day adjustments
if hour < 6 or hour > 20:  # Nighttime
    self.config.MIN_CONFIDENCE *= 0.9
```

### Integration with Other Systems

The service publishes standard MQTT messages, making integration easy:

```python
# Subscribe to triggers in your code
def on_trigger(msg):
    trigger_data = json.loads(msg.payload)
    cameras = trigger_data['consensus_cameras']
    confidence = trigger_data['confidence']
    # Your custom logic here
```

## Testing and Validation

### Simulation Mode

Test consensus logic without real fires:

1. Publish test detections to MQTT
2. Watch consensus decisions
3. Verify trigger behavior

### Manual Testing
```bash
# Publish test detection
mosquitto_pub -t "fire/detection" -m '{
  "camera_id": "test_cam",
  "confidence": 0.8,
  "bounding_box": [0.1, 0.1, 0.05, 0.05]
}'
```

## Learn More

### Fire Detection Theory
- [Wildfire Behavior](https://www.nwcg.gov/publications/pms437/fire-behavior)
- [Computer Vision for Fire Detection](https://arxiv.org/abs/2106.13943)
- [Consensus Algorithms](https://en.wikipedia.org/wiki/Consensus_algorithm)

### Related Documentation
- [MQTT Protocol](https://mqtt.org/)
- [Frigate NVR](https://docs.frigate.video/)
- [Docker Compose](https://docs.docker.com/compose/)

## Getting Help

If consensus isn't working:
1. Check camera connectivity
2. Verify MQTT broker is running
3. Review detection thresholds
4. Enable debug logging
5. Monitor health reports

Remember: The system is designed to be conservative by default. It's better to miss a tiny fire than to trigger false alarms. Adjust settings based on your specific risk tolerance and environment.
