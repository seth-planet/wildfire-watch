# Hailo Troubleshooting Guide

## Common Issues and Solutions

### 1. Device Detection Issues

#### Problem: Hailo device not found
```bash
$ ls /dev/hailo*
ls: cannot access '/dev/hailo*': No such file or directory
```

**Solutions:**

1. **Check PCIe connection**
   ```bash
   # Verify device in PCIe bus
   lspci | grep Hailo
   # Expected: XX:XX.X Processing accelerators: Hailo Technologies Ltd. Hailo-8 AI Processor
   
   # If not found, reseat the M.2 card
   ```

2. **Install/reinstall drivers**
   ```bash
   cd /opt/hailort/drivers
   sudo ./uninstall_pcie_driver.sh
   sudo ./install_pcie_driver.sh
   sudo modprobe hailo_pci
   
   # Verify driver loaded
   lsmod | grep hailo
   ```

3. **Check permissions**
   ```bash
   # Fix device permissions
   sudo chmod 666 /dev/hailo0
   
   # Add user to video group
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

### 2. API and Import Errors

#### Problem: ImportError - No module named 'hailo'
```python
ImportError: No module named 'hailo'
```

**Solution:**
```python
# Use hailo_platform instead
from hailo_platform import VDevice, HEF, ConfigureParams
# NOT: import hailo
```

#### Problem: AttributeError with VStreams
```python
AttributeError: 'ConfiguredNetwork' object has no attribute 'create_input_vstreams'
```

**Solution:**
```python
# Use underscore version and activate first
network_group.activate()
network_group.wait_for_activation(5000)
input_vstreams = network_group._create_input_vstreams(params)
```

### 3. Network Group Activation Errors

#### Problem: HailoRTStatusException: 69
```
[HailoRT] [error] CHECK failed - Trying to write to vstream before its network group is activated
```

**Solutions:**

1. **For underscore API - Activate before creating vstreams**
   ```python
   # Correct order:
   network_group.activate()
   network_group.wait_for_activation(5000)
   # THEN create vstreams
   input_vstreams = network_group._create_input_vstreams(params)
   ```

2. **For regular API - Start/stop vstreams**
   ```python
   # Start before inference
   input_vstream.start()
   output_vstream.start()
   
   # Run inference
   input_vstream.send(data)
   output = output_vstream.recv()
   
   # Stop after inference
   input_vstream.stop()
   output_vstream.stop()
   ```

### 4. Performance Issues

#### Problem: Low FPS / High Latency
**Diagnostic steps:**
```bash
# Check current performance
hailortcli measure-power-info
hailortcli monitor

# Check thermal throttling
cat /sys/class/hwmon/hwmon*/temp1_input
```

**Solutions:**

1. **Optimize batch size**
   ```python
   # Increase batch size for better throughput
   params.batch_size = 8  # Default is 1
   ```

2. **Check input preprocessing**
   ```python
   # Ensure efficient preprocessing
   # Bad: Multiple operations
   image = cv2.imread(path)
   image = cv2.resize(image, (640, 640))
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   image = image.astype(np.float32) / 255.0
   
   # Good: Combined operations
   image = cv2.resize(cv2.imread(path), (640, 640))
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
   ```

3. **Reduce unnecessary operations**
   ```python
   # Don't start/stop vstreams for each frame if using regular API
   # Start once, run many inferences, stop once
   ```

### 5. Temperature and Stability Issues

#### Problem: High temperature (>85°C)
**Solutions:**

1. **Improve cooling**
   ```bash
   # Add active cooling
   # Install M.2 heatsink with fan
   
   # Reduce workload
   echo 20 > /sys/class/hwmon/hwmon*/pwm1  # Reduce to 20% workload
   ```

2. **Reduce inference rate**
   ```python
   # Add frame skipping
   frame_skip = 2  # Process every 3rd frame
   if frame_id % (frame_skip + 1) == 0:
       detections = model.infer(frame)
   ```

#### Problem: Random crashes/segfaults
**Solutions:**

1. **Check memory usage**
   ```bash
   # Monitor memory during inference
   watch -n 1 'free -h && echo && ps aux | grep python | head -5'
   ```

2. **Fix memory leaks**
   ```python
   # Ensure proper cleanup
   class HailoInference:
       def __del__(self):
           if hasattr(self, 'network_group'):
               self.network_group.shutdown()
   ```

3. **Use process isolation**
   ```python
   # Run inference in separate process
   from multiprocessing import Process, Queue
   
   def inference_worker(hef_path, input_queue, output_queue):
       model = HailoModel(hef_path)
       while True:
           image = input_queue.get()
           if image is None:
               break
           result = model.infer(image)
           output_queue.put(result)
   ```

### 6. Docker-Specific Issues

#### Problem: Device not accessible in container
**Solution:**
```yaml
# docker-compose.yml
services:
  hailo-app:
    devices:
      - /dev/hailo0:/dev/hailo0
    group_add:
      - video
    privileged: true  # If needed
```

#### Problem: Driver version mismatch
**Solution:**
```dockerfile
# Ensure container uses same HailoRT version
FROM ubuntu:20.04
ARG HAILORT_VERSION=4.21.0
RUN apt-get update && \
    apt-get install -y hailort=${HAILORT_VERSION}
```

### 7. Model Conversion Issues

#### Problem: Accuracy degradation after quantization
**Solutions:**

1. **Improve calibration dataset**
   ```python
   # Use representative data
   calibration_data = load_wildfire_images()  # Not random data
   
   # Ensure sufficient samples
   assert len(calibration_data) >= 100  # Minimum
   ```

2. **Try different quantization methods**
   ```bash
   # Use QAT instead of PTQ
   python convert_model.py --qat --epochs 10
   ```

#### Problem: Model not compatible with target
```
Error: Model compiled for HAILO8 but running on HAILO8L
```

**Solution:**
```bash
# Recompile for correct target
hailo_compiler.compile(
    onnx_path="model.onnx",
    target="hailo8l",  # Specific target
    output_path="model_hailo8l.hef"
)
```

## Diagnostic Commands

### System Information
```bash
# Hailo device info
hailortcli device-info

# Driver version
modinfo hailo_pci | grep version

# Runtime version
python -c "import hailo_platform; print(hailo_platform.__version__)"

# Temperature and power
watch -n 1 'hailortcli measure-power-info'
```

### Performance Profiling
```bash
# Profile inference
hailortcli profile --hef model.hef --count 100

# Measure latency distribution
python -c "
import time
import numpy as np
latencies = []
for _ in range(100):
    start = time.perf_counter()
    # Run inference
    latencies.append((time.perf_counter() - start) * 1000)
print(f'P50: {np.percentile(latencies, 50):.1f}ms')
print(f'P95: {np.percentile(latencies, 95):.1f}ms')
print(f'P99: {np.percentile(latencies, 99):.1f}ms')
"
```

### Debug Logging
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('hailo').setLevel(logging.DEBUG)

# Or set environment variable
os.environ['HAILO_LOG_LEVEL'] = 'DEBUG'
```

## Best Practices for Stability

1. **Error Handling**
   ```python
   try:
       result = model.infer(image)
   except hailo_platform.HailoRTException as e:
       if e.status_code == 69:  # Network not activated
           model.reactivate()
       else:
           logger.error(f"Hailo error: {e}")
   ```

2. **Resource Management**
   ```python
   # Use context managers
   with HailoModel(hef_path) as model:
       results = model.infer(images)
   # Automatic cleanup
   ```

3. **Health Monitoring**
   ```python
   # Regular health checks
   def check_hailo_health():
       temp = device.get_temperature()
       if temp > 80:
           logger.warning(f"High temperature: {temp}°C")
       
       # Check inference latency
       test_latency = model.benchmark()
       if test_latency > 30:
           logger.warning(f"High latency: {test_latency}ms")
   ```

## Getting Help

1. **Collect Diagnostics**
   ```bash
   # Generate diagnostic report
   hailortcli diagnostic-report --output hailo_diagnostics.tar.gz
   ```

2. **Check Logs**
   ```bash
   # Kernel logs
   sudo dmesg | grep hailo
   
   # System logs
   journalctl -u hailo
   ```

3. **Community Resources**
   - Hailo Community Forum: https://community.hailo.ai/
   - GitHub Issues: https://github.com/hailo-ai/hailort/issues
   - Developer Docs: https://hailo.ai/developer-zone/documentation/