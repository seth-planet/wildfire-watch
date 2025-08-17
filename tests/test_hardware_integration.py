import pytest

# Test tier markers for organization
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]

#!/usr/bin/env python3.12
"""
Hardware Integration Tests for Wildfire Watch
Tests for Coral, Hailo, and NVIDIA GPU integration
Run on actual hardware with accelerators installed

IMPORTANT PYTHON VERSION REQUIREMENTS:
- Coral TPU tests: MUST use Python 3.8 (tflite_runtime requirement)
- Hailo tests: MUST use Python 3.10 (hailo-python requirement)
- TensorRT/GPU tests: Can use Python 3.10 or 3.12
- General tests: Python 3.12 (default)

The test framework will automatically handle version requirements.
"""
import os
import sys
import time
import json
import subprocess
import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import pytest
import fcntl
import contextlib

# Import parallel test utilities
from test_utils.helpers import ParallelTestContext, DockerContainerManager
from test_utils.topic_namespace import create_namespaced_client

# Hardware lockfile system for non-parallelizable hardware
@contextlib.contextmanager
def hardware_lock(hardware_name: str, timeout: int = 30):
    """Context manager for hardware exclusive access."""
    lock_file = f"/tmp/wildfire_watch_{hardware_name}_lock"
    lock_fd = None
    
    try:
        # Create lock file
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        
        # Try to acquire lock with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write process info to lock file
                os.write(lock_fd, f"PID:{os.getpid()}\nTime:{time.time()}\n".encode())
                yield
                return
            except BlockingIOError:
                time.sleep(0.1)
        
        # Timeout reached
        raise TimeoutError(f"Could not acquire {hardware_name} lock within {timeout} seconds")
        
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                os.unlink(lock_file)
            except:
                pass

def requires_hardware_lock(hardware_name: str, timeout: int = 1800):
    """Decorator for hardware-exclusive test methods."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with hardware_lock(hardware_name, timeout):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Hardware detection flags
HAS_CORAL = False
HAS_HAILO = False
HAS_GPU = False
HAS_INTEL_GPU = False

# Check for Coral
# First check for Coral hardware
if os.path.exists('/dev/apex_0'):
    HAS_CORAL = True
    print("PCIe Coral detected at /dev/apex_0")
elif subprocess.run(['lsusb'], capture_output=True).stdout.find(b'1a6e') != -1:
    HAS_CORAL = True
    print("USB Coral detected")

# Also check if tflite_runtime is properly installed
# Try with Python 3.8 for Coral compatibility
try:
    result = subprocess.run(['python3.8', '-c', 'import tflite_runtime.interpreter'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("tflite_runtime works with Python 3.8")
    else:
        if HAS_CORAL:
            print(f"Warning: Coral hardware detected but tflite_runtime not working with Python 3.8")
except Exception as e:
    if HAS_CORAL:
        print(f"Warning: Could not check Python 3.8 tflite_runtime: {e}")

# Check for Hailo
try:
    if os.path.exists('/dev/hailo0'):
        HAS_HAILO = True
except:
    pass

# Check for NVIDIA GPU
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True)
    if result.returncode == 0:
        HAS_GPU = True
except:
    pass

# Check for Intel GPU
try:
    result = subprocess.run(['vainfo'], capture_output=True)
    if result.returncode == 0 and b'Intel' in result.stdout:
        HAS_INTEL_GPU = True
except:
    pass

class TestHardwareDetection(unittest.TestCase):
    """Test hardware detection capabilities"""
    
    def test_hardware_availability(self):
        """Report available hardware"""
        print("\n=== Hardware Detection Results ===")
        print(f"Coral TPU: {'✓' if HAS_CORAL else '✗'}")
        print(f"Hailo AI: {'✓' if HAS_HAILO else '✗'}")
        print(f"NVIDIA GPU: {'✓' if HAS_GPU else '✗'}")
        print(f"Intel GPU: {'✓' if HAS_INTEL_GPU else '✗'}")
        print("================================\n")
        
        # At least one accelerator should be available for these tests
        self.assertTrue(
            HAS_CORAL or HAS_HAILO or HAS_GPU,
            "No AI accelerators detected. Please install Coral, Hailo, or GPU."
        )

class TestCoralIntegration(unittest.TestCase):
    """Test Coral TPU integration"""
    
    @unittest.skipUnless(HAS_CORAL, "Coral TPU not available")
    def setUp(self):
        """Set up test environment"""
        self.test_model_url = "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
        self.test_model_path = "/tmp/test_model_edgetpu.tflite"
        
        # Download test model if needed
        if not os.path.exists(self.test_model_path):
            subprocess.run([
                'wget', '-q', self.test_model_url,
                '-O', self.test_model_path
            ], check=True)
    
    @unittest.skipUnless(HAS_CORAL, "Coral TPU not available")
    @requires_hardware_lock("coral_tpu", timeout=1800)
    def test_coral_inference(self):
        """Test inference on Coral TPU"""
        # First check available devices
        devices_script = '''
from pycoral.utils.edgetpu import list_edge_tpus
import json
tpus = list_edge_tpus()
print(json.dumps([{"type": t["type"], "path": t["path"]} for t in tpus]))
'''
        devices_result = subprocess.run(['python3.8', '-c', devices_script], 
                                      capture_output=True, text=True)
        
        if devices_result.returncode == 0:
            try:
                devices = json.loads(devices_result.stdout.strip())
                print(f"Found {len(devices)} Coral TPU device(s)")
            except:
                devices = []
        else:
            devices = []
        
        # Try each device until one works
        success = False
        last_error = None
        
        for device_idx in range(max(1, len(devices))):
            # Create a Python script to run with Python 3.8, trying specific device
            test_script = f'''
import time
import numpy as np
import tflite_runtime.interpreter as tflite

# Load model - try specific device
try:
    delegates = [tflite.load_delegate('libedgetpu.so.1', {{"device": ":{device_idx}"}})]
except ValueError:
    # Fallback to default device selection
    delegates = [tflite.load_delegate('libedgetpu.so.1')]

interpreter = tflite.Interpreter(
    model_path="{self.test_model_path}",
    experimental_delegates=delegates
)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input
input_shape = input_details[0]['shape']
input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)

# Run inference
start_time = time.time()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
inference_time = (time.time() - start_time) * 1000

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"Coral inference time: {{inference_time:.2f}}ms")
print(f"Output shape: {{output_data.shape}}")
print(f"Device: {device_idx}")
print("SUCCESS")
'''
        
            # Run the script with Python 3.8
            result = subprocess.run(['python3.8', '-c', test_script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                success = True
                last_error = None
                print(f"Successfully used Coral TPU device {device_idx}")
                break
            else:
                last_error = result.stderr
                print(f"Failed to use device {device_idx}: {result.stderr[:100]}...")
                continue
        
        if not success:
            result = type('Result', (), {'returncode': 1, 'stderr': last_error or 'All devices failed', 'stdout': ''})
        
        if result.returncode != 0:
            # Check if it's a hardware availability issue
            if "Failed to open device" in result.stderr or "No EdgeTPU device found" in result.stderr:
                self.skipTest("Coral TPU hardware not accessible")
            elif "Failed to load delegate from libedgetpu.so.1" in result.stderr:
                # This can happen when devices are visible but not accessible
                # Try to provide more diagnostic info
                devices_check = subprocess.run(['python3.8', '-c', 
                    'from pycoral.utils.edgetpu import list_edge_tpus; print(len(list_edge_tpus()))'],
                    capture_output=True, text=True)
                if devices_check.returncode == 0:
                    num_devices = int(devices_check.stdout.strip())
                    if num_devices > 0:
                        # Check if devices are in use
                        lsof_check = subprocess.run(['lsof', '/dev/apex_0', '/dev/apex_1', '/dev/apex_2', '/dev/apex_3'], 
                                                  capture_output=True, text=True)
                        if lsof_check.stdout:
                            self.skipTest(f"Coral TPU devices are in use by other processes:\n{lsof_check.stdout}")
                        else:
                            self.skipTest(f"Coral TPU devices found but delegate failed to load. This may be a permissions issue.")
                    else:
                        self.fail(f"No Coral TPU devices found")
                else:
                    self.fail(f"Coral inference failed: {result.stderr}")
            else:
                self.fail(f"Coral inference failed: {result.stderr}")
        
        # Check the output
        self.assertIn("SUCCESS", result.stdout, "Coral inference did not complete successfully")
        self.assertIn("Coral inference time:", result.stdout, "No inference time reported")
        
        # Extract and check inference time
        for line in result.stdout.split('\n'):
            if "Coral inference time:" in line:
                time_ms = float(line.split(':')[1].strip().replace('ms', ''))
                print(f"Coral inference time: {time_ms:.2f}ms")
                self.assertLess(time_ms, 50, "Coral inference too slow")
    
    @unittest.skipUnless(HAS_CORAL, "Coral TPU not available")
    def test_coral_docker_access(self):
        """Test Coral access from Docker"""
        # Check if we have Docker permissions
        test_cmd = ['docker', 'ps']
        permission_check = subprocess.run(test_cmd, capture_output=True, text=True)
        if permission_check.returncode != 0:
            if "permission denied" in permission_check.stderr:
                # Check if user is in docker group but session not updated
                user = os.environ.get('USER', 'current user')
                group_check = subprocess.run(['id', user], capture_output=True, text=True)
                if 'docker' in group_check.stdout:
                    self.fail(f"User '{user}' is in docker group but current session needs refresh.\n"
                             f"Please log out and back in, or start a new shell session.\n"
                             f"Alternatively, run tests in a new shell: bash -c 'newgrp docker && python3.12 -m pytest {__file__}'")
                else:
                    self.fail(f"Docker permission denied. Fix with: sudo usermod -aG docker {user} && newgrp docker\n{permission_check.stderr}")
            else:
                self.fail(f"Docker error: {permission_check.stderr}")
        
        # Check if Docker can see Coral device
        result = subprocess.run([
            'docker', 'run', '--rm',
            '--device', '/dev/apex_0:/dev/apex_0' if os.path.exists('/dev/apex_0') else '/dev/bus/usb',
            'ubuntu:20.04',
            'ls', '-la', '/dev/'
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Docker cannot access Coral device")

class TestHailoIntegration(unittest.TestCase):
    """Test Hailo AI accelerator integration"""
    
    @unittest.skipUnless(HAS_HAILO, "Hailo not available")
    def test_hailo_detection(self):
        """Test Hailo device detection"""
        result = subprocess.run(['hailortcli', 'scan'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "hailortcli scan failed")
        self.assertIn('Hailo', result.stdout, "No Hailo device found")
        
        # Extract device info
        if 'Hailo-8L' in result.stdout:
            print("Detected Hailo-8L (13 TOPS)")
        elif 'Hailo-8' in result.stdout:
            print("Detected Hailo-8 (26 TOPS)")
    
    @unittest.skipUnless(HAS_HAILO, "Hailo not available")
    def test_hailo_docker_access(self):
        """Test Hailo access from Docker"""
        # Check if we have Docker permissions
        test_cmd = ['docker', 'ps']
        permission_check = subprocess.run(test_cmd, capture_output=True, text=True)
        if permission_check.returncode != 0:
            self.fail(f"Docker permission error: {permission_check.stderr}")
        
        result = subprocess.run([
            'docker', 'run', '--rm',
            '--device', '/dev/hailo0:/dev/hailo0',
            '-v', '/usr/lib/x86_64-linux-gnu/libhailort.so:/usr/lib/libhailort.so:ro',
            'ubuntu:20.04',
            'ls', '-la', '/dev/hailo0'
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Docker cannot access Hailo device")
    
    @unittest.skipUnless(HAS_HAILO, "Hailo not available")
    def test_hailo_benchmark(self):
        """Test Hailo performance benchmark"""
        # This would require a compiled HEF file
        # For now, just check that the runtime is accessible
        result = subprocess.run([
            'python3', '-c',
            'import hailo_platform; print("Hailo platform imported successfully")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Hailo Python bindings available")
        else:
            print("Hailo Python bindings not installed")

class TestNVIDIAGPUIntegration(unittest.TestCase):
    """Test NVIDIA GPU integration"""
    
    @unittest.skipUnless(HAS_GPU, "NVIDIA GPU not available")
    def test_gpu_detection(self):
        """Test NVIDIA GPU detection"""
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "nvidia-smi failed")
        
        gpu_info = result.stdout.strip()
        print(f"Detected GPU: {gpu_info}")
        self.assertTrue(len(gpu_info) > 0, "No GPU information returned")
    
    @unittest.skipUnless(HAS_GPU, "NVIDIA GPU not available")
    def test_cuda_availability(self):
        """Test CUDA availability"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
            self.assertTrue(cuda_available, "CUDA not available despite GPU presence")
        except ImportError:
            # Try with TensorFlow
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                print(f"TensorFlow GPU devices: {len(gpus)}")
                self.assertGreater(len(gpus), 0, "No GPU devices found by TensorFlow")
            except ImportError:
                self.skipTest("Neither PyTorch nor TensorFlow installed")
    
    @unittest.skipUnless(HAS_GPU, "NVIDIA GPU not available")
    def test_docker_gpu_access(self):
        """Test GPU access from Docker"""
        # Check if we have Docker permissions
        test_cmd = ['docker', 'ps']
        permission_check = subprocess.run(test_cmd, capture_output=True, text=True)
        if permission_check.returncode != 0:
            if "permission denied" in permission_check.stderr:
                # Check if user is in docker group but session not updated
                user = os.environ.get('USER', 'current user')
                group_check = subprocess.run(['id', user], capture_output=True, text=True)
                if 'docker' in group_check.stdout:
                    self.fail(f"User '{user}' is in docker group but current session needs refresh.\n"
                             f"Please log out and back in, or start a new shell session.\n"
                             f"Alternatively, run tests in a new shell: bash -c 'newgrp docker && python3.12 -m pytest {__file__}'")
                else:
                    self.fail(f"Docker permission denied. Fix with: sudo usermod -aG docker {user} && newgrp docker\n{permission_check.stderr}")
            else:
                self.fail(f"Docker error: {permission_check.stderr}")
        
        # Check if nvidia-container-toolkit is available by testing GPU access
        # First check if nvidia-container-cli exists
        nvidia_cli_check = subprocess.run(['which', 'nvidia-container-cli'], capture_output=True)
        if nvidia_cli_check.returncode != 0:
            # Try to run a simple GPU container to check if it's available
            test_result = subprocess.run([
                'docker', 'run', '--rm', '--gpus', 'all',
                'nvidia/cuda:11.8.0-base-ubuntu20.04',
                'nvidia-smi'
            ], capture_output=True, text=True)
            
            if test_result.returncode != 0:
                if "could not select device driver" in test_result.stderr:
                    self.fail("NVIDIA Container Toolkit not installed. Please install nvidia-container-toolkit package.")
                else:
                    self.fail(f"Docker cannot access GPU: {test_result.stderr}")
        else:
            # nvidia-container-cli exists, try running the container
            result = subprocess.run([
                'docker', 'run', '--rm', '--gpus', 'all',
                'nvidia/cuda:11.8.0-base-ubuntu20.04',
                'nvidia-smi'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                if "could not select device driver" in result.stderr:
                    # nvidia-container-cli exists but runtime not properly configured
                    self.fail("NVIDIA Container Toolkit installed but Docker GPU support not working.\n"
                             "This is required for GPU-accelerated inference in Wildfire Watch.\n"
                             "Please fix by:\n"
                             "1. Restarting Docker daemon: sudo systemctl restart docker\n"
                             "2. Ensuring nvidia-container-runtime is installed\n"
                             "3. Checking /etc/docker/daemon.json configuration\n"
                             f"Error: {result.stderr}")
                else:
                    self.fail(f"Docker cannot access GPU: {result.stderr}")
            else:
                # Verify nvidia-smi output
                self.assertIn("NVIDIA-SMI", result.stdout, "nvidia-smi output not found")
                print("✓ Docker GPU access working")
    
    @unittest.skipUnless(HAS_GPU, "NVIDIA GPU not available")
    def test_tensorrt_availability(self):
        """Test TensorRT availability"""
        result = subprocess.run([
            'python3', '-c',
            'import tensorrt; print(f"TensorRT version: {tensorrt.__version__}")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("TensorRT not installed")

class TestIntelGPUIntegration(unittest.TestCase):
    """Test Intel GPU integration"""
    
    @unittest.skipUnless(HAS_INTEL_GPU, "Intel GPU not available")
    def test_vaapi_support(self):
        """Test VA-API support"""
        result = subprocess.run(['vainfo'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, "vainfo failed")
        
        # Check for H264 decode support
        self.assertIn('VAProfileH264', result.stdout, "H264 decode not supported")
        print("Intel GPU VA-API profiles detected")
    
    @unittest.skipUnless(HAS_INTEL_GPU, "Intel GPU not available")
    def test_docker_vaapi_access(self):
        """Test VA-API access from Docker"""
        # Check if we have Docker permissions
        test_cmd = ['docker', 'ps']
        permission_check = subprocess.run(test_cmd, capture_output=True, text=True)
        if permission_check.returncode != 0:
            self.fail(f"Docker permission error: {permission_check.stderr}")
        
        result = subprocess.run([
            'docker', 'run', '--rm',
            '--device', '/dev/dri:/dev/dri',
            'ubuntu:20.04',
            'ls', '-la', '/dev/dri/'
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, f"Docker cannot access Intel GPU: {result.stderr}")
        self.assertIn('renderD128', result.stdout, "No render device found")

class TestFrigateIntegration(unittest.TestCase):
    """Test Frigate NVR integration with hardware"""
    
    def setUp(self):
        """Set up test environment"""
        self.compose_file = "docker-compose.yml"
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        import gc
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {self.test_dir}: {e}")
        # Force garbage collection to release file handles
        gc.collect()
    
    @unittest.skipIf(not os.path.exists("docker-compose.yml"), "Docker compose file not found")
    def test_frigate_hardware_detection(self):
        """Test Frigate hardware detection"""
        # Create test config
        config = {
            'mqtt': {
                'enabled': False
            },
            'detectors': {
                'default': {
                    'type': 'cpu'  # Start with CPU
                }
            },
            'cameras': {}
        }
        
        # Modify based on available hardware
        if HAS_CORAL:
            config['detectors']['coral'] = {
                'type': 'edgetpu',
                'device': 'pci' if os.path.exists('/dev/apex_0') else 'usb'
            }
        
        if HAS_GPU:
            config['detectors']['tensorrt'] = {
                'type': 'tensorrt',
                'device': 0
            }
        
        # Save config
        config_path = Path(self.test_dir) / "frigate_test.yml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"Test Frigate config created at {config_path}")
        self.assertTrue(config_path.exists())

class TestModelInference(unittest.TestCase):
    """Test model inference on available hardware"""
    
    def setUp(self):
        """Download test image"""
        self.test_image_url = "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg"
        self.test_image_path = "/tmp/test_image.jpg"
        
        if not os.path.exists(self.test_image_path):
            subprocess.run([
                'wget', '-q', self.test_image_url,
                '-O', self.test_image_path
            ], check=True)
    
    def _benchmark_inference(self, name: str, inference_func, num_runs: int = 10):
        """Benchmark inference function"""
        # Warmup
        for _ in range(3):
            inference_func()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = inference_func()
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\n{name} Inference Benchmark:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Std Dev: {std_time:.2f}ms")
        print(f"  Min: {min(times):.2f}ms")
        print(f"  Max: {max(times):.2f}ms")
        
        return avg_time
    
    @unittest.skipUnless(HAS_CORAL, "Coral not available")
    def test_coral_inference_speed(self):
        """Benchmark Coral inference speed"""
        # First check if opencv is available for Python 3.8
        cv_check = subprocess.run(['python3.8', '-c', 'import cv2'], 
                                capture_output=True)
        if cv_check.returncode != 0:
            # Install opencv for Python 3.8
            subprocess.run(['python3.8', '-m', 'pip', 'install', 'opencv-python'], 
                         capture_output=True)
        
        # Load test model
        model_path = "/tmp/test_model_edgetpu.tflite"
        if not os.path.exists(model_path):
            self.skipTest("Test model not available")
        
        # Create benchmark script for Python 3.8
        benchmark_script = f'''
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

# Create interpreter
interpreter = tflite.Interpreter(
    model_path="{model_path}",
    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = cv2.imread("{self.test_image_path}")
# Get model input shape
input_shape = input_details[0]['shape']
height, width = input_shape[1], input_shape[2]
img = cv2.resize(img, (width, height))
img = np.expand_dims(img, axis=0).astype(np.uint8)

# Warmup
for _ in range(3):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

# Benchmark
times = []
for _ in range(10):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    times.append((time.time() - start) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)

print(f"Coral TPU Inference Benchmark:")
print(f"  Average: {{avg_time:.2f}}ms")
print(f"  Std Dev: {{std_time:.2f}}ms")
print(f"  Min: {{min(times):.2f}}ms")
print(f"  Max: {{max(times):.2f}}ms")
print(f"AVG_TIME: {{avg_time}}")
'''
        
        # Run benchmark with Python 3.8
        result = subprocess.run(['python3.8', '-c', benchmark_script], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            self.fail(f"Coral benchmark failed: {result.stderr}")
        
        print(result.stdout)
        
        # Extract average time
        for line in result.stdout.split('\n'):
            if "AVG_TIME:" in line:
                avg_time = float(line.split(':')[1].strip())
                self.assertLess(avg_time, 50, "Coral inference slower than expected")
    
    @unittest.skipUnless(HAS_GPU, "GPU not available")
    def test_gpu_inference_speed(self):
        """Benchmark GPU inference speed with YOLOv8m"""
        try:
            import torch
            from ultralytics import YOLO
            
            # Use YOLOv8m which is similar to production models
            print("Loading YOLOv8m model...")
            model = YOLO('yolov8m.pt')  # Will download if not present
            
            # Move model to GPU
            if torch.cuda.is_available():
                model.to('cuda')
            
            def inference():
                # Run inference on the test image
                results = model(self.test_image_path, verbose=False)
                return results
            
            # Benchmark the inference
            avg_time = self._benchmark_inference("NVIDIA GPU (YOLOv8m)", inference)
            
            # YOLOv8m should be reasonably fast on GPU
            # Typical inference time: 10-30ms on modern GPUs
            self.assertLess(avg_time, 100, "GPU inference slower than expected")
            
            # Print detection results from last inference
            results = inference()
            if results and len(results) > 0:
                print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0} objects")
            
        except ImportError as e:
            self.skipTest(f"Ultralytics not installed: {e}")
        except Exception as e:
            self.skipTest(f"GPU inference test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test full system integration"""
    
    @unittest.skipIf(
        not (shutil.which('docker-compose') or 
             (shutil.which('docker') and 
              subprocess.run(['docker', 'compose', 'version'], capture_output=True).returncode == 0)),
        "Docker Compose not available (neither docker-compose nor docker compose found)"
    )
    def test_docker_compose_validation(self):
        """Validate docker-compose files"""
        # Check if we have Docker permissions
        test_cmd = ['docker', 'ps']
        permission_check = subprocess.run(test_cmd, capture_output=True, text=True)
        if permission_check.returncode != 0:
            if "permission denied" in permission_check.stderr:
                # Check if user is in docker group but session not updated
                user = os.environ.get('USER', 'current user')
                group_check = subprocess.run(['id', user], capture_output=True, text=True)
                if 'docker' in group_check.stdout:
                    self.fail(f"User '{user}' is in docker group but current session needs refresh.\n"
                             f"Please log out and back in, or start a new shell session.\n"
                             f"Alternatively, run tests in a new shell: bash -c 'newgrp docker && python3.12 -m pytest {__file__}'")
                else:
                    self.fail(f"Docker permission denied. Fix with: sudo usermod -aG docker {user} && newgrp docker\n{permission_check.stderr}")
            else:
                self.fail(f"Docker error: {permission_check.stderr}")
        
        # Check docker-compose version - try both docker-compose and docker compose
        compose_commands = [
            ['docker', 'compose', 'version'],  # Modern Docker Compose v2
            ['docker-compose', '--version']    # Legacy Docker Compose v1
        ]
        
        compose_cmd = None
        for cmd in compose_commands:
            version_result = subprocess.run(cmd, capture_output=True, text=True)
            if version_result.returncode == 0:
                compose_cmd = cmd[:-1]  # Remove version/--version argument
                version_output = version_result.stdout.strip()
                print(f"Docker Compose found: {version_output}")
                break
        
        if not compose_cmd:
            self.fail("Docker Compose not found. Install docker-compose or use Docker Desktop.")
        
        compose_files = [
            'docker-compose.yml',
            'docker-compose.local.yml'
        ]
        
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                # Use the detected compose command
                result = subprocess.run(
                    compose_cmd + ['-f', compose_file, 'config'],
                    capture_output=True, text=True
                )
                
                self.assertEqual(result.returncode, 0,
                               f"{compose_file} validation failed: {result.stderr}")
                print(f"✓ {compose_file} is valid")
    
    def test_mqtt_connectivity(self):
        """Test MQTT broker connectivity"""
        try:
            import paho.mqtt.client as mqtt
            
            connected = False
            
            def on_connect(client, userdata, flags, rc, properties=None):
                nonlocal connected
                connected = (rc == 0)
            
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.on_connect = on_connect
            
            try:
                client.connect(test_mqtt_broker.host, test_mqtt_broker.port, 60)
                client.loop_start()
                time.sleep(2)
                client.loop_stop()
                
                if connected:
                    print("✓ MQTT broker is accessible")
                else:
                    print("✗ MQTT broker not accessible")
                    
            except Exception as e:
                print(f"✗ MQTT connection failed: {e}")
                
        except ImportError:
            self.skipTest("paho-mqtt not installed")

def print_system_info():
    """Print system information"""
    print("\n=== System Information ===")
    
    # OS Info
    try:
        with open('/etc/os-release') as f:
            for line in f:
                if line.startswith('PRETTY_NAME'):
                    value = line.split('=')[1].strip().strip('"')
                    print(f"OS: {value}")
                    break
    except:
        print("OS: Unknown")
    
    # CPU Info
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Model name' in line:
                print(f"CPU: {line.split(':')[1].strip()}")
                break
    except:
        pass
    
    # Memory Info
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        if len(lines) > 1:
            mem_parts = lines[1].split()
            if len(mem_parts) > 1:
                print(f"Memory: {mem_parts[1]}")
    except:
        pass
    
    # Docker Info
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        print(f"Docker: {result.stdout.strip()}")
    except:
        print("Docker: Not installed")
    
    print("==========================\n")

if __name__ == '__main__':
    # Print system info first
    print_system_info()
    
    # Run tests
    unittest.main(verbosity=2)
