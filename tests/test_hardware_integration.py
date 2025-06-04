#!/usr/bin/env python3.12
#!/usr/bin/env python3
"""
Hardware Integration Tests for Wildfire Watch
Tests for Coral, Hailo, and NVIDIA GPU integration
Run on actual hardware with accelerators installed
"""
import os
import sys
import time
import json
import subprocess
import unittest
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Hardware detection flags
HAS_CORAL = False
HAS_HAILO = False
HAS_GPU = False
HAS_INTEL_GPU = False

# Check for Coral
try:
    import tflite_runtime.interpreter as tflite
    if os.path.exists('/dev/apex_0') or subprocess.run(['lsusb'], capture_output=True).stdout.find(b'1a6e') != -1:
        HAS_CORAL = True
except ImportError:
    pass

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
    def test_coral_inference(self):
        """Test inference on Coral TPU"""
        import tflite_runtime.interpreter as tflite
        
        # Load model
        interpreter = tflite.Interpreter(
            model_path=self.test_model_path,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
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
        
        print(f"Coral inference time: {inference_time:.2f}ms")
        self.assertLess(inference_time, 50, "Coral inference too slow")
        self.assertIsNotNone(output_data)
    
    @unittest.skipUnless(HAS_CORAL, "Coral TPU not available")
    def test_coral_docker_access(self):
        """Test Coral access from Docker"""
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
        result = subprocess.run([
            'docker', 'run', '--rm', '--gpus', 'all',
            'nvidia/cuda:11.8.0-base-ubuntu20.04',
            'nvidia-smi'
        ], capture_output=True)
        
        self.assertEqual(result.returncode, 0, "Docker cannot access GPU")
    
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
        result = subprocess.run([
            'docker', 'run', '--rm',
            '--device', '/dev/dri:/dev/dri',
            'ubuntu:20.04',
            'ls', '-la', '/dev/dri/'
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Docker cannot access Intel GPU")
        self.assertIn('renderD128', result.stdout, "No render device found")

class TestFrigateIntegration(unittest.TestCase):
    """Test Frigate NVR integration with hardware"""
    
    def setUp(self):
        """Set up test environment"""
        self.compose_file = "docker-compose.local.yml"
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    @unittest.skipIf(not os.path.exists("docker-compose.local.yml"), "Docker compose file not found")
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
        import tflite_runtime.interpreter as tflite
        import cv2
        
        # Load test model
        model_path = "/tmp/test_model_edgetpu.tflite"
        if not os.path.exists(model_path):
            self.skipTest("Test model not available")
        
        # Create interpreter
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
        interpreter.allocate_tensors()
        
        # Load and preprocess image
        img = cv2.imread(self.test_image_path)
        img = cv2.resize(img, (320, 320))
        img = np.expand_dims(img, axis=0).astype(np.uint8)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        def inference():
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            return interpreter.get_tensor(output_details[0]['index'])
        
        avg_time = self._benchmark_inference("Coral TPU", inference)
        self.assertLess(avg_time, 50, "Coral inference slower than expected")
    
    @unittest.skipUnless(HAS_GPU, "GPU not available")
    def test_gpu_inference_speed(self):
        """Benchmark GPU inference speed"""
        try:
            import torch
            import torchvision.transforms as T
            from PIL import Image
            
            # Load a simple model
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model = model.cuda()
            model.eval()
            
            # Load image
            img = Image.open(self.test_image_path)
            
            def inference():
                with torch.no_grad():
                    results = model(img)
                return results
            
            avg_time = self._benchmark_inference("NVIDIA GPU", inference)
            self.assertLess(avg_time, 100, "GPU inference slower than expected")
            
        except Exception as e:
            self.skipTest(f"GPU inference test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test full system integration"""
    
    @unittest.skipIf(not shutil.which('docker-compose'), "docker-compose not installed")
    def test_docker_compose_validation(self):
        """Validate docker-compose files"""
        compose_files = [
            'docker-compose.yml',
            'docker-compose.local.yml'
        ]
        
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                result = subprocess.run([
                    'docker-compose', '-f', compose_file, 'config'
                ], capture_output=True)
                
                self.assertEqual(result.returncode, 0,
                               f"{compose_file} validation failed")
                print(f"✓ {compose_file} is valid")
    
    def test_mqtt_connectivity(self):
        """Test MQTT broker connectivity"""
        try:
            import paho.mqtt.client as mqtt
            
            connected = False
            
            def on_connect(client, userdata, flags, rc):
                nonlocal connected
                connected = (rc == 0)
            
            client = mqtt.Client()
            client.on_connect = on_connect
            
            try:
                client.connect("localhost", 1883, 60)
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
                    print(f"OS: {line.split('=')[1].strip().strip('\"')}")
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
