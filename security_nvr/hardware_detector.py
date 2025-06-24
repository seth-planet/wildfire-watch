#!/usr/bin/env python3
"""Hardware detection and configuration for Frigate NVR AI accelerators.

This module automatically detects available hardware acceleration capabilities
and generates optimal Frigate configuration. It probes for various AI accelerators
(Coral TPU, Hailo-8, NVIDIA GPU) and video decoding hardware (Intel/AMD VA-API,
NVIDIA NVDEC, Raspberry Pi V4L2) to maximize performance on edge devices.

The detector uses common Linux command-line utilities to probe hardware. This
approach is portable but can be fragile if tool output formats change. The
script is designed to fail gracefully when tools are missing or inaccessible.

Hardware Priority (for AI detection):
    1. NVIDIA GPU with TensorRT - Highest performance
    2. Hailo-8/8L - Purpose-built AI accelerator
    3. Google Coral TPU - Efficient edge AI
    4. CPU - Fallback option

Video Decoding Priority:
    1. NVIDIA NVDEC - Hardware decoding on NVIDIA GPUs
    2. Intel/AMD VA-API - Hardware decoding via Video Acceleration API
    3. Raspberry Pi V4L2 - Hardware decoding on Pi 4/5
    4. Software decoding - CPU fallback

Required Command-Line Tools:
    - lspci, lsusb: Hardware enumeration (pciutils, usbutils packages)
    - vainfo: VA-API capability detection (libva-utils package)
    - nvidia-smi: NVIDIA GPU detection (nvidia-driver package)
    - hailortcli: Hailo device detection (hailo-driver package)

Container Requirements:
    - Device access: /dev/dri/*, /dev/bus/usb/*, /dev/hailo*, /dev/apex_*
    - For NVIDIA: nvidia-container-runtime or device mapping
    - For Coral USB: Privileged mode or specific USB device mapping

Known Issues:
    1. Missing hwaccel configuration: Current implementation doesn't set
       hardware video decoding flags, forcing CPU decoding
    2. Fragile GPU detection: Relies on string parsing of 'ls' output
    3. No permission checks: Doesn't verify if detected devices are accessible
    4. Hailo duplication: May report same device twice if both methods succeed

Example:
    Run detection and print results:
        $ python hardware_detector.py
        
    Export configuration to file:
        $ python hardware_detector.py --export
        
    Use in Docker:
        $ docker run --device /dev/dri --device /dev/bus/usb \\
            wildfire-nvr python hardware_detector.py

Note:
    For production use, ensure the container has appropriate device
    mappings and the required command-line tools are installed.
"""
import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import centralized command runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.command_runner import run_command, CommandError

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detects available hardware acceleration for AI inference and video decoding.
    
    This class probes the system for various hardware components that can
    accelerate wildfire detection. It runs detection methods sequentially
    during initialization and stores results for configuration generation.
    
    The detector is designed to work in containerized environments where
    not all hardware may be accessible. It fails gracefully when tools
    are missing or devices are not mapped into the container.
    
    Attributes:
        detected_hardware (Dict): Complete hardware inventory with subkeys:
            - cpu: Processor information and capabilities
            - gpu: Graphics card for video decoding and AI
            - coral: Google Coral TPU devices (USB and PCIe)
            - hailo: Hailo-8 AI processors
            - memory: System RAM information
            - platform: OS and container environment
            
    Detection Methods:
        Each hardware type has a dedicated detection method that returns
        a dictionary with device-specific information. Methods handle
        missing tools and inaccessible devices gracefully.
        
    Thread Safety:
        This class is not thread-safe. Create separate instances for
        concurrent use.
    """
    
    def __init__(self):
        """Initialize detector and run all hardware detection methods.
        
        Executes all detection methods sequentially and stores results.
        Detection failures are logged but don't prevent initialization.
        The detected_hardware dictionary is always populated, though
        values may indicate no hardware found.
        """
        self.detected_hardware = {
            'cpu': self._detect_cpu(),
            'gpu': self._detect_gpu(),
            'coral': self._detect_coral(),
            'hailo': self._detect_hailo(),
            'memory': self._detect_memory(),
            'platform': self._detect_platform(),
        }
        
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information"""
        info = {
            'vendor': 'unknown',
            'model': 'unknown',
            'cores': os.cpu_count() or 1,
            'architecture': os.uname().machine,
        }
        
        # Check CPU info
        try:
            _, cpuinfo, _ = run_command(['cat', '/proc/cpuinfo'], check=False)
            if cpuinfo:
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['model'] = line.split(':')[1].strip()
                    elif 'vendor_id' in line:
                        info['vendor'] = line.split(':')[1].strip()
        except (FileNotFoundError, PermissionError):
            logger.warning("Could not read /proc/cpuinfo")
        except CommandError as e:
            logger.debug(f"Error reading cpuinfo: {e}")
                    
        # Check for Raspberry Pi
        if os.path.exists('/proc/device-tree/model'):
            try:
                _, model, _ = run_command(['cat', '/proc/device-tree/model'], check=False)
                if model and 'Raspberry Pi' in model:
                    info['platform'] = 'raspberry_pi'
                    if 'Pi 5' in model:
                        info['model'] = 'Raspberry Pi 5'
                        info['has_v4l2'] = True
                        info['has_h265_decode'] = True
            except (FileNotFoundError, PermissionError, CommandError):
                pass
                    
        return info
    
    def _get_render_devices(self) -> List[str]:
        """Get list of available DRM render devices.
        
        Returns:
            List of render device paths (e.g., ['/dev/dri/renderD128', '/dev/dri/renderD129'])
        """
        render_devices = []
        try:
            import glob
            devices = glob.glob('/dev/dri/renderD*')
            render_devices = sorted(devices)  # Sort for consistency
        except Exception as e:
            logger.debug(f"Error listing render devices: {e}")
        return render_devices
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities for video decoding and AI inference.
        
        Probes for Intel, AMD, and NVIDIA GPUs using various methods.
        For Intel/AMD, uses VA-API (Video Acceleration API) to determine
        hardware decoding capabilities. For NVIDIA, uses nvidia-smi.
        
        Returns:
            dict: GPU information with keys:
                - vendor: 'intel', 'amd', 'nvidia', or None
                - model: GPU model name (NVIDIA only)
                - driver: 'vaapi' or 'nvdec'
                - decode: List of supported codecs ['h264', 'h265']
                - encode: List of supported encoding (currently unused)
                
        Detection Methods:
            - Intel/AMD: Checks for /dev/dri/renderD* devices and runs vainfo
            - NVIDIA: Runs nvidia-smi to query GPU name
            
        Known Issues:
            - AMD detection relies on fragile 'ls' output parsing
            - Assumes renderD128 exists, but there may be multiple devices
            - No verification that detected devices are accessible
            
        Note:
            VA-API detection requires vainfo tool from libva-utils package.
            NVIDIA detection requires nvidia-smi from driver package.
        """
        gpu = {
            'vendor': None,
            'model': None,
            'driver': None,
            'decode': [],
            'encode': [],
            'render_device': None,
        }
        
        # Get available render devices
        render_devices = self._get_render_devices()
        if render_devices:
            gpu['render_device'] = render_devices[0]  # Use first available device
        
        # Check for Intel GPU
        try:
            _, vainfo, _ = run_command(['vainfo'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            vainfo = ""
        if vainfo and 'Intel' in vainfo:
            gpu['vendor'] = 'intel'
            gpu['driver'] = 'vaapi'
            if 'VAProfileH264' in vainfo:
                gpu['decode'].append('h264')
            if 'VAProfileHEVC' in vainfo:
                gpu['decode'].append('h265')
                
        # Check for AMD GPU
        if render_devices:
            try:
                _, gpu_info, _ = run_command(['ls', '-la', '/dev/dri/'], check=False)
            except (FileNotFoundError, PermissionError, CommandError):
                gpu_info = ""
            if gpu_info and 'amdgpu' in gpu_info:
                gpu['vendor'] = 'amd'
                gpu['driver'] = 'vaapi'
                
        # Check for NVIDIA GPU
        try:
            _, nvidia_smi, _ = run_command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            nvidia_smi = ""
        if nvidia_smi:
            gpu['vendor'] = 'nvidia'
            gpu['model'] = nvidia_smi
            gpu['driver'] = 'nvdec'
            gpu['decode'] = ['h264', 'h265']
            
        return gpu
    
    def _detect_coral(self) -> Dict:
        """Detect Coral AI accelerators"""
        coral = {
            'usb': [],
            'pcie': [],
        }
        
        # Check USB Coral
        try:
            _, lsusb, _ = run_command(['lsusb'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            lsusb = ""
        if lsusb:
            # Global Unichip Corp (1a6e) or Google Inc (18d1)
            if '1a6e:089a' in lsusb or '18d1:9302' in lsusb:
                coral['usb'].append({
                    'id': '1a6e:089a',
                    'name': 'Coral USB Accelerator',
                    'path': '/dev/bus/usb',
                })
                
        # Check PCIe Coral
        try:
            _, lspci, _ = run_command(['lspci', '-nn'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            lspci = ""
        if lspci and '089a' in lspci:
            coral['pcie'].append({
                'name': 'Coral PCIe Accelerator',
                'path': '/dev/apex_0',
            })
            
        return coral
    
    def _detect_hailo(self) -> Dict:
        """Detect Hailo AI accelerators"""
        hailo = {
            'devices': [],
        }
        
        # Check for Hailo devices
        if os.path.exists('/dev/hailo0'):
            hailo['devices'].append({
                'name': 'Hailo-8',
                'path': '/dev/hailo0',
                'tops': 26,
            })
            
        # Check Hailo CLI
        try:
            _, hailortcli, _ = run_command(['hailortcli', 'scan'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            hailortcli = ""
        if hailortcli:
            if 'Hailo-8L' in hailortcli:
                hailo['devices'].append({
                    'name': 'Hailo-8L',
                    'path': '/dev/hailo0',
                    'tops': 13,
                })
                
        return hailo
    
    def _detect_memory(self) -> Dict:
        """Detect system memory"""
        memory = {
            'total_mb': 0,
            'available_mb': 0,
        }
        
        try:
            _, meminfo, _ = run_command(['cat', '/proc/meminfo'], check=False)
        except (FileNotFoundError, PermissionError, CommandError):
            meminfo = ""
        if meminfo:
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    memory['total_mb'] = int(line.split()[1]) // 1024
                elif 'MemAvailable:' in line:
                    memory['available_mb'] = int(line.split()[1]) // 1024
                    
        return memory
    
    def _detect_platform(self) -> Dict:
        """Detect platform information"""
        platform = {
            'os': 'linux',
            'arch': os.uname().machine,
            'hostname': os.uname().nodename,
        }
        
        # Check if running in Docker
        if os.path.exists('/.dockerenv'):
            platform['container'] = 'docker'
            
        # Check for Balena
        if os.environ.get('BALENA_DEVICE_UUID'):
            platform['container'] = 'balena'
            platform['device_uuid'] = os.environ.get('BALENA_DEVICE_UUID')
            
        return platform
    
    def get_recommended_config(self) -> Dict:
        """Generate optimal Frigate configuration based on detected hardware.
        
        Analyzes detected hardware and returns configuration that maximizes
        performance. Selects best available options for both AI detection
        and video decoding independently.
        
        Returns:
            dict: Frigate configuration with keys:
                - detector_type: AI accelerator type ('cpu', 'edgetpu', 'hailo', 'tensorrt')
                - detector_device: Device index/path
                - model_path: Path to model file for selected detector
                - hwaccel_args: FFmpeg hardware acceleration arguments
                - record_codec: Recording codec ('copy' to avoid transcoding)
                - record_preset: Encoding preset if transcoding
                - record_quality: Encoding quality if transcoding
                
        Priority Logic:
            AI Detection (in order):
                1. NVIDIA GPU with TensorRT (if available)
                2. Hailo-8/8L AI processor
                3. Google Coral TPU (USB or PCIe)
                4. CPU fallback
                
            Video Decoding (in order):
                1. NVIDIA NVDEC
                2. Intel/AMD VA-API
                3. Raspberry Pi V4L2
                4. Software decoding
                
        Critical Issue:
            Current implementation ONLY configures AI detection, completely
            ignoring hardware video decoding. This forces CPU decoding even
            when hardware acceleration is available, causing unnecessary
            CPU load and reducing camera capacity.
            
        Note:
            Model paths assume a standard directory structure. Actual paths
            may need adjustment based on deployment.
        """
        config = {
            'detector_type': 'cpu',
            'detector_device': '0',
            'model_path': '/models/wildfire/wildfire_cpu.tflite',
            'hwaccel_args': [],
            'record_codec': 'copy',
            'record_preset': 'fast',
            'record_quality': '23'
        }
        
        # Select best detector
        if self.detected_hardware['gpu']['vendor'] == 'nvidia':
            config['detector_type'] = 'tensorrt'
        elif self.detected_hardware['hailo']['devices']:
            config['detector_type'] = 'hailo'
            config['model_path'] = '/models/wildfire/wildfire_hailo8.hef'
        elif self.detected_hardware['coral']['usb'] or self.detected_hardware['coral']['pcie']:
            config['detector_type'] = 'edgetpu'
            config['model_path'] = '/models/wildfire/wildfire_coral_lite.tflite'

        # Select best hardware acceleration for video
        gpu = self.detected_hardware['gpu']
        if gpu['vendor'] == 'nvidia' and 'nvdec' in gpu.get('driver', ''):
            config['hwaccel_args'] = [
                '-c:v',
                'h264_cuvid',  # Or h265_cuvid depending on stream
            ]
        elif gpu['vendor'] in ['intel', 'amd'] and 'vaapi' in gpu.get('driver', ''):
            # Use detected render device or fallback to default
            render_device = gpu.get('render_device', '/dev/dri/renderD128')
            config['hwaccel_args'] = [
                '-hwaccel',
                'vaapi',
                '-hwaccel_device',
                render_device,
                '-hwaccel_output_format',
                'yuv420p'
            ]
        elif self.detected_hardware.get('cpu', {}).get('platform') == 'raspberry_pi':
            config['hwaccel_args'] = [
                '-c:v',
                'h264_v4l2m2m'
            ]

        return config

if __name__ == '__main__':
    detector = HardwareDetector()
    config = detector.get_recommended_config()
    
    if '--export' in sys.argv:
        with open('/tmp/hardware_config.json', 'w') as f:
            json.dump({'detected': detector.detected_hardware, 'recommended': config}, f)
    else:
        # Print hardware info for debugging/testing
        output = {
            'hardware': detector.detected_hardware,
            'recommended_config': config
        }
        print(json.dumps(output, indent=2))
