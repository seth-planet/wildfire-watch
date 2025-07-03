#!/usr/bin/env python3.12
"""Improved Hardware Detection with Multi-Device Support

This module provides robust hardware detection without hardcoded assumptions.
It properly enumerates multiple devices and handles various hardware configurations.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Import centralized command runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.command_runner import run_command, CommandError

logger = logging.getLogger(__name__)

@dataclass
class GPUDevice:
    """Represents a GPU device with its properties"""
    vendor: str  # 'nvidia', 'amd', 'intel'
    index: int
    device_path: str
    name: Optional[str] = None
    memory_mb: Optional[int] = None
    driver: Optional[str] = None
    decode_formats: List[str] = None
    encode_formats: List[str] = None
    
    def __post_init__(self):
        if self.decode_formats is None:
            self.decode_formats = []
        if self.encode_formats is None:
            self.encode_formats = []

@dataclass 
class TPUDevice:
    """Represents a TPU/AI accelerator device"""
    vendor: str  # 'google', 'hailo'
    type: str   # 'usb', 'pcie', 'm2'
    index: int
    device_path: str
    name: Optional[str] = None
    tops: Optional[float] = None  # Tera operations per second

@dataclass
class CameraResolution:
    """Represents a camera resolution capability"""
    width: int
    height: int
    fps: Optional[float] = None
    format: Optional[str] = None
    
    def __str__(self):
        if self.fps:
            return f"{self.width}x{self.height}@{self.fps}fps"
        return f"{self.width}x{self.height}"

class ImprovedHardwareDetector:
    """Hardware detector without hardcoded assumptions"""
    
    def __init__(self):
        self.detected_hardware = {
            'cpu': self._detect_cpu(),
            'gpus': self._enumerate_gpus(),
            'tpus': self._enumerate_tpus(),
            'memory': self._detect_memory(),
            'platform': self._detect_platform(),
            'cameras': self._enumerate_camera_capabilities()
        }
        
    def _detect_cpu(self) -> Dict:
        """Detect CPU information"""
        cpu_info = {
            'model': 'Unknown',
            'cores': os.cpu_count() or 1,
            'architecture': os.uname().machine,
        }
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_info['model'] = line.split(':', 1)[1].strip()
                        break
        except Exception as e:
            logger.debug(f"Could not read CPU info: {e}")
            
        return cpu_info
    
    def _enumerate_gpus(self) -> List[GPUDevice]:
        """Enumerate all available GPUs without assumptions"""
        gpus = []
        
        # Try NVIDIA GPUs first
        gpus.extend(self._enumerate_nvidia_gpus())
        
        # Then AMD GPUs
        gpus.extend(self._enumerate_amd_gpus())
        
        # Finally Intel GPUs
        gpus.extend(self._enumerate_intel_gpus())
        
        return gpus
    
    def _enumerate_nvidia_gpus(self) -> List[GPUDevice]:
        """Enumerate NVIDIA GPUs using nvidia-ml-py if available"""
        gpus = []
        
        # First try nvidia-ml-py for proper enumeration
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu = GPUDevice(
                    vendor='nvidia',
                    index=i,
                    device_path=f'gpu:{i}',  # CUDA device syntax
                    name=name,
                    memory_mb=memory_info.total // (1024 * 1024),
                    driver='cuda',
                    decode_formats=['h264', 'h265', 'vp8', 'vp9'],
                    encode_formats=['h264', 'h265']
                )
                gpus.append(gpu)
                logger.info(f"Found NVIDIA GPU {i}: {name}")
                
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.debug("pynvml not available, trying nvidia-smi")
            # Fallback to nvidia-smi
            gpus.extend(self._enumerate_nvidia_gpus_fallback())
        except Exception as e:
            logger.debug(f"NVIDIA enumeration via pynvml failed: {e}")
            gpus.extend(self._enumerate_nvidia_gpus_fallback())
            
        return gpus
    
    def _enumerate_nvidia_gpus_fallback(self) -> List[GPUDevice]:
        """Fallback NVIDIA enumeration using nvidia-smi"""
        gpus = []
        
        try:
            # Query all GPUs
            result = run_command([
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total',
                '--format=csv,noheader,nounits'
            ])
            
            if result[0] == 0 and result[1]:
                for line in result[1].strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        index = int(parts[0])
                        name = parts[1]
                        memory_mb = int(parts[2])
                        
                        gpu = GPUDevice(
                            vendor='nvidia',
                            index=index,
                            device_path=f'gpu:{index}',
                            name=name,
                            memory_mb=memory_mb,
                            driver='cuda',
                            decode_formats=['h264', 'h265'],
                            encode_formats=['h264', 'h265']
                        )
                        gpus.append(gpu)
                        
        except Exception as e:
            logger.debug(f"nvidia-smi enumeration failed: {e}")
            
        return gpus
    
    def _enumerate_amd_gpus(self) -> List[GPUDevice]:
        """Enumerate AMD GPUs using multiple methods"""
        gpus = []
        seen_devices = set()
        
        # Method 1: Check /sys/class/drm for amdgpu devices
        try:
            drm_path = Path('/sys/class/drm')
            if drm_path.exists():
                for card_path in drm_path.glob('card*'):
                    # Check if it's an AMD GPU
                    driver_path = card_path / 'device' / 'driver'
                    if driver_path.is_symlink():
                        driver_name = driver_path.resolve().name
                        if driver_name == 'amdgpu':
                            # Found AMD GPU
                            card_num = int(card_path.name[4:])  # Extract number from 'cardN'
                            render_path = f"/dev/dri/renderD{128 + card_num}"
                            
                            if render_path not in seen_devices:
                                seen_devices.add(render_path)
                                
                                # Try to get more info
                                gpu_name = "AMD GPU"
                                try:
                                    vendor_path = card_path / 'device' / 'vendor'
                                    device_path = card_path / 'device' / 'device'
                                    if vendor_path.exists() and device_path.exists():
                                        vendor_id = vendor_path.read_text().strip()
                                        device_id = device_path.read_text().strip()
                                        # Could look up PCI IDs here
                                        gpu_name = f"AMD GPU {device_id}"
                                except Exception:
                                    pass
                                
                                gpu = GPUDevice(
                                    vendor='amd',
                                    index=card_num,
                                    device_path=render_path,
                                    name=gpu_name,
                                    driver='vaapi',
                                    decode_formats=['h264', 'h265'],
                                    encode_formats=['h264']
                                )
                                gpus.append(gpu)
                                logger.info(f"Found AMD GPU {card_num}: {gpu_name}")
                                
        except Exception as e:
            logger.debug(f"AMD GPU enumeration via /sys failed: {e}")
        
        # Method 2: Try rocm-smi if available
        try:
            result = run_command(['rocm-smi', '--showid'])
            if result[0] == 0 and result[1]:
                # Parse rocm-smi output
                for line in result[1].split('\n'):
                    if 'GPU' in line and ':' in line:
                        try:
                            gpu_id = int(line.split(':')[0].replace('GPU', '').strip())
                            # rocm-smi uses different indexing, map to render device
                            render_path = f"/dev/dri/renderD{128 + gpu_id}"
                            
                            if render_path not in seen_devices:
                                gpu = GPUDevice(
                                    vendor='amd',
                                    index=gpu_id,
                                    device_path=render_path,
                                    name=f"AMD GPU {gpu_id} (ROCm)",
                                    driver='vaapi',
                                    decode_formats=['h264', 'h265'],
                                    encode_formats=['h264']
                                )
                                gpus.append(gpu)
                                
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.debug(f"rocm-smi enumeration failed: {e}")
            
        return gpus
    
    def _enumerate_intel_gpus(self) -> List[GPUDevice]:
        """Enumerate Intel GPUs"""
        gpus = []
        
        # Check for Intel GPU via VA-API
        try:
            # First check if we have any Intel render devices
            drm_path = Path('/sys/class/drm')
            if drm_path.exists():
                for card_path in drm_path.glob('card*'):
                    driver_path = card_path / 'device' / 'driver'
                    if driver_path.is_symlink():
                        driver_name = driver_path.resolve().name
                        if driver_name in ['i915', 'xe']:  # Intel GPU drivers
                            card_num = int(card_path.name[4:])
                            render_path = f"/dev/dri/renderD{128 + card_num}"
                            
                            # Verify with vainfo
                            gpu_name = "Intel GPU"
                            has_vaapi = False
                            
                            try:
                                env = os.environ.copy()
                                env['LIBVA_DRIVER_NAME'] = 'iHD'  # Intel Media Driver
                                result = run_command(['vainfo'], env=env)
                                
                                if result[0] == 0 and 'Intel' in result[1]:
                                    has_vaapi = True
                                    # Extract GPU name if possible
                                    for line in result[1].split('\n'):
                                        if 'Intel' in line and 'Graphics' in line:
                                            gpu_name = line.strip()
                                            break
                                            
                            except Exception:
                                pass
                            
                            gpu = GPUDevice(
                                vendor='intel',
                                index=card_num,
                                device_path=render_path,
                                name=gpu_name,
                                driver='vaapi' if has_vaapi else 'i915',
                                decode_formats=['h264', 'h265'] if has_vaapi else [],
                                encode_formats=['h264'] if has_vaapi else []
                            )
                            gpus.append(gpu)
                            logger.info(f"Found Intel GPU {card_num}: {gpu_name}")
                            
        except Exception as e:
            logger.debug(f"Intel GPU enumeration failed: {e}")
            
        return gpus
    
    def _enumerate_tpus(self) -> List[TPUDevice]:
        """Enumerate TPU/AI accelerator devices"""
        tpus = []
        
        # Enumerate Google Coral devices
        tpus.extend(self._enumerate_coral_devices())
        
        # Enumerate Hailo devices
        tpus.extend(self._enumerate_hailo_devices())
        
        return tpus
    
    def _enumerate_coral_devices(self) -> List[TPUDevice]:
        """Enumerate all Coral TPU devices"""
        tpus = []
        
        # Check USB Coral devices
        try:
            result = run_command(['lsusb'])
            if result[0] == 0 and result[1]:
                # Look for all Coral USB devices
                usb_index = 0
                for line in result[1].split('\n'):
                    if '1a6e:089a' in line or '18d1:9302' in line:
                        # Extract bus and device numbers
                        parts = line.split()
                        if len(parts) >= 6:
                            bus = parts[1]
                            device = parts[3].rstrip(':')
                            device_path = f"/dev/bus/usb/{bus}/{device}"
                            
                            tpu = TPUDevice(
                                vendor='google',
                                type='usb',
                                index=usb_index,
                                device_path=device_path,
                                name=f"Coral USB Accelerator {usb_index}",
                                tops=4.0  # 4 TOPS for USB version
                            )
                            tpus.append(tpu)
                            usb_index += 1
                            logger.info(f"Found Coral USB TPU {usb_index}")
                            
        except Exception as e:
            logger.debug(f"USB Coral enumeration failed: {e}")
        
        # Check PCIe/M.2 Coral devices
        try:
            # Check all possible apex devices
            for i in range(8):  # Check up to 8 devices
                apex_path = f"/dev/apex_{i}"
                if os.path.exists(apex_path):
                    tpu = TPUDevice(
                        vendor='google',
                        type='pcie',
                        index=i,
                        device_path=apex_path,
                        name=f"Coral PCIe/M.2 Accelerator {i}",
                        tops=8.0  # 8 TOPS for PCIe version
                    )
                    tpus.append(tpu)
                    logger.info(f"Found Coral PCIe TPU {i}")
                    
        except Exception as e:
            logger.debug(f"PCIe Coral enumeration failed: {e}")
            
        # Try pycoral for more detailed info if available
        try:
            from pycoral.utils import edgetpu
            edge_tpus = edgetpu.list_edge_tpus()
            
            # Merge with our detected devices
            for i, etpu in enumerate(edge_tpus):
                # Update existing entries with more info
                for tpu in tpus:
                    if tpu.device_path == etpu['path']:
                        tpu.name = f"{etpu['type']} TPU {i}"
                        break
                else:
                    # New device not found by other methods
                    tpu_type = 'usb' if 'usb' in etpu['type'].lower() else 'pcie'
                    tpu = TPUDevice(
                        vendor='google',
                        type=tpu_type,
                        index=i,
                        device_path=etpu['path'],
                        name=f"{etpu['type']} TPU {i}",
                        tops=4.0 if tpu_type == 'usb' else 8.0
                    )
                    tpus.append(tpu)
                    
        except ImportError:
            logger.debug("pycoral not available for detailed TPU info")
        except Exception as e:
            logger.debug(f"pycoral enumeration failed: {e}")
            
        return tpus
    
    def _enumerate_hailo_devices(self) -> List[TPUDevice]:
        """Enumerate Hailo AI processors"""
        tpus = []
        
        # Check for Hailo devices in /dev
        try:
            for i in range(8):  # Check up to 8 devices
                hailo_path = f"/dev/hailo{i}"
                if os.path.exists(hailo_path):
                    # Determine Hailo model and TOPS
                    model = "Hailo-8"
                    tops = 26.0
                    
                    # Try hailortcli for more info
                    try:
                        result = run_command(['hailortcli', 'scan'])
                        if result[0] == 0 and result[1]:
                            if 'Hailo-8L' in result[1]:
                                model = "Hailo-8L"
                                tops = 13.0
                    except Exception:
                        pass
                    
                    tpu = TPUDevice(
                        vendor='hailo',
                        type='pcie',
                        index=i,
                        device_path=hailo_path,
                        name=f"{model} AI Processor {i}",
                        tops=tops
                    )
                    tpus.append(tpu)
                    logger.info(f"Found {model} at {hailo_path}")
                    
        except Exception as e:
            logger.debug(f"Hailo enumeration failed: {e}")
            
        return tpus
    
    def _enumerate_camera_capabilities(self) -> List[Dict]:
        """Enumerate camera capabilities without assuming resolutions"""
        cameras = []
        
        # This would be populated by the camera detector service
        # For now, return empty list as cameras are discovered dynamically
        return cameras
    
    def _detect_memory(self) -> Dict:
        """Detect system memory"""
        memory = {
            'total_mb': 0,
            'available_mb': 0,
        }
        
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal:' in line:
                        memory['total_mb'] = int(line.split()[1]) // 1024
                    elif 'MemAvailable:' in line:
                        memory['available_mb'] = int(line.split()[1]) // 1024
        except Exception as e:
            logger.debug(f"Memory detection failed: {e}")
            
        return memory
    
    def _detect_platform(self) -> Dict:
        """Detect platform information"""
        platform = {
            'os': 'linux',
            'arch': os.uname().machine,
            'hostname': os.uname().nodename,
            'kernel': os.uname().release,
            'container': None
        }
        
        # Detect container environment
        if os.path.exists('/.dockerenv'):
            platform['container'] = 'docker'
        elif os.environ.get('KUBERNETES_SERVICE_HOST'):
            platform['container'] = 'kubernetes'
        elif os.environ.get('BALENA_DEVICE_UUID'):
            platform['container'] = 'balena'
            
        return platform
    
    def get_best_gpu_for_decode(self) -> Optional[GPUDevice]:
        """Get the best GPU for video decoding"""
        if not self.detected_hardware['gpus']:
            return None
            
        # Priority: NVIDIA > AMD > Intel
        for vendor in ['nvidia', 'amd', 'intel']:
            for gpu in self.detected_hardware['gpus']:
                if gpu.vendor == vendor and gpu.decode_formats:
                    return gpu
                    
        return self.detected_hardware['gpus'][0]
    
    def get_best_ai_accelerator(self) -> Optional[Dict]:
        """Get the best AI accelerator (GPU or TPU)"""
        candidates = []
        
        # Add GPUs that support AI
        for gpu in self.detected_hardware['gpus']:
            if gpu.vendor == 'nvidia':
                candidates.append({
                    'type': 'gpu',
                    'device': gpu,
                    'score': 100 + (gpu.memory_mb or 0) / 1000
                })
        
        # Add TPUs
        for tpu in self.detected_hardware['tpus']:
            # Prefer PCIe over USB
            score = 50 + (tpu.tops or 0)
            if tpu.type == 'pcie':
                score += 20
            candidates.append({
                'type': 'tpu',
                'device': tpu,
                'score': score
            })
        
        if not candidates:
            return None
            
        # Return highest scoring candidate
        return max(candidates, key=lambda x: x['score'])
    
    def generate_frigate_config(self) -> Dict:
        """Generate Frigate configuration for detected hardware"""
        config = {
            'detectors': {},
            'ffmpeg': {
                'hwaccel_args': []
            }
        }
        
        # Configure AI detector
        ai_accel = self.get_best_ai_accelerator()
        if ai_accel:
            if ai_accel['type'] == 'gpu' and ai_accel['device'].vendor == 'nvidia':
                # TensorRT detector for NVIDIA
                gpu = ai_accel['device']
                config['detectors'][f'tensorrt_{gpu.index}'] = {
                    'type': 'tensorrt',
                    'device': gpu.index
                }
            elif ai_accel['type'] == 'tpu':
                # Coral detector
                tpu = ai_accel['device']
                config['detectors'][f'coral_{tpu.index}'] = {
                    'type': 'edgetpu',
                    'device': tpu.device_path
                }
        else:
            # CPU fallback
            config['detectors']['cpu'] = {
                'type': 'cpu',
                'model': {
                    'path': '/models/yolov5s.tflite'
                }
            }
        
        # Configure hardware video decoding
        decode_gpu = self.get_best_gpu_for_decode()
        if decode_gpu:
            if decode_gpu.vendor == 'nvidia':
                config['ffmpeg']['hwaccel_args'] = [
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', str(decode_gpu.index),
                    '-hwaccel_output_format', 'cuda'
                ]
            elif decode_gpu.driver == 'vaapi':
                config['ffmpeg']['hwaccel_args'] = [
                    '-hwaccel', 'vaapi',
                    '-hwaccel_device', decode_gpu.device_path,
                    '-hwaccel_output_format', 'vaapi'
                ]
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert detection results to dictionary"""
        result = {
            'cpu': self.detected_hardware['cpu'],
            'gpus': [asdict(gpu) for gpu in self.detected_hardware['gpus']],
            'tpus': [asdict(tpu) for tpu in self.detected_hardware['tpus']],
            'memory': self.detected_hardware['memory'],
            'platform': self.detected_hardware['platform']
        }
        return result

def main():
    """Main entry point for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    detector = ImprovedHardwareDetector()
    
    # Print detected hardware
    print("\n=== Detected Hardware ===")
    print(json.dumps(detector.to_dict(), indent=2))
    
    # Print best devices
    print("\n=== Best Devices ===")
    
    decode_gpu = detector.get_best_gpu_for_decode()
    if decode_gpu:
        print(f"Video Decode: {decode_gpu.vendor} GPU {decode_gpu.index} ({decode_gpu.name})")
    else:
        print("Video Decode: CPU (no hardware acceleration)")
    
    ai_accel = detector.get_best_ai_accelerator()
    if ai_accel:
        device = ai_accel['device']
        print(f"AI Inference: {ai_accel['type'].upper()} - {device.name}")
    else:
        print("AI Inference: CPU (no accelerator)")
    
    # Print Frigate config
    print("\n=== Frigate Configuration ===")
    print(json.dumps(detector.generate_frigate_config(), indent=2))

if __name__ == "__main__":
    main()