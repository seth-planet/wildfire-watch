#!/usr/bin/env python3
"""
Hardware Detection Script for Frigate NVR
Automatically detects and configures available hardware acceleration
"""
import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class HardwareDetector:
    def __init__(self):
        self.detected_hardware = {
            'cpu': self._detect_cpu(),
            'gpu': self._detect_gpu(),
            'coral': self._detect_coral(),
            'hailo': self._detect_hailo(),
            'memory': self._detect_memory(),
            'platform': self._detect_platform(),
        }
        
    def _run_command(self, cmd: List[str]) -> Optional[str]:
        """Run command and return output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Command {' '.join(cmd)} failed: {e}")
        return None
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information"""
        info = {
            'vendor': 'unknown',
            'model': 'unknown',
            'cores': os.cpu_count() or 1,
            'architecture': os.uname().machine,
        }
        
        # Check CPU info
        cpuinfo = self._run_command(['cat', '/proc/cpuinfo'])
        if cpuinfo:
            for line in cpuinfo.split('\n'):
                if 'model name' in line:
                    info['model'] = line.split(':')[1].strip()
                elif 'vendor_id' in line:
                    info['vendor'] = line.split(':')[1].strip()
                    
        # Check for Raspberry Pi
        if os.path.exists('/proc/device-tree/model'):
            model = self._run_command(['cat', '/proc/device-tree/model'])
            if model and 'Raspberry Pi' in model:
                info['platform'] = 'raspberry_pi'
                if 'Pi 5' in model:
                    info['model'] = 'Raspberry Pi 5'
                    info['has_v4l2'] = True
                    info['has_h265_decode'] = True
                    
        return info
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU capabilities"""
        gpu = {
            'vendor': None,
            'model': None,
            'driver': None,
            'decode': [],
            'encode': [],
        }
        
        # Check for Intel GPU
        vainfo = self._run_command(['vainfo'])
        if vainfo and 'Intel' in vainfo:
            gpu['vendor'] = 'intel'
            gpu['driver'] = 'vaapi'
            if 'VAProfileH264' in vainfo:
                gpu['decode'].append('h264')
            if 'VAProfileHEVC' in vainfo:
                gpu['decode'].append('h265')
                
        # Check for AMD GPU
        if os.path.exists('/dev/dri/renderD128'):
            gpu_info = self._run_command(['ls', '-la', '/dev/dri/'])
            if gpu_info and 'amdgpu' in gpu_info:
                gpu['vendor'] = 'amd'
                gpu['driver'] = 'vaapi'
                
        # Check for NVIDIA GPU
        nvidia_smi = self._run_command(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
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
        lsusb = self._run_command(['lsusb'])
        if lsusb:
            # Global Unichip Corp (1a6e) or Google Inc (18d1)
            if '1a6e:089a' in lsusb or '18d1:9302' in lsusb:
                coral['usb'].append({
                    'id': '1a6e:089a',
                    'name': 'Coral USB Accelerator',
                    'path': '/dev/bus/usb',
                })
                
        # Check PCIe Coral
        lspci = self._run_command(['lspci', '-nn'])
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
        hailortcli = self._run_command(['hailortcli', 'scan'])
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
        
        meminfo = self._run_command(['cat', '/proc/meminfo'])
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
            
        return
