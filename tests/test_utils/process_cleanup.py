#!/usr/bin/env python3.12
"""
Enhanced Process Cleanup Utility for Wildfire Watch Tests

This utility addresses the critical process leak issue where hundreds of 'python' 
processes accumulate during test runs, consuming all available RAM.

Key Features:
- Targets generic 'python' processes (primary leak source)
- Graceful cleanup with SIGTERM → SIGKILL progression
- Docker container cleanup
- MQTT broker process management
- Safe process filtering to avoid system processes
"""

import os
import sys
import time
import signal
import psutil
import logging
import subprocess
from typing import List, Set, Dict, Optional
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessCleaner:
    """Enhanced process cleanup for test environments"""
    
    def __init__(self):
        self.cleaned_processes = set()
        self.protected_processes = set()
        self._identify_protected_processes()
    
    def _identify_protected_processes(self):
        """Identify processes that should never be killed"""
        try:
            # System and session processes to protect
            for proc in psutil.process_iter(['pid', 'name', 'ppid', 'cmdline']):
                try:
                    info = proc.info
                    name = info['name'] or ''
                    cmdline = ' '.join(info['cmdline'] or [])
                    
                    # Protect system processes
                    if any(pattern in name.lower() for pattern in [
                        'systemd', 'kernel', 'init', 'dbus', 'networkmanager',
                        'ssh', 'sshd', 'getty', 'cron', 'rsyslog'
                    ]):
                        self.protected_processes.add(info['pid'])
                    
                    # Protect current Python session
                    if info['pid'] == os.getpid():
                        self.protected_processes.add(info['pid'])
                        
                    # Protect parent processes
                    current_pid = os.getpid()
                    parent = psutil.Process(current_pid).parent()
                    while parent:
                        self.protected_processes.add(parent.pid)
                        try:
                            parent = parent.parent()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            break
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                    continue
                    
        except Exception as e:
            logger.warning(f"Error identifying protected processes: {e}")
    
    def find_leaked_python_processes(self) -> List[psutil.Process]:
        """Find generic 'python' processes that are likely leaked"""
        leaked_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'ppid', 'cmdline', 'create_time']):
            try:
                info = proc.info
                pid = info['pid']
                name = info['name'] or ''
                cmdline = info['cmdline'] or []
                
                # Skip protected processes
                if pid in self.protected_processes:
                    continue
                
                # Target generic 'python' processes (not python3.x)
                if name == 'python' and cmdline:
                    # Check if it's a test-related process
                    cmdline_str = ' '.join(cmdline)
                    if any(pattern in cmdline_str.lower() for pattern in [
                        'test', 'pytest', 'consensus.py', 'detect.py', 
                        'trigger.py', 'telemetry.py', 'convert', 'mosquitto'
                    ]):
                        leaked_processes.append(proc)
                        logger.info(f"Found leaked process: PID {pid}, CMD: {cmdline_str[:100]}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        
        return leaked_processes
    
    def find_orphaned_mqtt_brokers(self) -> List[psutil.Process]:
        """Find orphaned mosquitto broker processes"""
        mqtt_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                info = proc.info
                name = info['name'] or ''
                cmdline = ' '.join(info['cmdline'] or [])
                
                if 'mosquitto' in name.lower() and any(pattern in cmdline for pattern in [
                    'test', '20000', '20001', '20002', '20003', '20004'  # Test ports
                ]):
                    mqtt_processes.append(proc)
                    logger.info(f"Found orphaned MQTT broker: PID {info['pid']}, CMD: {cmdline[:100]}")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        
        return mqtt_processes
    
    def cleanup_docker_containers(self) -> int:
        """Clean up test-related Docker containers for current worker only"""
        cleaned_count = 0
        
        # Try to get worker ID from environment or pytest
        worker_id = os.environ.get('PYTEST_XDIST_WORKER', '')
        if not worker_id:
            # If no worker ID, skip Docker cleanup to avoid removing other workers' containers
            logger.warning("No worker ID found, skipping Docker cleanup to prevent cross-worker interference")
            return 0
        
        try:
            # Find only this worker's containers (e.g., wfgw0-, wfgw1-, etc.)
            result = subprocess.run([
                'docker', 'ps', '-a', '--filter', f'name=wf{worker_id}-', 
                '--format', '{{.Names}}'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                container_names = result.stdout.strip().split('\n')
                for name in container_names:
                    if name and ('test' in name or 'wf-' in name):
                        try:
                            # Force remove container
                            subprocess.run([
                                'docker', 'rm', '-f', name
                            ], capture_output=True, timeout=15)
                            cleaned_count += 1
                            logger.info(f"Removed container: {name}")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Timeout removing container: {name}")
                        except Exception as e:
                            logger.warning(f"Error removing container {name}: {e}")
            
            # Clean up orphaned containers
            result = subprocess.run([
                'docker', 'container', 'prune', '-f'
            ], capture_output=True, text=True, timeout=30)
            
        except Exception as e:
            logger.warning(f"Error cleaning Docker containers: {e}")
        
        return cleaned_count
    
    def terminate_process_gracefully(self, proc: psutil.Process, timeout: int = 5) -> bool:
        """Terminate a process gracefully with fallback to force kill"""
        try:
            pid = proc.pid
            
            # Skip if already cleaned
            if pid in self.cleaned_processes:
                return True
            
            # Try graceful termination first
            proc.terminate()
            
            try:
                proc.wait(timeout=timeout)
                self.cleaned_processes.add(pid)
                logger.info(f"Gracefully terminated process {pid}")
                return True
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                proc.kill()
                proc.wait(timeout=2)
                self.cleaned_processes.add(pid)
                logger.info(f"Force killed process {pid}")
                return True
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process already gone or no permission
            return True
        except Exception as e:
            logger.error(f"Error terminating process {proc.pid}: {e}")
            return False
    
    def cleanup_all(self) -> Dict[str, int]:
        """Perform comprehensive cleanup of leaked processes"""
        stats = {
            'python_processes': 0,
            'mqtt_brokers': 0,
            'docker_containers': 0,
            'total_processes': 0
        }
        
        logger.info("Starting comprehensive process cleanup...")
        
        # 1. Clean up Docker containers first
        stats['docker_containers'] = self.cleanup_docker_containers()
        
        # 2. Clean up leaked Python processes
        python_processes = self.find_leaked_python_processes()
        for proc in python_processes:
            if self.terminate_process_gracefully(proc):
                stats['python_processes'] += 1
        
        # 3. Clean up orphaned MQTT brokers
        mqtt_processes = self.find_orphaned_mqtt_brokers()
        for proc in mqtt_processes:
            if self.terminate_process_gracefully(proc):
                stats['mqtt_brokers'] += 1
        
        stats['total_processes'] = stats['python_processes'] + stats['mqtt_brokers']
        
        logger.info(f"Cleanup complete: {stats}")
        return stats


def cleanup_signal_handler(signum, frame):
    """Handle cleanup on script interruption"""
    logger.info(f"Received signal {signum}, performing cleanup...")
    cleaner = ProcessCleaner()
    cleaner.cleanup_all()
    sys.exit(0)


@contextmanager
def managed_subprocess(*args, **kwargs):
    """Context manager for subprocess that ensures cleanup"""
    proc = None
    try:
        proc = subprocess.Popen(*args, **kwargs)
        yield proc
    finally:
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except:
                    pass


def main():
    """Main cleanup execution"""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, cleanup_signal_handler)
    signal.signal(signal.SIGINT, cleanup_signal_handler)
    
    cleaner = ProcessCleaner()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        # Monitor mode - continuous cleanup
        logger.info("Starting monitoring mode - will cleanup every 30 seconds")
        try:
            while True:
                stats = cleaner.cleanup_all()
                if stats['total_processes'] > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Cleaned: {stats['total_processes']} processes, {stats['docker_containers']} containers")
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
    else:
        # One-time cleanup
        stats = cleaner.cleanup_all()
        print(f"Cleanup Results:")
        print(f"  Python processes: {stats['python_processes']}")
        print(f"  MQTT brokers: {stats['mqtt_brokers']}")
        print(f"  Docker containers: {stats['docker_containers']}")
        print(f"  Total processes: {stats['total_processes']}")


if __name__ == '__main__':
    main()