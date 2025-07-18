#!/usr/bin/env python3.12
"""
Enhanced Process Cleanup Utility for Wildfire Watch Test Suite

This utility provides comprehensive cleanup of leaked processes, with specific
focus on generic 'python' processes that cause resource exhaustion.

CRITICAL FIXES:
1. Targets generic 'python' processes specifically
2. Enhanced zombie process cleanup  
3. Robust Docker container cleanup
4. Subprocess cleanup with proper signal handling
5. Memory-efficient process identification
"""

import os
import sys
import time
import subprocess
import logging
import signal
import atexit
from typing import List, Set, Dict
import docker
import psutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class EnhancedProcessCleaner:
    """Enhanced process cleanup focusing on generic 'python' process leaks."""
    
    def __init__(self):
        self.docker_client = None
        self.cleanup_registry = []
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
        
        # Register cleanup on exit
        atexit.register(self.cleanup_all)
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful cleanup on interruption."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, performing cleanup...")
            self.cleanup_all()
            # Restore original handler and re-raise
            if signum == signal.SIGINT and self.original_sigint_handler:
                signal.signal(signal.SIGINT, self.original_sigint_handler)
            elif signum == signal.SIGTERM and self.original_sigterm_handler:
                signal.signal(signal.SIGTERM, self.original_sigterm_handler)
            os.kill(os.getpid(), signum)
        
        self.original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)
        self.original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)
    
    def register_for_cleanup(self, resource_type: str, resource_id: str, cleanup_func):
        """Register a resource for cleanup."""
        self.cleanup_registry.append({
            'type': resource_type,
            'id': resource_id,
            'cleanup': cleanup_func
        })
    
    def cleanup_generic_python_processes(self) -> int:
        """
        CRITICAL: Clean up generic 'python' processes that cause resource leaks.
        
        This specifically targets processes launched with 'python' instead of 'python3.12'.
        These are the primary cause of zombie processes and resource exhaustion.
        """
        cleaned = 0
        current_pid = os.getpid()
        
        try:
            # Method 1: Use pgrep to find generic python processes
            try:
                result = subprocess.run(['pgrep', '-f', '^python [^3]'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                    for pid in pids:
                        if pid and int(pid) != current_pid:
                            try:
                                # Get process info before killing
                                proc_info = subprocess.run(['ps', '-p', pid, '-o', 'pid,ppid,cmd'], 
                                                         capture_output=True, text=True, timeout=5)
                                logger.info(f"Killing generic python process {pid}: {proc_info.stdout}")
                                
                                # Graceful termination first
                                subprocess.run(['kill', '-TERM', pid], timeout=2)
                                time.sleep(0.5)
                                
                                # Check if still running, force kill if needed
                                if subprocess.run(['kill', '-0', pid], capture_output=True).returncode == 0:
                                    subprocess.run(['kill', '-KILL', pid], timeout=1)
                                
                                cleaned += 1
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                                pass  # Process already dead or inaccessible
            except subprocess.TimeoutExpired:
                logger.warning("Timeout finding generic python processes with pgrep")
            
            # Method 2: Use psutil for detailed inspection
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid', 'status']):
                    try:
                        info = proc.info
                        
                        # Target generic 'python' processes (not python3.x)
                        if (info['name'] == 'python' and 
                            info['pid'] != current_pid):
                            
                            cmdline = info.get('cmdline', [])
                            if cmdline:
                                # Avoid killing essential system processes
                                cmd_str = ' '.join(cmdline)
                                system_excludes = [
                                    '/usr/bin/python3',
                                    'networkd-dispatcher', 
                                    'unattended-upgrade'
                                ]
                                
                                if not any(exclude in cmd_str for exclude in system_excludes):
                                    logger.info(f"Terminating generic python process {info['pid']}: {cmd_str}")
                                    
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=3)
                                    except psutil.TimeoutExpired:
                                        proc.kill()
                                        proc.wait(timeout=1)
                                    
                                    cleaned += 1
                                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                        
            except Exception as e:
                logger.warning(f"Error using psutil for python cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error cleaning generic python processes: {e}")
        
        return cleaned
    
    def cleanup_zombie_processes(self) -> int:
        """Clean up zombie processes specifically."""
        cleaned = 0
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'status', 'ppid']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        pid = proc.info['pid']
                        ppid = proc.info['ppid']
                        
                        logger.info(f"Found zombie process {pid} (parent: {ppid})")
                        
                        # Try to clean up zombie by signaling parent
                        try:
                            parent = psutil.Process(ppid)
                            # Send SIGCHLD to parent to collect zombie
                            parent.send_signal(signal.SIGCHLD)
                            time.sleep(0.1)
                            
                            # If zombie still exists, more aggressive cleanup
                            if proc.is_running() and proc.status() == psutil.STATUS_ZOMBIE:
                                # Try killing parent if it's a test process
                                parent_cmdline = ' '.join(parent.cmdline())
                                if any(test_term in parent_cmdline for test_term in ['pytest', 'test_', '/tests/']):
                                    logger.info(f"Killing parent test process {ppid} to clean zombie {pid}")
                                    parent.terminate()
                                    parent.wait(timeout=2)
                            
                            cleaned += 1
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                    
        except Exception as e:
            logger.warning(f"Error cleaning zombie processes: {e}")
        
        return cleaned
    
    def cleanup_test_docker_containers(self) -> int:
        """Enhanced Docker container cleanup with better filtering."""
        cleaned = 0
        
        if not self.docker_client:
            return 0
        
        try:
            # Get all containers
            all_containers = self.docker_client.containers.list(all=True)
            
            test_indicators = [
                'com.wildfire.test=true',  # Our test label
                'test-',                   # Test prefix
                'e2e-',                   # E2E test prefix
                'wf-gw',                  # Our test networks
                'wf-master',              # Our test networks
                'mqtt_test_',             # MQTT test containers
            ]
            
            for container in all_containers:
                try:
                    should_clean = False
                    
                    # Check labels
                    labels = container.labels or {}
                    if 'com.wildfire.test' in labels:
                        should_clean = True
                    
                    # Check name patterns
                    if any(indicator in container.name for indicator in test_indicators):
                        should_clean = True
                    
                    # Check if using generic python command
                    if hasattr(container, 'attrs'):
                        config = container.attrs.get('Config', {})
                        cmd = config.get('Cmd', [])
                        if cmd and len(cmd) > 0 and cmd[0] == 'python':
                            logger.info(f"Found container using generic python: {container.name}")
                            should_clean = True
                    
                    if should_clean:
                        logger.info(f"Cleaning up test container: {container.name}")
                        
                        if container.status == 'running':
                            container.stop(timeout=5)
                        
                        container.remove(force=True)
                        cleaned += 1
                        
                except Exception as e:
                    logger.warning(f"Error cleaning container {container.name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning Docker containers: {e}")
        
        return cleaned
    
    def cleanup_subprocess_processes(self) -> int:
        """Clean up subprocess processes that may be leaked."""
        cleaned = 0
        current_pid = os.getpid()
        
        try:
            # Find processes with subprocess-related patterns
            subprocess_patterns = [
                'mosquitto.*mqtt_test_',
                'python.*-c.*import',
                'python.*pytest',
            ]
            
            for pattern in subprocess_patterns:
                try:
                    result = subprocess.run(['pgrep', '-f', pattern], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
                        for pid in pids:
                            if pid and int(pid) != current_pid:
                                try:
                                    # Get process info
                                    proc_info = subprocess.run(['ps', '-p', pid, '-o', 'pid,cmd'], 
                                                             capture_output=True, text=True, timeout=2)
                                    logger.info(f"Cleaning subprocess {pid}: {proc_info.stdout.strip()}")
                                    
                                    subprocess.run(['kill', '-TERM', pid], timeout=1)
                                    time.sleep(0.2)
                                    
                                    # Force kill if needed
                                    if subprocess.run(['kill', '-0', pid], capture_output=True).returncode == 0:
                                        subprocess.run(['kill', '-KILL', pid], timeout=1)
                                    
                                    cleaned += 1
                                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                                    pass
                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout finding processes with pattern: {pattern}")
                    
        except Exception as e:
            logger.warning(f"Error cleaning subprocess processes: {e}")
        
        return cleaned
    
    @contextmanager
    def subprocess_manager(self, *args, **kwargs):
        """Context manager for subprocess with automatic cleanup."""
        proc = None
        try:
            proc = subprocess.Popen(*args, **kwargs)
            self.register_for_cleanup('subprocess', str(proc.pid), 
                                    lambda: self._cleanup_subprocess(proc))
            yield proc
        finally:
            if proc:
                self._cleanup_subprocess(proc)
    
    def _cleanup_subprocess(self, proc):
        """Clean up a specific subprocess."""
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
        except Exception as e:
            logger.warning(f"Error cleaning subprocess {proc.pid}: {e}")
    
    def cleanup_all(self) -> Dict[str, int]:
        """Perform comprehensive cleanup with focus on process leaks."""
        results = {}
        
        logger.info("Starting enhanced process cleanup (targeting generic python processes)...")
        
        # Priority order - most critical first
        results['generic_python'] = self.cleanup_generic_python_processes()
        results['zombie_processes'] = self.cleanup_zombie_processes()
        results['docker_containers'] = self.cleanup_test_docker_containers()
        results['subprocess_processes'] = self.cleanup_subprocess_processes()
        
        # Clean up registered resources
        registry_cleaned = 0
        for resource in self.cleanup_registry[:]:  # Copy list to avoid modification during iteration
            try:
                resource['cleanup']()
                registry_cleaned += 1
                self.cleanup_registry.remove(resource)
            except Exception as e:
                logger.warning(f"Error cleaning registered resource {resource['id']}: {e}")
        
        results['registered_resources'] = registry_cleaned
        
        total_cleaned = sum(results.values())
        logger.info(f"Enhanced cleanup complete. Total items cleaned: {total_cleaned}")
        logger.info(f"Breakdown: {results}")
        
        return results

# Global instance for easy access
_cleaner_instance = None

def get_process_cleaner() -> EnhancedProcessCleaner:
    """Get global process cleaner instance."""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = EnhancedProcessCleaner()
    return _cleaner_instance

def cleanup_on_test_failure():
    """Cleanup function to be called on test failures."""
    cleaner = get_process_cleaner()
    results = cleaner.cleanup_all()
    print(f"Emergency cleanup performed: {results}")

def main():
    """Main cleanup function."""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    cleaner = EnhancedProcessCleaner()
    results = cleaner.cleanup_all()
    
    print("Enhanced process cleanup results:")
    for category, count in results.items():
        print(f"  {category}: {count} items cleaned")
    
    total = sum(results.values())
    print(f"Total items cleaned: {total}")
    
    if total > 0:
        print("✅ Process leak cleanup completed successfully")
        return 0
    else:
        print("ℹ️  No leaked processes found")
        return 0

if __name__ == '__main__':
    sys.exit(main())