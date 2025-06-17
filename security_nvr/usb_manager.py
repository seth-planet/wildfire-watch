#!/usr/bin/env python3
"""USB storage management for Frigate NVR video recording expansion.

This module provides automatic detection and mounting of USB storage devices
for Frigate's video recordings. It solves the storage limitation problem on
edge devices like Raspberry Pi by allowing hot-pluggable external storage.

The manager monitors USB events and automatically mounts the largest available
drive when inserted. It creates the required Frigate directory structure and
monitors disk usage to prevent storage exhaustion.

Key Features:
    - Automatic USB drive detection using udev
    - Intelligent drive selection (largest available)
    - Filesystem support: ext4, ext3, NTFS, FAT32
    - Hot-plug support with automatic mounting
    - Storage monitoring with configurable alerts
    - Frigate directory structure creation

Security Considerations:
    - Requires root/sudo for mounting operations
    - Mounts with restrictive permissions (755)
    - No automatic execution of files from USB
    - Filesystem-specific security options applied

Integration:
    This service runs independently but integrates with Frigate by:
    - Mounting drives to Frigate's expected recording path
    - Creating required directory structure
    - Monitoring storage to prevent recording failures

Dependencies:
    - pyudev: For USB device monitoring
    - Linux commands: mount, umount, df, blockdev

Example:
    Run as service with auto-mounting:
        $ sudo python usb_manager.py monitor
        
    List available USB drives:
        $ sudo python usb_manager.py list
        
    Mount largest drive manually:
        $ sudo python usb_manager.py mount

Note:
    This service must run with root privileges for mount operations.
    In Docker, requires privileged mode or specific capabilities.
"""
import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pyudev

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class USBStorageManager:
    """Manages USB storage devices for Frigate NVR recordings.
    
    Provides automatic detection, mounting, and monitoring of USB storage
    devices. Designed to expand storage capacity on edge devices with
    limited internal storage by utilizing external USB drives.
    
    The manager uses udev for device detection and standard Linux mount
    commands for filesystem operations. It automatically selects the
    largest available drive and creates Frigate's required directory
    structure.
    
    Attributes:
        mount_path (str): Target mount point for USB storage
        context (pyudev.Context): udev context for device enumeration
        monitor (pyudev.Monitor): udev monitor for hot-plug events
        
    Mount Strategy:
        1. Detects all USB block devices with partitions
        2. Filters out already mounted devices
        3. Selects largest unmounted drive by capacity
        4. Mounts with appropriate filesystem options
        5. Creates Frigate directory structure
        
    Thread Safety:
        Not thread-safe. Use single instance per process.
    """
    
    def __init__(self, mount_path: str = "/media/frigate"):
        """Initialize USB storage manager.
        
        Sets up udev context and monitor for USB device detection.
        The monitor is configured to only watch block devices with
        partitions (not whole disks).
        
        Args:
            mount_path: Directory where USB drives will be mounted.
                       Must be accessible and writable by the process.
                       
        Side Effects:
            - Creates pyudev context and monitor
            - Does not start monitoring (call setup_auto_mount for that)
        """
        self.mount_path = mount_path
        self.context = pyudev.Context()
        self.monitor = pyudev.Monitor.from_netlink(self.context)
        self.monitor.filter_by(subsystem='block', device_type='partition')
        
    def find_usb_drives(self) -> List[Dict]:
        """Find all USB storage devices"""
        drives = []
        
        for device in self.context.list_devices(subsystem='block', DEVTYPE='partition'):
            if self._is_usb_device(device):
                drive_info = self._get_drive_info(device)
                if drive_info:
                    drives.append(drive_info)
                    
        return drives
    
    def _is_usb_device(self, device) -> bool:
        """Check if device is USB storage"""
        parent = device.find_parent('usb', 'usb_device')
        return parent is not None
    
    def _get_drive_info(self, device) -> Optional[Dict]:
        """Get information about a drive"""
        try:
            info = {
                'device': device.device_node,
                'partition': device.get('DEVNAME'),
                'label': device.get('ID_FS_LABEL', 'Unknown'),
                'uuid': device.get('ID_FS_UUID'),
                'filesystem': device.get('ID_FS_TYPE'),
                'size_bytes': self._get_partition_size(device.device_node),
                'mounted': self._is_mounted(device.device_node),
                'mount_point': self._get_mount_point(device.device_node),
            }
            
            # Calculate human-readable size
            if info['size_bytes']:
                info['size_human'] = self._bytes_to_human(info['size_bytes'])
                
            return info
            
        except Exception as e:
            logger.error(f"Error getting drive info: {e}")
            return None
    
    def _get_partition_size(self, device_path: str) -> Optional[int]:
        """Get partition size in bytes"""
        try:
            result = subprocess.run(
                ['blockdev', '--getsize64', device_path],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None
    
    def _is_mounted(self, device_path: str) -> bool:
        """Check if device is mounted"""
        try:
            with open('/proc/mounts', 'r') as f:
                mounts = f.read()
                return device_path in mounts
        except:
            return False
    
    def _get_mount_point(self, device_path: str) -> Optional[str]:
        """Get mount point for device"""
        try:
            with open('/proc/mounts', 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts[0] == device_path:
                        return parts[1]
        except:
            pass
        return None
    
    def _bytes_to_human(self, bytes_size: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def mount_largest_drive(self) -> Optional[str]:
        """Mount the largest available USB drive for Frigate storage.
        
        Implements intelligent drive selection by finding all USB drives,
        filtering out mounted ones, and selecting the largest by capacity.
        If a drive is already mounted at the target path, returns success.
        
        Returns:
            Device path (e.g., '/dev/sdb1') if successful, None otherwise
            
        Selection Algorithm:
            1. Find all USB storage devices
            2. Filter out already mounted drives
            3. Sort by size (largest first)
            4. Mount the largest drive
            5. Create Frigate directory structure
            
        Side Effects:
            - Mounts filesystem at self.mount_path
            - Creates Frigate directories
            - Logs selection process
            
        Error Handling:
            - Returns None if no suitable drives found
            - Returns existing device if already mounted correctly
            - Mount failures logged and None returned
        """
        drives = self.find_usb_drives()
        
        # Filter unmounted drives
        unmounted = [d for d in drives if not d['mounted']]
        if not unmounted:
            # Check if already mounted at correct location
            for drive in drives:
                if drive['mount_point'] == self.mount_path:
                    logger.info(f"Drive already mounted at {self.mount_path}")
                    return drive['device']
                    
            logger.warning("No unmounted USB drives found")
            return None
        
        # Sort by size and get largest
        unmounted.sort(key=lambda d: d['size_bytes'] or 0, reverse=True)
        largest = unmounted[0]
        
        logger.info(f"Selected drive: {largest['device']} ({largest['size_human']})")
        
        # Mount the drive
        return self.mount_drive(largest['device'], largest['filesystem'])
    
    def mount_drive(self, device: str, filesystem: str) -> Optional[str]:
        """Mount a specific drive"""
        try:
            # Create mount point
            os.makedirs(self.mount_path, exist_ok=True)
            
            # Mount command
            mount_cmd = ['mount', '-t', filesystem, device, self.mount_path]
            
            # Add filesystem-specific options
            if filesystem == 'ntfs':
                mount_cmd.extend(['-o', 'permissions'])
            elif filesystem in ['ext4', 'ext3']:
                mount_cmd.extend(['-o', 'defaults,noatime'])
                
            result = subprocess.run(mount_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully mounted {device} to {self.mount_path}")
                
                # Set permissions
                os.chmod(self.mount_path, 0o755)
                
                # Create Frigate directories
                self._create_frigate_dirs()
                
                return device
            else:
                logger.error(f"Failed to mount: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Mount error: {e}")
            
        return None
    
    def _create_frigate_dirs(self):
        """Create directory structure for Frigate"""
        dirs = [
            'recordings',
            'clips',
            'exports',
            'logs',
            'cache',
        ]
        
        for dir_name in dirs:
            dir_path = os.path.join(self.mount_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            os.chmod(dir_path, 0o755)
            
        logger.info("Created Frigate directory structure")
    
    def unmount_drive(self) -> bool:
        """Unmount the current drive"""
        try:
            result = subprocess.run(
                ['umount', self.mount_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully unmounted {self.mount_path}")
                return True
            else:
                logger.error(f"Failed to unmount: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Unmount error: {e}")
            
        return False
    
    def get_storage_stats(self) -> Optional[Dict]:
        """Get storage statistics"""
        try:
            result = subprocess.run(
                ['df', '-B1', self.mount_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 6:
                        total = int(parts[1])
                        used = int(parts[2])
                        available = int(parts[3])
                        percent = parts[4].rstrip('%')
                        
                        return {
                            'total_bytes': total,
                            'used_bytes': used,
                            'available_bytes': available,
                            'used_percent': int(percent),
                            'total_human': self._bytes_to_human(total),
                            'used_human': self._bytes_to_human(used),
                            'available_human': self._bytes_to_human(available),
                        }
                        
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            
        return None
    
    def monitor_storage(self, warning_percent: int = 90, critical_percent: int = 95):
        """Monitor storage usage and log warnings"""
        stats = self.get_storage_stats()
        
        if stats:
            used_percent = stats['used_percent']
            
            if used_percent >= critical_percent:
                logger.critical(f"Storage critically low: {used_percent}% used")
            elif used_percent >= warning_percent:
                logger.warning(f"Storage running low: {used_percent}% used")
            else:
                logger.info(f"Storage usage: {used_percent}% ({stats['used_human']} / {stats['total_human']})")
                
            return stats
            
        return None
    
    def setup_auto_mount(self):
        """Setup automatic mounting on USB insertion with storage monitoring.
        
        Starts a udev monitor that watches for USB device insertion/removal
        events. When a USB drive is inserted, automatically mounts it if
        no drive is currently mounted. Also performs periodic storage
        monitoring to alert on low disk space.
        
        This method blocks indefinitely until interrupted (Ctrl+C).
        
        Monitoring Features:
            - Hot-plug detection via udev
            - Automatic mounting on insertion
            - Periodic storage usage checks (every 60s)
            - Warning at 90% usage, critical at 95%
            
        Side Effects:
            - Starts background udev observer thread
            - Blocks main thread in monitoring loop
            - Logs storage statistics periodically
            - Mounts/unmounts drives on USB events
            
        Signal Handling:
            - KeyboardInterrupt (Ctrl+C) stops monitoring gracefully
            - Observer thread is properly stopped on exit
        """
        logger.info("Starting USB monitor...")
        
        observer = pyudev.MonitorObserver(self.monitor, self._handle_usb_event)
        observer.start()
        
        try:
            while True:
                time.sleep(60)
                # Periodic storage check
                self.monitor_storage()
                
        except KeyboardInterrupt:
            observer.stop()
            logger.info("USB monitor stopped")
    
    def _handle_usb_event(self, action, device):
        """Handle USB insertion/removal events"""
        if action == 'add' and self._is_usb_device(device):
            logger.info(f"USB device inserted: {device.device_node}")
            time.sleep(2)  # Wait for device to settle
            
            # Check if we need to mount
            if not self._is_mounted(self.mount_path):
                self.mount_largest_drive()
                
        elif action == 'remove':
            logger.info(f"USB device removed: {device.device_node}")

def main():
    """Command-line interface for USB storage management.
    
    Provides utilities for managing USB storage devices for Frigate NVR.
    Must be run with root/sudo privileges for mount operations.
    
    Commands:
        list: Display all detected USB storage devices
        mount: Mount the largest available USB drive
        unmount: Unmount the current drive
        stats: Show storage usage statistics
        monitor: Start auto-mount service (blocks)
        (no command): Mount largest drive and start monitoring
        
    Exit Codes:
        0: Success
        1: Mount/unmount operation failed
        
    Examples:
        List USB drives:
            $ sudo python usb_manager.py list
            
        Mount and monitor (service mode):
            $ sudo python usb_manager.py monitor
            
        Check storage usage:
            $ sudo python usb_manager.py stats
            
    Permissions:
        All mount operations require root/sudo. In Docker, requires:
        - Privileged mode, or
        - CAP_SYS_ADMIN capability
        - Device access to /dev/bus/usb/*
    """
    manager = USBStorageManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'list':
            # List all USB drives
            drives = manager.find_usb_drives()
            print("\nUSB Storage Devices:")
            print("-" * 60)
            
            for drive in drives:
                print(f"Device: {drive['device']}")
                print(f"  Label: {drive['label']}")
                print(f"  Size: {drive['size_human']}")
                print(f"  Filesystem: {drive['filesystem']}")
                print(f"  Mounted: {'Yes' if drive['mounted'] else 'No'}")
                if drive['mount_point']:
                    print(f"  Mount Point: {drive['mount_point']}")
                print()
                
        elif command == 'mount':
            # Mount largest drive
            device = manager.mount_largest_drive()
            if device:
                print(f"Mounted {device} to {manager.mount_path}")
            else:
                print("Failed to mount drive")
                sys.exit(1)
                
        elif command == 'unmount':
            # Unmount current drive
            if manager.unmount_drive():
                print(f"Unmounted {manager.mount_path}")
            else:
                print("Failed to unmount drive")
                sys.exit(1)
                
        elif command == 'stats':
            # Show storage statistics
            stats = manager.get_storage_stats()
            if stats:
                print("\nStorage Statistics:")
                print("-" * 40)
                print(f"Total: {stats['total_human']}")
                print(f"Used: {stats['used_human']} ({stats['used_percent']}%)")
                print(f"Available: {stats['available_human']}")
            else:
                print("No storage mounted")
                
        elif command == 'monitor':
            # Start monitoring
            manager.setup_auto_mount()
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: usb_manager.py [list|mount|unmount|stats|monitor]")
            
    else:
        # Default: mount and monitor
        manager.mount_largest_drive()
        manager.setup_auto_mount()

if __name__ == '__main__':
    main()
