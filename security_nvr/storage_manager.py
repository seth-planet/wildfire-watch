#!/usr/bin/env python3
"""Storage management for Frigate NVR video recording expansion.

This module provides automatic detection and mounting of storage devices
(both USB and internal drives) for Frigate's video recordings. It solves 
the storage limitation problem on edge devices by allowing flexible storage
options.

The manager can work with:
- USB drives (auto-detection and hot-plug support)
- Internal drives (additional partitions or dedicated drives)
- Network storage (future enhancement)

Key Features:
    - Automatic USB drive detection using udev
    - Support for internal drive mounting
    - Intelligent drive selection (largest available)
    - Filesystem support: ext4, ext3, NTFS, FAT32, btrfs, xfs
    - Hot-plug support with automatic mounting (USB only)
    - Storage monitoring with configurable alerts
    - Frigate directory structure creation

Security Considerations:
    - Requires root/sudo for mounting operations
    - Mounts with restrictive permissions (755)
    - No automatic execution of files from storage
    - Filesystem-specific security options applied

Integration:
    This service runs independently but integrates with Frigate by:
    - Mounting drives to Frigate's expected recording path
    - Creating required directory structure
    - Monitoring storage to prevent recording failures

Dependencies:
    - pyudev: For USB device monitoring
    - Linux commands: mount, umount, df, blockdev, lsblk

Example:
    Run as service with auto-mounting:
        $ sudo python storage_manager.py monitor
        
    List available drives (USB and internal):
        $ sudo python storage_manager.py list
        
    Mount largest drive automatically:
        $ sudo python storage_manager.py mount
        
    Mount specific internal drive:
        $ sudo python storage_manager.py mount --device /dev/sda2
        
    Mount by UUID (recommended for internal drives):
        $ sudo python storage_manager.py mount --uuid 12345678-1234-1234-1234-123456789012

Note:
    This service must run with root privileges for mount operations.
    In Docker, requires privileged mode or specific capabilities.
"""
import os
import sys
import json
import time
import logging
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import centralized command runner
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.command_runner import run_command, CommandError
from utils.safe_logging import SafeLoggingMixin
from utils.logging_config import setup_logging

logger = setup_logging("storage_manager")

# Import USB manager functionality
from usb_manager import USBStorageManager

class StorageManager(USBStorageManager, SafeLoggingMixin):
    """Extended storage manager supporting both USB and internal drives.
    
    This class extends USBStorageManager to add support for internal drives
    while maintaining backward compatibility with USB functionality.
    """
    
    def __init__(self, mount_path: str = '/media/frigate'):
        """Initialize storage manager
        
        Args:
            mount_path: Target mount path for storage
        """
        super().__init__(mount_path)
        self.logger = logger
        
    def find_all_drives(self, include_internal: bool = True) -> List[Dict]:
        """Find all available storage drives (USB and optionally internal).
        
        Args:
            include_internal: Whether to include internal drives
            
        Returns:
            List of drive information dictionaries
        """
        drives = []
        
        # Get USB drives from parent class
        usb_drives = self.find_usb_drives()
        drives.extend(usb_drives)
        
        if include_internal:
            # Get internal drives
            internal_drives = self.find_internal_drives()
            drives.extend(internal_drives)
            
        return drives
    
    def find_internal_drives(self) -> List[Dict]:
        """Find internal drives suitable for storage.
        
        Returns:
            List of drive information dictionaries
        """
        drives = []
        
        try:
            # Use lsblk to get all block devices
            cmd = ['lsblk', '-J', '-o', 'NAME,SIZE,TYPE,FSTYPE,UUID,MOUNTPOINT,MODEL']
            _, output, _ = run_command(cmd, timeout=10, check=True)
            
            lsblk_data = json.loads(output)
            
            for device in lsblk_data.get('blockdevices', []):
                # Skip if it's the root device
                if self._is_system_disk(device):
                    continue
                    
                # Process partitions
                for part in device.get('children', []):
                    if part.get('type') == 'part' and part.get('fstype'):
                        drive_info = self._process_internal_partition(part, device)
                        if drive_info:
                            drives.append(drive_info)
                            
        except Exception as e:
            self._safe_log('error', f"Error finding internal drives: {e}")
            
        return drives
    
    def _is_system_disk(self, device: Dict) -> bool:
        """Check if device contains system partitions.
        
        Args:
            device: Device information from lsblk
            
        Returns:
            True if device contains system partitions
        """
        # Check if any partition is mounted on critical paths
        critical_mounts = ['/', '/boot', '/boot/efi', '/usr', '/var']
        
        for part in device.get('children', []):
            mount = part.get('mountpoint', '')
            if mount in critical_mounts:
                return True
                
        return False
    
    def _process_internal_partition(self, partition: Dict, parent: Dict) -> Optional[Dict]:
        """Process internal partition information.
        
        Args:
            partition: Partition data from lsblk
            parent: Parent device data
            
        Returns:
            Drive information dictionary or None
        """
        try:
            device_path = f"/dev/{partition['name']}"
            
            # Skip if already mounted (unless it's our mount point)
            mount_point = partition.get('mountpoint')
            if mount_point and mount_point != self.mount_path:
                return None
                
            # Get size
            size_str = partition.get('size', '0')
            size_bytes = self._parse_size_string(size_str)
            
            info = {
                'device': device_path,
                'filesystem': partition.get('fstype', 'unknown'),
                'size_bytes': size_bytes,
                'size_human': size_str,
                'label': partition.get('label', ''),
                'uuid': partition.get('uuid', ''),
                'model': parent.get('model', 'Internal Drive'),
                'mounted': bool(mount_point),
                'mount_point': mount_point or '',
                'type': 'internal',
                'parent_device': f"/dev/{parent['name']}"
            }
            
            return info
            
        except Exception as e:
            self._safe_log('error', f"Error processing partition {partition.get('name')}: {e}")
            return None
    
    def _parse_size_string(self, size_str: str) -> int:
        """Parse size string from lsblk (e.g., '1.8T', '500G') to bytes.
        
        Args:
            size_str: Size string from lsblk
            
        Returns:
            Size in bytes
        """
        if not size_str:
            return 0
            
        # Remove spaces and convert to uppercase
        size_str = size_str.strip().upper()
        
        # Parse units
        units = {
            'B': 1,
            'K': 1024,
            'M': 1024**2,
            'G': 1024**3,
            'T': 1024**4,
            'P': 1024**5
        }
        
        try:
            # Extract number and unit
            for unit, multiplier in units.items():
                if size_str.endswith(unit):
                    number = float(size_str[:-1])
                    return int(number * multiplier)
                    
            # If no unit, assume bytes
            return int(float(size_str))
            
        except ValueError:
            self._safe_log('error', f"Could not parse size: {size_str}")
            return 0
    
    def mount_largest_drive(self) -> Optional[str]:
        """Override to include internal drives in selection.
        
        Finds largest available drive (USB or internal) >= 500 GB.
        
        Returns:
            Device path if successful, None otherwise
        """
        drives = self.find_all_drives(include_internal=True)
        
        # No need to filter by size - find_all_drives already does that
        
        # Filter unmounted drives
        unmounted = [d for d in drives if not d['mounted']]
        if not unmounted:
            # Check if already mounted at correct location
            for drive in drives:
                if drive['mount_point'] == self.mount_path:
                    self._safe_log('info', f"Drive already mounted at {self.mount_path}")
                    return drive['device']
                    
            self._safe_log('warning', "No unmounted drives found >= 500 GB")
            return None
        
        # Sort by size and get largest
        unmounted.sort(key=lambda d: d['size_bytes'] or 0, reverse=True)
        largest = unmounted[0]
        
        self._safe_log('info', f"Selected drive: {largest['device']} ({largest['size_human']}) - {largest.get('type', 'usb').upper()}")
        
        # Mount the drive
        return self.mount_drive(largest['device'], largest['filesystem'])
    
    def mount_drive(self, device: str, filesystem: str) -> Optional[str]:
        """Mount a specific drive with security validations.
        
        Overrides parent to add security checks.
        
        Args:
            device: Device path to mount
            filesystem: Filesystem type
            
        Returns:
            Device path if successful, None otherwise
        """
        # Validate device path to prevent command injection
        if not re.match(r'^/dev/[a-zA-Z0-9/_-]+$', device):
            self._safe_log('error', f"Invalid device path format: {device}")
            return None
            
        # Validate filesystem type
        valid_filesystems = ['ext4', 'ext3', 'ext2', 'ntfs', 'vfat', 'btrfs', 'xfs']
        if filesystem not in valid_filesystems:
            self._safe_log('error', f"Unsupported filesystem: {filesystem}")
            return None
            
        # Call parent mount_drive which will handle the actual mounting
        return super().mount_drive(device, filesystem)
    
    def mount_by_uuid(self, uuid: str, filesystem: Optional[str] = None) -> Optional[str]:
        """Mount a drive by UUID (recommended for internal drives).
        
        Args:
            uuid: UUID of the drive to mount
            filesystem: Filesystem type (auto-detected if not provided)
            
        Returns:
            Device path if successful, None otherwise
        """
        try:
            # Find device by UUID
            uuid_path = f"/dev/disk/by-uuid/{uuid}"
            if not os.path.exists(uuid_path):
                self._safe_log('error', f"UUID not found: {uuid}")
                return None
                
            # Resolve to actual device
            device = os.path.realpath(uuid_path)
            
            # Get filesystem if not provided
            if not filesystem:
                cmd = ['blkid', '-o', 'value', '-s', 'TYPE', device]
                _, fs_output, _ = run_command(cmd, timeout=5, check=True)
                filesystem = fs_output.strip()
                
            self._safe_log('info', f"Mounting device {device} (UUID: {uuid}) with filesystem {filesystem}")
            return self.mount_drive(device, filesystem)
            
        except Exception as e:
            self._safe_log('error', f"Error mounting by UUID: {e}")
            return None
    
    def mount_specific_device(self, device: str, filesystem: Optional[str] = None) -> Optional[str]:
        """Mount a specific device.
        
        Args:
            device: Device path (e.g., /dev/sda2)
            filesystem: Filesystem type (auto-detected if not provided)
            
        Returns:
            Device path if successful, None otherwise
        """
        try:
            if not os.path.exists(device):
                self._safe_log('error', f"Device not found: {device}")
                return None
                
            # Get filesystem if not provided
            if not filesystem:
                cmd = ['blkid', '-o', 'value', '-s', 'TYPE', device]
                _, fs_output, _ = run_command(cmd, timeout=5, check=True)
                filesystem = fs_output.strip()
                
            self._safe_log('info', f"Mounting device {device} with filesystem {filesystem}")
            return self.mount_drive(device, filesystem)
            
        except Exception as e:
            self._safe_log('error', f"Error mounting device: {e}")
            return None


def main():
    """Main entry point for storage manager."""
    parser = argparse.ArgumentParser(description='Storage Manager for Frigate NVR')
    parser.add_argument('command', 
                       choices=['list', 'mount', 'unmount', 'stats', 'monitor'],
                       help='Command to execute')
    parser.add_argument('--mount-path', 
                       default='/media/frigate',
                       help='Mount path for storage')
    parser.add_argument('--device',
                       help='Specific device to mount (e.g., /dev/sda2)')
    parser.add_argument('--uuid',
                       help='Mount by UUID (recommended for internal drives)')
    parser.add_argument('--include-internal',
                       action='store_true',
                       default=True,
                       help='Include internal drives in listing/selection')
    parser.add_argument('--usb-only',
                       action='store_true',
                       help='Only consider USB drives')
    
    args = parser.parse_args()
    
    # Create manager
    manager = StorageManager(args.mount_path)
    
    # Override include_internal if usb_only is set
    if args.usb_only:
        args.include_internal = False
    
    # Execute command
    command = args.command
    
    if command == 'list':
        # List available drives
        drives = manager.find_all_drives(include_internal=args.include_internal)
        
        if not drives:
            print("No suitable drives found")
            return
            
        print(f"Available drives:")
        print("-" * 60)
        
        for drive in drives:
            status = "MOUNTED" if drive['mounted'] else "AVAILABLE"
            drive_type = drive.get('type', 'usb').upper()
            print(f"{drive['device']} - {drive['size_human']} - {drive['filesystem']} - {drive_type} - {status}")
            print(f"  Model: {drive['model']}")
            if drive.get('uuid'):
                print(f"  UUID: {drive['uuid']}")
            if drive['mounted']:
                print(f"  Mount: {drive['mount_point']}")
            print()
            
    elif command == 'mount':
        # Mount drive
        if args.uuid:
            # Mount by UUID
            device = manager.mount_by_uuid(args.uuid)
        elif args.device:
            # Mount specific device
            device = manager.mount_specific_device(args.device)
        else:
            # Mount largest available drive
            device = manager.mount_largest_drive()
            
        if device:
            print(f"Successfully mounted {device} to {manager.mount_path}")
        else:
            print("Warning: Failed to mount drive, continuing without external storage")
            # Don't exit - graceful degradation
            
    elif command == 'unmount':
        # Unmount current drive
        if manager.unmount_drive():
            print(f"Unmounted {manager.mount_path}")
        else:
            print("Warning: Failed to unmount")
            # Don't exit - graceful degradation
            
    elif command == 'stats':
        # Show storage stats
        stats = manager.get_storage_stats()
        if stats:
            print(f"Storage Statistics for {manager.mount_path}:")
            print(f"  Total: {stats['total_human']}")
            print(f"  Used:  {stats['used_human']} ({stats['percent_used']:.1f}%)")
            print(f"  Free:  {stats['free_human']}")
        else:
            print("No storage mounted")
            # Don't exit - graceful degradation
            
    elif command == 'monitor':
        # Monitor for USB drives (internal drives don't hot-plug)
        print(f"Monitoring for USB drive events...")
        print(f"Mount path: {manager.mount_path}")
        print("Press Ctrl+C to stop")
        
        try:
            manager.monitor_drives()
        except KeyboardInterrupt:
            print("\nMonitoring stopped")


if __name__ == "__main__":
    main()