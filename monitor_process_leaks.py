#!/usr/bin/env python3.12
"""
Process Leak Monitor for Wildfire Watch Test Suite

This script continuously monitors for process leaks during testing,
specifically focusing on generic 'python' processes and zombies.
"""

import time
import subprocess
import psutil
import sys
from datetime import datetime
from typing import Dict, List
import signal

class ProcessLeakMonitor:
    """Monitor for process leaks during testing."""
    
    def __init__(self, interval: int = 10):
        self.interval = interval
        self.running = True
        self.baseline_processes = self.get_python_processes()
        self.leak_threshold = 5  # Alert if more than 5 new processes
        self.history = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n📡 Received signal {signum}, stopping monitor...")
        self.running = False
    
    def get_python_processes(self) -> Dict[str, List[Dict]]:
        """Get current Python processes categorized by type."""
        processes = {
            'generic_python': [],
            'specific_python': [],
            'zombies': [],
            'test_processes': []
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
                try:
                    info = proc.info
                    name = info.get('name', '')
                    cmdline = info.get('cmdline', [])
                    status = info.get('status', '')
                    
                    if name == 'python':
                        processes['generic_python'].append({
                            'pid': info['pid'],
                            'cmdline': ' '.join(cmdline) if cmdline else '',
                            'status': status
                        })
                    elif name.startswith('python3'):
                        processes['specific_python'].append({
                            'pid': info['pid'],
                            'name': name,
                            'cmdline': ' '.join(cmdline) if cmdline else '',
                            'status': status
                        })
                    
                    if status == psutil.STATUS_ZOMBIE:
                        processes['zombies'].append({
                            'pid': info['pid'],
                            'name': name,
                            'status': status
                        })
                    
                    # Check for test processes
                    if cmdline and any(test_term in ' '.join(cmdline) for test_term in ['pytest', 'test_', '/tests/']):
                        processes['test_processes'].append({
                            'pid': info['pid'],
                            'name': name,
                            'cmdline': ' '.join(cmdline) if cmdline else '',
                            'status': status
                        })
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except Exception as e:
            print(f"Error getting processes: {e}")
        
        return processes
    
    def detect_leaks(self, current: Dict, baseline: Dict) -> Dict:
        """Detect process leaks compared to baseline."""
        leaks = {}
        
        for category in ['generic_python', 'specific_python', 'zombies', 'test_processes']:
            current_count = len(current.get(category, []))
            baseline_count = len(baseline.get(category, []))
            
            if current_count > baseline_count:
                leak_count = current_count - baseline_count
                leaks[category] = {
                    'count': leak_count,
                    'current': current_count,
                    'baseline': baseline_count,
                    'processes': current.get(category, [])
                }
        
        return leaks
    
    def print_status(self, processes: Dict, leaks: Dict):
        """Print current status."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Summary line
        generic_count = len(processes.get('generic_python', []))
        zombie_count = len(processes.get('zombies', []))
        test_count = len(processes.get('test_processes', []))
        
        status_line = f"[{timestamp}] Generic Python: {generic_count} | Zombies: {zombie_count} | Test Processes: {test_count}"
        
        if leaks:
            status_line += " | 🚨 LEAKS DETECTED"
        
        print(status_line)
        
        # Detailed leak information
        if leaks:
            for category, leak_info in leaks.items():
                if leak_info['count'] >= self.leak_threshold:
                    print(f"  🚨 {category}: +{leak_info['count']} processes (total: {leak_info['current']})")
                    
                    # Show details for generic python leaks (most critical)
                    if category == 'generic_python' and leak_info['count'] > 0:
                        print("    Details:")
                        for proc in leak_info['processes'][-3:]:  # Show last 3
                            cmdline = proc['cmdline'][:80] + "..." if len(proc['cmdline']) > 80 else proc['cmdline']
                            print(f"      PID {proc['pid']}: {cmdline}")
    
    def run_cleanup_if_needed(self, leaks: Dict):
        """Run cleanup if leak threshold is exceeded."""
        generic_leaks = leaks.get('generic_python', {}).get('count', 0)
        zombie_leaks = leaks.get('zombies', {}).get('count', 0)
        
        if generic_leaks >= self.leak_threshold or zombie_leaks >= self.leak_threshold:
            print(f"🧹 Leak threshold exceeded, running cleanup...")
            
            try:
                sys.path.insert(0, '/home/seth/wildfire-watch/tests')
                from enhanced_process_cleanup import get_process_cleaner
                
                cleaner = get_process_cleaner()
                if cleaner:
                    results = cleaner.cleanup_all()
                    cleaned_total = sum(results.values())
                    
                    if cleaned_total > 0:
                        print(f"✅ Cleaned up {cleaned_total} leaked processes")
                        # Reset baseline after cleanup
                        self.baseline_processes = self.get_python_processes()
                    else:
                        print("ℹ️  No processes to clean up")
                        
            except Exception as e:
                print(f"❌ Cleanup failed: {e}")
    
    def monitor(self):
        """Main monitoring loop."""
        print("🔍 Process Leak Monitor Started")
        print("=" * 60)
        print("Monitoring for:")
        print("  • Generic 'python' processes (main leak source)")
        print("  • Zombie processes")  
        print("  • Test processes")
        print(f"  • Cleanup threshold: {self.leak_threshold} processes")
        print("=" * 60)
        
        # Print baseline
        baseline_summary = {k: len(v) for k, v in self.baseline_processes.items()}
        print(f"Baseline: {baseline_summary}")
        print()
        
        try:
            while self.running:
                current_processes = self.get_python_processes()
                leaks = self.detect_leaks(current_processes, self.baseline_processes)
                
                self.print_status(current_processes, leaks)
                
                # Store history
                self.history.append({
                    'timestamp': time.time(),
                    'processes': current_processes,
                    'leaks': leaks
                })
                
                # Keep only last 100 entries
                if len(self.history) > 100:
                    self.history.pop(0)
                
                # Run cleanup if needed
                if leaks:
                    self.run_cleanup_if_needed(leaks)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n📡 Monitor stopped by user")
        
        self.print_summary()
    
    def print_summary(self):
        """Print monitoring summary."""
        print("\n📊 MONITORING SUMMARY")
        print("=" * 40)
        
        if not self.history:
            print("No data collected")
            return
        
        # Find peak leaks
        max_generic_leak = 0
        max_zombie_leak = 0
        total_cleanups = 0
        
        for entry in self.history:
            leaks = entry['leaks']
            generic_count = leaks.get('generic_python', {}).get('count', 0)
            zombie_count = leaks.get('zombies', {}).get('count', 0)
            
            max_generic_leak = max(max_generic_leak, generic_count)
            max_zombie_leak = max(max_zombie_leak, zombie_count)
        
        print(f"Peak generic python leaks: {max_generic_leak}")
        print(f"Peak zombie process leaks: {max_zombie_leak}")
        print(f"Monitoring duration: {len(self.history) * self.interval} seconds")
        
        # Current status
        if self.history:
            final_processes = self.history[-1]['processes']
            final_counts = {k: len(v) for k, v in final_processes.items()}
            print(f"Final process counts: {final_counts}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor for process leaks during testing")
    parser.add_argument('--interval', type=int, default=10, 
                       help='Monitoring interval in seconds (default: 10)')
    parser.add_argument('--threshold', type=int, default=5,
                       help='Leak threshold for cleanup (default: 5)')
    
    args = parser.parse_args()
    
    monitor = ProcessLeakMonitor(interval=args.interval)
    monitor.leak_threshold = args.threshold
    
    try:
        monitor.monitor()
    except Exception as e:
        print(f"Monitor error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())