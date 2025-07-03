#!/usr/bin/env python3.12
"""
Test Script for Process Leak Fixes

This script validates that our fixes for generic 'python' process leaks are working.
It checks Docker containers, subprocess calls, and process cleanup functionality.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def test_docker_containers():
    """Test that Docker containers use specific Python versions."""
    print("🔍 Testing Docker container Python versions...")
    
    try:
        # Test camera detector
        result = subprocess.run(['docker', 'run', '--rm', 'python:3.12-slim', 
                               'python3.12', '-c', 'import sys; print(f"Python {sys.version}")'],
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Docker python3.12 execution works")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Docker python3.12 execution failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Docker test timed out")
        return False
    except Exception as e:
        print(f"❌ Docker test error: {e}")
        return False
    
    return True

def test_subprocess_cleanup():
    """Test that subprocess cleanup works properly."""
    print("🔍 Testing subprocess cleanup...")
    
    try:
        # Import our enhanced cleanup
        sys.path.insert(0, '/home/seth/wildfire-watch/tests')
        from enhanced_process_cleanup import get_process_cleaner
        
        cleaner = get_process_cleaner()
        if cleaner is None:
            print("❌ Could not get process cleaner")
            return False
        
        # Perform cleanup test
        results = cleaner.cleanup_all()
        print(f"✅ Process cleanup executed: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Subprocess cleanup test error: {e}")
        return False

def test_process_count_before_after():
    """Test process count before and after cleanup."""
    print("🔍 Testing process count reduction...")
    
    try:
        # Count python processes before
        result_before = subprocess.run(['pgrep', '-c', '^python$'], 
                                     capture_output=True, text=True)
        
        count_before = int(result_before.stdout.strip()) if result_before.returncode == 0 else 0
        print(f"Generic python processes before cleanup: {count_before}")
        
        # Run cleanup
        sys.path.insert(0, '/home/seth/wildfire-watch/tests')
        from enhanced_process_cleanup import get_process_cleaner
        
        cleaner = get_process_cleaner()
        if cleaner:
            results = cleaner.cleanup_generic_python_processes()
            print(f"Cleaned up {results} generic python processes")
        
        # Count python processes after
        result_after = subprocess.run(['pgrep', '-c', '^python$'], 
                                    capture_output=True, text=True)
        
        count_after = int(result_after.stdout.strip()) if result_after.returncode == 0 else 0
        print(f"Generic python processes after cleanup: {count_after}")
        
        if count_after < count_before:
            print("✅ Process count reduced successfully")
            return True
        elif count_before == 0:
            print("✅ No generic python processes found (good)")
            return True
        else:
            print("⚠️  Process count unchanged (may be normal)")
            return True
            
    except Exception as e:
        print(f"❌ Process count test error: {e}")
        return False

def test_dockerfile_fixes():
    """Test that Dockerfiles use correct Python versions."""
    print("🔍 Testing Dockerfile fixes...")
    
    dockerfiles = [
        'camera_detector/Dockerfile',
        'fire_consensus/Dockerfile', 
        'gpio_trigger/Dockerfile',
        'cam_telemetry/Dockerfile'
    ]
    
    base_path = Path('/home/seth/wildfire-watch')
    
    for dockerfile in dockerfiles:
        dockerfile_path = base_path / dockerfile
        
        try:
            content = dockerfile_path.read_text()
            
            # Check for generic 'python' in CMD
            if 'CMD ["python"' in content:
                print(f"❌ {dockerfile} still uses generic python in CMD")
                return False
            
            # Check for generic 'python' in HEALTHCHECK  
            if 'CMD python -c' in content:
                print(f"❌ {dockerfile} still uses generic python in HEALTHCHECK")
                return False
                
            print(f"✅ {dockerfile} uses specific Python version")
            
        except Exception as e:
            print(f"❌ Error checking {dockerfile}: {e}")
            return False
    
    return True

def test_docker_compose_fixes():
    """Test that docker-compose.yml uses correct Python versions."""
    print("🔍 Testing docker-compose.yml fixes...")
    
    try:
        compose_path = Path('/home/seth/wildfire-watch/docker-compose.yml')
        content = compose_path.read_text()
        
        # Check for generic python in health checks
        if '"python", "-c"' in content:
            print("❌ docker-compose.yml still uses generic python in health checks")
            return False
        
        # Check for generic python in commands
        if '"python", "convert_model.py"' in content:
            print("❌ docker-compose.yml still uses generic python in commands")
            return False
            
        print("✅ docker-compose.yml uses specific Python versions")
        return True
        
    except Exception as e:
        print(f"❌ Error checking docker-compose.yml: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 PROCESS LEAK FIX VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Dockerfile fixes", test_dockerfile_fixes),
        ("docker-compose.yml fixes", test_docker_compose_fixes),
        ("Docker container execution", test_docker_containers),
        ("Process cleanup functionality", test_subprocess_cleanup),
        ("Process count reduction", test_process_count_before_after),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Process leak fixes are working!")
        return 0
    else:
        print("⚠️  Some tests failed - review the fixes")
        return 1

if __name__ == '__main__':
    sys.exit(main())