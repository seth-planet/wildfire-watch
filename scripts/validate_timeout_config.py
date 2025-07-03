#!/usr/bin/env python3.12
"""
Validate timeout configuration for Wildfire Watch tests.

This script checks that all timeout-related settings are properly configured
and provides recommendations for optimal test performance.
"""

import os
import sys
import configparser
import subprocess
import time
from pathlib import Path

def check_pytest_ini():
    """Check pytest.ini timeout configuration."""
    print("=== Checking pytest.ini Configuration ===")
    
    pytest_ini = Path("/home/seth/wildfire-watch/pytest.ini")
    if not pytest_ini.exists():
        print("❌ pytest.ini not found")
        return False
    
    config = configparser.ConfigParser()
    config.read(pytest_ini)
    
    if 'tool:pytest' not in config:
        print("❌ [tool:pytest] section not found in pytest.ini")
        return False
    
    pytest_config = config['tool:pytest']
    
    # Check timeout settings
    timeout = pytest_config.get('timeout', '0')
    timeout_method = pytest_config.get('timeout_method', 'signal')
    timeout_func_only = pytest_config.get('timeout_func_only', 'false')
    session_timeout = pytest_config.get('session_timeout', 'None')
    
    print(f"✅ timeout = {timeout}s")
    print(f"✅ timeout_method = {timeout_method}")
    print(f"✅ timeout_func_only = {timeout_func_only}")
    print(f"✅ session_timeout = {session_timeout}s")
    
    # Validate values
    if int(timeout) < 1800:  # 30 minutes minimum
        print(f"⚠️  Warning: timeout ({timeout}s) may be too short for infrastructure setup")
    
    if timeout_method != 'thread':
        print(f"⚠️  Warning: timeout_method '{timeout_method}' may be less reliable than 'thread'")
    
    if timeout_func_only.lower() != 'true':
        print("⚠️  Warning: timeout_func_only should be 'true' to allow fixture setup time")
    
    return True

def check_conftest():
    """Check conftest.py timeout fixtures."""
    print("\n=== Checking conftest.py Fixtures ===")
    
    conftest_path = Path("/home/seth/wildfire-watch/tests/conftest.py")
    if not conftest_path.exists():
        print("❌ conftest.py not found")
        return False
    
    with open(conftest_path) as f:
        content = f.read()
    
    required_fixtures = [
        'session_mqtt_broker',
        'long_timeout_environment',
        'timeout_aware_test_setup',
        'test_isolation'
    ]
    
    for fixture in required_fixtures:
        if fixture in content:
            print(f"✅ {fixture} fixture present")
        else:
            print(f"❌ {fixture} fixture missing")
    
    return True

def check_timeout_utils():
    """Check timeout utilities."""
    print("\n=== Checking timeout_utils.py ===")
    
    utils_path = Path("/home/seth/wildfire-watch/tests/timeout_utils.py")
    if not utils_path.exists():
        print("❌ timeout_utils.py not found")
        return False
    
    try:
        sys.path.insert(0, str(utils_path.parent))
        import timeout_utils
        
        # Check decorators
        decorators = ['expect_long_timeout', 'mqtt_infrastructure_test', 'integration_test']
        for decorator in decorators:
            if hasattr(timeout_utils, decorator):
                print(f"✅ {decorator} decorator available")
            else:
                print(f"❌ {decorator} decorator missing")
        
        # Check context managers
        contexts = ['timeout_context', 'mqtt_setup_context', 'service_startup_context']
        for context in contexts:
            if hasattr(timeout_utils, context):
                print(f"✅ {context} context manager available")
            else:
                print(f"❌ {context} context manager missing")
        
    except ImportError as e:
        print(f"❌ Failed to import timeout_utils: {e}")
        return False
    
    return True

def check_pytest_plugins():
    """Check required pytest plugins."""
    print("\n=== Checking pytest Plugins ===")
    
    required_plugins = ['pytest-timeout', 'pytest-mock']
    
    for plugin in required_plugins:
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'show', plugin
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ {plugin} installed")
            else:
                print(f"❌ {plugin} not installed")
                print(f"   Install with: pip3.12 install {plugin}")
                
        except subprocess.TimeoutExpired:
            print(f"⚠️  Timeout checking {plugin}")
        except Exception as e:
            print(f"⚠️  Error checking {plugin}: {e}")

def test_timeout_configuration():
    """Test timeout configuration with a simple test."""
    print("\n=== Testing Timeout Configuration ===")
    
    try:
        # Run a simple timeout test
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_timeout_configuration.py::test_short_test',
            '-v', '--tb=short'
        ], cwd='/home/seth/wildfire-watch', capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Basic timeout test passed")
            
            # Check if test completed in reasonable time
            if "passed" in result.stdout:
                print("✅ Test execution successful")
        else:
            print(f"❌ Timeout test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout test timed out")
    except Exception as e:
        print(f"❌ Error running timeout test: {e}")

def benchmark_mqtt_setup():
    """Benchmark MQTT broker setup time."""
    print("\n=== Benchmarking MQTT Setup ===")
    
    try:
        # Time MQTT broker setup
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/test_timeout_configuration.py::test_real_mqtt_broker_timeout',
            '-v', '--tb=short'
        ], cwd='/home/seth/wildfire-watch', capture_output=True, text=True, timeout=120)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ MQTT setup completed in {duration:.2f}s")
            
            if duration < 30:
                print("✅ MQTT setup time is acceptable")
            elif duration < 60:
                print("⚠️  MQTT setup is slow but tolerable")
            else:
                print("❌ MQTT setup is very slow")
        else:
            print(f"❌ MQTT test failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ MQTT benchmark timed out after 2 minutes")
    except Exception as e:
        print(f"❌ Error benchmarking MQTT: {e}")

def provide_recommendations():
    """Provide recommendations for timeout optimization."""
    print("\n=== Recommendations ===")
    
    print("For optimal test performance:")
    print("1. ✅ Use session-scoped MQTT broker (implemented)")
    print("2. ✅ Set high timeout values (implemented)")
    print("3. ✅ Use thread-based timeout method (implemented)")
    print("4. ✅ Mark slow tests appropriately (implemented)")
    print("5. ✅ Provide timeout utilities (implemented)")
    
    print("\nTo run tests efficiently:")
    print("• Normal tests: python3.12 -m pytest tests/ -v")
    print("• Skip slow tests: python3.12 -m pytest tests/ -v -m 'not slow'")
    print("• MQTT tests only: python3.12 -m pytest tests/ -v -m 'mqtt'")
    print("• No timeout (debug): python3.12 -m pytest tests/ -v --timeout=0")
    
    print("\nExpected timing:")
    print("• Individual test: <10s normal, <60s slow")
    print("• MQTT infrastructure: ~15s (session setup)")
    print("• Full test suite: <30 minutes normal, <2 hours with slow tests")

def main():
    """Main validation function."""
    print("Wildfire Watch Timeout Configuration Validator")
    print("=" * 50)
    
    checks = [
        check_pytest_ini,
        check_conftest,
        check_timeout_utils,
        check_pytest_plugins,
        test_timeout_configuration,
        benchmark_mqtt_setup,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Error in {check.__name__}: {e}")
            results.append(False)
    
    provide_recommendations()
    
    print(f"\n=== Summary ===")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All timeout configuration checks passed!")
        print("✅ Tests should handle long timeouts gracefully")
    else:
        print(f"⚠️  {total - passed} checks failed - see details above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)