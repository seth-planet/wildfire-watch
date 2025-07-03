#!/usr/bin/env python3.12
"""
Script to apply comprehensive fixes to the failing tests in wildfire-watch.
This addresses the root causes identified in the test failure analysis.
"""

import os
import sys
import subprocess

def fix_consensus_test_issues():
    """Fix the consensus test issues including empty list bug and thread cleanup."""
    print("Fixing consensus test issues...")
    
    # The health monitoring test fix has already been applied via the message_monitor fixture
    # The test is now passing
    
    # Fix the thread cleanup issue in consensus.py
    consensus_file = "fire_consensus/consensus.py"
    
    # Read the current file
    with open(consensus_file, 'r') as f:
        content = f.read()
    
    # Add proper thread cleanup with timeout
    if "_shutdown = True" in content and "# Wait for threads to finish" not in content:
        # Find the cleanup method and add thread wait
        cleanup_section = """        # Set shutdown flag to stop timer reschedules
        self._shutdown = True
        
        # Cancel any active timers
        if self._health_timer and self._health_timer.is_alive():
            self._health_timer.cancel()
        if self._cleanup_timer and self._cleanup_timer.is_alive():
            self._cleanup_timer.cancel()"""
        
        new_cleanup_section = """        # Set shutdown flag to stop timer reschedules
        self._shutdown = True
        
        # Cancel any active timers
        if self._health_timer and self._health_timer.is_alive():
            self._health_timer.cancel()
        if self._cleanup_timer and self._cleanup_timer.is_alive():
            self._cleanup_timer.cancel()
        
        # Wait for threads to finish with timeout
        for timer in [self._health_timer, self._cleanup_timer]:
            if timer and timer.is_alive():
                timer.join(timeout=1.0)"""
        
        content = content.replace(cleanup_section, new_cleanup_section)
        
        with open(consensus_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed thread cleanup in {consensus_file}")


def fix_telemetry_thread_cleanup():
    """Fix the telemetry service thread cleanup issue."""
    print("Fixing telemetry thread cleanup...")
    
    telemetry_file = "cam_telemetry/telemetry.py"
    
    # Read the current file
    with open(telemetry_file, 'r') as f:
        content = f.read()
    
    # Add thread tracking and proper cleanup
    if "active_timer = None" not in content:
        # Add active timer tracking
        timer_section = """    # Schedule next telemetry
    timer = threading.Timer(TELEMETRY_INT, publish_telemetry)
    timer.daemon = True
    timer.start()"""
        
        new_timer_section = """    # Schedule next telemetry
    global active_timer
    active_timer = threading.Timer(TELEMETRY_INT, publish_telemetry)
    active_timer.daemon = True
    active_timer.start()"""
        
        # Add global variable
        globals_section = """# ─────────────────────────────────────────────────────────────
#  MQTT setup with LWT
# ─────────────────────────────────────────────────────────────"""
        
        new_globals_section = """# ─────────────────────────────────────────────────────────────
#  Global state
# ─────────────────────────────────────────────────────────────
active_timer = None

# ─────────────────────────────────────────────────────────────
#  MQTT setup with LWT
# ─────────────────────────────────────────────────────────────"""
        
        # Update main cleanup
        cleanup_section = """    finally:
        client.loop_stop()
        client.disconnect()"""
        
        new_cleanup_section = """    finally:
        # Cancel active timer if running
        global active_timer
        if active_timer and active_timer.is_alive():
            active_timer.cancel()
        client.loop_stop()
        client.disconnect()"""
        
        content = content.replace(timer_section, new_timer_section)
        content = content.replace(globals_section, new_globals_section)
        content = content.replace(cleanup_section, new_cleanup_section)
        
        with open(telemetry_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Fixed thread cleanup in {telemetry_file}")


def fix_detect_optimized_scope_issue():
    """Fix the session scope issue in test_detect_optimized.py"""
    print("Fixing detect_optimized scope issue...")
    
    detect_file = "tests/test_detect_optimized.py"
    
    # Read the current file
    with open(detect_file, 'r') as f:
        content = f.read()
    
    # Fix the shared_mqtt_pool fixture to use session_mqtt_broker
    old_fixture = """@pytest.fixture(scope="session")
def shared_mqtt_pool(test_mqtt_broker):
    """
    
    new_fixture = """@pytest.fixture(scope="session")
def shared_mqtt_pool(session_mqtt_broker):
    """
    
    content = content.replace(old_fixture, new_fixture)
    
    # Also update the broker reference
    content = content.replace(
        "pool = SharedMQTTPool(test_mqtt_broker.get_connection_params())",
        "pool = SharedMQTTPool(session_mqtt_broker.get_connection_params())"
    )
    
    with open(detect_file, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed session scope issue in {detect_file}")


def run_test_verification():
    """Run a subset of previously failing tests to verify fixes."""
    print("\nVerifying fixes with test runs...")
    
    test_commands = [
        # Health monitoring test (previously failing with empty list)
        "python3.12 -m pytest tests/test_consensus.py::TestHealthMonitoring::test_health_report_generation -xvs",
        
        # Telemetry test (previously failing with thread issues)
        "python3.12 -m pytest tests/test_telemetry.py::test_multiple_telemetry_publishes -xvs",
        
        # Detect optimized test (previously failing with scope issue)
        "python3.12 -m pytest tests/test_detect_optimized.py::test_initialization_fast -xvs",
    ]
    
    results = []
    for cmd in test_commands:
        print(f"\nRunning: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        passed = "passed" in result.stdout and result.returncode == 0
        results.append((cmd, passed))
        print("✓ PASSED" if passed else "✗ FAILED")
    
    return results


def main():
    """Apply all test fixes."""
    print("Applying comprehensive test fixes...\n")
    
    # Apply fixes
    fix_consensus_test_issues()
    fix_telemetry_thread_cleanup()
    fix_detect_optimized_scope_issue()
    
    # Verify fixes
    print("\n" + "="*60)
    results = run_test_verification()
    
    # Summary
    print("\n" + "="*60)
    print("Fix Summary:")
    passed = sum(1 for _, p in results if p)
    print(f"✓ {passed}/{len(results)} verification tests passed")
    
    if passed < len(results):
        print("\nFailed tests:")
        for cmd, passed in results:
            if not passed:
                print(f"  - {cmd.split('::')[-1]}")
    
    print("\nNext steps:")
    print("1. Run the full test suite to check all fixes")
    print("2. Address any remaining failures individually")
    print("3. Update conftest.py to add better thread cleanup helpers")


if __name__ == "__main__":
    main()