"""
Pytest plugin for automatic process cleanup
"""
import pytest
import logging
import os
import sys

# Add tests directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

def pytest_configure(config):
    """Configure the cleanup plugin."""
    config.addinivalue_line("markers", "cleanup_critical: mark test as requiring aggressive cleanup")

def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test if marked as cleanup_critical."""
    if item.get_closest_marker("cleanup_critical"):
        from test_utils.process_cleanup import ProcessCleaner
        cleaner = ProcessCleaner()
        results = cleaner.cleanup_all()
        if sum(results.values()) > 0:
            logger.info(f"Post-test cleanup for {item.name}: {results}")

def pytest_sessionfinish(session, exitstatus):
    """Perform final cleanup at session end."""
    from test_utils.process_cleanup import ProcessCleaner
    
    logger.info("Performing final session cleanup...")
    cleaner = ProcessCleaner()
    results = cleaner.cleanup_all()
    
    if sum(results.values()) > 0:
        print(f"\nFinal cleanup results: {results}")
    else:
        print("\nNo cleanup needed at session end")

@pytest.fixture(autouse=True)
def resource_monitor(request):
    """Monitor and clean up resources for each test."""
    # Pre-test: record baseline
    import psutil
    import time
    
    initial_processes = len(psutil.pids())
    initial_memory = psutil.virtual_memory().used
    
    yield
    
    # Post-test: check for significant increases
    time.sleep(0.5)  # Brief delay to allow cleanup
    
    final_processes = len(psutil.pids())
    final_memory = psutil.virtual_memory().used
    
    process_increase = final_processes - initial_processes
    memory_increase = final_memory - initial_memory
    
    # If significant increase, trigger cleanup
    if process_increase > 5 or memory_increase > 100 * 1024 * 1024:  # 100MB
        logger.warning(f"Test {request.node.name} caused resource increase: "
                      f"+{process_increase} processes, +{memory_increase/1024/1024:.1f}MB")
        
        # Trigger cleanup for resource-heavy tests
        from test_utils.process_cleanup import ProcessCleaner
        cleaner = ProcessCleaner()
        cleaner.cleanup_mosquitto_processes()
        cleaner.cleanup_docker_containers()