"""
Pytest plugin to ensure integration tests run sequentially.

This prevents parallelization issues with Docker containers,
MQTT brokers, and fixed resource names.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add xdist markers for integration tests.
    
    This ensures that:
    1. Integration tests within the same class run in the same worker
    2. Docker-based tests don't conflict with each other
    3. MQTT broker ports don't collide
    """
    
    # Group tests by their class to ensure they run together
    for item in items:
        # Check if this is an integration or e2e test
        if any(marker in item.keywords for marker in ['integration', 'e2e', 'docker']):
            # Get the test class name
            if hasattr(item, 'cls') and item.cls:
                group_name = f"integration_{item.cls.__name__}"
            else:
                # For module-level tests, use the module name
                group_name = f"integration_{item.module.__name__}"
            
            # Add xdist_group marker to ensure tests run in same worker
            item.add_marker(pytest.mark.xdist_group(group_name))
            
            # For particularly problematic tests, force sequential execution
            problematic_tests = [
                'test_integration_e2e_improved',
                'test_e2e_hardware_docker',
                'test_integration_docker_sdk',
                'test_hardware_integration'
            ]
            
            if any(test_name in str(item.fspath) for test_name in problematic_tests):
                # Mark to run in the main worker only
                item.add_marker(pytest.mark.xdist_group("integration_sequential"))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "xdist_group(name): mark tests to run in the same worker process"
    )