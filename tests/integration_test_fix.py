#!/usr/bin/env python3.12
"""
Fix for integration test parallelization issues.

The main issues:
1. Fixed container names causing conflicts
2. Fixed MQTT ports causing conflicts  
3. Services trying to publish before MQTT is ready
4. Tests not using unique topics

This module provides utilities to make integration tests parallel-safe.
"""
import uuid
import os
from typing import Dict, Optional


class ParallelSafeIntegrationSetup:
    """Integration test setup that's safe for parallel execution"""
    
    def __init__(self, worker_id: str = None):
        """
        Initialize with unique identifiers for parallel safety.
        
        Args:
            worker_id: pytest-xdist worker ID (e.g., 'gw0', 'gw1', etc.)
        """
        # Generate unique suffix for this test instance
        if worker_id:
            # Use worker ID for consistency within a worker
            self.unique_suffix = worker_id
        else:
            # Fallback to UUID for uniqueness
            self.unique_suffix = uuid.uuid4().hex[:8]
            
        # Container name mapping
        self.container_names = {
            'mqtt': f'mqtt-broker-test-{self.unique_suffix}',
            'camera_detector': f'camera-detector-test-{self.unique_suffix}',
            'fire_consensus': f'fire-consensus-test-{self.unique_suffix}',
            'gpio_trigger': f'gpio-trigger-test-{self.unique_suffix}',
            'frigate': f'frigate-test-{self.unique_suffix}'
        }
        
        # Unique network name
        self.network_name = f'wildfire-test-net-{self.unique_suffix}'
        
        # Port allocation based on worker ID
        if worker_id and worker_id.startswith('gw'):
            # Extract worker number (gw0 -> 0, gw1 -> 1, etc.)
            try:
                worker_num = int(worker_id[2:])
                # Allocate ports with 10-port spacing to avoid conflicts
                self.mqtt_port = 11883 + (worker_num * 10)
                self.mqtt_tls_port = 18883 + (worker_num * 10)
            except ValueError:
                # Fallback to random high ports
                import random
                base = random.randint(20000, 30000)
                self.mqtt_port = base
                self.mqtt_tls_port = base + 1
        else:
            # Use default ports for non-parallel execution
            self.mqtt_port = 11883
            self.mqtt_tls_port = 18883
            
    def get_container_env(self, service: str) -> Dict[str, str]:
        """
        Get environment variables for a service with proper MQTT configuration.
        
        Args:
            service: Service name (camera_detector, fire_consensus, etc.)
            
        Returns:
            Dictionary of environment variables
        """
        base_env = {
            'MQTT_BROKER': 'localhost',  # Using host networking
            'MQTT_PORT': str(self.mqtt_port),
            'MQTT_TLS': 'false',
            'LOG_LEVEL': 'DEBUG',
            # Add unique client ID to prevent conflicts
            'MQTT_CLIENT_ID': f'{service}_{self.unique_suffix}'
        }
        
        # Service-specific environment variables
        if service == 'camera_detector':
            base_env.update({
                'CAMERA_CREDENTIALS': os.getenv('CAMERA_CREDENTIALS', 'admin:S3thrule'),
                'DISCOVERY_INTERVAL': '30'
            })
        elif service == 'fire_consensus':
            base_env.update({
                'CONSENSUS_THRESHOLD': '2',
                'TIME_WINDOW': '30',
                'MIN_CONFIDENCE': '0.6'
            })
        elif service == 'gpio_trigger':
            base_env.update({
                'GPIO_SIMULATION': 'true',
                'MAX_ENGINE_RUNTIME': '30'
            })
            
        return base_env
        
    def get_unique_topic(self, base_topic: str) -> str:
        """
        Generate a unique topic for this test instance.
        
        Args:
            base_topic: Base topic name
            
        Returns:
            Unique topic with test instance suffix
        """
        return f'test_{self.unique_suffix}/{base_topic}'


def fix_integration_test_class(test_class_path: str):
    """
    Add parallel-safety markers to an integration test class.
    
    Args:
        test_class_path: Path to the test file
    """
    import re
    
    with open(test_class_path, 'r') as f:
        content = f.read()
        
    # Add xdist_group marker if not present
    if '@pytest.mark.xdist_group' not in content:
        # Find class definition
        class_pattern = r'(@pytest\.mark\.\w+\s*\n)*class\s+(\w+):'
        
        def add_xdist_marker(match):
            markers = match.group(1) or ''
            class_name = match.group(2)
            # Add xdist_group marker with unique group name
            xdist_marker = f'@pytest.mark.xdist_group("{class_name.lower()}")\n'
            return markers + xdist_marker + f'class {class_name}:'
            
        content = re.sub(class_pattern, add_xdist_marker, content)
        
        with open(test_class_path, 'w') as f:
            f.write(content)
            
        print(f"✅ Added xdist_group marker to {test_class_path}")


# Pytest plugin hooks for parallel safety
def pytest_configure(config):
    """Configure pytest for parallel-safe integration tests"""
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "xdist_group(name): mark tests to run in same worker"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for parallel safety"""
    # Group integration tests by class to run in same worker
    for item in items:
        if 'integration' in item.keywords or 'e2e' in item.keywords:
            # Add xdist_group marker if not present
            class_name = item.cls.__name__ if item.cls else 'default'
            item.add_marker(pytest.mark.xdist_group(class_name.lower()))


if __name__ == '__main__':
    # Fix known integration test files
    import glob
    
    test_files = [
        'tests/test_integration_e2e_improved.py',
        'tests/test_e2e_hardware_docker.py',
        'tests/test_integration_docker_sdk.py',
        'tests/test_e2e_hardware_integration.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            fix_integration_test_class(test_file)