#!/usr/bin/env python3.12
"""
Docker Container Lifecycle Management Fixes

This module provides enhanced Docker container lifecycle management to address
common issues in integration tests:
1. Container name conflicts
2. Network cleanup race conditions
3. Proper health checking
4. Resource cleanup on test failures
"""

import os
import time
import docker
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from contextlib import contextmanager
import atexit
import signal

logger = logging.getLogger(__name__)


class EnhancedDockerContainerManager:
    """
    Enhanced Docker container manager with better lifecycle handling
    
    Improvements:
    - Atomic container operations with locks
    - Proper network cleanup with retry logic
    - Health check improvements
    - Better error handling and logging
    - Automatic cleanup on process exit
    """
    
    def __init__(self, client: docker.DockerClient = None, worker_id: str = 'master'):
        self.client = client or docker.from_env()
        self.worker_id = worker_id
        self.container_prefix = f"wf-{worker_id}"
        self.created_containers = []
        self.created_networks = []
        self.created_images = []
        self._lock = threading.RLock()  # Reentrant lock
        self._cleanup_registered = False
        
        # Register cleanup handlers
        self._register_cleanup_handlers()
        
    def _register_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown"""
        if not self._cleanup_registered:
            atexit.register(self._emergency_cleanup)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            self._cleanup_registered = True
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, cleaning up containers...")
        self._emergency_cleanup()
        
    def _emergency_cleanup(self):
        """Emergency cleanup on process exit"""
        try:
            self.cleanup(force=True, timeout=5)
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def cleanup_old_container(self, container_name: str, timeout: int = 10, retry_count: int = 3) -> None:
        """
        Enhanced container cleanup with retry logic
        
        Args:
            container_name: Name of the container to remove
            timeout: Timeout for stopping the container
            retry_count: Number of retries for cleanup
        """
        for attempt in range(retry_count):
            try:
                with self._lock:
                    container = self.client.containers.get(container_name)
                    
                    # Check if container is being removed by another process
                    container.reload()
                    if container.attrs.get('State', {}).get('Status') == 'removing':
                        logger.info(f"Container {container_name} already being removed, waiting...")
                        time.sleep(2)
                        continue
                    
                    # Stop container if running
                    if container.status == 'running':
                        logger.info(f"Stopping container: {container_name}")
                        container.stop(timeout=timeout)
                    
                    # Wait a bit for stop to complete
                    time.sleep(0.5)
                    
                    # Remove container
                    logger.info(f"Removing container: {container_name}")
                    container.remove(force=True)
                    
                    # Wait for removal to complete
                    time.sleep(1)
                    
                    # Verify removal
                    try:
                        self.client.containers.get(container_name)
                        # If we get here, container still exists
                        if attempt < retry_count - 1:
                            logger.warning(f"Container {container_name} still exists, retrying...")
                            continue
                    except docker.errors.NotFound:
                        # Success - container removed
                        logger.info(f"Successfully removed container: {container_name}")
                        return
                        
            except docker.errors.NotFound:
                # Container doesn't exist - success
                return
            except docker.errors.APIError as e:
                if "removal of container" in str(e) and "already in progress" in str(e):
                    logger.info(f"Container {container_name} removal already in progress")
                    time.sleep(3)
                    return
                elif attempt < retry_count - 1:
                    logger.warning(f"Failed to remove container {container_name}: {e}, retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"Failed to remove container {container_name} after {retry_count} attempts: {e}")
                    raise
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Unexpected error removing container {container_name}: {e}, retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"Failed to remove container {container_name}: {e}")
                    raise
    
    def start_container(self, image: str, name: str, config: Dict[str, Any],
                       wait_timeout: int = 30, health_check_fn: Callable = None,
                       retry_count: int = 3) -> docker.models.containers.Container:
        """
        Enhanced container start with better error handling
        
        Args:
            image: Image to use
            name: Container name
            config: Container configuration
            wait_timeout: Time to wait for container to be ready
            health_check_fn: Optional health check function
            retry_count: Number of retries for container start
            
        Returns:
            Started container instance
        """
        # Always cleanup old container first
        self.cleanup_old_container(name)
        
        for attempt in range(retry_count):
            try:
                with self._lock:
                    logger.info(f"Starting container: {name} (attempt {attempt + 1}/{retry_count})")
                    
                    # Merge configuration
                    config = config.copy()
                    config['name'] = name
                    config['image'] = image
                    config['detach'] = True
                    
                    # Add labels for tracking
                    config.setdefault('labels', {})
                    config['labels'].update({
                        'com.wildfire.test': 'true',
                        'com.wildfire.worker': self.worker_id,
                        'com.wildfire.created': str(time.time())
                    })
                    
                    # Add resource limits if not specified
                    if 'mem_limit' not in config:
                        config['mem_limit'] = '1g'
                    if 'cpu_quota' not in config:
                        config['cpu_quota'] = 100000  # 100% of one CPU
                    
                    # Ensure remove=False to prevent auto-removal
                    config['remove'] = False
                    
                    # Start container
                    container = self.client.containers.run(**config)
                    self.created_containers.append(container)
                    
                    # Wait for container to initialize
                    if not self._wait_for_container_ready(container, wait_timeout, health_check_fn):
                        raise RuntimeError(f"Container {name} failed to become ready")
                    
                    logger.info(f"✓ Container {name} is running and healthy")
                    return container
                    
            except Exception as e:
                logger.error(f"Failed to start container {name}: {e}")
                
                # Cleanup failed container
                try:
                    if 'container' in locals():
                        container.stop(timeout=5)
                        container.remove(force=True)
                except:
                    pass
                
                if attempt < retry_count - 1:
                    logger.info(f"Retrying container start for {name}...")
                    time.sleep(3)
                else:
                    raise
    
    def _wait_for_container_ready(self, container: docker.models.containers.Container,
                                  timeout: int, health_check_fn: Optional[Callable]) -> bool:
        """
        Wait for container to be ready with improved health checking
        
        Args:
            container: Container to check
            timeout: Maximum time to wait
            health_check_fn: Optional custom health check
            
        Returns:
            True if container is ready, False otherwise
        """
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                # Reload container state
                container.reload()
                
                # Check if container exited
                if container.status == 'exited':
                    logs = container.logs(tail=100).decode()
                    logger.error(f"Container exited. Logs:\n{logs}")
                    return False
                
                # Check built-in health status
                health = container.attrs.get('State', {}).get('Health', {})
                health_status = health.get('Status', 'none')
                
                if health_status != last_status:
                    logger.info(f"Container health status: {health_status}")
                    last_status = health_status
                
                # If container has health check and it's healthy
                if health_status == 'healthy':
                    return True
                
                # If container has health check but it's unhealthy
                if health_status == 'unhealthy':
                    logger.warning("Container reported unhealthy")
                    # Get last health check log
                    if health.get('Log'):
                        last_check = health['Log'][-1]
                        logger.warning(f"Last health check: {last_check.get('Output', 'No output')}")
                
                # Run custom health check if provided
                if health_check_fn:
                    try:
                        if health_check_fn(container):
                            return True
                    except Exception as e:
                        logger.debug(f"Custom health check error: {e}")
                
                # If no health check, just ensure it's running
                elif container.status == 'running' and health_status == 'none':
                    # Wait a bit more for startup
                    time.sleep(3)
                    return True
                
            except Exception as e:
                logger.error(f"Error checking container health: {e}")
            
            time.sleep(2)
        
        # Timeout reached
        try:
            logs = container.logs(tail=50).decode()
            logger.error(f"Container failed to become ready. Last logs:\n{logs}")
        except:
            pass
            
        return False
    
    def create_test_network(self, network_name: str = None, retry_count: int = 3) -> docker.models.networks.Network:
        """
        Create test network with cleanup and retry logic
        
        Args:
            network_name: Network name (auto-generated if not provided)
            retry_count: Number of retries for network creation
            
        Returns:
            Created network
        """
        if network_name is None:
            network_name = self.get_network_name()
        
        for attempt in range(retry_count):
            try:
                with self._lock:
                    # Remove old network if exists
                    try:
                        old_net = self.client.networks.get(network_name)
                        
                        # Disconnect all containers first
                        for container in old_net.containers:
                            try:
                                old_net.disconnect(container, force=True)
                            except:
                                pass
                        
                        # Remove network
                        old_net.remove()
                        time.sleep(1)
                    except docker.errors.NotFound:
                        pass
                    
                    # Create new network
                    network = self.client.networks.create(
                        name=network_name,
                        driver="bridge",
                        labels={
                            'com.wildfire.test': 'true',
                            'com.wildfire.worker': self.worker_id
                        }
                    )
                    
                    self.created_networks.append(network)
                    logger.info(f"Created test network: {network_name}")
                    return network
                    
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Failed to create network: {e}, retrying...")
                    time.sleep(2)
                else:
                    raise
    
    def cleanup(self, force: bool = False, timeout: int = 10):
        """
        Enhanced cleanup with better error handling
        
        Args:
            force: Force cleanup even if errors occur
            timeout: Timeout for container stop operations
        """
        logger.info(f"Starting cleanup for worker {self.worker_id}")
        
        with self._lock:
            # Stop and remove containers
            for container in self.created_containers[:]:  # Copy list to avoid modification during iteration
                try:
                    if isinstance(container, str):
                        try:
                            container = self.client.containers.get(container)
                        except docker.errors.NotFound:
                            continue
                    
                    container_name = container.name
                    logger.debug(f"Cleaning up container: {container_name}")
                    
                    # Stop container
                    if container.status == 'running':
                        container.stop(timeout=timeout)
                    
                    # Remove container
                    container.remove(force=True)
                    logger.debug(f"Removed container: {container_name}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up container: {e}")
                    if not force:
                        raise
            
            # Clear container list
            self.created_containers.clear()
            
            # Remove networks
            for network in self.created_networks[:]:
                try:
                    if isinstance(network, str):
                        try:
                            network = self.client.networks.get(network)
                        except docker.errors.NotFound:
                            continue
                    
                    network_name = network.name
                    logger.debug(f"Cleaning up network: {network_name}")
                    
                    # Disconnect all containers
                    network.reload()
                    for container in network.containers:
                        try:
                            network.disconnect(container, force=True)
                        except:
                            pass
                    
                    # Remove network
                    network.remove()
                    logger.debug(f"Removed network: {network_name}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up network: {e}")
                    if not force:
                        raise
            
            # Clear network list
            self.created_networks.clear()
            
        logger.info(f"Cleanup completed for worker {self.worker_id}")
    
    def get_container_name(self, service: str) -> str:
        """Get unique container name for service"""
        return f"{self.container_prefix}-{service}"
    
    def get_network_name(self) -> str:
        """Get unique network name"""
        return f"{self.container_prefix}-network"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(force=True)


@contextmanager
def managed_docker_containers(worker_id: str = 'master'):
    """
    Context manager for Docker containers with automatic cleanup
    
    Usage:
        with managed_docker_containers(worker_id) as manager:
            container = manager.start_container(...)
            # Use containers
        # Automatic cleanup on exit
    """
    manager = EnhancedDockerContainerManager(worker_id=worker_id)
    try:
        yield manager
    finally:
        manager.cleanup(force=True)


# Fixture for pytest
def pytest_fixture_docker_manager(worker_id):
    """
    Pytest fixture for enhanced Docker container manager
    
    Usage in conftest.py:
        from docker_lifecycle_fix import pytest_fixture_docker_manager
        
        @pytest.fixture
        def docker_container_manager(worker_id):
            return pytest_fixture_docker_manager(worker_id)
    """
    manager = EnhancedDockerContainerManager(worker_id=worker_id)
    yield manager
    manager.cleanup(force=True)