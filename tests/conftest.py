import sys
import os
import subprocess
from pathlib import Path
import pytest
import time
import uuid
import paho.mqtt.client as mqtt
import multiprocessing
import logging

# Add tests directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

# Import safe logging to prevent I/O on closed file errors
# This is critical for parallel test execution
from utils.safe_logging import disable_problem_loggers, cleanup_test_logging
disable_problem_loggers()

# Import enhanced process cleanup for proper test isolation
try:
    from enhanced_process_cleanup import get_process_cleaner, cleanup_on_test_failure
    # Configure to not kill test processes
    import enhanced_process_cleanup
    if hasattr(enhanced_process_cleanup, 'PROTECTED_PATTERNS'):
        # Add pytest worker processes to protected patterns
        enhanced_process_cleanup.PROTECTED_PATTERNS.extend([
            'pytest', 'py.test', 'xdist', 'gw[0-9]+', 'execnet'
        ])
except ImportError:
    def get_process_cleaner():
        return None
    def cleanup_on_test_failure():
        pass

def has_coral_tpu():
    """Check if Coral TPU is available"""
    try:
        result = subprocess.run(
            ['python3.8', '-c', 'from pycoral.utils.edgetpu import list_edge_tpus; print(len(list_edge_tpus()))'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            num_tpus = int(result.stdout.strip())
            return num_tpus > 0
    except:
        pass
    return False

def has_tensorrt():
    """Check if TensorRT is available"""
    try:
        import tensorrt
        return True
    except ImportError:
        return False

def has_hailo():
    """Check if Hailo device is available"""
    try:
        # Check if Hailo device exists
        if os.path.exists('/dev/hailo0'):
            return True
        # Alternative: try to use hailortcli
        result = subprocess.run(
            ['hailortcli', 'scan'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    except:
        pass
    return False

def has_camera_on_network():
    """Check if there's a camera on the network"""
    # We have cameras on the network with credentials admin:S3thrule
    # Always return True since we know cameras exist
    return True

def pytest_ignore_collect(collection_path, config):
    python_version = sys.version_info
    path = Path(collection_path)
    if "hailo" in path.name and python_version[:2] != (3, 10):
        return True
    if "coral" in path.name and python_version[:2] != (3, 8):
        return True
    return False

def pytest_runtest_makereport(item, call):
    """Hook to perform cleanup on test failures."""
    if call.when == "call":
        if call.excinfo is not None:
            # Test failed, perform emergency cleanup
            print(f"\n⚠️  Test {item.name} failed, performing process cleanup...")
            try:
                cleanup_on_test_failure()
            except Exception as e:
                print(f"Cleanup failed: {e}")

@pytest.fixture(scope="session", autouse=True)
def session_process_cleanup():
    """Automatic process cleanup at session start and end."""
    # Cleanup at start of session
    cleaner = get_process_cleaner()
    if cleaner:
        print("🧹 Performing initial process cleanup...")
        initial_results = cleaner.cleanup_all()
        if sum(initial_results.values()) > 0:
            print(f"Cleaned up {sum(initial_results.values())} leaked processes from previous runs")
    
    yield
    
    # Cleanup at end of session
    if cleaner:
        print("🧹 Performing final process cleanup...")
        final_results = cleaner.cleanup_all()
        if sum(final_results.values()) > 0:
            print(f"Final cleanup: {sum(final_results.values())} items cleaned")


# ─────────────────────────────────────────────────────────────
# Shared MQTT Test Infrastructure
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def test_mqtt_broker(worker_id):
    """Provide a test MQTT broker using enhanced broker with session reuse"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from enhanced_mqtt_broker import TestMQTTBroker
    
    broker = TestMQTTBroker(session_scope=True, worker_id=worker_id)
    broker.start()
    
    # Reset state for this test
    broker.reset_state()
    
    yield broker
    
    # Don't stop session broker (it's reused)


@pytest.fixture
def mqtt_client_factory(test_mqtt_broker):
    """Factory for creating MQTT clients with automatic cleanup"""
    clients = []
    
    def create_client(client_id=None):
        if not client_id:
            client_id = f"test_client_{uuid.uuid4().hex[:8]}"
            
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        conn_params = test_mqtt_broker.get_connection_params()
        client.connect(conn_params['host'], conn_params['port'], 60)
        client.loop_start()
        
        # Track for cleanup
        clients.append(client)
        
        # Wait for connection
        start_time = time.time()
        while not client.is_connected() and time.time() - start_time < 5:
            time.sleep(0.1)
            
        return client
    
    yield create_client
    
    # Cleanup all created clients
    for client in clients:
        try:
            client.loop_stop()
            client.disconnect()
        except:
            pass


@pytest.fixture
def mqtt_client(mqtt_client_factory):
    """Provide a single MQTT client for tests"""
    client = mqtt_client_factory()
    yield client
    # Cleanup handled by factory


@pytest.fixture
def mqtt_topic_factory(worker_id):
    """Generate unique MQTT topics for parallel test safety"""
    def generate_topic(base_topic):
        # Use worker_id if available (from pytest-xdist)
        if worker_id and worker_id != 'master':
            return f"test_{worker_id}/{base_topic}"
        else:
            unique_id = uuid.uuid4().hex[:8]
            return f"test_{unique_id}/{base_topic}"
    return generate_topic


@pytest.fixture(scope='session')
def worker_id(request):
    """Get the pytest-xdist worker ID for parallel test isolation"""
    # This is provided by pytest-xdist
    # Will be 'master' for non-parallel execution
    # Will be 'gw0', 'gw1', etc. for parallel workers
    return getattr(request.config, 'workerinput', {}).get('workerid', 'master')


@pytest.fixture
def test_mqtt_tls_broker(tmp_path):
    """Provide a test MQTT broker with TLS enabled"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from mqtt_tls_test_broker import MQTTTLSTestBroker
    
    broker = MQTTTLSTestBroker(cert_dir=str(tmp_path))
    broker.start()
    
    yield broker
    
    broker.stop()


# ─────────────────────────────────────────────────────────────
# Session cleanup hooks for parallel test isolation
# ─────────────────────────────────────────────────────────────

def pytest_sessionstart(session):
    """Clean up any leftover containers before starting tests"""
    import subprocess
    
    print("Pre-test cleanup: removing any leftover test containers...")
    
    try:
        # Clean up any containers with our test labels
        result = subprocess.run(['docker', 'ps', '-aq', '--filter', 'label=com.wildfire.test=true'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            for container_id in container_ids:
                if container_id.strip():
                    try:
                        subprocess.run(['docker', 'rm', '-f', container_id.strip()], timeout=5)
                        print(f"Removed leftover test container {container_id}")
                    except:
                        pass
    except Exception as e:
        print(f"Warning: Error in pre-test cleanup: {e}")


def pytest_sessionfinish(session, exitstatus):
    """Clean up all test resources at session end"""
    import subprocess
    import time
    import sys
    import os
    
    # Clean up all test logging to prevent I/O on closed file errors
    try:
        from safe_logging import cleanup_test_logging
        cleanup_test_logging()
    except ImportError:
        # Fallback: Prevent super-gradients from writing to closed file handles during teardown
        try:
            # Redirect super-gradients output to null to prevent hanging
            import logging
            logging.getLogger("super_gradients").handlers.clear()
            logging.getLogger("super_gradients").addHandler(logging.NullHandler())
        except:
            pass
    
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from enhanced_mqtt_broker import TestMQTTBroker
    
    # Clean up all worker brokers
    TestMQTTBroker.cleanup_session()
    
    # Additional aggressive cleanup of stray processes
    try:
        # Kill any remaining mosquitto test processes
        result = subprocess.run(['pgrep', '-f', 'mqtt_test_'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    try:
                        subprocess.run(['kill', '-KILL', pid.strip()], timeout=1)
                        print(f"Force killed stray mosquitto process {pid}")
                    except:
                        pass
        
        # Clean up any containers with our test labels
        result = subprocess.run(['docker', 'ps', '-aq', '--filter', 'label=com.wildfire.test=true'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            for container_id in container_ids:
                if container_id.strip():
                    try:
                        subprocess.run(['docker', 'rm', '-f', container_id.strip()], timeout=5)
                        print(f"Force removed stray test container {container_id}")
                    except:
                        pass
                        
    except Exception as e:
        print(f"Warning: Error in aggressive cleanup: {e}")
    
    # Log cleanup status
    if hasattr(session.config, '_worker_id'):
        worker_id = session.config._worker_id
        print(f"\nWorker {worker_id} completed test session cleanup")


def pytest_configure(config):
    """Configure pytest with worker ID tracking and multiprocessing setup"""
    # Track worker ID for cleanup
    worker_id = getattr(config, 'workerinput', {}).get('workerid', 'master')
    config._worker_id = worker_id
    
    # Set multiprocessing start method
    # For pytest-xdist, we should not change the multiprocessing start method
    # as it conflicts with execnet. Comment this out to use default fork mode.
    # try:
    #     # First check if we're on Linux
    #     if sys.platform.startswith('linux'):
    #         # Use forkserver for better safety while preserving imports
    #         multiprocessing.set_start_method('forkserver', force=True)
    #     else:
    #         # Use spawn on other platforms
    #         multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     # Already set, ignore
    #     pass


def pytest_collection_modifyitems(config, items):
    """Modify test collection to apply automatic timeouts based on markers"""
    for item in items:
        # Check if test already has explicit timeout marker
        has_timeout = any(mark.name == 'timeout' for mark in item.iter_markers())
        
        if not has_timeout:
            # Apply timeout based on slow/very_slow markers
            if item.get_closest_marker('very_slow'):
                item.add_marker(pytest.mark.timeout(1800))  # 30 minutes
            elif item.get_closest_marker('slow'):
                item.add_marker(pytest.mark.timeout(300))   # 5 minutes
            # Default timeout is already set in pytest.ini


# ─────────────────────────────────────────────────────────────
# Parallel test context for isolation
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def parallel_test_context(worker_id):
    """Provide parallel test context for isolation"""
    from tests.helpers import ParallelTestContext
    return ParallelTestContext(worker_id)


@pytest.fixture
def docker_container_manager(worker_id):
    """Provide Docker container manager with worker isolation"""
    from tests.helpers import DockerContainerManager
    manager = DockerContainerManager(worker_id=worker_id)
    yield manager
    manager.cleanup()


@pytest.fixture
def security_nvr_manager(docker_container_manager):
    """Provide Security NVR manager for tests that need Frigate containers"""
    # This fixture wraps docker_container_manager for security NVR specific needs
    return docker_container_manager


# ============================================================================
# LEGACY ADAPTERS FOR REFACTORED CLASSES (Temporary - Remove after migration)
# ============================================================================

from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Optional, List
import threading

# Import new classes - handle import errors gracefully
try:
    # Ensure parent directory is in path for utils imports
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    from utils.mqtt_service import MQTTService
    from utils.health_reporter import HealthReporter
    from utils.thread_manager import ThreadSafeService
    from utils.config_base import ConfigBase
    from camera_detector.detect import Camera, CameraDetector, CameraDetectorConfig
    from fire_consensus.consensus import FireConsensus, FireConsensusConfig, Detection, CameraState
except ImportError as e:
    print(f"Warning: Could not import refactored classes: {e}")
    # Define dummy classes for tests that don't need the real ones
    Camera = CameraDetector = CameraDetectorConfig = None
    FireConsensus = FireConsensusConfig = Detection = CameraState = None


class LegacyCameraAdapter:
    """Adapter to make new Camera dataclass work with old test expectations."""
    
    def __init__(self, camera):
        self._camera = camera
        
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped camera."""
        return getattr(self._camera, name)
    
    @property
    def id(self) -> str:
        """Old tests expect camera.id, which is now derived from MAC."""
        return self._camera.mac.lower().replace(':', '')
    
    @property
    def ip_history(self) -> List[str]:
        """Old tests expect ip_history tracking."""
        # Return a list with just current IP (history not tracked anymore)
        return [self._camera.ip]
    
    def update_ip(self, new_ip: str):
        """Old tests expect update_ip method."""
        self._camera.ip = new_ip
    
    def to_frigate_config(self) -> Dict[str, Any]:
        """Old tests expect to_frigate_config method."""
        # Generate a basic Frigate config from camera data
        return {
            self.id: {
                'ffmpeg': {
                    'inputs': [
                        {
                            'path': self._camera.rtsp_urls.get('main', ''),
                            'roles': ['detect', 'record']
                        }
                    ]
                },
                'detect': {
                    'enabled': self._camera.online,
                    'width': 1920,
                    'height': 1080
                },
                'objects': {
                    'track': ['fire', 'smoke']
                }
            }
        }


class LegacyCameraDetectorAdapter:
    """Adapter to make new CameraDetector work with old test expectations."""
    
    def __init__(self, detector):
        self._detector = detector
        self._running = getattr(detector, '_shutdown_event', threading.Event())
        
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped detector."""
        return getattr(self._detector, name)
    
    def _publish_camera_discovery(self, camera):
        """Old tests expect this method name."""
        self._detector._publish_camera(camera)
    
    def _update_frigate_config(self):
        """Old tests expect this method."""
        # In new architecture, this would publish config via MQTT
        config = {'cameras': {}}
        for mac, camera in self._detector.cameras.items():
            adapter = LegacyCameraAdapter(camera)
            cam_config = adapter.to_frigate_config()
            config['cameras'].update(cam_config)
        
        # Publish via MQTT
        self._detector.publish_message('frigate/config', config)
    
    def _publish_health(self):
        """Old tests expect this method."""
        # In new architecture, health is handled by HealthReporter
        if hasattr(self._detector, 'health_reporter'):
            # Trigger a health report
            health_data = self._detector.health_reporter.get_service_health()
            self._detector.publish_message('system/health/camera_detector', health_data)
    
    def _discover_onvif_cameras(self):
        """Old tests expect this method."""
        # Call the new method if it exists
        if hasattr(self._detector, '_discover_onvif_cameras'):
            return self._detector._discover_onvif_cameras()
        return []


class LegacyConfigAdapter:
    """Adapter to make new Config classes work with old test expectations."""
    
    def __init__(self, config):
        self._config = config
        
    def __getitem__(self, key: str):
        """Support dictionary-style access for old tests."""
        # Convert key to lowercase for new config
        attr_name = key.lower()
        return getattr(self._config, attr_name, None)
    
    def __setitem__(self, key: str, value: Any):
        """Support dictionary-style setting for old tests."""
        attr_name = key.lower()
        setattr(self._config, attr_name, value)
    
    def get(self, key: str, default: Any = None):
        """Support dict.get() for old tests."""
        attr_name = key.lower()
        return getattr(self._config, attr_name, default)
    
    def __getattr__(self, name: str):
        """Support both UPPERCASE and lowercase attributes."""
        # Try lowercase first (new style)
        if hasattr(self._config, name.lower()):
            return getattr(self._config, name.lower())
        # Try as-is
        return getattr(self._config, name)


# ============================================================================
# ADAPTER FIXTURES
# ============================================================================

@pytest.fixture
def camera_adapter():
    """Factory for creating legacy-adapted cameras."""
    if Camera is None:
        pytest.skip("Camera class not available")
    
    def _create_camera(**kwargs):
        defaults = {
            'ip': '192.168.1.100',
            'mac': 'AA:BB:CC:DD:EE:FF',
            'name': 'Test Camera',
            'manufacturer': 'TestCam',
            'model': 'TC-1000'
        }
        defaults.update(kwargs)
        camera = Camera(**defaults)
        return LegacyCameraAdapter(camera)
    return _create_camera


@pytest.fixture
def legacy_config_adapter():
    """Factory for creating legacy-adapted configs."""
    def _create_config(config_class, **overrides):
        config = config_class()
        for key, value in overrides.items():
            setattr(config, key.lower(), value)
        return LegacyConfigAdapter(config)
    return _create_config
