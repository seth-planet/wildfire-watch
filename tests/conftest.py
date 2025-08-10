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

# Add project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Also add tests directory for test utils
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)


# Import safe logging to prevent I/O on closed file errors
# This is critical for parallel test execution
try:
    from test_utils.safe_logging import (
        disable_problem_loggers, 
        cleanup_test_logging,
        cleanup_all_registered_loggers,
        SafeStreamHandler
    )
    disable_problem_loggers()
except ImportError as e:
    print(f"WARNING: Could not import safe_logging: {e}")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Project root: {project_root}")
    print(f"  sys.path: {sys.path[:3]}")
    # Define dummy functions to prevent further errors
    def disable_problem_loggers():
        pass
    def cleanup_test_logging():
        pass
    def cleanup_all_registered_loggers():
        pass
    SafeStreamHandler = None

# Import enhanced process cleanup for proper test isolation
try:
    from test_utils.enhanced_process_cleanup import get_process_cleaner, cleanup_on_test_failure
    # Configure to not kill test processes
    import test_utils.enhanced_process_cleanup as enhanced_process_cleanup
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
    
    # Build Docker images if needed
    if os.path.exists('/home/seth/wildfire-watch/scripts/build_test_images.sh'):
        print("🔨 Building Docker images for tests...")
        result = subprocess.run(['/home/seth/wildfire-watch/scripts/build_test_images.sh'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to build Docker images: {result.stderr}")
    
    yield
    
    # Cleanup at end of session
    if cleaner:
        print("🧹 Performing final process cleanup...")
        final_results = cleaner.cleanup_all()
        if sum(final_results.values()) > 0:
            print(f"Final cleanup: {sum(final_results.values())} items cleaned")


@pytest.fixture(scope="session", autouse=True)
def docker_cleanup_at_session_start():
    """Clean up any stale Docker containers from previous test runs"""
    import docker
    
    try:
        client = docker.from_env()
        
        print("🧹 Cleaning up stale Docker containers from previous runs...")
        cleaned = 0
        
        # Remove all containers with our test prefix
        for container in client.containers.list(all=True, filters={"name": "wf-"}):
            try:
                print(f"  Removing stale container: {container.name}")
                container.remove(force=True)
                cleaned += 1
            except Exception as e:
                print(f"  Warning: Could not remove container {container.name}: {e}")
        
        if cleaned > 0:
            print(f"✅ Removed {cleaned} stale test containers")
        else:
            print("✅ No stale containers found")
            
    except Exception as e:
        print(f"⚠️  Docker cleanup error: {e}")
        # Don't fail the test session if Docker cleanup fails
    
    yield  # Run tests
    
    # No cleanup needed at end since DockerContainerManager handles it


# ─────────────────────────────────────────────────────────────
# Shared MQTT Test Infrastructure
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def test_mqtt_broker(worker_id):
    """Provide a test MQTT broker using enhanced broker with session reuse"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from test_utils.enhanced_mqtt_broker import TestMQTTBroker
    
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


@pytest.fixture(autouse=True)
def thread_cleanup():
    """Enhanced thread cleanup fixture for all tests.
    
    Automatically tracks threads created during tests and ensures
    they are properly cleaned up to prevent resource leaks.
    """
    import gc
    import threading
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Record initial state
    initial_threads = {t.ident for t in threading.enumerate() if t.is_alive()}
    initial_count = threading.active_count()
    
    yield
    
    # Force garbage collection first
    gc.collect()
    
    # Give threads a moment to finish naturally
    time.sleep(0.1)
    
    # Check for orphaned threads
    current_threads = threading.enumerate()
    orphaned = [t for t in current_threads 
                if t.ident not in initial_threads and t.is_alive()]
    
    if orphaned:
        logger.debug(f"Found {len(orphaned)} orphaned threads after test")
        
        # Try graceful shutdown
        for thread in orphaned:
            # Standard thread stop methods
            if hasattr(thread, 'stop') and callable(thread.stop):
                try:
                    thread.stop()
                except Exception:
                    pass
            
            # Timer threads
            if hasattr(thread, 'cancel') and callable(thread.cancel):
                try:
                    thread.cancel()
                except Exception:
                    pass
            
            # Set common stop flags
            for attr in ['_stop_event', '_shutdown', 'shutdown_flag']:
                if hasattr(thread, attr):
                    flag = getattr(thread, attr)
                    if hasattr(flag, 'set'):
                        flag.set()
                    elif isinstance(flag, bool):
                        setattr(thread, attr, True)
        
        # Wait briefly for cleanup
        time.sleep(0.2)
        
        # Check again
        still_alive = [t for t in orphaned if t.is_alive()]
        if still_alive:
            logger.warning(f"{len(still_alive)} threads still alive after cleanup: "
                         f"{[t.name for t in still_alive]}")
    
    # Final garbage collection
    gc.collect()


@pytest.fixture
def test_mqtt_tls_broker(tmp_path):
    """Provide a test MQTT broker with TLS enabled"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from test_utils.mqtt_tls_test_broker import MQTTTLSTestBroker
    
    broker = MQTTTLSTestBroker(cert_dir=str(tmp_path))
    broker.start()
    
    yield broker
    
    broker.stop()


@pytest.fixture
def mqtt_tls_broker(worker_id):
    """Provide a test MQTT broker with TLS enabled using project certificates"""
    # Import here to avoid circular dependencies
    sys.path.insert(0, os.path.dirname(__file__))
    from test_utils.mqtt_tls_test_broker import MQTTTLSTestBroker
    
    # Use the project's actual certificates
    cert_dir = os.path.join(project_root, 'certs')
    
    # Create broker with worker-specific ports for parallel test safety
    broker = MQTTTLSTestBroker(cert_dir=cert_dir)
    broker.start()
    
    # Wait for TLS port to be ready
    if not broker.wait_for_tls_ready(timeout=30):
        raise RuntimeError("TLS broker failed to start")
    
    yield broker
    
    broker.stop()


# ─────────────────────────────────────────────────────────────
# Session cleanup hooks for parallel test isolation
# ─────────────────────────────────────────────────────────────

def pytest_sessionstart(session):
    """Clean up any leftover containers before starting tests"""
    import subprocess
    
    # Get worker ID for targeted cleanup
    worker_id = getattr(session.config, 'workerinput', {}).get('workerid', 'master')
    
    print(f"Pre-test cleanup for worker {worker_id}: removing any leftover test containers...")
    
    try:
        # Clean up only this worker's containers with both test and worker labels
        result = subprocess.run([
            'docker', 'ps', '-aq', 
            '--filter', 'label=com.wildfire.test=true',
            '--filter', f'label=com.wildfire.worker={worker_id}'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            for container_id in container_ids:
                if container_id.strip():
                    try:
                        subprocess.run(['docker', 'rm', '-f', container_id.strip()], timeout=5)
                        print(f"Removed leftover test container {container_id} for worker {worker_id}")
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
        from test_utils.safe_logging import cleanup_test_logging, cleanup_all_registered_loggers
        # Clean up registered loggers first
        cleanup_all_registered_loggers()
        # Then do general test logging cleanup
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
    from test_utils.enhanced_mqtt_broker import TestMQTTBroker
    
    # Get worker ID for targeted cleanup
    worker_id = None
    if hasattr(session.config, '_worker_id'):
        worker_id = session.config._worker_id
    
    # Clean up worker-specific broker
    TestMQTTBroker.cleanup_session(worker_id=worker_id)
    
    # Additional aggressive cleanup of stray processes
    try:
        # Kill any remaining mosquitto test processes for this worker only
        if worker_id:
            pattern = f'mqtt_test_{worker_id}_'
            result = subprocess.run(['pgrep', '-f', pattern], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        try:
                            subprocess.run(['kill', '-KILL', pid.strip()], timeout=1)
                            print(f"Force killed stray mosquitto process {pid} for worker {worker_id}")
                        except:
                            pass
        
        # Clean up only this worker's containers with both test and worker labels
        if worker_id:
            result = subprocess.run([
                'docker', 'ps', '-aq', 
                '--filter', 'label=com.wildfire.test=true',
                '--filter', f'label=com.wildfire.worker={worker_id}'
            ], capture_output=True, text=True)
        else:
            # Fallback for master/single worker - only clean containers without worker label
            result = subprocess.run([
                'docker', 'ps', '-aq', 
                '--filter', 'label=com.wildfire.test=true',
                '--filter', 'label=com.wildfire.worker=master'
            ], capture_output=True, text=True)
            
        if result.returncode == 0 and result.stdout.strip():
            container_ids = result.stdout.strip().split('\n')
            for container_id in container_ids:
                if container_id.strip():
                    try:
                        subprocess.run(['docker', 'rm', '-f', container_id.strip()], timeout=5)
                        print(f"Force removed stray test container {container_id} for worker {worker_id or 'master'}")
                    except:
                        pass
                        
    except Exception as e:
        print(f"Warning: Error in aggressive cleanup: {e}")
    
    # Log cleanup status
    if worker_id:
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
    from test_utils.helpers import ParallelTestContext
    return ParallelTestContext(worker_id)


@pytest.fixture
def docker_container_manager(worker_id):
    """Provide Docker container manager with worker isolation"""
    from test_utils.helpers import DockerContainerManager
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
    parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Debug output
    debug_env = os.environ.get('DEBUG_IMPORTS', '')
    if debug_env:
        print(f"DEBUG: tests/conftest.py - parent_dir: {parent_dir}")
        print(f"DEBUG: tests/conftest.py - sys.path[0]: {sys.path[0]}")
        print(f"DEBUG: tests/conftest.py - utils exists: {os.path.exists(os.path.join(parent_dir, 'utils'))}")
        
    from test_utils.mqtt_service import MQTTService
    from test_utils.health_reporter import HealthReporter
    from test_utils.thread_manager import ThreadSafeService
    from test_utils.config_base import ConfigBase
    from camera_detector.detect import Camera, CameraDetector, CameraDetectorConfig
    from fire_consensus.consensus import FireConsensus, FireConsensusConfig, Detection, CameraState
except ImportError as e:
    # Skip warning for now - the trigger test will handle its own imports
    pass  # print(f"Warning: Could not import refactored classes: {e}")
    # Define dummy classes for tests that don't need the real ones
    Camera = CameraDetector = CameraDetectorConfig = None
    FireConsensus = FireConsensusConfig = Detection = CameraState = None
    MQTTService = HealthReporter = ThreadSafeService = ConfigBase = None


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


@pytest.fixture
def pump_controller_factory(test_mqtt_broker, mqtt_topic_factory, monkeypatch):
    """Factory fixture for creating PumpController instances with proper test configuration.
    
    This fixture creates PumpController instances with dependency injection,
    avoiding the legacy CONFIG dictionary approach.
    
    Usage:
        controller = pump_controller_factory()
        # or with custom config:
        controller = pump_controller_factory(max_engine_runtime=60)
    """
    from tests.test_utils.helpers import create_pump_controller_with_config
    
    def _create_controller(**kwargs):
        # Get connection parameters from test broker
        conn_params = test_mqtt_broker.get_connection_params()
        
        # Get unique topic prefix for test isolation
        full_topic = mqtt_topic_factory("dummy")
        topic_prefix = full_topic.rsplit('/', 1)[0]
        
        # Set environment variables for the configuration
        test_env = {
            'MQTT_BROKER': conn_params['host'],
            'MQTT_PORT': str(conn_params['port']),
            'MQTT_TLS': 'false',
            'MQTT_TOPIC_PREFIX': topic_prefix,
        }
        
        # Add any custom configuration from kwargs
        for key, value in kwargs.items():
            env_key = key.upper()
            test_env[env_key] = str(value)
        
        # Apply environment variables
        for key, value in test_env.items():
            monkeypatch.setenv(key, value)
        
        # Create and return the controller
        return create_pump_controller_with_config(
            test_env=test_env,
            mqtt_conn_params=conn_params,
            topic_prefix=topic_prefix,
            auto_connect=kwargs.get('auto_connect', False)
        )
    
    return _create_controller


# ==============================
# GPIO Test Fixtures
# ==============================

@pytest.fixture
def gpio_test_setup():
    """Setup GPIO for testing - uses real GPIO module or simulation.
    
    BEST PRACTICE: This is NOT a mock - it uses the real GPIO module when available
    or the built-in simulation mode when running on non-Pi hardware.
    """
    import threading
    import os
    # Import the current GPIO from trigger module to ensure we use the right instance
    from gpio_trigger.trigger import GPIO
    
    # GPIO is now always available (real or simulated)
    # Ensure GPIO lock exists (for compatibility with older code)
    if not hasattr(GPIO, '_lock'):
        GPIO._lock = threading.RLock()
    
    # Set BCM mode
    GPIO.setmode(GPIO.BCM)
    
    with GPIO._lock:
        # Clear any existing state
        if hasattr(GPIO, '_state'):
            GPIO._state.clear()
        
        # Set all pins to LOW initially using default pin values
        pin_defaults = {
            'MAIN_VALVE_PIN': 18,
            'IGN_START_PIN': 23,
            'IGN_ON_PIN': 24,
            'IGN_OFF_PIN': 25,
            'REFILL_VALVE_PIN': 22,
            'PRIMING_VALVE_PIN': 26,
            'RPM_REDUCE_PIN': 27,
        }
        
        for pin_name, default_pin in pin_defaults.items():
            # Get pin from environment or use default
            pin = int(os.getenv(pin_name, str(default_pin)))
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    
    yield GPIO
    
    # Cleanup after test
    try:
        GPIO.cleanup()
    except Exception:
        pass  # Ignore cleanup errors
    if hasattr(GPIO, '_state'):
        with GPIO._lock:
            GPIO._state.clear()


def wait_for_state(controller, state, timeout=5):
    """Wait for controller to reach specific state"""
    import time
    from gpio_trigger.trigger import PumpState
    
    start = time.time()
    while time.time() - start < timeout:
        if controller._state == state:
            return True
        # Log ERROR state entries for analysis
        if controller._state == PumpState.ERROR:
            import inspect
            import logging
            frame = inspect.currentframe()
            caller = frame.f_back.f_code.co_name if frame.f_back else "unknown"
            logging.getLogger(__name__).warning(f"ERROR state reached in test: {caller}, waiting for: {state.name}")
        time.sleep(0.01)
    return False


def wait_for_any_state(controller, states, timeout=5):
    """Wait for controller to reach any of the specified states"""
    import time
    from gpio_trigger.trigger import PumpState
    
    start = time.time()
    while time.time() - start < timeout:
        if controller._state in states:
            return controller._state
        # Log ERROR state entries for analysis
        if controller._state == PumpState.ERROR:
            import inspect
            import logging
            frame = inspect.currentframe()
            caller = frame.f_back.f_code.co_name if frame.f_back else "unknown"
            logging.getLogger(__name__).warning(f"ERROR state reached in test: {caller}, waiting for any of: {[s.name for s in states]}")
        time.sleep(0.01)
    return None
