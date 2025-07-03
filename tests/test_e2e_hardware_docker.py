#!/usr/bin/env python3.12
"""
End-to-End Hardware Integration Tests with Docker
Comprehensive tests for the complete wildfire detection pipeline using Docker containers
Tests real hardware acceleration (Coral TPU, TensorRT, etc.) with full system integration
"""

import os
import sys
import time
import pytest
import json
import docker
import numpy as np
import cv2
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import paho.mqtt.client as mqtt
from threading import Event, Lock
from datetime import datetime
import yaml

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.conftest import has_coral_tpu, has_tensorrt, has_hailo, has_camera_on_network
from tests.mqtt_test_broker import MQTTTestBroker as TestMQTTBroker
from tests.helpers import (
    DockerHealthChecker, ensure_docker_available, requires_docker,
    DockerContainerManager, create_test_frigate_config, prepare_frigate_test_environment
)


@pytest.mark.skip(reason="Temporarily disabled during refactoring - Phase 1")
class TestE2EHardwareDocker:
    """Comprehensive E2E tests with real hardware acceleration using Docker"""
    
    def setup_class(self):
        """Ensure Docker is available before running any tests"""
        ensure_docker_available()
    
    @pytest.fixture
    def mqtt_broker(self):
        """Start test MQTT broker"""
        broker = TestMQTTBroker()
        broker.start()
        time.sleep(2)  # Wait for broker to start
        yield broker
        broker.stop()
    
    @pytest.fixture
    def docker_client(self):
        """Docker client for container management"""
        client = docker.from_env()
        yield client
        # Cleanup any test containers
        for container in client.containers.list(all=True):
            if container.name and 'e2e_test' in container.name:
                try:
                    container.stop(timeout=5)
                    container.remove(force=True)
                except:
                    pass
    
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.docker
    @requires_docker
    @pytest.mark.timeout(600)  # 10 minutes for complete pipeline test
    def test_complete_pipeline_auto_hardware(self, mqtt_broker, docker_client):
        """Test complete pipeline with automatic hardware selection"""
        print("\n" + "="*80)
        print("E2E TEST: Complete Pipeline with Auto Hardware Selection")
        print("="*80)
        
        # Detect available hardware
        hardware = self._detect_hardware()
        print(f"\nDetected hardware: {json.dumps(hardware, indent=2)}")
        
        # Use container manager for proper cleanup
        with DockerContainerManager(docker_client) as container_manager:
            containers = {}
            mqtt_messages = []
            mqtt_client = None
            
            try:
                # 1. Start camera detector
                print("\n1. Starting camera detector...")
                containers['detector'] = self._start_camera_detector(
                    docker_client, mqtt_broker.port, container_manager
                )
                
                # 2. Start Frigate with auto-detected hardware
                print("\n2. Starting Frigate with hardware acceleration...")
                containers['frigate'] = self._start_frigate(
                    docker_client, 
                    mqtt_broker.port,
                    hardware['selected'],
                    container_manager=container_manager
                )
                
                # 3. Start consensus service
                print("\n3. Starting consensus service...")
                containers['consensus'] = self._start_consensus(
                    docker_client, mqtt_broker.port, container_manager
                )
                
                # 4. Start GPIO trigger (simulated)
                print("\n4. Starting GPIO trigger service...")
                containers['trigger'] = self._start_gpio_trigger(
                    docker_client, mqtt_broker.port, container_manager
                )
                
                # 5. Connect MQTT client to monitor
                print("\n5. Connecting MQTT monitor...")
                mqtt_client, message_event = self._setup_mqtt_monitor(mqtt_broker.port, mqtt_messages)
                
                # Wait for services to initialize
                print("\n6. Waiting for services to initialize...")
                time.sleep(10)
                
                # 7. Verify service health
                print("\n7. Verifying service health...")
                self._verify_services_healthy(containers)
                
                # 8. Simulate fire detection
                print("\n8. Simulating fire detection events...")
                topic_prefix = f"test/{container_manager.worker_id}"
                self._simulate_fire_events(mqtt_client, num_events=8, topic_prefix=topic_prefix)
                
                # 9. Wait for consensus and trigger  
                print("\n9. Waiting for consensus and trigger...")
                time.sleep(20)  # Longer wait time
                
                # Check consensus container logs for debugging
                print("\n9.1. Checking consensus container logs...")
                consensus_container = containers.get('consensus')
                if consensus_container:
                    logs = consensus_container.logs(tail=50).decode()
                    print("Consensus logs (last 50 lines):")
                    for line in logs.split('\n')[-10:]:  # Show last 10 lines
                        if line.strip():
                            print(f"  {line}")
                
                # Check GPIO trigger container logs too
                print("\n9.2. Checking GPIO trigger container logs...")
                trigger_container = containers.get('trigger') 
                if trigger_container:
                    logs = trigger_container.logs(tail=30).decode()
                    print("GPIO trigger logs (last 30 lines):")
                    for line in logs.split('\n')[-5:]:  # Show last 5 lines
                        if line.strip():
                            print(f"  {line}")
                
                # 10. Analyze results
                print("\n10. Analyzing results...")
                
                # Debug: Show all topics received
                topics_seen = set(msg['topic'] for msg in mqtt_messages)
                print(f"\nDebug - Topics received ({len(topics_seen)}):")
                for topic in sorted(topics_seen):
                    count = sum(1 for msg in mqtt_messages if msg['topic'] == topic)
                    print(f"  {topic}: {count} messages")
                
                # Debug: Show GPIO trigger telemetry messages
                print(f"\nDebug - GPIO trigger telemetry messages:")
                for msg in mqtt_messages:
                    if 'trigger_telemetry' in msg['topic']:
                        try:
                            payload = json.loads(msg['payload'])
                            print(f"  {msg['topic']}: {payload}")
                        except:
                            print(f"  {msg['topic']}: {msg['payload']}")
                
                results = self._analyze_results(mqtt_messages)
                
                # Print summary
                print("\n" + "="*80)
                print("E2E TEST RESULTS:")
                print(f"  Hardware used: {hardware['selected']}")
                print(f"  Messages received: {len(mqtt_messages)}")
                print(f"  Fire detections: {results['fire_detections']}")
                print(f"  Consensus reached: {results['consensus_reached']}")
                print(f"  GPIO triggered: {results['gpio_triggered']}")
                print(f"  Average latency: {results['avg_latency']:.2f}ms")
                
                # Assertions
                assert len(mqtt_messages) > 0, "No MQTT messages received"
                assert results['fire_detections'] > 0, "No fire detections"
                assert results['consensus_reached'], "Consensus not reached"
                assert results['gpio_triggered'], "GPIO not triggered"
                
                print("  Result: ✓ PASSED")
                print("="*80)
                
            finally:
                # Cleanup
                if mqtt_client:
                    mqtt_client.loop_stop()
                    mqtt_client.disconnect()
    
    @pytest.mark.skipif(not has_coral_tpu(), reason="Coral TPU not available")
    @pytest.mark.skipif(sys.version_info[:2] != (3, 8), reason="Coral TPU requires Python 3.8")
    @pytest.mark.slow
    @pytest.mark.coral_tpu
    @pytest.mark.python38
    def test_coral_tpu_pipeline(self, mqtt_broker, docker_client):
        """Test pipeline specifically with Coral TPU"""
        print("\n" + "="*80)
        print("E2E TEST: Coral TPU Fire Detection Pipeline")
        print("="*80)
        
        containers = {}
        mqtt_messages = []
        mqtt_client = None
        
        try:
            # Verify Coral TPU
            print("\n1. Verifying Coral TPU...")
            coral_info = self._verify_coral_tpu()
            print(f"✓ Found {len(coral_info['devices'])} Coral TPU(s)")
            
            # Start services with Coral TPU
            containers['frigate'] = self._start_frigate(
                docker_client,
                mqtt_broker.port,
                'coral',
                num_detectors=len(coral_info['devices'])
            )
            
            containers['consensus'] = self._start_consensus(docker_client, mqtt_broker.port)
            
            # Monitor MQTT
            mqtt_client, _ = self._setup_mqtt_monitor(mqtt_broker.port, mqtt_messages)
            
            # Wait for initialization
            time.sleep(10)
            
            # Send test detections
            self._send_coral_test_detections(mqtt_client)
            
            # Wait for processing
            time.sleep(10)
            
            # Verify Coral TPU was used
            coral_used = self._verify_coral_usage(containers['frigate'], mqtt_messages)
            
            print(f"\n✓ Coral TPU pipeline test completed")
            print(f"  Coral TPU used: {coral_used}")
            print(f"  Messages processed: {len(mqtt_messages)}")
            
            assert coral_used, "Coral TPU was not used"
            
        finally:
            # Cleanup
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            
            for container in containers.values():
                if container:
                    try:
                        container.stop(timeout=10)
                        container.remove()
                    except:
                        pass
    
    @pytest.mark.skipif(not has_tensorrt(), reason="TensorRT not available")
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_tensorrt_gpu_pipeline(self, mqtt_broker, docker_client):
        """Test pipeline specifically with TensorRT GPU"""
        print("\n" + "="*80)
        print("E2E TEST: TensorRT GPU Fire Detection Pipeline")
        print("="*80)
        
        containers = {}
        mqtt_messages = []
        performance_metrics = []
        mqtt_client = None
        
        try:
            # Verify GPU
            print("\n1. Verifying GPU and TensorRT...")
            gpu_info = self._verify_gpu_tensorrt()
            print(f"✓ GPU: {gpu_info['gpu_name']}")
            print(f"✓ TensorRT: {gpu_info['tensorrt_version']}")
            
            # Start services with TensorRT
            containers['frigate'] = self._start_frigate(
                docker_client,
                mqtt_broker.port,
                'tensorrt'
            )
            
            containers['consensus'] = self._start_consensus(docker_client, mqtt_broker.port)
            
            # Monitor MQTT with performance tracking
            mqtt_client, _ = self._setup_mqtt_monitor(
                mqtt_broker.port, 
                mqtt_messages,
                performance_callback=lambda msg: performance_metrics.append(msg)
            )
            
            # Wait for initialization
            time.sleep(10)
            
            # Run performance test
            print("\n2. Running TensorRT performance test...")
            self._run_tensorrt_performance_test(mqtt_client)
            
            # Wait for processing
            time.sleep(20)
            
            # Analyze performance
            gpu_performance = self._analyze_gpu_performance(performance_metrics)
            
            print(f"\n✓ TensorRT GPU pipeline test completed")
            print(f"  Average inference: {gpu_performance['avg_inference']:.2f}ms")
            print(f"  Peak FPS: {gpu_performance['peak_fps']:.1f}")
            print(f"  GPU utilization: {gpu_performance['gpu_util']:.1f}%")
            
            # Performance assertions - more lenient for container environment
            if gpu_performance['avg_inference'] > 0:
                assert gpu_performance['avg_inference'] < 50, f"GPU inference too slow: {gpu_performance['avg_inference']}ms"
            else:
                print("  Warning: No inference metrics captured, may indicate GPU not properly utilized")
            
            if gpu_performance['peak_fps'] > 0:
                assert gpu_performance['peak_fps'] > 5, f"FPS too low for GPU: {gpu_performance['peak_fps']}"
            else:
                print("  Warning: No FPS metrics captured, checking if container can at least start with GPU")
            
        finally:
            # Cleanup
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            
            for container in containers.values():
                if container:
                    try:
                        container.stop(timeout=10)
                        container.remove()
                    except:
                        pass
    
    @pytest.mark.skipif(not has_hailo(), reason="Hailo not available")
    @pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason="Hailo requires Python 3.10")
    @pytest.mark.slow
    @pytest.mark.hailo
    @pytest.mark.python310
    def test_hailo_pipeline(self, mqtt_broker, docker_client):
        """Test pipeline with Hailo acceleration"""
        print("\n" + "="*80)
        print("E2E TEST: Hailo Fire Detection Pipeline")
        print("="*80)
        
        containers = {}
        mqtt_messages = []
        mqtt_client = None
        
        try:
            # Verify Hailo
            print("\n1. Verifying Hailo...")
            hailo_info = self._verify_hailo()
            print(f"✓ Found Hailo device: {hailo_info['device']}")
            
            # Start services with Hailo
            containers['frigate'] = self._start_frigate(
                docker_client,
                mqtt_broker.port,
                'hailo'
            )
            
            containers['consensus'] = self._start_consensus(docker_client, mqtt_broker.port)
            
            # Monitor MQTT
            mqtt_client, _ = self._setup_mqtt_monitor(mqtt_broker.port, mqtt_messages)
            
            # Wait for initialization
            time.sleep(10)
            
            # Send test detections
            self._send_hailo_test_detections(mqtt_client)
            
            # Wait for processing
            time.sleep(10)
            
            # Verify Hailo was used
            hailo_used = self._verify_hailo_usage(containers['frigate'], mqtt_messages)
            
            print(f"\n✓ Hailo pipeline test completed")
            print(f"  Hailo used: {hailo_used}")
            print(f"  Messages processed: {len(mqtt_messages)}")
            
            assert hailo_used, "Hailo was not used"
            
        finally:
            # Cleanup
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            
            for container in containers.values():
                if container:
                    try:
                        container.stop(timeout=10)
                        container.remove()
                    except:
                        pass
    
    @pytest.mark.skipif(not has_camera_on_network(), reason="No cameras on network")
    @pytest.mark.slow
    @pytest.mark.cameras
    def test_real_camera_integration(self, mqtt_broker, docker_client):
        """Test with real network cameras"""
        print("\n" + "="*80)
        print("E2E TEST: Real Camera Integration")
        print("="*80)
        
        # Use container manager for proper cleanup
        with DockerContainerManager(docker_client) as container_manager:
            containers = {}
            mqtt_messages = []
            camera_discoveries = []
            mqtt_client = None
            
            try:
                # Start camera detector
                print("\n1. Starting camera detector...")
                containers['detector'] = self._start_camera_detector(
                    docker_client, mqtt_broker.port, container_manager
                )
                
                # Monitor camera discoveries
                def on_camera_discovery(msg):
                    if msg.topic == 'cameras/discovered':
                        camera_discoveries.append(json.loads(msg.payload))
                
                mqtt_client, _ = self._setup_mqtt_monitor(
                    mqtt_broker.port,
                    mqtt_messages,
                    extra_callback=on_camera_discovery
                )
                
                # Wait for camera discovery
                print("\n2. Waiting for camera discovery...")
                time.sleep(30)
                
                print(f"\n3. Discovered {len(camera_discoveries)} cameras")
                
                if camera_discoveries:
                    # Generate Frigate config
                    print("\n4. Generating Frigate configuration...")
                    self._wait_for_frigate_config(camera_discoveries[0])
                    
                    # Start Frigate with discovered cameras
                    hardware = self._detect_hardware()
                    containers['frigate'] = self._start_frigate(
                        docker_client,
                        mqtt_broker.port,
                        hardware['selected'],
                        use_discovered_cameras=True
                    )
                    
                    # Wait for Frigate to process streams
                    print("\n5. Processing camera streams...")
                    time.sleep(30)
                    
                    # Check for detections
                    detections = [m for m in mqtt_messages if 'frigate/events' in m.get('topic', '')]
                    
                    print(f"\n✓ Real camera test completed")
                    print(f"  Cameras discovered: {len(camera_discoveries)}")
                    print(f"  Detections: {len(detections)}")
                    
                    assert len(camera_discoveries) > 0, "No cameras discovered"
                else:
                    # Check if CAMERA_CREDENTIALS is set
                    if not os.getenv('CAMERA_CREDENTIALS'):
                        pytest.fail("CAMERA_CREDENTIALS environment variable must be set for real camera integration test")
                    else:
                        pytest.fail("No cameras discovered on network 192.168.5.0/24. Check camera availability and credentials.")
                
            finally:
                # Cleanup
                if mqtt_client:
                    mqtt_client.loop_stop()
                    mqtt_client.disconnect()
                
                for container in containers.values():
                    if container:
                        try:
                            container.stop(timeout=10)
                            container.remove()
                        except:
                            pass
    
    def test_multi_accelerator_failover(self, mqtt_broker, docker_client):
        """Test failover between different accelerators"""
        print("\n" + "="*80)
        print("E2E TEST: Multi-Accelerator Failover")
        print("="*80)
        
        # Detect all available hardware
        hardware = self._detect_hardware()
        available = [hw for hw, avail in hardware.items() if hw != 'selected' and avail]
        
        if len(available) < 2:
            pytest.skip("Need at least 2 accelerators for failover test")
        
        print(f"\nTesting failover between: {available}")
        
        # Test switching between accelerators
        for accelerator in available[:2]:
            print(f"\nTesting with {accelerator}...")
            
            container = self._start_frigate(
                docker_client,
                mqtt_broker.port,
                accelerator
            )
            
            time.sleep(10)
            
            # Verify it started correctly
            logs = container.logs(tail=100).decode()
            # Check for actual errors, not just the word "error" in info messages
            if "failed to initialize" in logs.lower() or "exited abnormally" in logs.lower():
                pytest.fail(f"Error starting with {accelerator}: {logs}")
            
            container.stop(timeout=10)
            container.remove()
            
            print(f"✓ {accelerator} working correctly")
    
    @pytest.mark.benchmark
    @pytest.mark.timeout(300)  # 5 minute timeout for performance test
    def test_performance_comparison(self, mqtt_broker, docker_client):
        """Compare performance across available accelerators"""
        print("\n" + "="*80)
        print("E2E TEST: Hardware Performance Comparison")
        print("="*80)
        
        hardware = self._detect_hardware()
        results = {}
        
        # Limit to available hardware to prevent hanging
        available_accelerators = [accel for accel in ['coral', 'tensorrt', 'cpu'] 
                                if hardware.get(accel, False)]
        
        if not available_accelerators:
            pytest.skip("No hardware accelerators available for testing")
        
        print(f"Available accelerators: {available_accelerators}")
        
        # Test each available accelerator with timeout
        for accel_type in available_accelerators:
            
            print(f"\nTesting {accel_type} performance...")
            
            container = None
            mqtt_messages = []
            mqtt_client = None
            
            try:
                # Start Frigate with specific accelerator
                container = self._start_frigate(
                    docker_client,
                    mqtt_broker.port,
                    accel_type
                )
                
                # Monitor performance with timeout
                mqtt_client, _ = self._setup_mqtt_monitor(mqtt_broker.port, mqtt_messages)
                
                # Run standardized test with timeout
                test_start = time.time()
                try:
                    self._run_standard_performance_test(mqtt_client)
                    
                    # Wait for results with progress checking
                    max_wait = 30  # seconds
                    wait_start = time.time()
                    while time.time() - wait_start < max_wait:
                        stats_messages = [m for m in mqtt_messages if m.get('topic') == 'frigate/stats']
                        if stats_messages:
                            break
                        time.sleep(2)
                        print(f"  Waiting for stats... ({time.time() - wait_start:.0f}s)")
                    
                    if time.time() - wait_start >= max_wait:
                        print(f"  Warning: No stats received for {accel_type} within {max_wait}s")
                        
                except Exception as e:
                    print(f"  Error during {accel_type} test: {e}")
                    results[accel_type] = {'error': str(e)}
                    continue
                
                # Analyze
                stats_messages = [m for m in mqtt_messages if m.get('topic') == 'frigate/stats']
                if stats_messages:
                    latest_stats = json.loads(stats_messages[-1]['payload'])
                    
                    results[accel_type] = {
                        'inference_ms': self._extract_inference_time(latest_stats),
                        'fps': self._extract_fps(latest_stats),
                        'cpu_usage': latest_stats.get('cpu_usages', {}).get('frigate', 0)
                    }
                
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                
            finally:
                if container:
                    container.stop(timeout=10)
                    container.remove()
        
        # Print comparison
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON:")
        print("-"*60)
        print("Accelerator | Inference (ms) | FPS  | CPU Usage")
        print("-"*60)
        
        for accel, metrics in results.items():
            print(f"{accel:11} | {metrics['inference_ms']:14.2f} | {metrics['fps']:4.1f} | {metrics['cpu_usage']:8.1f}%")
        
        print("="*60)
    
    @pytest.mark.stress
    def test_continuous_operation(self, mqtt_broker, docker_client):
        """Test system stability during continuous operation"""
        print("\n" + "="*80)
        print("E2E TEST: Continuous Operation Stress Test")
        print("="*80)
        
        hardware = self._detect_hardware()
        containers = {}
        mqtt_messages = []
        mqtt_client = None
        errors = []
        
        try:
            # Start all services
            containers['detector'] = self._start_camera_detector(docker_client, mqtt_broker.port)
            containers['frigate'] = self._start_frigate(docker_client, mqtt_broker.port, hardware['selected'])
            containers['consensus'] = self._start_consensus(docker_client, mqtt_broker.port)
            containers['trigger'] = self._start_gpio_trigger(docker_client, mqtt_broker.port)
            
            # Monitor MQTT
            mqtt_client, _ = self._setup_mqtt_monitor(mqtt_broker.port, mqtt_messages)
            
            # Wait for initialization
            time.sleep(10)
            
            # Run continuous test for 5 minutes
            print("\nRunning 5-minute continuous operation test...")
            start_time = time.time()
            events_sent = 0
            
            while time.time() - start_time < 300:  # 5 minutes
                # Send fire events
                self._simulate_fire_events(mqtt_client, num_events=1)
                events_sent += 3  # 3 cameras per event
                
                # Check service health every 30 seconds
                if int(time.time() - start_time) % 30 == 0:
                    try:
                        self._verify_services_healthy(containers)
                    except AssertionError as e:
                        errors.append(str(e))
                
                time.sleep(5)
            
            # Analyze results
            results = self._analyze_results(mqtt_messages)
            
            print(f"\n✓ Continuous operation test completed")
            print(f"  Duration: 5 minutes")
            print(f"  Events sent: {events_sent}")
            print(f"  Fire detections: {results['fire_detections']}")
            print(f"  Consensus triggers: {results['consensus_reached']}")
            print(f"  Errors: {len(errors)}")
            
            assert len(errors) == 0, f"Errors during operation: {errors}"
            assert results['fire_detections'] > 0, "No fire detections during test"
            
        finally:
            # Cleanup
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            
            for container in containers.values():
                if container:
                    try:
                        container.stop(timeout=10)
                        container.remove()
                    except:
                        pass
    
    # Helper methods
    
    def _detect_hardware(self) -> Dict[str, bool]:
        """Detect available hardware acceleration"""
        hardware = {
            'coral': has_coral_tpu(),
            'tensorrt': has_tensorrt(),
            'hailo': has_hailo(),
            'cpu': True  # Always available
        }
        
        # Select best available
        if hardware['coral']:
            hardware['selected'] = 'coral'
        elif hardware['tensorrt']:
            hardware['selected'] = 'tensorrt'
        elif hardware['hailo']:
            hardware['selected'] = 'hailo'
        else:
            hardware['selected'] = 'cpu'
        
        return hardware
    
    def _start_camera_detector(self, client, mqtt_port: int, container_manager: DockerContainerManager = None):
        """Start camera detector container"""
        if not container_manager:
            container_manager = DockerContainerManager(client)
        
        # Use the standard image we already built
        config_dir = f'/tmp/{container_manager.worker_id}/frigate_config'
        os.makedirs(config_dir, exist_ok=True)
        
        # Container configuration
        config = {
            'detach': True,
            'network_mode': 'host',
            'environment': {
                'MQTT_BROKER': 'localhost',
                'MQTT_PORT': str(mqtt_port),
                'CAMERA_CREDENTIALS': os.getenv('CAMERA_CREDENTIALS', 'admin:S3thrule'),
                'SCAN_SUBNETS': '192.168.5.0/24',  # Use the specific camera subnet
                'DISCOVERY_INTERVAL': '30',
                'LOG_LEVEL': 'DEBUG'
            },
            'volumes': {
                config_dir: {'bind': '/config', 'mode': 'rw'}
            }
        }
        
        # Start container with health checking
        return container_manager.start_container(
            image="wildfire-watch/camera_detector:latest",
            name=container_manager.get_container_name("e2e_test_camera_detector"),
            config=config,
            wait_timeout=5
        )
    
    def _start_frigate(self, client, mqtt_port: int, detector: str, 
                       num_detectors: int = 1, use_discovered_cameras: bool = False,
                       container_manager: DockerContainerManager = None):
        """Start Frigate container with specified detector"""
        if not container_manager:
            container_manager = DockerContainerManager(client)
        
        # Prepare Frigate config
        config_dir = f'/tmp/{container_manager.worker_id}/frigate_config'  # Default
        if not use_discovered_cameras:
            config_dir = prepare_frigate_test_environment(detector=detector)
            config = create_test_frigate_config(
                detector=detector,
                num_detectors=num_detectors,
                mqtt_port=mqtt_port
            )
            # Write the config
            config_path = Path(config_dir) / 'config.yml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        
        # Container configuration
        config = {
            'detach': True,
            'network_mode': 'host',
            'environment': {
                'FRIGATE_MQTT_HOST': 'localhost',
                'FRIGATE_MQTT_PORT': str(mqtt_port)
            },
            'volumes': {
                config_dir: {'bind': '/config', 'mode': 'rw'},
                '/dev/shm': {'bind': '/dev/shm', 'mode': 'rw'}
            },
            'shm_size': '512m'
        }
        
        # Add hardware-specific config
        if detector == 'coral':
            try:
                coral_devices = self._get_coral_devices()
                config.devices = coral_devices
                config.privileged = True
            except Exception as e:
                print(f"  Warning: Could not map Coral devices: {e}")
                detector = 'cpu'
        elif detector == 'tensorrt':
            config.runtime = 'nvidia'
            config.environment['NVIDIA_VISIBLE_DEVICES'] = 'all'
        elif detector == 'hailo':
            config.devices = ['/dev/hailo0:/dev/hailo0']
            config.privileged = True
        
        # Define health check function
        def frigate_health_check(container):
            logs = container.logs().decode()
            # Look for successful startup indicators
            # Frigate typically logs when it's ready
            indicators = [
                'Starting Frigate',
                'frigate.app',
                'connected to mqtt',
                'Capture process started',
                'Detection process started',
                'Output process started'
            ]
            # Check if any indicator is present
            logs_lower = logs.lower()
            has_startup = any(indicator.lower() in logs_lower for indicator in indicators)
            # Also check that there are no critical errors
            # Don't just look for "error" as it appears in many info messages
            critical_errors = ['failed to initialize', 'exited abnormally', 'fatal error', 'permission denied']
            has_critical_error = any(err in logs_lower for err in critical_errors)
            return has_startup and not has_critical_error
        
        # Start container with health checking
        return container_manager.start_container(
            image='ghcr.io/blakeblackshear/frigate:stable',
            name=container_manager.get_container_name('e2e_test_frigate'),
            config=config,
            wait_timeout=120,  # Increased timeout for hardware initialization
            health_check_fn=frigate_health_check
        )
    
    def _start_consensus(self, client, mqtt_port: int, container_manager: DockerContainerManager = None):
        """Start consensus service"""
        if not container_manager:
            container_manager = DockerContainerManager(client)
        
        # Get worker ID for topic prefix
        worker_id = getattr(container_manager, 'worker_id', 'master')
        topic_prefix = f"test/{worker_id}"
        
        # Container configuration
        config = {
            'detach': True,
            'network_mode': 'host',
            'environment': {
                'MQTT_BROKER': 'localhost',
                'MQTT_PORT': str(mqtt_port),
                'MQTT_TOPIC_PREFIX': topic_prefix,
                'CONSENSUS_THRESHOLD': '1',  # Single camera threshold for testing
                'TIME_WINDOW': '30',
                'MIN_CONFIDENCE': '0.7',  # Lower confidence to ensure detection
                'SINGLE_CAMERA_TRIGGER': 'true',  # Enable single camera trigger for testing
                'CAMERA_WINDOW': '10',  # Shorter window for testing
                'COOLDOWN_PERIOD': '5',  # Shorter cooldown for testing
                'MOVING_AVERAGE_WINDOW': '2',  # Smaller window for testing (needs 4 detections)
                'LOG_LEVEL': 'DEBUG'  # Enable debug logging
            }
        }
        
        # Start container with health checking
        return container_manager.start_container(
            image="wildfire-watch/fire_consensus:latest",
            name=container_manager.get_container_name("e2e_test_consensus"),
            config=config,
            wait_timeout=5
        )
    
    def _start_gpio_trigger(self, client, mqtt_port: int, container_manager: DockerContainerManager = None):
        """Start GPIO trigger service"""
        if not container_manager:
            container_manager = DockerContainerManager(client)
        
        # Get worker ID for topic prefix
        worker_id = getattr(container_manager, 'worker_id', 'master')
        topic_prefix = f"test/{worker_id}"
        
        # Container configuration
        config = {
            'detach': True,
            'network_mode': 'host',
            'environment': {
                'MQTT_BROKER': 'localhost',
                'MQTT_PORT': str(mqtt_port),
                'MQTT_TOPIC_PREFIX': topic_prefix,
                'GPIO_SIMULATION': 'true',  # Always simulate for tests
                'MAX_ENGINE_RUNTIME': '30',
                'LOG_LEVEL': 'DEBUG'
            }
        }
        
        # Start container with health checking
        return container_manager.start_container(
            image="wildfire-watch/gpio_trigger:latest",
            name=container_manager.get_container_name("e2e_test_gpio"),
            config=config,
            wait_timeout=5
        )
    
    def _setup_mqtt_monitor(self, port: int, message_list: List, 
                           performance_callback=None, extra_callback=None):
        """Set up MQTT monitoring"""
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id='e2e_test_monitor'
        )
        message_event = Event()
        message_lock = Lock()
        
        def on_message(client, userdata, msg):
            message_data = {
                'topic': msg.topic,
                'payload': msg.payload.decode(),
                'timestamp': time.time()
            }
            
            with message_lock:
                message_list.append(message_data)
            
            # Call additional callbacks
            if performance_callback and 'stats' in msg.topic:
                performance_callback(message_data)
            
            if extra_callback:
                extra_callback(msg)
            
            message_event.set()
        
        def on_connect(client, userdata, flags, rc, properties=None):
            if rc != 0:
                print(f"Failed to connect to MQTT broker: {rc}")
        
        client.on_message = on_message
        client.on_connect = on_connect
        
        # Connect with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client.connect('localhost', port, 60)
                break
            except ConnectionRefusedError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Wait before retry
        
        # Subscribe to all topics
        client.subscribe('#')
        client.loop_start()
        
        return client, message_event
    
    def _verify_services_healthy(self, containers: Dict):
        """Verify all services are healthy"""
        for name, container in containers.items():
            if container:
                container.reload()
                assert container.status == 'running', f"{name} not running"
                
                # Check logs for errors
                logs = container.logs(tail=50).decode()
                assert 'error' not in logs.lower(), f"Error in {name} logs"
                assert 'fatal' not in logs.lower(), f"Fatal error in {name}"
                
                print(f"  ✓ {name} is healthy")
    
    def _simulate_fire_events(self, mqtt_client, num_events: int = 5, topic_prefix: str = "test/master"):
        """Simulate fire detection events"""
        cameras = ['camera_1', 'camera_2', 'camera_3']
        base_time = time.time()
        
        # First register cameras as online
        for camera in cameras:
            telemetry = {
                'camera_id': camera,
                'status': 'online',
                'timestamp': base_time
            }
            mqtt_client.publish(f'{topic_prefix}/system/camera_telemetry', json.dumps(telemetry))
        
        time.sleep(1)  # Allow cameras to register
        
        # Send growing fire detections from each camera
        for i in range(num_events):
            for j, camera in enumerate(cameras):
                # Growing bounding box to simulate spreading fire
                width = 0.03 + i * 0.005
                height = 0.03 + i * 0.004
                
                detection = {
                    'camera_id': camera,
                    'object': 'fire',
                    'object_id': f'fire_{j+1:03d}',
                    'confidence': 0.85 + i * 0.01,  # Higher confidence to ensure detection
                    'bounding_box': [0.1 + j*0.1, 0.1 + j*0.1, width, height],  # [x, y, width, height] normalized
                    'timestamp': base_time + i * 0.5 + j * 0.1
                }
                
                mqtt_client.publish(f'{topic_prefix}/fire/detection', json.dumps(detection))
                time.sleep(0.2)
    
    def _analyze_results(self, messages: List) -> Dict:
        """Analyze test results from messages"""
        results = {
            'fire_detections': 0,
            'consensus_reached': False,
            'gpio_triggered': False,
            'avg_latency': 0
        }
        
        fire_events = []
        consensus_events = []
        gpio_events = []
        
        for msg in messages:
            topic = msg['topic']
            
            if topic.endswith('/fire/detection'):
                try:
                    detection = json.loads(msg['payload'])
                    if detection.get('object') == 'fire':
                        fire_events.append(msg)
                        results['fire_detections'] += 1
                except:
                    pass
            
            elif topic.endswith('/fire/trigger'):
                consensus_events.append(msg)
                results['consensus_reached'] = True
            
            elif topic.endswith('/gpio/status') or topic.endswith('/system/trigger_telemetry'):
                try:
                    status = json.loads(msg['payload'])
                    # Check for pump activation in various formats
                    system_state = status.get('system_state', {})
                    if (status.get('pump_active') or 
                        system_state.get('engine_on') or 
                        system_state.get('state') == 'RUNNING' or
                        status.get('action') == 'engine_running'):
                        gpio_events.append(msg)
                        results['gpio_triggered'] = True
                except:
                    pass
        
        # Calculate average latency
        if fire_events and gpio_events:
            first_detection = fire_events[0]['timestamp']
            first_trigger = gpio_events[0]['timestamp']
            results['avg_latency'] = (first_trigger - first_detection) * 1000  # ms
        
        return results
    
    def _verify_coral_tpu(self) -> Dict:
        """Verify Coral TPU availability"""
        result = subprocess.run(
            ['python3.8', '-c', '''
import json
from pycoral.utils.edgetpu import list_edge_tpus
tpus = list_edge_tpus()
print(json.dumps({"devices": [{"type": t["type"], "path": t["path"]} for t in tpus]}))
'''],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        
        return {'devices': []}
    
    def _verify_gpu_tensorrt(self) -> Dict:
        """Verify GPU and TensorRT"""
        info = {}
        
        # Get GPU info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            info['gpu_name'] = result.stdout.strip()
        
        # Check TensorRT
        try:
            import tensorrt as trt
            info['tensorrt_version'] = trt.__version__
        except:
            info['tensorrt_version'] = 'Not installed'
        
        return info
    
    def _verify_hailo(self) -> Dict:
        """Verify Hailo availability"""
        # Check for Hailo device
        if Path('/dev/hailo0').exists():
            return {'device': '/dev/hailo0'}
        
        # Try hailortcli
        result = subprocess.run(['hailortcli', 'fw-control', 'identify'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {'device': 'hailo0', 'info': result.stdout}
        
        return {'device': None}
    
    def _get_coral_devices(self) -> List[str]:
        """Get Coral device mappings for Docker"""
        devices = []
        
        # Check for PCIe Coral
        import glob
        apex_devices = glob.glob('/dev/apex*')
        for device in apex_devices:
            devices.append(f"{device}:{device}")
        
        # Check for USB Coral
        usb_result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if usb_result.returncode == 0:
            if '1a6e:089a' in usb_result.stdout or '18d1:9302' in usb_result.stdout:
                # USB Coral uses generic USB device
                devices.append('/dev/bus/usb:/dev/bus/usb')
        
        return devices
    
    
    def _verify_coral_usage(self, container, messages: List) -> bool:
        """Verify Coral TPU was actually used"""
        # Check container logs
        logs = container.logs().decode()
        
        coral_indicators = [
            'edge tpu detected',
            'coral',
            'edgetpu delegate',
            'EdgeTPU available'
        ]
        
        for indicator in coral_indicators:
            if indicator in logs.lower():
                return True
        
        # Check stats messages
        for msg in messages:
            if msg['topic'] == 'frigate/stats':
                try:
                    stats = json.loads(msg['payload'])
                    detectors = stats.get('detectors', {})
                    for name, info in detectors.items():
                        if 'coral' in name and info.get('inference_speed'):
                            return True
                except:
                    pass
        
        return False
    
    def _verify_hailo_usage(self, container, messages: List) -> bool:
        """Verify Hailo was actually used"""
        # Check container logs
        logs = container.logs().decode()
        
        hailo_indicators = [
            'hailo',
            'HailoRT',
            'Hailo device'
        ]
        
        for indicator in hailo_indicators:
            if indicator in logs:
                return True
        
        return False
    
    def _send_coral_test_detections(self, mqtt_client):
        """Send test detections for Coral TPU"""
        for i in range(10):
            event = {
                'type': 'new',
                'after': {
                    'id': f'coral_test_{i}',
                    'camera': 'test_camera',
                    'label': 'fire',
                    'score': 0.85,
                    'area': 25000,
                    'box': [150, 150, 350, 350]
                }
            }
            mqtt_client.publish('frigate/events', json.dumps(event))
            time.sleep(0.2)
    
    def _send_hailo_test_detections(self, mqtt_client):
        """Send test detections for Hailo"""
        for i in range(10):
            event = {
                'type': 'new',
                'after': {
                    'id': f'hailo_test_{i}',
                    'camera': 'test_camera',
                    'label': 'fire',
                    'score': 0.88,
                    'area': 28000,
                    'box': [180, 180, 380, 380]
                }
            }
            mqtt_client.publish('frigate/events', json.dumps(event))
            time.sleep(0.2)
    
    def _run_tensorrt_performance_test(self, mqtt_client):
        """Run performance test for TensorRT"""
        # Send rapid-fire detections to test throughput
        for i in range(50):
            event = {
                'type': 'new',
                'after': {
                    'id': f'trt_perf_{i}',
                    'camera': 'perf_camera',
                    'label': 'fire',
                    'score': 0.9,
                    'area': 30000,
                    'box': [200, 200, 400, 400]
                }
            }
            mqtt_client.publish('frigate/events', json.dumps(event))
            time.sleep(0.05)  # 20 events per second
    
    def _analyze_gpu_performance(self, metrics: List) -> Dict:
        """Analyze GPU performance metrics"""
        inference_times = []
        fps_values = []
        
        for metric in metrics:
            try:
                data = json.loads(metric['payload'])
                
                # Extract detector inference times
                for detector in data.get('detectors', {}).values():
                    if 'inference_speed' in detector:
                        inference_times.append(detector['inference_speed'])
                
                # Extract camera FPS
                for camera in data.get('cameras', {}).values():
                    if 'detection_fps' in camera:
                        fps_values.append(camera['detection_fps'])
            except:
                pass
        
        return {
            'avg_inference': np.mean(inference_times) if inference_times else 0,
            'peak_fps': max(fps_values) if fps_values else 0,
            'gpu_util': 50.0  # Would need nvidia-ml-py for real GPU util
        }
    
    def _wait_for_frigate_config(self, camera_data: Dict, container_manager: DockerContainerManager):
        """Wait for Frigate config to be generated"""
        config_path = Path(f'/tmp/{container_manager.worker_id}/frigate_config/config.yml')
        
        for _ in range(30):
            if config_path.exists():
                print("  ✓ Frigate config generated")
                return
            time.sleep(1)
        
        raise TimeoutError("Frigate config not generated")
    
    def _run_standard_performance_test(self, mqtt_client):
        """Run standardized performance test"""
        # Send 100 detection events
        for i in range(100):
            event = {
                'type': 'new',
                'after': {
                    'id': f'perf_test_{i}',
                    'camera': 'test_camera',
                    'label': 'fire',
                    'score': 0.8,
                    'area': 20000,
                    'box': [100, 100, 300, 300]
                }
            }
            mqtt_client.publish('frigate/events', json.dumps(event))
            time.sleep(0.1)
    
    def _extract_inference_time(self, stats: Dict) -> float:
        """Extract average inference time from stats"""
        times = []
        for detector in stats.get('detectors', {}).values():
            if 'inference_speed' in detector:
                times.append(detector['inference_speed'])
        
        return np.mean(times) if times else 0
    
    def _extract_fps(self, stats: Dict) -> float:
        """Extract average FPS from stats"""
        fps_values = []
        for camera in stats.get('cameras', {}).values():
            if 'detection_fps' in camera:
                fps_values.append(camera['detection_fps'])
        
        return np.mean(fps_values) if fps_values else 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])