#!/usr/bin/env python3.12
"""
Comprehensive tests for MQTT Broker configuration and functionality
Tests configuration consistency, multi-node support, security features, and performance settings
"""
import os
import sys
import time
import json
import socket
import threading
import subprocess
import tempfile
import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path

# Test imports
import paho.mqtt.client as mqtt

# ─────────────────────────────────────────────────────────────────
# Test Fixtures and Utilities
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def temp_config_dir():
    """Create temporary directory for test configs"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def base_mosquitto_config():
    """Base mosquitto configuration for testing"""
    return """
# Test configuration
persistence true
persistence_location /mosquitto/data/
autosave_interval 30
max_inflight_messages 40
max_queued_messages 10000
listener 1883
allow_anonymous true
"""

@pytest.fixture
def mock_mqtt_client():
    """Mock MQTT client for testing"""
    client = MagicMock(spec=mqtt.Client)
    client.connect = MagicMock(return_value=(0, None))
    client.subscribe = MagicMock(return_value=(0, None))
    client.publish = MagicMock(return_value=(0, None))
    return client

# ─────────────────────────────────────────────────────────────────
# Configuration File Tests
# ─────────────────────────────────────────────────────────────────
class TestConfigurationFiles:
    def test_main_mosquitto_config_exists(self):
        """Test that main mosquitto.conf exists and is valid"""
        config_path = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        assert os.path.exists(config_path), "mosquitto.conf not found"
        
        # Read and validate config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check required settings
        assert "persistence true" in config_content
        assert "listener 1883" in config_content
        assert "include_dir /mosquitto/config/conf.d" in config_content
        assert "autosave_interval" in config_content
        assert "max_queued_messages" in config_content
    
    def test_tls_config_exists(self):
        """Test TLS configuration file"""
        tls_config_path = "/home/seth/wildfire-watch/mqtt_broker/conf.d/tls.conf"
        assert os.path.exists(tls_config_path), "tls.conf not found"
        
        with open(tls_config_path, 'r') as f:
            tls_config = f.read()
        
        # Verify TLS settings as documented
        assert "listener 8883" in tls_config
        assert "cafile /mnt/data/certs/ca.crt" in tls_config
        assert "certfile /mnt/data/certs/server.crt" in tls_config
        assert "keyfile /mnt/data/certs/server.key" in tls_config
        assert "require_certificate false" in tls_config, "Should allow anonymous per README"
        assert "tls_version tlsv1.2" in tls_config
    
    def test_websocket_config_exists(self):
        """Test WebSocket configuration file"""
        ws_config_path = "/home/seth/wildfire-watch/mqtt_broker/conf.d/websockets.conf"
        assert os.path.exists(ws_config_path), "websockets.conf not found"
        
        with open(ws_config_path, 'r') as f:
            ws_config = f.read()
        
        # Verify WebSocket settings
        assert "listener 9001" in ws_config
        assert "protocol websockets" in ws_config
        assert "allow_anonymous true" in ws_config
        # Check for commented WSS config
        assert "# listener 9443" in ws_config
    
    def test_bridge_config_template(self):
        """Test bridge configuration template for multi-node"""
        bridge_config_path = "/home/seth/wildfire-watch/mqtt_broker/conf.d/bridge.conf"
        assert os.path.exists(bridge_config_path), "bridge.conf not found"
        
        with open(bridge_config_path, 'r') as f:
            bridge_config = f.read()
        
        # Verify bridge template (should be commented out)
        assert "# connection wildfire_cloud_bridge" in bridge_config
        assert "# topic fire/# out 2" in bridge_config
        assert "# topic system/# out 1" in bridge_config
        assert "# topic gpio/# none" in bridge_config, "GPIO topics should not be bridged"
        assert "# cleansession false" in bridge_config
        assert "# queue_qos0_messages true" in bridge_config
    
    def test_entrypoint_script(self):
        """Test entrypoint.sh script functionality"""
        entrypoint_path = "/home/seth/wildfire-watch/mqtt_broker/entrypoint.sh"
        assert os.path.exists(entrypoint_path), "entrypoint.sh not found"
        
        with open(entrypoint_path, 'r') as f:
            script = f.read()
        
        # Check for required functionality
        assert "dbus-daemon --system --fork" in script, "D-Bus startup missing"
        assert "avahi-daemon" in script, "mDNS support missing"
        assert "avahi-publish-service" in script, "Service publishing missing"
        assert "mkdir -p /mosquitto/data /mosquitto/log" in script
        assert "chown -R mosquitto:mosquitto" in script
        assert 'exec "$@"' in script

# ─────────────────────────────────────────────────────────────────
# Configuration Consistency Tests
# ─────────────────────────────────────────────────────────────────
class TestConfigurationConsistency:
    def test_port_consistency(self):
        """Test that ports mentioned in README match config files"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        tls_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/tls.conf"
        ws_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/websockets.conf"
        
        # Read README
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check port 1883 (plain MQTT)
        assert "Port 1883" in readme
        with open(mosquitto_conf, 'r') as f:
            assert "listener 1883" in f.read()
        
        # Check port 8883 (TLS)
        assert "Port 8883" in readme
        with open(tls_conf, 'r') as f:
            assert "listener 8883" in f.read()
        
        # Check port 9001 (WebSocket)
        assert "Port 9001" in readme
        with open(ws_conf, 'r') as f:
            assert "listener 9001" in f.read()
    
    def test_certificate_paths_consistency(self):
        """Test certificate paths are consistent across configs"""
        tls_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/tls.conf"
        entrypoint = "/home/seth/wildfire-watch/mqtt_broker/entrypoint.sh"
        
        cert_base_path = "/mnt/data/certs"
        
        # Check TLS config
        with open(tls_conf, 'r') as f:
            tls_content = f.read()
            assert f"{cert_base_path}/ca.crt" in tls_content
            assert f"{cert_base_path}/server.crt" in tls_content
            assert f"{cert_base_path}/server.key" in tls_content
        
        # Check entrypoint script
        with open(entrypoint, 'r') as f:
            script = f.read()
            assert cert_base_path in script
    
    def test_persistence_settings(self):
        """Test persistence configuration consistency"""
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        
        with open(mosquitto_conf, 'r') as f:
            config = f.read()
        
        # Verify persistence settings
        assert "persistence true" in config
        assert "persistence_location /mosquitto/data/" in config
        assert "autosave_interval 30" in config
        assert "autosave_on_changes true" in config
        assert "retain_available true" in config
        assert "queue_qos0_messages true" in config

# ─────────────────────────────────────────────────────────────────
# Multi-Node Support Tests
# ─────────────────────────────────────────────────────────────────
class TestMultiNodeSupport:
    def test_bridge_configuration_template(self):
        """Test bridge configuration supports multi-node as per docs/multi-node.md"""
        bridge_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/bridge.conf"
        multi_node_doc = "/home/seth/wildfire-watch/docs/multi-node.md"
        
        # Read both files
        with open(bridge_conf, 'r') as f:
            bridge_config = f.read()
        
        with open(multi_node_doc, 'r') as f:
            multi_node = f.read()
        
        # Verify bridge supports required topics from multi-node.md
        assert "topic fire/# out" in bridge_config
        assert "topic system/# out" in bridge_config
        assert "bridge_protocol_version mqttv311" in bridge_config
        assert "bridge_cafile" in bridge_config
        assert "restart_timeout" in bridge_config
        assert "keepalive_interval" in bridge_config
    
    def test_edge_to_central_bridging(self):
        """Test edge to central bridging configuration"""
        bridge_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/bridge.conf"
        
        with open(bridge_conf, 'r') as f:
            config = f.read()
        
        # Check for edge node bridging patterns
        assert "# start_type automatic" in config
        assert "# cleansession false" in config
        assert "# max_queued_messages" in config
        assert "# queue_qos0_messages true" in config
        
        # Verify security topics are not bridged
        assert "# topic gpio/# none" in config
    
    def test_high_availability_settings(self):
        """Test settings support HA deployments"""
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        
        with open(mosquitto_conf, 'r') as f:
            config = f.read()
        
        # Check for HA-friendly settings
        assert "max_connections -1" in config, "Should support unlimited connections"
        assert "max_inflight_messages" in config
        assert "max_queued_messages" in config
        assert "keepalive_interval 60" in config
        assert "max_keepalive 120" in config

# ─────────────────────────────────────────────────────────────────
# Security Configuration Tests
# ─────────────────────────────────────────────────────────────────
class TestSecurityConfiguration:
    def test_tls_security_settings(self):
        """Test TLS security configuration"""
        tls_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/tls.conf"
        
        with open(tls_conf, 'r') as f:
            config = f.read()
        
        # Verify security settings match README
        assert "require_certificate false" in config, "Should allow graceful degradation"
        assert "use_identity_as_username false" in config
        assert "allow_anonymous true" in config
        assert "tls_version tlsv1.2" in config
        
        # Check cipher suites
        assert "ciphers" in config
        assert "TLS_AES_256_GCM_SHA384" in config
        assert "ECDHE-RSA-AES256-GCM-SHA384" in config
    
    def test_default_security_warning(self):
        """Test that default certificate warning is documented"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check for security warnings
        assert "Security Note" in readme
        assert "INSECURE" in readme
        assert "default certificates" in readme.lower()
        assert "must be replaced for production" in readme.lower()
    
    def test_anonymous_access_configuration(self):
        """Test anonymous access settings across configs"""
        configs = [
            "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf",
            "/home/seth/wildfire-watch/mqtt_broker/conf.d/tls.conf",
            "/home/seth/wildfire-watch/mqtt_broker/conf.d/websockets.conf"
        ]
        
        for config_path in configs:
            with open(config_path, 'r') as f:
                config = f.read()
                if "listener" in config:
                    assert "allow_anonymous true" in config, f"Anonymous access not configured in {config_path}"

# ─────────────────────────────────────────────────────────────────
# Performance and Scaling Tests
# ─────────────────────────────────────────────────────────────────
class TestPerformanceConfiguration:
    def test_message_limits(self):
        """Test message size and queue limits"""
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        
        with open(mosquitto_conf, 'r') as f:
            config = f.read()
        
        # Check message limits
        assert "max_packet_size 1048576" in config, "1MB packet size expected"
        assert "message_size_limit 0" in config, "Unlimited message size expected"
        assert "max_inflight_messages 40" in config
        assert "max_queued_messages 10000" in config
        
        # Memory limits
        assert "memory_limit 0" in config
        assert "max_inflight_bytes 0" in config
        assert "max_queued_bytes 0" in config
    
    def test_network_resilience_settings(self):
        """Test settings for edge network resilience"""
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        bridge_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/bridge.conf"
        
        with open(mosquitto_conf, 'r') as f:
            main_config = f.read()
        
        # Check keepalive settings for flaky connections
        assert "keepalive_interval 60" in main_config
        assert "max_keepalive 120" in main_config
        
        # Check bridge resilience settings
        with open(bridge_conf, 'r') as f:
            bridge_config = f.read()
            assert "# restart_timeout 5 30" in bridge_config
            assert "# idle_timeout 60" in bridge_config
            assert "# keepalive_interval 30" in bridge_config
    
    def test_scaling_configurations_in_readme(self):
        """Test that README documents scaling configurations"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check for scaling guidance
        assert "Small Networks" in readme
        assert "Medium Networks" in readme
        assert "Large Networks" in readme
        assert "max_connections" in readme
        assert "max_inflight_messages" in readme
        assert "max_queued_messages" in readme

# ─────────────────────────────────────────────────────────────────
# Service Discovery Tests
# ─────────────────────────────────────────────────────────────────
class TestServiceDiscovery:
    def test_mdns_configuration(self):
        """Test mDNS service discovery setup"""
        entrypoint = "/home/seth/wildfire-watch/mqtt_broker/entrypoint.sh"
        
        with open(entrypoint, 'r') as f:
            script = f.read()
        
        # Check Avahi daemon setup
        assert "avahi-daemon" in script
        assert "--no-drop-root" in script
        assert "--daemonize" in script
        assert "--no-chroot" in script
        
        # Check service publishing
        assert "avahi-publish-service" in script
        assert '"Wildfire MQTT Broker"' in script
        assert "_mqtt._tcp" in script
        assert "1883" in script
        assert '"Wildfire MQTT-TLS Broker"' in script
        assert "_secure-mqtt._tcp" in script
        assert "8883" in script
    
    def test_dbus_requirement(self):
        """Test D-Bus daemon requirement for Avahi"""
        entrypoint = "/home/seth/wildfire-watch/mqtt_broker/entrypoint.sh"
        
        with open(entrypoint, 'r') as f:
            script = f.read()
        
        # Check D-Bus startup
        assert "dbus-daemon --system --fork" in script
        assert "if [ ! -f /var/run/dbus/pid ]" in script
        assert "sleep 1" in script, "Should wait for D-Bus"

# ─────────────────────────────────────────────────────────────────
# Topic Structure Tests
# ─────────────────────────────────────────────────────────────────
class TestTopicStructure:
    def test_documented_topics(self):
        """Test that README documents all required topics"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check for all required topics
        required_topics = [
            "fire/detection/{camera_id}",
            "fire/trigger",
            "fire/consensus",
            "camera/discovery/{camera_id}",
            "camera/status/{camera_id}",
            "system/+/health",
            "frigate/events"
        ]
        
        for topic in required_topics:
            assert topic in readme, f"Topic {topic} not documented"
    
    def test_bridge_topic_filtering(self):
        """Test bridge filters sensitive topics"""
        bridge_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/bridge.conf"
        
        with open(bridge_conf, 'r') as f:
            config = f.read()
        
        # Verify sensitive topics are filtered
        assert "# topic gpio/# none" in config
        assert "# topic fire/# out" in config, "Fire topics should be outgoing only"
        assert "# topic system/# out" in config, "System topics should be outgoing only"

# ─────────────────────────────────────────────────────────────────
# Docker Integration Tests
# ─────────────────────────────────────────────────────────────────
class TestDockerIntegration:
    def test_dockerfile_exists(self):
        """Test Dockerfile exists and is properly configured"""
        dockerfile = "/home/seth/wildfire-watch/mqtt_broker/Dockerfile"
        assert os.path.exists(dockerfile), "Dockerfile not found"
        
        with open(dockerfile, 'r') as f:
            content = f.read()
        
        # Basic checks
        assert "mosquitto" in content.lower()
        assert "COPY" in content
        assert "ENTRYPOINT" in content or "CMD" in content
    
    def test_volume_mounts_documented(self):
        """Test that required volume mounts are documented"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check for volume mount documentation
        assert "/mosquitto/data" in readme
        assert "/mosquitto/log" in readme
        assert "/mnt/data/certs" in readme

# ─────────────────────────────────────────────────────────────────
# Logging Configuration Tests
# ─────────────────────────────────────────────────────────────────
class TestLoggingConfiguration:
    def test_logging_settings(self):
        """Test logging configuration"""
        mosquitto_conf = "/home/seth/wildfire-watch/mqtt_broker/mosquitto.conf"
        
        with open(mosquitto_conf, 'r') as f:
            config = f.read()
        
        # Check logging configuration
        assert "log_dest stdout" in config
        assert "log_dest file /mosquitto/log/mosquitto.log" in config
        assert "log_type all" in config
        assert "log_timestamp true" in config
        assert "log_timestamp_format %Y-%m-%d %H:%M:%S" in config
    
    def test_debug_logging_documented(self):
        """Test debug logging is documented"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        assert "Enable Debug Logging" in readme
        assert "log_type all" in readme
        assert "log_dest file" in readme

# ─────────────────────────────────────────────────────────────────
# WebSocket Support Tests
# ─────────────────────────────────────────────────────────────────
class TestWebSocketSupport:
    def test_websocket_configuration(self):
        """Test WebSocket configuration"""
        ws_conf = "/home/seth/wildfire-watch/mqtt_broker/conf.d/websockets.conf"
        
        with open(ws_conf, 'r') as f:
            config = f.read()
        
        # Check WebSocket settings
        assert "listener 9001" in config
        assert "protocol websockets" in config
        assert "websockets_max_frame_size 0" in config
        assert "socket_domain ipv4" in config
    
    def test_websocket_examples_in_readme(self):
        """Test WebSocket usage examples in README"""
        readme_path = "/home/seth/wildfire-watch/mqtt_broker/README.md"
        
        with open(readme_path, 'r') as f:
            readme = f.read()
        
        # Check for WebSocket examples
        assert "ws://" in readme
        assert "mqtt.connect" in readme
        assert "9001" in readme
        assert "MQTT.js" in readme

if __name__ == '__main__':
    pytest.main([__file__, '-v'])