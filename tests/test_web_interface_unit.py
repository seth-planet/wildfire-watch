#!/usr/bin/env python3.12
"""Unit tests for the Web Interface service.

Tests individual components in isolation, mocking only external dependencies.
"""

import pytest
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules to test
from web_interface.models import (
    MQTTEvent, ServiceHealth, GPIOState, SystemStatus,
    EventType, ServiceStatus, HealthCheckResponse
)
from web_interface.security import (
    LANOnlyMiddleware, RateLimitMiddleware, DebugAuthMiddleware,
    AuditLogger, generate_csrf_token, verify_csrf_token
)
from web_interface.config import WebInterfaceConfig


class TestModels:
    """Test data models and serialization."""
    
    def test_mqtt_event_creation(self):
        """Test MQTTEvent model creation and display formatting."""
        event = MQTTEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            topic="fire/trigger",
            payload={"camera": "cam1", "confidence": 0.95},
            event_type=EventType.FIRE
        )
        
        display = event.to_display_dict()
        assert display['timestamp'] == '2024-01-01 12:00:00 UTC'
        assert display['topic'] == 'fire/trigger'
        assert display['type'] == 'fire'
        assert 'payload' in display
        
    def test_service_health_staleness(self):
        """Test service health staleness detection."""
        # Fresh service
        fresh_service = ServiceHealth(
            name="test_service",
            status=ServiceStatus.HEALTHY,
            last_seen=datetime.utcnow(),
            uptime=1.5
        )
        assert not fresh_service.is_stale
        
        # Stale service (3 minutes old)
        from datetime import timedelta
        old_time = datetime.utcnow() - timedelta(minutes=3)
        stale_service = ServiceHealth(
            name="test_service",
            status=ServiceStatus.HEALTHY,
            last_seen=old_time,
            uptime=1.5
        )
        assert stale_service.is_stale
        
    def test_gpio_state_display(self):
        """Test GPIO state display formatting."""
        gpio = GPIOState(
            pin_id="main_valve",
            name="Main Water Valve",
            state=True,
            last_change=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        display = gpio.to_display_dict()
        assert display['id'] == 'main_valve'
        assert display['name'] == 'Main Water Valve'
        assert display['state'] == 'ON'
        assert display['state_bool'] is True
        assert display['last_change'] == '12:00:00'
        
    def test_system_status_health_percentage(self):
        """Test system health percentage calculation."""
        status = SystemStatus(
            fire_active=False,
            service_count=5,
            healthy_services=4,
            camera_count=3,
            active_cameras=2,
            mqtt_connected=True,
            buffer_size=100
        )
        
        assert status.system_health_percentage == 80  # 4/5 = 80%
        
        # Test edge case with no services
        empty_status = SystemStatus(
            fire_active=False,
            service_count=0,
            healthy_services=0,
            camera_count=0,
            active_cameras=0,
            mqtt_connected=False,
            buffer_size=0
        )
        assert empty_status.system_health_percentage == 0


class TestSecurity:
    """Test security middleware and utilities."""
    
    def test_lan_only_middleware_allows_local(self):
        """Test LAN-only middleware allows local IPs."""
        from fastapi import FastAPI, Request
        
        app = FastAPI()
        middleware = LANOnlyMiddleware(app, allowed_networks={'127.0.0.1', '192.168.1.'})
        
        # Test allowed IPs
        assert middleware._is_allowed_ip('127.0.0.1')
        assert middleware._is_allowed_ip('192.168.1.100')
        assert not middleware._is_allowed_ip('8.8.8.8')
        assert not middleware._is_allowed_ip('192.168.2.100')
        
    def test_lan_only_middleware_cidr(self):
        """Test LAN-only middleware with CIDR notation."""
        from fastapi import FastAPI
        
        app = FastAPI()
        middleware = LANOnlyMiddleware(app, allowed_networks={'10.0.0.0/8'})
        
        assert middleware._is_allowed_ip('10.1.2.3')
        assert middleware._is_allowed_ip('10.255.255.255')
        assert not middleware._is_allowed_ip('11.0.0.0')
        
    def test_rate_limiter_token_bucket(self):
        """Test rate limiter token bucket algorithm."""
        from fastapi import FastAPI
        
        app = FastAPI()
        limiter = RateLimitMiddleware(app, requests_per_minute=60, burst_size=5)
        
        # Should allow burst
        for _ in range(5):
            assert limiter._check_rate_limit('127.0.0.1')
            
        # 6th request should be rate limited
        assert not limiter._check_rate_limit('127.0.0.1')
        
        # Different IP should have its own bucket
        assert limiter._check_rate_limit('192.168.1.1')
        
    def test_debug_auth_token_verification(self):
        """Test debug authentication token verification."""
        with patch('web_interface.security.get_config') as mock_config:
            mock_config.return_value.debug_mode = True
            mock_config.return_value.debug_token = 'test-token-1234567890123456789012'
            
            auth = DebugAuthMiddleware()
            
            # Valid token
            assert auth.verify_debug_token('test-token-1234567890123456789012', '127.0.0.1')
            
            # Invalid token
            assert not auth.verify_debug_token('wrong-token', '127.0.0.1')
            
            # Too many failures
            for _ in range(5):
                auth.verify_debug_token('wrong', '192.168.1.1')
            assert not auth.verify_debug_token('test-token-1234567890123456789012', '192.168.1.1')
            
    def test_csrf_token_generation_and_verification(self):
        """Test CSRF token generation and verification."""
        token1 = generate_csrf_token()
        token2 = generate_csrf_token()
        
        # Tokens should be unique
        assert token1 != token2
        
        # Token should be verifiable
        assert verify_csrf_token(token1, token1)
        assert not verify_csrf_token(token1, token2)
        assert not verify_csrf_token('', token1)
        assert not verify_csrf_token(token1, '')


class TestConfiguration:
    """Test configuration management."""
    
    @patch.dict(os.environ, {
        'STATUS_PANEL_HTTP_PORT': '9090',
        'STATUS_PANEL_DEBUG_MODE': 'true',
        'STATUS_PANEL_DEBUG_TOKEN': 'test-token-1234567890123456789012345',
        'STATUS_PANEL_MQTT_BUFFER_SIZE': '500'
    })
    def test_config_loading_from_env(self):
        """Test configuration loading from environment variables."""
        config = WebInterfaceConfig()
        
        assert config.http_port == 9090
        assert config.debug_mode is True
        assert config.debug_token == 'test-token-1234567890123456789012345'
        assert config.mqtt_buffer_size == 500
        
    def test_config_validation(self):
        """Test configuration validation."""
        with patch.dict(os.environ, {
            'STATUS_PANEL_DEBUG_MODE': 'true',
            'STATUS_PANEL_DEBUG_TOKEN': 'short'  # Too short
        }):
            with pytest.raises(Exception) as exc_info:
                WebInterfaceConfig()
            assert 'token too weak' in str(exc_info.value)
            
    def test_config_mqtt_topics(self):
        """Test MQTT topic generation."""
        config = WebInterfaceConfig()
        topics = config.get_mqtt_topics()
        
        assert 'system/+/health' in topics
        assert 'fire/trigger' in topics
        assert 'gpio/status' in topics
        assert len(topics) > 10  # Should have many topics
        
    def test_config_security_defaults(self):
        """Test security-focused default configuration."""
        config = WebInterfaceConfig()
        
        # Should default to localhost only
        assert config.http_host == '127.0.0.1'
        assert config.allowed_networks == ['127.0.0.1', 'localhost']
        assert config.debug_mode is False
        assert config.enable_csrf is True
        assert config.audit_log_enabled is True
        assert config.allow_remote_control is False


class TestMQTTHandler:
    """Test MQTT handler functionality."""
    
    @patch('web_interface.mqtt_handler.get_config')
    def test_event_buffer_circular(self, mock_config):
        """Test that event buffer is circular and respects max size."""
        from web_interface.mqtt_handler import MQTTHandler
        
        # Configure small buffer for testing
        mock_config.return_value.mqtt_buffer_size = 5
        mock_config.return_value.get_mqtt_topics.return_value = []
        
        with patch('web_interface.mqtt_handler.MQTTService.__init__'):
            handler = MQTTHandler()
            
            # Add more events than buffer size
            for i in range(10):
                event = MQTTEvent(
                    timestamp=datetime.utcnow(),
                    topic=f"test/topic{i}",
                    payload={"count": i},
                    event_type=EventType.OTHER
                )
                with handler._event_buffer_lock:
                    handler._event_buffer.append(event)
                    
            # Buffer should only contain last 5 events
            assert len(handler._event_buffer) == 5
            events = list(handler._event_buffer)
            assert events[0].payload['count'] == 5
            assert events[4].payload['count'] == 9
            
    @patch('web_interface.mqtt_handler.get_config')
    def test_rate_limiting_per_topic(self, mock_config):
        """Test MQTT message rate limiting per topic."""
        from web_interface.mqtt_handler import MQTTHandler
        
        mock_config.return_value.get_mqtt_topics.return_value = []
        mock_config.return_value.mqtt_buffer_size = 1000
        
        with patch('web_interface.mqtt_handler.MQTTService.__init__', return_value=None):
            handler = MQTTHandler()
            # Initialize required attributes manually since we mocked __init__
            handler._event_window_start = time.time()
            handler._event_counts = {}
            
            # Fire events should have high limit (100)
            for i in range(99):
                assert handler._check_rate_limit('fire/trigger')
                
            # 100th should still be allowed
            assert handler._check_rate_limit('fire/trigger')
            
            # 101st should be rate limited
            assert not handler._check_rate_limit('fire/trigger')
            
            # Regular telemetry should have lower limit
            for i in range(20):
                assert handler._check_rate_limit('telemetry/test')
                
            # 21st should be rate limited
            assert not handler._check_rate_limit('telemetry/test')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])