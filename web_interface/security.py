#!/usr/bin/env python3.12
"""Security middleware and utilities for the Web Interface service.

This module provides security features including:
- LAN-only access control
- Rate limiting
- CSRF protection
- Debug mode authentication
- Audit logging
"""

import os
import time
import logging
import hashlib
import secrets
import ipaddress
from typing import Optional, Dict, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_config
from .models import APIResponse

# Import safe logging
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_logging import SafeLoggingMixin, safe_log

logger = logging.getLogger(__name__)


class LANOnlyMiddleware(BaseHTTPMiddleware, SafeLoggingMixin):
    """Middleware to restrict access to LAN IP addresses only.
    
    This is critical for safety - the web interface should never be
    exposed to the public internet.
    """
    
    def __init__(self, app, allowed_networks: Optional[Set[str]] = None):
        """Initialize LAN-only middleware.
        
        Args:
            app: FastAPI application
            allowed_networks: Set of allowed network prefixes
        """
        super().__init__(app)
        self.config = get_config()
        self.logger = logger  # Set logger for SafeLoggingMixin
        
        # Use config or provided networks
        if allowed_networks:
            self.allowed_networks = allowed_networks
        else:
            self.allowed_networks = set(self.config.allowed_networks)
            
        # Always ensure localhost is allowed
        self.allowed_networks.update(['127.0.0.1', 'localhost', '::1'])
        
        # Parse CIDR networks
        self.allowed_cidrs = []
        for network in self.allowed_networks:
            if '/' in network:  # CIDR notation
                try:
                    self.allowed_cidrs.append(ipaddress.ip_network(network))
                except ValueError:
                    self._safe_log('warning', f"Invalid CIDR network: {network}")
                    
        self._safe_log('info', f"LAN-only middleware initialized with networks: {self.allowed_networks}")
        
    async def dispatch(self, request: Request, call_next):
        """Check if request is from allowed network.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response or 403 error
        """
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check if allowed
        if not self._is_allowed_ip(client_ip):
            self._safe_log('warning', f"Blocked access from non-LAN IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Access restricted to local network only"}
            )
            
        # Add client IP to request state for logging
        request.state.client_ip = client_ip
        
        # Continue to next middleware
        response = await call_next(request)
        return response
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP from request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client IP address
        """
        # CRITICAL: For safety-critical systems, NEVER trust X-Forwarded-For
        # Only use direct client IP to prevent spoofing
        
        # Handle TestClient which uses 'testclient' as host
        if hasattr(request.client, 'host'):
            return request.client.host
        return 'unknown'
        
    def _is_allowed_ip(self, ip: str) -> bool:
        """Check if IP is in allowed networks.
        
        Args:
            ip: IP address to check
            
        Returns:
            True if allowed, False otherwise
        """
        # Check exact matches first
        if ip in self.allowed_networks:
            return True
            
        # Check prefix matches (e.g., '192.168.1.')
        for allowed in self.allowed_networks:
            if allowed.endswith('.') and ip.startswith(allowed):
                return True
                
        # Check CIDR networks with proper validation
        try:
            ip_obj = ipaddress.ip_address(ip)
            for cidr in self.allowed_cidrs:
                if ip_obj in cidr:
                    return True
        except ValueError:
            self._safe_log('warning', f"Invalid IP address: {ip}")
            
        return False


class RateLimitMiddleware(BaseHTTPMiddleware, SafeLoggingMixin):
    """Rate limiting middleware to prevent DoS attacks."""
    
    def __init__(self, app, 
                 requests_per_minute: int = 60,
                 burst_size: int = 10,
                 enabled: bool = True):
        """Initialize rate limiter.
        
        Args:
            app: FastAPI application
            requests_per_minute: Maximum sustained request rate
            burst_size: Maximum burst capacity
            enabled: Whether rate limiting is enabled
        """
        super().__init__(app)
        self.config = get_config()
        self.logger = logger  # Set logger for SafeLoggingMixin
        self.enabled = enabled and self.config.rate_limit_enabled
        self.requests_per_minute = self.config.rate_limit_requests
        self.burst_size = burst_size
        
        # Token bucket implementation
        self._buckets: Dict[str, Tuple[float, float, float]] = {}  # ip -> (tokens, last_update, timestamp)
        self._cleanup_interval = 300  # Cleanup old IPs every 5 minutes
        self._last_cleanup = time.time()
        
        self._safe_log('info', f"Rate limiting {'enabled' if self.enabled else 'disabled'}: "
                      f"{self.requests_per_minute} req/min, burst {self.burst_size}")
        
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response or 429 error
        """
        if not self.enabled:
            return await call_next(request)
            
        # Get client IP
        client_ip = getattr(request.state, 'client_ip', request.client.host)
        
        # Check rate limit
        if not self._check_rate_limit(client_ip):
            self._safe_log('warning', f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
            
        # Continue to next middleware
        response = await call_next(request)
        return response
        
    def _check_rate_limit(self, ip: str) -> bool:
        """Check if request is within rate limit.
        
        Args:
            ip: Client IP address
            
        Returns:
            True if allowed, False if rate limited
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_buckets(current_time)
            
        # Get or create bucket
        if ip in self._buckets:
            tokens, last_update, _ = self._buckets[ip]
        else:
            tokens = float(self.burst_size)
            last_update = current_time
            
        # Calculate token replenishment
        time_passed = current_time - last_update
        rate_per_second = self.requests_per_minute / 60.0
        tokens = min(self.burst_size, tokens + time_passed * rate_per_second)
        
        # Check if we have tokens
        if tokens >= 1.0:
            # Consume a token
            tokens -= 1.0
            self._buckets[ip] = (tokens, current_time, current_time)
            return True
        else:
            # No tokens available
            self._buckets[ip] = (tokens, current_time, current_time)
            return False
            
    def _cleanup_buckets(self, current_time: float):
        """Remove old IP entries to prevent memory growth.
        
        Args:
            current_time: Current timestamp
        """
        cutoff = current_time - 3600  # Remove entries older than 1 hour
        self._buckets = {
            ip: bucket for ip, bucket in self._buckets.items()
            if bucket[2] > cutoff
        }
        self._last_cleanup = current_time


class DebugAuthMiddleware(SafeLoggingMixin):
    """Authentication for debug mode endpoints."""
    
    def __init__(self):
        """Initialize debug authentication."""
        self.config = get_config()
        self.logger = logger  # Set logger for SafeLoggingMixin
        self._failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
    def verify_debug_token(self, token: str, client_ip: str) -> bool:
        """Verify debug token with rate limiting.
        
        Args:
            token: Provided debug token
            client_ip: Client IP for rate limiting
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.debug_mode:
            return False
            
        # Check recent failed attempts
        recent_failures = self._failed_attempts[client_ip]
        if len(recent_failures) >= 5:
            # Check if all failures were in last 5 minutes
            five_minutes_ago = time.time() - 300
            recent_count = sum(1 for t in recent_failures if t > five_minutes_ago)
            if recent_count >= 5:
                self._safe_log('warning', f"Too many failed debug auth attempts from {client_ip}")
                return False
                
        # Constant-time comparison to prevent timing attacks
        expected_token = self.config.debug_token
        if not expected_token:
            return False
            
        # Use secrets.compare_digest for constant-time comparison
        is_valid = secrets.compare_digest(token, expected_token)
        
        if not is_valid:
            self._failed_attempts[client_ip].append(time.time())
            self._safe_log('warning', f"Invalid debug token from {client_ip}")
        else:
            # Clear failed attempts on success
            self._failed_attempts[client_ip].clear()
            
        return is_valid


class AuditLogger(SafeLoggingMixin):
    """Audit logging for security-critical actions."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.config = get_config()
        self.enabled = self.config.audit_log_enabled
        self.audit_logger = logging.getLogger('audit')
        self.logger = logger  # Set logger for SafeLoggingMixin (for warnings)
        
        # Setup audit log handler if enabled
        if self.enabled:
            log_path = '/mnt/data/logs/web_interface_audit.log'
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                handler = logging.FileHandler(log_path)
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
                self.audit_logger.addHandler(handler)
                self.audit_logger.setLevel(logging.INFO)
            except (PermissionError, OSError) as e:
                # Fall back to console logging if file logging fails
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    'AUDIT - %(asctime)s - %(levelname)s - %(message)s'
                ))
                self.audit_logger.addHandler(console_handler)
                self.audit_logger.setLevel(logging.INFO)
                self._safe_log('warning', f"Could not create audit log file at {log_path}: {e}. Using console logging.")
            
    def log_access(self, request: Request, response_status: int):
        """Log access attempt.
        
        Args:
            request: FastAPI request
            response_status: HTTP response status code
        """
        if not self.enabled:
            return
            
        client_ip = getattr(request.state, 'client_ip', request.client.host)
        safe_log(self.audit_logger, 'info',
            f"ACCESS - IP: {client_ip} - Method: {request.method} - "
            f"Path: {request.url.path} - Status: {response_status}"
        )
        
    def log_debug_action(self, action: str, client_ip: str, success: bool):
        """Log debug mode action.
        
        Args:
            action: Action attempted
            client_ip: Client IP address
            success: Whether action succeeded
        """
        if not self.enabled:
            return
            
        status = "SUCCESS" if success else "FAILED"
        safe_log(self.audit_logger, 'warning',
            f"DEBUG_ACTION - IP: {client_ip} - Action: {action} - Status: {status}"
        )
        
    def log_security_event(self, event_type: str, details: str, client_ip: str):
        """Log security-related event.
        
        Args:
            event_type: Type of security event
            details: Event details
            client_ip: Client IP address
        """
        if not self.enabled:
            return
            
        safe_log(self.audit_logger, 'warning',
            f"SECURITY - IP: {client_ip} - Event: {event_type} - Details: {details}"
        )


def generate_csrf_token() -> str:
    """Generate a CSRF token.
    
    Returns:
        Secure random token
    """
    return secrets.token_urlsafe(32)


def verify_csrf_token(provided_token: str, expected_token: str) -> bool:
    """Verify CSRF token.
    
    Args:
        provided_token: Token from request
        expected_token: Expected token
        
    Returns:
        True if valid, False otherwise
    """
    if not provided_token or not expected_token:
        return False
    return secrets.compare_digest(provided_token, expected_token)


def hash_password(password: str) -> str:
    """Hash a password for storage.
    
    WARNING: This implementation uses SHA-256 which is NOT suitable for
    password hashing in production. Use bcrypt, scrypt, or argon2id instead.
    This is a placeholder implementation only.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    # CRITICAL WARNING: SHA-256 is NOT suitable for password hashing
    # This is vulnerable to GPU-based attacks
    # TODO: Replace with bcrypt or argon2id before production use
    safe_log(logger, 'warning', "Using weak password hashing - replace with bcrypt/argon2id for production!")
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"{salt}${password_hash}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against hash.
    
    Args:
        password: Plain text password
        password_hash: Stored hash
        
    Returns:
        True if valid, False otherwise
    """
    try:
        salt, expected_hash = password_hash.split('$')
        actual_hash = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return secrets.compare_digest(actual_hash, expected_hash)
    except (ValueError, AttributeError):
        return False