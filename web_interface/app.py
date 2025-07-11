#!/usr/bin/env python3.12
"""Main FastAPI application for the Web Interface service.

This module provides a read-only web dashboard for monitoring the
Wildfire Watch system. It displays system status, service health,
GPIO states, and event logs.

CRITICAL: This interface is READ-ONLY by design for safety.
No control operations are exposed through the web interface.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .mqtt_handler import MQTTHandler
from .models import (
    SystemStatus, ServiceHealth, MQTTEvent, EventFilter,
    APIResponse, HealthCheckResponse, EventType
)
from .security import (
    LANOnlyMiddleware, RateLimitMiddleware, 
    AuditLogger, DebugAuthMiddleware, generate_csrf_token
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import safe logging utility
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_logging import safe_log

# Create module-level safe logging wrapper
def _safe_log(level: str, message: str, exc_info: bool = False) -> None:
    """Module-level safe logging wrapper."""
    safe_log(logger, level, message, exc_info)

# Global instances
config = get_config()
mqtt_handler = MQTTHandler()
audit_logger = AuditLogger()
debug_auth = DebugAuthMiddleware()

# Service start time for uptime calculation
SERVICE_START_TIME = time.time()

# Templates - use absolute path for reliability
template_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=template_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    _safe_log('info', "Starting Web Interface service...")
    
    # Initialize MQTT handler (non-blocking now)
    try:
        mqtt_handler.initialize()
        _safe_log('info', "MQTT handler initialized, connection will happen in background")
    except Exception as e:
        _safe_log('error', f"Error initializing MQTT handler: {e}")
    
    _safe_log('info', "Web Interface service started successfully")
    
    yield
    
    # Shutdown
    _safe_log('info', "Shutting down Web Interface service...")
    mqtt_handler.shutdown()
    _safe_log('info', "Web Interface service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Wildfire Watch Status Panel",
    description="Read-only monitoring interface for Wildfire Watch system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if not config.show_debug_info else "/docs",
    redoc_url=None if not config.show_debug_info else "/redoc"
)

# Add middleware in correct order
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LANOnlyMiddleware, allowed_networks=set(config.allowed_networks))

# CORS - only if explicitly needed (disabled by default for security)
if os.getenv('ENABLE_CORS', 'false').lower() == 'true':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080"],  # Restrict to specific origins
        allow_credentials=False,
        allow_methods=["GET"],  # Read-only
        allow_headers=["*"],
    )

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependency for audit logging
async def log_request(request: Request):
    """Log request for audit trail."""
    # This runs after the response
    # Actual logging happens in middleware
    pass


# Main dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    try:
        # Get current system status
        system_status = mqtt_handler.get_system_status()
        service_health = mqtt_handler.get_service_health()
        gpio_states = mqtt_handler.get_gpio_states()
        recent_events = mqtt_handler.get_recent_events(limit=50)
        
        # Generate CSRF token for any forms
        csrf_token = generate_csrf_token() if config.enable_csrf else ""
        
        context = {
            "request": request,
            "system_status": system_status.to_display_dict(),
            "services": [s.to_display_dict() for s in service_health],
            "gpio_states": {k: v.to_display_dict() for k, v in gpio_states.items()},
            "recent_events": [e.to_display_dict() for e in recent_events],
            "csrf_token": csrf_token,
            "refresh_interval": config.refresh_interval,
            "debug_mode": config.debug_mode,
            "show_debug_info": config.show_debug_info,
            "audit_log_enabled": config.audit_log_enabled
        }
        
        return templates.TemplateResponse("index.html", context)
        
    except Exception as e:
        _safe_log('error', f"Error rendering dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error loading dashboard"
        )


# API endpoints for data retrieval
@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get current system status."""
    return mqtt_handler.get_system_status()


@app.get("/api/services", response_model=List[ServiceHealth])
async def get_services():
    """Get service health status."""
    return mqtt_handler.get_service_health()


@app.get("/api/events")
async def get_events(
    event_type: Optional[EventType] = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    max_age: Optional[int] = Query(None, ge=1, le=86400, description="Maximum age in seconds")
):
    """Get recent events with optional filtering."""
    events = mqtt_handler.get_recent_events(
        limit=limit,
        event_type=event_type.value if event_type else None,
        max_age_seconds=max_age
    )
    
    return {
        "events": [e.to_display_dict() for e in events],
        "count": len(events),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/gpio")
async def get_gpio_states():
    """Get current GPIO pin states."""
    states = mqtt_handler.get_gpio_states()
    return {
        "states": {k: v.to_display_dict() for k, v in states.items()},
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check endpoint
@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    uptime = time.time() - SERVICE_START_TIME
    
    # Check MQTT connection status safely
    try:
        mqtt_connected = mqtt_handler.is_connected if hasattr(mqtt_handler, 'is_connected') else False
    except:
        mqtt_connected = False
    
    # During startup (first 30 seconds), report OK even if MQTT isn't connected yet
    # This allows the container to become healthy while MQTT connects in background
    if uptime < 30:
        status = "ok"
    else:
        status = "ok" if mqtt_connected else "degraded"
    
    return HealthCheckResponse(
        status=status,
        service="web_interface",
        version="1.0.0",
        mqtt_connected=mqtt_connected,
        uptime_seconds=uptime
    )


# Debug endpoints (only if debug mode is enabled)
if config.debug_mode:
    _safe_log('warning', "DEBUG MODE ENABLED - This should NEVER be used in production!")
    
    @app.get("/debug", response_class=HTMLResponse)
    async def debug_panel(request: Request, token: str = Query(...)):
        """Debug panel (requires token)."""
        client_ip = request.client.host
        
        # Verify token
        if not debug_auth.verify_debug_token(token, client_ip):
            audit_logger.log_debug_action("debug_panel_access", client_ip, False)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid debug token"
            )
            
        audit_logger.log_debug_action("debug_panel_access", client_ip, True)
        
        # Generate CSRF token
        csrf_token = generate_csrf_token()
        
        context = {
            "request": request,
            "csrf_token": csrf_token,
            "system_status": mqtt_handler.get_system_status().to_display_dict(),
            "mqtt_topics": config.get_mqtt_topics()
        }
        
        return templates.TemplateResponse("debug.html", context)
    
    # CRITICAL WARNING: No control endpoints are implemented
    # The system is designed to be read-only for safety
    # Any control operations must be performed through physical access


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    _safe_log('error', f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for audit trail."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log access
    process_time = time.time() - start_time
    audit_logger.log_access(request, response.status_code)
    
    # Add process time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


if __name__ == "__main__":
    # This should not be run directly in production
    # Use the Docker container with proper configuration
    import uvicorn
    
    _safe_log('warning', "Running directly - use Docker container for production!")
    
    uvicorn.run(
        app,
        host=config.http_host,
        port=config.http_port,
        log_level="info",
        access_log=config.access_log
    )