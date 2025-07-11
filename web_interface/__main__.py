#!/usr/bin/env python3.12
"""Entry point for running web_interface as a module."""

import uvicorn
import logging
import os
import sys

# Import safe logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_logging import safe_log

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print to stderr to ensure visibility
print("web_interface.__main__ module loaded", file=sys.stderr)

if __name__ == "__main__":
    # Import app here to ensure all initialization happens after logging setup
    safe_log(logger, 'info', "Starting web_interface module...")
    
    try:
        from web_interface.app import app
        
        # Get configuration from environment
        host = os.getenv('STATUS_PANEL_HTTP_HOST', '0.0.0.0')
        port = int(os.getenv('STATUS_PANEL_HTTP_PORT', '8080'))
        log_level = os.getenv('LOG_LEVEL', 'warning').lower()
        
        safe_log(logger, 'info', f"Starting uvicorn server on {host}:{port}")
        
        # This allows running with: python -m web_interface
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,
            loop="uvloop",
            log_level=log_level
        )
    except Exception as e:
        safe_log(logger, 'error', f"Failed to start web interface: {e}")
        import traceback
        traceback.print_exc()
        raise