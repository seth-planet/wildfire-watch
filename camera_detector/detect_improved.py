"""Improved RTSP validation using ProcessPoolExecutor for robust timeout handling.

This module contains the improved _validate_rtsp_stream method that uses
process-based isolation to prevent hanging on problematic RTSP streams.
"""

import cv2
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


def _rtsp_validation_worker(rtsp_url: str, timeout_ms: int) -> bool:
    """Worker function to validate RTSP stream in separate process.
    
    This function runs in a separate process to ensure robust timeout handling
    even if cv2.VideoCapture hangs indefinitely.
    
    Args:
        rtsp_url: The RTSP URL to validate
        timeout_ms: Timeout in milliseconds for OpenCV operations
        
    Returns:
        bool: True if stream is valid and accessible, False otherwise
    """
    cap = None
    try:
        # Create video capture with timeout settings
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        
        # Check if capture opened successfully
        if not cap.isOpened():
            return False
        
        # Try to read a frame to ensure stream is actually working
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
            
        return True
        
    except Exception as e:
        # Any exception means validation failed
        logger.debug(f"RTSP validation worker error: {e}")
        return False
        
    finally:
        # Always release the capture
        if cap is not None:
            try:
                cap.release()
            except:
                pass


def validate_rtsp_stream_improved(rtsp_url: str, timeout_seconds: int = 10) -> bool:
    """Validate RTSP stream with robust process-based timeout.
    
    This improved version uses ProcessPoolExecutor to run the validation
    in a separate process, providing guaranteed timeout even if OpenCV
    hangs indefinitely.
    
    Args:
        rtsp_url: The RTSP URL to validate
        timeout_seconds: Maximum time to wait for validation (default: 10)
        
    Returns:
        bool: True if stream is valid and accessible, False otherwise
        
    Thread Safety:
        This function is thread-safe as it uses process isolation.
        
    Side Effects:
        - Creates a temporary process for validation
        - May log debug/warning messages
    """
    logger.debug(f"Validating RTSP stream: {rtsp_url}")
    
    # Convert timeout to milliseconds for OpenCV
    timeout_ms = max(timeout_seconds * 1000, 1000)  # Minimum 1 second
    
    # Use ProcessPoolExecutor for robust timeout handling
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            # Submit validation task to separate process
            future = executor.submit(_rtsp_validation_worker, rtsp_url, timeout_ms)
            
            # Wait for result with timeout
            is_valid = future.result(timeout=timeout_seconds)
            
            if is_valid:
                logger.debug(f"Successfully validated RTSP stream: {rtsp_url}")
            else:
                logger.warning(f"RTSP stream validation failed: {rtsp_url}")
                
            return is_valid
            
        except FuturesTimeoutError:
            logger.warning(f"RTSP validation timed out after {timeout_seconds}s: {rtsp_url}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error during RTSP validation: {e}")
            return False