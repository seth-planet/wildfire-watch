#!/usr/bin/env python3.12
"""
Centralized subprocess command runner with robust error handling.

This module provides a standardized way to execute external commands with:
- Configurable timeouts
- Retry logic for transient failures
- Clear error messages
- Graceful degradation when tools are missing

Thread Safety:
    All functions in this module are thread-safe.
"""
import subprocess
import logging
import time
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# Add a null handler as fallback to prevent I/O errors during cleanup
null_handler = logging.NullHandler()
logger.addHandler(null_handler)

# Safe logging helper
def safe_log(message, level=logging.INFO):
    """Safely log messages, catching I/O errors during teardown."""
    try:
        logger.log(level, message)
    except (ValueError, OSError, IOError):
        # Ignore logging errors during teardown
        pass
    except Exception:
        # Ignore any other logging errors
        pass

class CommandError(Exception):
    """Custom exception for command execution errors."""
    def __init__(self, message, returncode=None, stdout=None, stderr=None):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

def run_command(
    cmd: List[str],
    timeout: int = 10,
    retries: int = 1,
    retry_delay: float = 0.5,
    check: bool = True,
    log_errors: bool = True
) -> Tuple[int, str, str]:
    """
    Executes a shell command with timeout, retries, and robust error handling.

    Args:
        cmd: Command and arguments as a list of strings.
        timeout: Command timeout in seconds.
        retries: Number of retry attempts for transient failures.
        retry_delay: Delay in seconds between retries.
        check: If True, raises CommandError on non-zero return codes.
        log_errors: If True, logs errors.

    Returns:
        A tuple of (return_code, stdout, stderr).

    Raises:
        CommandError: If the command fails after all retries and `check` is True.
        FileNotFoundError: If the command executable is not found (not retryable).
        PermissionError: If command execution is denied (not retryable).
    """
    cmd_str = ' '.join(cmd)
    last_exception = None

    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # We will check the return code manually
            )

            if result.returncode == 0:
                return result.returncode, result.stdout.strip(), result.stderr.strip()

            error_message = (
                f"Command '{cmd_str}' failed with code {result.returncode}. "
                f"Stderr: {result.stderr.strip()}"
            )
            last_exception = CommandError(
                error_message,
                returncode=result.returncode,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip()
            )

        except subprocess.TimeoutExpired as e:
            error_message = f"Command '{cmd_str}' timed out after {timeout}s."
            last_exception = CommandError(error_message, stderr=str(e))
        except FileNotFoundError as e:
            error_message = f"Command not found: '{cmd[0]}'. Please ensure it is installed and in the PATH."
            if log_errors:
                safe_log(error_message, logging.ERROR)
            raise e
        except PermissionError as e:
            error_message = f"Permission denied executing '{cmd_str}'. Check file permissions."
            if log_errors:
                safe_log(error_message, logging.ERROR)
            raise e
        except Exception as e:
            error_message = f"An unexpected error occurred while running '{cmd_str}': {e}"
            last_exception = CommandError(error_message, stderr=str(e))

        if attempt < retries:
            if log_errors:
                safe_log(f"{last_exception} Retrying in {retry_delay}s... (Attempt {attempt + 1}/{retries})", logging.WARNING)
            time.sleep(retry_delay)
        else:
            if log_errors:
                safe_log(f"Command '{cmd_str}' failed after {retries} retries. Final error: {last_exception}", logging.ERROR)
            if check:
                raise last_exception
            return last_exception.returncode or -1, last_exception.stdout or "", last_exception.stderr or str(last_exception)
    
    # Fallback, should not be reached
    final_error = last_exception or CommandError(f"Command '{cmd_str}' failed for an unknown reason.")
    if check:
        raise final_error
    return -1, "", str(final_error)