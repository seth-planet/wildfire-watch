#!/usr/bin/env python3.12
"""
Intelligent test runner that automatically uses correct Python versions.

This script analyzes tests and runs them with the appropriate Python version:
- Python 3.12: Most tests (camera_detector, fire_consensus, etc.)
- Python 3.10: YOLO-NAS and super-gradients tests  
- Python 3.8: Coral TPU and TensorFlow Lite tests

Usage:
    python3.12 scripts/run_tests_with_correct_python.py [options]
    python3.12 scripts/run_tests_with_correct_python.py --all
    python3.12 scripts/run_tests_with_correct_python.py --version 3.12
    python3.12 scripts/run_tests_with_correct_python.py --test tests/test_detect.py
"""

import argparse
import subprocess
import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add tests directory to path for importing pytest plugin
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

try:
    from pytest_python_versions import list_tests_by_python_version, get_python_version_for_test
except ImportError:
    print("Warning: Could not import pytest_python_versions plugin")
    list_tests_by_python_version = lambda: {'3.12': [], '3.10': [], '3.8': []}
    get_python_version_for_test = lambda x: ['3.12']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Python Version Detection and Validation
# ─────────────────────────────────────────────────────────────

def check_python_version_available(version: str) -> Tuple[bool, str]:
    """Check if a specific Python version is available on the system."""
    python_cmd = f"python{version}"
    
    try:
        result = subprocess.run(
            [python_cmd, "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            actual_version = result.stdout.strip()
            return True, actual_version
        else:
            return False, f"Command failed: {result.stderr}"
            
    except FileNotFoundError:
        return False, f"{python_cmd} not found in PATH"
    except subprocess.TimeoutExpired:
        return False, f"{python_cmd} --version timed out"
    except Exception as e:
        return False, f"Error checking {python_cmd}: {e}"

def validate_python_environment():
    """Validate that all required Python versions are available."""
    logger.info("Validating Python environment...")
    
    required_versions = ['3.12', '3.10', '3.8']
    available_versions = {}
    missing_versions = []
    
    for version in required_versions:
        available, info = check_python_version_available(version)
        
        if available:
            available_versions[version] = info
            logger.info(f"✅ Python {version}: {info}")
        else:
            missing_versions.append(version)
            logger.warning(f"❌ Python {version}: {info}")
    
    if missing_versions:
        logger.warning(f"Missing Python versions: {missing_versions}")
        logger.info("Some tests may be skipped due to missing Python versions")
    
    return available_versions, missing_versions

# ─────────────────────────────────────────────────────────────
# Test Execution Engine
# ─────────────────────────────────────────────────────────────

class TestRunner:
    """Manages execution of tests with appropriate Python versions."""
    
    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = Path(working_dir or os.getcwd())
        self.available_versions, self.missing_versions = validate_python_environment()
        self.results = {}
    
    def run_tests_for_version(self, version: str, test_files: List[str], 
                            extra_args: List[str] = None) -> Tuple[bool, str, float]:
        """Run tests for a specific Python version."""
        
        if version not in self.available_versions:
            logger.warning(f"Skipping Python {version} tests - version not available")
            return False, f"Python {version} not available", 0.0
        
        python_cmd = f"python{version}"
        extra_args = extra_args or []
        
        # Build pytest command
        cmd = [
            python_cmd, '-m', 'pytest'
        ] + test_files + [
            '-v',
            '--tb=short',
            f'-m python{version.replace(".", "")}',  # Only run tests marked for this version
        ] + extra_args
        
        logger.info(f"Running Python {version} tests: {' '.join(test_files)}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if success:
                logger.info(f"✅ Python {version} tests passed ({duration:.1f}s)")
            else:
                logger.error(f"❌ Python {version} tests failed ({duration:.1f}s)")
                logger.error("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
                logger.error("STDERR:", result.stderr[-1000:])
            
            return success, result.stdout + result.stderr, duration
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"❌ Python {version} tests timed out ({duration:.1f}s)")
            return False, "Tests timed out after 1 hour", duration
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Error running Python {version} tests: {e}")
            return False, str(e), duration
    
    def run_all_tests(self, extra_args: List[str] = None) -> Dict[str, Tuple[bool, str, float]]:
        """Run all tests with appropriate Python versions."""
        logger.info("Starting comprehensive test run with automatic Python version selection")
        
        # Get tests organized by Python version
        tests_by_version = list_tests_by_python_version()
        
        results = {}
        total_start_time = time.time()
        
        for version in ['3.12', '3.10', '3.8']:
            test_files = tests_by_version.get(version, [])
            
            if not test_files:
                logger.info(f"No tests found for Python {version}")
                continue
            
            if version in self.missing_versions:
                logger.warning(f"Skipping Python {version} - not available")
                results[version] = (False, f"Python {version} not available", 0.0)
                continue
            
            # Run tests for this version
            success, output, duration = self.run_tests_for_version(
                version, test_files, extra_args
            )
            
            results[version] = (success, output, duration)
        
        total_duration = time.time() - total_start_time
        
        # Report summary
        self._report_summary(results, total_duration)
        
        return results
    
    def run_specific_tests(self, test_patterns: List[str], 
                          extra_args: List[str] = None) -> Dict[str, Tuple[bool, str, float]]:
        """Run specific tests with automatic Python version detection."""
        logger.info(f"Running specific tests: {test_patterns}")
        
        # Group test patterns by required Python version
        tests_by_version = {'3.12': [], '3.10': [], '3.8': []}
        
        for pattern in test_patterns:
            # If it's a file pattern, analyze it
            if pattern.endswith('.py') and Path(pattern).exists():
                versions = get_python_version_for_test(pattern)
                for version in versions:
                    if pattern not in tests_by_version[version]:
                        tests_by_version[version].append(pattern)
            else:
                # For test name patterns, add to all versions and let pytest filter
                for version in tests_by_version:
                    tests_by_version[version].append(pattern)
        
        results = {}
        
        for version, test_files in tests_by_version.items():
            if not test_files:
                continue
                
            success, output, duration = self.run_tests_for_version(
                version, test_files, extra_args
            )
            
            results[version] = (success, output, duration)
        
        return results
    
    def run_version_specific_tests(self, version: str, 
                                 extra_args: List[str] = None) -> Tuple[bool, str, float]:
        """Run tests for a specific Python version only."""
        tests_by_version = list_tests_by_python_version()
        test_files = tests_by_version.get(version, [])
        
        if not test_files:
            logger.warning(f"No tests found for Python {version}")
            return True, f"No tests for Python {version}", 0.0
        
        return self.run_tests_for_version(version, test_files, extra_args)
    
    def _report_summary(self, results: Dict[str, Tuple[bool, str, float]], 
                       total_duration: float):
        """Report test execution summary."""
        logger.info("=" * 60)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for success, _, _ in results.values() if success)
        
        for version, (success, output, duration) in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"Python {version}: {status} ({duration:.1f}s)")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} versions passed")
        logger.info(f"Total time: {total_duration:.1f}s")
        
        if passed_tests < total_tests:
            logger.error("Some test versions failed - check logs above")
        else:
            logger.info("All test versions passed! 🎉")

# ─────────────────────────────────────────────────────────────
# Command Line Interface
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Wildfire Watch tests with correct Python versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          # Run all tests with correct Python versions
  %(prog)s --version 3.12                 # Run only Python 3.12 tests
  %(prog)s --test tests/test_detect.py    # Run specific test file
  %(prog)s --test test_yolo_nas           # Run tests matching pattern
  %(prog)s --list                         # List tests by Python version
  %(prog)s --validate                     # Validate Python environment
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all tests with appropriate Python versions')
    
    parser.add_argument('--version', choices=['3.12', '3.10', '3.8'],
                       help='Run tests for specific Python version only')
    
    parser.add_argument('--test', action='append', dest='tests',
                       help='Run specific test file or pattern (can be used multiple times)')
    
    parser.add_argument('--list', action='store_true',
                       help='List tests grouped by Python version')
    
    parser.add_argument('--validate', action='store_true',
                       help='Validate Python environment and exit')
    
    parser.add_argument('--timeout', type=int, default=0,
                       help='Override pytest timeout (0 = no timeout)')
    
    parser.add_argument('--no-cov', action='store_true',
                       help='Disable coverage reporting')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    runner = TestRunner()
    
    if args.validate:
        logger.info("Python environment validation complete")
        return 0
    
    if args.list:
        tests_by_version = list_tests_by_python_version()
        for version, test_files in tests_by_version.items():
            print(f"\nPython {version} tests:")
            for test_file in sorted(test_files):
                print(f"  {test_file}")
        return 0
    
    # Build extra pytest arguments
    extra_args = []
    if args.timeout:
        extra_args.extend(['--timeout', str(args.timeout)])
    elif args.timeout == 0:
        extra_args.extend(['--timeout', '0'])
    
    if args.no_cov:
        extra_args.append('--no-cov')
    
    if args.dry_run:
        extra_args.append('--collect-only')
    
    # Execute tests
    try:
        if args.all:
            results = runner.run_all_tests(extra_args)
            # Return non-zero if any version failed
            return 0 if all(success for success, _, _ in results.values()) else 1
            
        elif args.version:
            success, output, duration = runner.run_version_specific_tests(args.version, extra_args)
            return 0 if success else 1
            
        elif args.tests:
            results = runner.run_specific_tests(args.tests, extra_args)
            return 0 if all(success for success, _, _ in results.values()) else 1
            
        else:
            # Default: run all tests
            results = runner.run_all_tests(extra_args)
            return 0 if all(success for success, _, _ in results.values()) else 1
            
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())