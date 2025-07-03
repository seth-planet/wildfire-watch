#!/usr/bin/env python3.12
"""
Test script to verify pytest collection is working correctly for each Python version.
"""

import subprocess
import sys
import re
from pathlib import Path

def test_pytest_collection(python_version, config_file):
    """Test pytest collection for a specific Python version."""
    print(f"\n{'='*60}")
    print(f"Testing pytest collection for Python {python_version}")
    print(f"Config file: {config_file}")
    print('='*60)
    
    # Check if config file exists
    if not Path(config_file).exists():
        print(f"❌ Config file {config_file} not found")
        return False
        
    # Run pytest collection
    cmd = [
        f"python{python_version}",
        "-m", "pytest",
        "-c", config_file,
        "--collect-only",
        "--quiet"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse output
        output = result.stdout + result.stderr
        
        # Look for collection summary
        collected_match = re.search(r'(\d+) tests? collected', output)
        error_match = re.search(r'(\d+) errors?', output)
        
        if collected_match:
            num_collected = int(collected_match.group(1))
            print(f"✅ Collected {num_collected} tests")
        else:
            print("❌ No collection summary found")
            num_collected = 0
            
        if error_match:
            num_errors = int(error_match.group(1))
            print(f"❌ {num_errors} collection errors")
        else:
            num_errors = 0
            
        # Check for specific problematic paths
        problematic_paths = [
            "converted_models/YOLO-NAS-pytorch/tests/",
            "tmp/test_",
            "output/test_",
            "scripts/test_"
        ]
        
        problematic_found = []
        for path in problematic_paths:
            if path in output:
                problematic_found.append(path)
                
        if problematic_found:
            print(f"⚠️  Found tests from excluded directories:")
            for path in problematic_found:
                print(f"   - {path}")
                
        # For Python 3.12, ensure NO tests from excluded dirs
        if python_version == "3.12" and problematic_found:
            print("❌ Python 3.12 should not collect from excluded directories!")
            return False
            
        # Success criteria
        success = num_collected > 0 and num_errors == 0
        
        if success:
            print(f"✅ Collection successful: {num_collected} tests, 0 errors")
        else:
            print(f"❌ Collection failed: {num_collected} tests, {num_errors} errors")
            
        # Show sample of collected tests
        if num_collected > 0:
            print("\nSample of collected tests:")
            test_lines = [line for line in output.split('\n') if line.strip() and '::' in line]
            for line in test_lines[:5]:
                print(f"  - {line.strip()}")
            if len(test_lines) > 5:
                print(f"  ... and {len(test_lines) - 5} more")
                
        return success
        
    except subprocess.TimeoutExpired:
        print("❌ Collection timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False


def main():
    """Test all Python version configurations."""
    configs = [
        ("3.12", "pytest-python312.ini"),
        ("3.10", "pytest-python310.ini"),
        ("3.8", "pytest-python38.ini"),
    ]
    
    results = {}
    
    for python_version, config_file in configs:
        # Check if Python version is available
        try:
            subprocess.run(
                [f"python{python_version}", "--version"],
                capture_output=True,
                check=True
            )
            results[python_version] = test_pytest_collection(python_version, config_file)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"\n⚠️  Python {python_version} not available, skipping")
            results[python_version] = None
            
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for version, result in results.items():
        if result is None:
            status = "⚠️  SKIPPED (Python not available)"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"Python {version}: {status}")
        
    # Overall result
    failed = [v for v, r in results.items() if r is False]
    if failed:
        print(f"\n❌ Collection failed for Python versions: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ All pytest collections working correctly!")
        sys.exit(0)


if __name__ == "__main__":
    main()