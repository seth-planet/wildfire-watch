#!/usr/bin/env python3
"""
Check and fix NumPy version for Python 3.10 environment.
Super-gradients requires NumPy <=1.23
This script installs the newest compatible version: 1.23.5
"""
import subprocess
import sys

def check_numpy_version():
    """Check current NumPy version in Python 3.10"""
    try:
        result = subprocess.run(
            ["python3.10", "-c", "import numpy; print(numpy.__version__)"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Current NumPy version in Python 3.10: {version}")
            return version
        else:
            print("NumPy not installed in Python 3.10")
            return None
    except Exception as e:
        print(f"Error checking NumPy version: {e}")
        return None

def fix_numpy_version():
    """Install the newest NumPy version compatible with super-gradients"""
    print("Installing NumPy 1.23.5 (newest version compatible with super-gradients)...")
    try:
        # First uninstall current version
        subprocess.run(
            ["python3.10", "-m", "pip", "uninstall", "-y", "numpy"],
            check=True
        )
        
        # Install compatible version (1.23.5 is the newest that satisfies numpy<=1.23)
        subprocess.run(
            ["python3.10", "-m", "pip", "install", "numpy==1.23.5"],
            check=True
        )
        
        # Verify installation
        new_version = check_numpy_version()
        if new_version and new_version.startswith("1.23"):
            print(f"✅ NumPy successfully installed: {new_version}")
            return True
        else:
            print(f"❌ NumPy version after install: {new_version}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fixing NumPy version: {e}")
        return False

def main():
    print("Checking NumPy compatibility for super-gradients...")
    
    current_version = check_numpy_version()
    if current_version:
        # Check if version is >= 1.26
        major, minor = map(int, current_version.split('.')[:2])
        if major > 1 or (major == 1 and minor >= 26):
            print(f"⚠️  NumPy {current_version} is incompatible with super-gradients (requires <1.26.0)")
            
            response = input("Do you want to install NumPy 1.23.5? (y/n): ")
            if response.lower() == 'y':
                if fix_numpy_version():
                    print("\n✅ NumPy version fixed successfully!")
                else:
                    print("\n❌ Failed to fix NumPy version")
                    sys.exit(1)
            else:
                print("\n⚠️  Skipping NumPy installation. Tests may fail.")
        else:
            print(f"✅ NumPy {current_version} is compatible with super-gradients")
    else:
        print("Installing NumPy 1.23.5...")
        subprocess.run(
            ["python3.10", "-m", "pip", "install", "numpy==1.23.5"],
            check=True
        )

if __name__ == "__main__":
    main()