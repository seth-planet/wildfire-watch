"""
Pytest plugin for strict test collection control.
This plugin ensures tests are only collected from allowed directories.
"""

import os
import sys
from pathlib import Path
import pytest

# Define allowed test directories based on Python version
PYTHON_VERSION_TEST_DIRS = {
    "3.12": ["tests"],  # Only collect from tests/ for Python 3.12
    "3.10": ["tests", "converted_models/tests"],  # Python 3.10 can access YOLO-NAS tests
    "3.8": ["tests"],   # Python 3.8 for Coral TPU tests
}

# Directories to always ignore
ALWAYS_IGNORE = {
    "tmp",
    "output", 
    "scripts",
    "demo_output",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "venv",
    ".venv",
    "certs",
    "docs",
    "mosquitto_data",
}


class StrictCollectionPlugin:
    """Plugin to enforce strict test collection rules."""
    
    def __init__(self, config):
        self.config = config
        self.project_root = Path.cwd()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.allowed_dirs = set(PYTHON_VERSION_TEST_DIRS.get(self.python_version, ["tests"]))
        
    def pytest_ignore_collect(self, collection_path, config):
        """
        Return True to prevent collection of the given path.
        This is called for every directory and file pytest encounters.
        """
        path = Path(collection_path)
        
        # Quick check: if it's not a Python file or directory, ignore
        if path.is_file() and not path.suffix == '.py':
            return True
            
        # Get relative path from project root
        try:
            rel_path = path.relative_to(self.project_root)
        except ValueError:
            # Path is outside project root, ignore it
            return True
            
        # Convert to string for easier checking
        rel_str = str(rel_path)
        path_parts = rel_path.parts
        
        # Check if any part of the path is in the always-ignore list
        if any(part in ALWAYS_IGNORE for part in path_parts):
            return True
            
        # For Python 3.12, strictly enforce only collecting from tests/
        if self.python_version == "3.12":
            # Must start with 'tests/' or be 'tests' itself
            if not (rel_str == 'tests' or rel_str.startswith('tests/')):
                return True
                
            # Special case: ignore YOLO-NAS even if it's somehow in tests/
            if 'YOLO-NAS-pytorch' in rel_str:
                return True
                
            # Ignore converted_models even if referenced
            if 'converted_models' in path_parts:
                return True
                
        # Check allowed directories
        first_part = path_parts[0] if path_parts else ""
        if first_part and first_part not in self.allowed_dirs:
            # Not in allowed directories, ignore it
            return True
            
        return False
        
    def pytest_collection_modifyitems(self, config, items):
        """
        Called after collection has been performed.
        Can be used to filter or re-order the items.
        """
        # Additional safety: remove any items that shouldn't have been collected
        filtered_items = []
        
        for item in items:
            # Get the file path of the test
            test_file = Path(item.fspath)
            
            # Apply same rules as ignore_collect
            if self.should_skip_item(test_file):
                continue
                
            filtered_items.append(item)
            
        # Update items in-place
        items[:] = filtered_items
        
        # Log what we collected
        if config.option.verbose:
            print(f"\nCollected {len(items)} tests for Python {self.python_version}")
            if items:
                unique_files = set(str(item.fspath.relative_to(self.project_root)) for item in items)
                print(f"From files: {sorted(unique_files)[:5]}...")  # Show first 5
                
    def should_skip_item(self, test_file):
        """Check if a test item should be skipped based on its file path."""
        try:
            rel_path = test_file.relative_to(self.project_root)
            rel_str = str(rel_path)
            
            # For Python 3.12, only allow tests from tests/ directory
            if self.python_version == "3.12":
                if not rel_str.startswith('tests/'):
                    return True
                    
                # Double-check for sneaky paths
                if any(ignore in rel_str for ignore in ['converted_models', 'tmp/', 'output/', 'scripts/']):
                    return True
                    
            return False
            
        except ValueError:
            # File is outside project root
            return True


def pytest_configure(config):
    """Register our plugin."""
    config.pluginmanager.register(StrictCollectionPlugin(config), "strict_collection")


def pytest_collection_finish(session):
    """Called after collection has been finished."""
    # Report what was collected
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"\nCollection finished for Python {py_version}: {len(session.items)} tests")
    
    # In verbose mode, show which files tests were collected from
    if session.config.option.verbose > 0:
        test_files = set()
        for item in session.items:
            try:
                rel_path = Path(item.fspath).relative_to(Path.cwd())
                test_files.add(str(rel_path))
            except:
                pass
                
        if test_files:
            print("Tests collected from:")
            for f in sorted(test_files)[:10]:  # Show up to 10 files
                print(f"  - {f}")
            if len(test_files) > 10:
                print(f"  ... and {len(test_files) - 10} more files")