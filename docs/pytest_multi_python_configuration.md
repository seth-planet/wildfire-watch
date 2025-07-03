# Pytest Multi-Python Version Configuration Guide

## Problem Summary

When running `scripts/run_tests_by_python_version.sh --all`, pytest was collecting tests from directories that should be excluded (like `converted_models/`, `tmp/`, `output/`, `scripts/`), causing hundreds of import errors. This was particularly problematic for Python 3.12 tests, which should only run tests from the `tests/` directory.

## Root Causes

1. **Pytest Collection Behavior**: Pytest's test discovery is aggressive and can override configuration settings
2. **Configuration Precedence**: Command-line arguments can override config file settings
3. **Path Resolution**: Relative vs absolute paths in ignore directives
4. **Multiple Collection Mechanisms**: `testpaths`, `norecursedirs`, `--ignore`, and `collect_ignore_glob` can conflict

## Solution Overview

The solution involves a multi-layered approach to ensure strict directory exclusion:

### 1. Updated pytest-python312.ini Configuration

```ini
[tool:pytest]
# Restrict test discovery to ONLY the tests directory
testpaths = tests

# Exclude directories from recursion
norecursedirs = .* build dist CVS _darcs {arch} *.egg tmp output converted_models scripts __pycache__ venv .venv demo_output certs docs mosquitto_data

# Test file patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Command line options with explicit ignores for all problematic directories
addopts = 
    -v
    --tb=short
    --strict-markers
    --color=yes
    --durations=10
    --maxfail=10
    --ignore=converted_models/
    --ignore=converted_models/YOLO-NAS-pytorch/
    --ignore=converted_models/YOLO-NAS-pytorch/tests/
    --ignore=tmp/
    --ignore=output/
    --ignore=scripts/
    --ignore=demo_output/
    --ignore=certs/
    --ignore=docs/
    --ignore=mosquitto_data/
    --ignore=__pycache__/
    --ignore=.git/
    --ignore=.pytest_cache/
    --ignore=venv/
    --ignore=.venv/
    -m "not (yolo_nas or super_gradients or coral_tpu or tflite_runtime or model_converter or hardware_integration or deployment or python310 or python38)"
```

### 2. Key Configuration Elements

#### testpaths
- **Purpose**: Restricts where pytest starts looking for tests
- **Value**: `tests` - Only look in the tests directory
- **Why it helps**: Prevents pytest from even starting collection in other directories

#### norecursedirs
- **Purpose**: Directories to skip during recursive collection
- **Value**: Includes all directories that should never contain tests
- **Why it helps**: Secondary defense against collecting from wrong directories

#### --ignore in addopts
- **Purpose**: Explicitly ignore specific paths during collection
- **Value**: Comprehensive list of all directories to exclude
- **Why it helps**: Most reliable way to prevent collection from specific paths

#### Marker exclusion (-m)
- **Purpose**: Exclude tests marked for other Python versions
- **Value**: Excludes all non-Python 3.12 markers
- **Why it helps**: Prevents running tests that require different Python versions

### 3. Additional Safeguards

#### Root conftest.py
Created a `conftest.py` in the project root with `pytest_ignore_collect` hook:

```python
def pytest_ignore_collect(collection_path):
    """Programmatically ignore specific directories during test collection."""
    path = Path(collection_path)
    
    # List of directories to completely ignore
    ignore_dirs = {
        'converted_models',
        'tmp',
        'output',
        'scripts',
        'demo_output',
        # ... etc
    }
    
    # Check if any part of the path contains ignored directories
    path_parts = set(rel_path.parts)
    if path_parts.intersection(ignore_dirs):
        return True
```

#### Collection Plugin (Optional)
Created `pytest_collection_plugin.py` for even stricter control:

```python
class StrictCollectionPlugin:
    """Plugin to enforce strict test collection rules."""
    
    def pytest_ignore_collect(self, collection_path, config):
        """Return True to prevent collection of the given path."""
        # Strict enforcement based on Python version
```

### 4. Python Version-Specific Configurations

#### Python 3.10 (pytest-python310.ini)
- Uses `-k` filter to only run specific test patterns
- Uses `-m` to only run tests with Python 3.10 markers
- Still excludes problematic directories

#### Python 3.8 (pytest-python38.ini)
- Similar approach to Python 3.10
- Focuses on Coral TPU and model conversion tests

## Best Practices

1. **Use Multiple Layers**: Combine `testpaths`, `norecursedirs`, and `--ignore` for defense in depth
2. **Be Explicit**: List all directories to ignore explicitly in `--ignore`
3. **Use Markers**: Mark tests with Python version requirements and filter by markers
4. **Test Collection**: Always verify collection with `--collect-only` after changes
5. **Monitor for Drift**: Regularly check that new test files aren't being added to wrong directories

## Verification

Use the provided test script to verify configuration:

```bash
python3.12 scripts/test_pytest_collection.py
```

This will test collection for all Python versions and report any issues.

## Common Issues and Solutions

### Issue: Tests still collected from excluded directories
**Solution**: Add explicit `--ignore` directives in addopts

### Issue: Import errors during collection
**Solution**: Ensure all directories with incompatible code are in ignore lists

### Issue: Tests not found
**Solution**: Check that `testpaths` is set correctly and not overridden

### Issue: Wrong tests run for Python version
**Solution**: Use both `-k` (keyword) and `-m` (marker) filtering

## Conclusion

The key to successful multi-Python pytest configuration is:
1. Restrict where pytest looks (`testpaths`)
2. Explicitly ignore problematic directories (`--ignore`)
3. Filter by markers for Python version-specific tests
4. Verify configuration regularly

This multi-layered approach ensures that each Python version only runs its appropriate tests without collection errors.