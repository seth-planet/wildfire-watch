# E2E Coral Frigate Test - Camera Authentication Fix

## Problem
The E2E test `test_e2e_coral_frigate.py` was failing with "401 Unauthorized" errors when trying to access cameras because the camera credentials were not being properly set.

## Root Cause
1. The test was checking for the `CAMERA_CREDENTIALS` environment variable but it wasn't being set
2. The fallback credentials in the code were incorrect ('admin:password' instead of 'admin:S3thrule')

## Solution Implemented

### 1. Added Auto-Setup Fixture
Added a pytest fixture that automatically sets the camera credentials for the test:

```python
@pytest.fixture(autouse=True)
def setup_camera_credentials(self, monkeypatch):
    """Set camera credentials for test"""
    monkeypatch.setenv('CAMERA_CREDENTIALS', 'admin:S3thrule')
```

### 2. Updated Default Credentials
Updated the default credentials in two places to use the correct values:

- In `_discover_and_validate_cameras()`:
  ```python
  camera_creds = os.getenv('CAMERA_CREDENTIALS', 'admin:S3thrule')
  ```

- In `_generate_frigate_config_fixed()`:
  ```python
  camera_creds = os.getenv('CAMERA_CREDENTIALS', 'admin:S3thrule')
  # ...
  else:
      username, password = 'admin', 'S3thrule'  # Default credentials
  ```

## Testing
The test can now be run in two ways:

1. **With environment variable** (original method):
   ```bash
   export CAMERA_CREDENTIALS="admin:S3thrule"
   python3.8 -m pytest tests/test_e2e_coral_frigate.py
   ```

2. **Without environment variable** (now works with fix):
   ```bash
   python3.8 -m pytest tests/test_e2e_coral_frigate.py
   ```

The test will now properly authenticate with the cameras using the correct credentials in both cases.

## Files Modified
- `/home/seth/wildfire-watch/tests/test_e2e_coral_frigate.py`
  - Added `setup_camera_credentials` fixture
  - Updated default credentials in `_discover_and_validate_cameras`
  - Updated default credentials in `_generate_frigate_config_fixed`