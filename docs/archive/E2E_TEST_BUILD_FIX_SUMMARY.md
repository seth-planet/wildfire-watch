# E2E Test Build Context Fix Summary

## Issue
The E2E test (`test_e2e_fire_detection_full.py`) was failing because:
1. It was building Docker images from individual service directories
2. The extended Frigate image and camera_detector both need access to the shared `utils` module
3. The utils module is in the project root, not in the service directories

## Changes Made

### 1. Updated Frigate/Security NVR Build
- Changed build path from `"security_nvr"` to `project_root`
- Changed dockerfile to `"security_nvr/Dockerfile.extended"`
- Changed image tag to `"wildfire-security-nvr-extended:test"`
- Updated container run to use the new image name

### 2. Created Extended Dockerfile for Camera Detector
- Created `camera_detector/Dockerfile.extended` that:
  - Builds from project root context
  - Copies and installs the utils module
  - Maintains all original functionality

### 3. Updated Camera Detector Build
- Changed build path from `"camera_detector"` to `project_root`
- Changed dockerfile to `"camera_detector/Dockerfile.extended"`
- Changed image tag to `"wildfire-camera-detector-extended:test"`
- Updated container run to use the new image name

### 4. Fixed All Service Build Paths
- Added `project_root` variable at the beginning of `build_docker_images()`
- Updated all service builds to use absolute paths:
  - MQTT: `os.path.join(project_root, "mqtt_broker")`
  - Fire Consensus: `os.path.join(project_root, "fire_consensus")`
  - GPIO Trigger: `os.path.join(project_root, "gpio_trigger")`

## Key Changes in test_e2e_fire_detection_full.py

1. **Line 96**: Added project_root definition at start of build_docker_images()
2. **Lines 101-115**: Updated Frigate build to use extended dockerfile from project root
3. **Lines 144-149**: Updated Camera Detector build to use extended dockerfile from project root
4. **Lines 122, 134**: Updated Fire Consensus and GPIO Trigger to use absolute paths
5. **Lines 279, 368**: Updated container run commands to use new image names

## Result
All Docker images will now be built with the correct context, allowing:
- Extended Frigate to access the utils module for enhanced hardware detection
- Camera Detector to access the utils module for command execution
- Consistent build paths for all services