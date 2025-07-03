# RTSP Test Error Fixes Summary

## Issues Identified

1. **RTSP 401 Unauthorized Errors**: Tests were attempting to connect to real IP addresses (192.168.1.100) which might have actual cameras on the network requiring authentication.

2. **FFMPEG Backend Warnings**: OpenCV's FFMPEG backend was generating warnings when it couldn't connect to RTSP streams.

3. **Process Isolation**: RTSP validation runs in a separate process (ProcessPoolExecutor), so environment variables set in the test process don't affect the worker process.

## Solutions Applied

### 1. Updated Test IPs to Non-Routable Addresses
- Changed test RTSP URLs from `192.168.1.100` to `192.0.2.100` (TEST-NET-1 range)
- Updated files:
  - `tests/test_detect.py`
  - `tests/test_frigate_integration.py`
  - `tests/test_security_nvr_integration.py`
  - `tests/test_integration_e2e.py`

### 2. Added OpenCV Warning Suppression
- Created `suppress_opencv_warnings` fixture in `tests/test_detect.py`
- Sets environment variables:
  - `OPENCV_FFMPEG_LOGLEVEL='quiet'`
  - `OPENCV_LOG_LEVEL='ERROR'`
  - `OPENCV_FFMPEG_CAPTURE_OPTIONS='loglevel;quiet'`

### 3. Updated RTSP Worker Process
- Modified `_rtsp_validation_worker` in `camera_detector/detect.py` to suppress warnings in the worker process
- Added environment variable setup at the beginning of the worker function

### 4. Fixed OpenCV Log Level Compatibility
- Added try/except blocks to handle different OpenCV versions
- Falls back to numeric value (0) if named constants aren't available

## Test Behavior

The tests now:
1. Use non-routable IP addresses that can't have real devices
2. Suppress all OpenCV/FFMPEG warnings during test execution
3. Still test real RTSP validation behavior (timeouts, process isolation)
4. Run cleanly without spurious error messages

## Verification

Run the tests to verify no warnings appear:
```bash
python3.12 -m pytest tests/test_detect.py::TestCameraDiscovery::test_rtsp_validation -xvs
```

The output should show:
- Test passing
- No `[rtsp @ ...] 401 Unauthorized` errors
- No `VIDEOIO(FFMPEG)` warnings
- Clean test execution