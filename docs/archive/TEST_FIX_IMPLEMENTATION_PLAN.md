# Test Fix Implementation Plan

## Overview
This plan outlines the specific steps to fix all failing tests and ensure compliance with integration testing requirements.

## Phase 1: Fix API Usage Tests ⏳ IN PROGRESS

### Task 1.1: Fix test_api_usage.py
- **Status**: ⏳ PENDING
- **Issue**: Heavy mocking of super-gradients components
- **Fix**: Use real super-gradients API with minimal training
- **Actions**:
  1. Create temporary dataset with synthetic images
  2. Use real Trainer with 1 epoch
  3. Test actual model instantiation
  4. Verify dataloader creation
  5. Test with Python 3.10

### Task 1.2: Fix test_yolo_nas_training_updated.py
- **Status**: ⏳ PENDING
- **Issue**: Similar mocking violations
- **Fix**: Real training with minimal epochs
- **Actions**:
  1. Create real dataset structure
  2. Use actual super-gradients models
  3. Test training parameters
  4. Verify checkpoint saving

## Phase 2: Fix RTSP Validation Tests

### Task 2.1: Fix test_rtsp_validation_improved.py
- **Status**: ⏳ PENDING
- **Issue**: Patching internal CameraDetector methods
- **Fix**: Use real camera testing
- **Actions**:
  1. Test with actual network cameras when available
  2. Create mock RTSP server for CI/CD
  3. Use CAMERA_CREDENTIALS from env
  4. No internal method patching

### Task 2.2: Fix test_rtsp_validation_timeout.py
- **Status**: ⏳ PENDING
- **Issue**: Mocking cv2.VideoCapture
- **Fix**: Real timeout testing
- **Actions**:
  1. Test with non-existent hosts
  2. Use process-based timeout implementation
  3. Real network failure scenarios

## Phase 3: Add Hardware Tests

### Task 3.1: Create Coral TPU Tests
- **Status**: ⏳ PENDING
- **Actions**:
  1. Create test_coral_tpu_hardware.py
  2. Test model loading with Python 3.8
  3. Verify inference performance (15-20ms)
  4. Test with real images

### Task 3.2: Create TensorRT Tests
- **Status**: ⏳ PENDING
- **Actions**:
  1. Create test_tensorrt_hardware.py
  2. Test engine creation
  3. Verify GPU inference (8-12ms)
  4. Test batch processing

### Task 3.3: Create Camera Hardware Tests
- **Status**: ⏳ PENDING
- **Actions**:
  1. Create test_camera_hardware.py
  2. Test ONVIF discovery
  3. Validate RTSP streams
  4. Test reconnection logic

## Phase 4: Fix Model Conversion Tests

### Task 4.1: Fix conversion mocking
- **Status**: ⏳ PENDING
- **Issue**: Tests mock conversion process
- **Fix**: Use real conversions with timeout
- **Actions**:
  1. Mark tests as @pytest.mark.slow
  2. Use 60-minute timeout
  3. Test with real calibration data
  4. Verify output models

## Phase 5: Integration Tests

### Task 5.1: Create E2E hardware test
- **Status**: ⏳ PENDING
- **Actions**:
  1. Test full pipeline with real hardware
  2. Camera → AI → MQTT → GPIO
  3. Use available accelerators
  4. Verify complete operation

## Phase 6: Fix Remaining Issues

### Task 6.1: Remove hardcoded credentials
- **Status**: ⏳ PENDING
- **Actions**:
  1. Search all test files
  2. Replace with env var usage
  3. Update documentation

### Task 6.2: Update test markers
- **Status**: ⏳ PENDING
- **Actions**:
  1. Add hardware markers
  2. Add slow test markers
  3. Update pytest.ini

## Phase 7: Validation

### Task 7.1: Run all tests individually
- **Status**: ⏳ PENDING
- **Actions**:
  1. Run each test file with 30-min timeout
  2. Document results
  3. Fix any failures

### Task 7.2: Run comprehensive test suite
- **Status**: ⏳ PENDING
- **Actions**:
  1. ./scripts/run_tests_by_python_version.sh --all
  2. Verify no timeouts
  3. Verify no skipped tests (except Hailo)

## Current Priority: Phase 1 - Fix API Usage Tests

Let's start with test_api_usage.py...