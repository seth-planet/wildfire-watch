# Hailo Test Files Summary

This directory contains the essential test files for Hailo-8L integration:

## Core Test Files

### test_hailo_nms_working.py
**Status**: ✅ Working
- The main working test that demonstrates successful fire detection
- Uses InferVStreams API pattern with proper NMS output handling
- Handles the 32-class YOLO model correctly (fire is class 26)
- Includes visualization of detection results
- Current performance: ~64ms inference time

### test_hailo_accuracy.py
**Status**: Ready to test
- Compares accuracy between ONNX and HEF models
- Validates that quantization doesn't degrade accuracy beyond 2%
- Uses the same test videos for consistent comparison

### test_hailo_e2e_fire_detection.py
**Status**: Ready to test (needs update for NMS format)
- End-to-end fire detection test with MQTT integration
- Tests the complete pipeline from video input to fire detection output
- Includes consensus logic testing

### test_hailo_fire_detection_final.py
**Status**: Ready to test (needs update for NMS format)
- Comprehensive fire detection test with performance metrics
- Tests long-running stability and temperature monitoring
- Includes detailed performance benchmarking

## Running the Tests

```bash
# Run the working NMS test
python3.10 tests/test_hailo_nms_working.py

# Run accuracy comparison (requires both ONNX and HEF models)
python3.10 tests/test_hailo_accuracy.py

# Run end-to-end test
python3.10 tests/test_hailo_e2e_fire_detection.py

# Run comprehensive test
python3.10 tests/test_hailo_fire_detection_final.py
```

## Key Findings

1. **Model Format**: The YOLO model has 32 classes, with fire at index 26
2. **NMS Configuration**: Models must be compiled without --no-nms flag
3. **Output Format**: Nested list structure requiring unwrapping
4. **Performance**: Currently ~64ms per frame (needs optimization for <25ms target)

## Test Utils

All tests use the shared `hailo_test_utils.py` which provides:
- Video download and caching
- Hailo device detection and temperature monitoring
- Common initialization patterns