# Camera Detector Test Cleanup Plan

## Current State
We have multiple camera detector test files with different approaches:

### Files to Review:
1. **test_camera_detector_real_mqtt.py** (NEW)
   - Created with real MQTT broker approach
   - 8/12 tests passing
   - Follows integration testing philosophy
   - This is our primary test file going forward

2. **test_camera_detector.py** (OLD) 
   - Extensive mocking of mqtt.Client
   - Violates integration testing philosophy
   - Should be removed or merged

3. **test_detect.py** (OLD)
   - May have additional test coverage
   - Likely has internal mocking
   - Should be reviewed and merged

4. **test_detect_optimized.py**
   - Unknown content, needs review

5. **camera_detector_optimized_fixture.py**
   - Fixture file, may be useful

## Action Plan:

### Phase 1: Analyze Old Tests
- Check test_camera_detector.py for unique tests not in new file
- Check test_detect.py for unique coverage
- Identify any valuable tests to migrate

### Phase 2: Migrate Valuable Tests
- Port unique tests to test_camera_detector_real_mqtt.py
- Ensure they follow integration testing philosophy
- Remove internal mocking

### Phase 3: Clean Up
- Delete test_camera_detector.py
- Delete test_detect.py if fully migrated
- Keep only test_camera_detector_real_mqtt.py

### Phase 4: Fix Remaining Issues
- Fix ONVIF discovery test (currently finding real cameras)
- Fix Frigate config publication test
- Fix TLS test timeout