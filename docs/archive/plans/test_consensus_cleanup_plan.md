# Consensus Test Cleanup Plan

## Current State
We have multiple consensus test files with varying levels of mocking and coverage:

### Files to Keep:
1. **test_consensus_integration.py** (renamed from test_consensus_real_mqtt_fixed.py)
   - 6 integration tests using real MQTT broker
   - Tests multi-camera consensus, cooldown, edge cases
   - Follows integration testing philosophy
   - All tests passing ✅

2. **test_consensus.py**
   - 40+ unit tests covering detailed functionality
   - Currently uses internal mocking (needs fixing)
   - Tests many edge cases not covered in integration tests:
     - Area calculation and validation
     - Camera state tracking details
     - Detection validation edge cases
     - MQTT disconnection/reconnection handling
     - Object tracking cleanup
     - Frigate event processing
     - Health reporting
     - Moving average calculations
     - Growth trend checking

3. **test_consensus_area_calculation.py**
   - Specialized tests for area calculation logic
   - May have unique coverage

### Files to Remove:
1. **test_consensus_enhanced.py** - Original version with heavy mocking
2. **test_consensus_enhanced_fixed.py** - Intermediate broken version
3. **test_consensus_debug.py** - Debug version
4. **debug_consensus_simple.py** - Debug script

## Action Plan:

### Phase 1: Clean up redundant files
```bash
rm tests/test_consensus_enhanced.py
rm tests/test_consensus_enhanced_fixed.py
rm tests/test_consensus_debug.py
rm tests/debug_consensus_simple.py
```

### Phase 2: Fix test_consensus.py
- Remove internal mocking of consensus module
- Use real MQTT broker fixture
- Keep the detailed unit test coverage
- Make it complementary to integration tests

### Phase 3: Verify coverage
- Ensure no unique tests were lost
- Run both test files to verify full coverage
- Document which file tests what

## Test Organization:
- **test_consensus_integration.py**: High-level integration tests with real MQTT
- **test_consensus.py**: Detailed unit tests for edge cases and internals
- Both files should work together for comprehensive coverage