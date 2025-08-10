# Test GPIO RPM Reduction Analysis

## Issues Found in test_gpio_rpm_reduction.py

### 1. Missing Import
- **Issue**: Test uses `CONFIG['RPM_REDUCE_PIN']` but doesn't import CONFIG
- **Fix**: Add `from gpio_trigger.trigger import CONFIG`

### 2. Non-existent FIRE_OFF_DELAY Feature
- **Issue**: Test expects automatic RPM reduction after `fire_off_delay - rpm_reduction_lead` seconds
- **Reality**: The implementation has NO fire_off_delay timer mechanism
- **Actual Behavior**: RPM reduction only occurs when:
  - `_shutdown_engine()` is explicitly called
  - Maximum runtime is reached
  - An error state is entered
  - Cleanup is performed

### 3. Environment Variable Mismatch
- **Issue**: Test sets `RPM_REDUCTION_LEAD` environment variable
- **Reality**: Config schema uses `rpm_reduction_duration` (mapped to `RPM_REDUCTION_LEAD` in legacy config)
- **Fix**: Use `RPM_REDUCTION_DURATION` environment variable

### 4. Incorrect Test Expectations
Several tests expect behaviors that don't exist in the implementation:

#### test_rpm_reduction_before_fire_off_delay
- **Expects**: Automatic transition to REDUCING_RPM after timer
- **Reality**: No such timer exists; need to call `_shutdown_engine()` explicitly

#### test_fire_trigger_during_rpm_reduction_cancels_shutdown
- **Expects**: Fire trigger during REDUCING_RPM cancels shutdown
- **Reality**: `handle_fire_trigger()` only works when state is IDLE and refill_complete is True

## Recommendations

### Option 1: Fix the Tests (Recommended)
Update tests to match actual implementation:
1. Import CONFIG properly
2. Remove references to FIRE_OFF_DELAY
3. Use explicit `_shutdown_engine()` calls to trigger RPM reduction
4. Update test expectations to match actual state machine behavior

See `test_gpio_rpm_reduction_fixed.py` for corrected version.

### Option 2: Implement Missing Features
If FIRE_OFF_DELAY is a required feature:
1. Add `fire_off_delay` to ConfigSchema
2. Implement timer in `handle_fire_trigger()` that:
   - Starts countdown when fire trigger received
   - Resets on subsequent fire triggers
   - Triggers RPM reduction `rpm_reduction_lead` seconds before expiry
3. Allow fire triggers to cancel shutdown during REDUCING_RPM state

### Test Isolation Issues
The tests appear to have proper isolation:
- Uses test_mqtt_broker fixture for MQTT isolation
- Uses mqtt_topic_factory for topic namespace isolation
- Proper cleanup in fixture teardown

## Key Implementation Details

### State Machine Flow
```
IDLE → (fire trigger) → PRIMING → STARTING → RUNNING
                                                 ↓
                                           (shutdown)
                                                 ↓
                                          REDUCING_RPM
                                                 ↓
                                            STOPPING
                                                 ↓
                                           REFILLING → IDLE
```

### RPM Reduction Trigger Points
1. Manual shutdown via `_shutdown_engine()`
2. Max runtime exceeded (background monitor)
3. Error state entry
4. Service cleanup

### Configuration Mapping
- Environment: `RPM_REDUCTION_DURATION` → Config: `rpm_reduction_duration` → Legacy: `RPM_REDUCTION_LEAD`
- No `FIRE_OFF_DELAY` configuration exists in current implementation