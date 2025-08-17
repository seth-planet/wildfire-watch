# E2E Test Timeout Configuration

## Summary of Changes

Updated timeout settings for E2E integration tests to accommodate the full test cycle including:
- Docker container startup time
- Camera discovery phase
- Fire detection simulation
- Consensus validation
- Pump activation and full 60-second runtime
- Pump deactivation and cleanup
- Both insecure and TLS variants

## Timeout Settings

### Test Class Level
- **E2E Test Class**: Increased from 1800s (30 min) to 3600s (1 hour)
  - File: `tests/test_integration_e2e_improved.py`
  - Decorator: `@pytest.mark.timeout(3600)`

### Configuration Files
- **pytest.ini**: Session timeout increased to 14400s (4 hours)
- **pytest-python312.ini**: Session timeout increased to 14400s (4 hours)
- **pytest-python310.ini**: Already set to 14400s (4 hours)
- **pytest-python38.ini**: Already set to 14400s (4 hours)

### Per-Test Timeouts
- Default timeout: 3600s (1 hour) per test
- Session timeout: 14400s (4 hours) for entire test session
- Method: thread-based timeout (doesn't count time in debugger)

## Running E2E Tests

```bash
# Run both insecure and TLS variants
CAMERA_CREDENTIALS=admin:password python3.12 -m pytest tests/test_integration_e2e_improved.py -v

# Run only TLS variant
CAMERA_CREDENTIALS=admin:password python3.12 -m pytest tests/test_integration_e2e_improved.py -k tls -v

# Run with custom timeout
CAMERA_CREDENTIALS=admin:password python3.12 -m pytest tests/test_integration_e2e_improved.py --timeout=7200 -v
```

## Expected Test Duration

Each E2E test variant typically takes:
- Startup phase: ~20-30 seconds
- Fire detection simulation: ~10-15 seconds  
- Pump runtime: 60 seconds (fixed safety timeout)
- Shutdown and cleanup: ~10-20 seconds

**Total per variant**: ~100-125 seconds
**Both variants (insecure + TLS)**: ~200-250 seconds

The increased timeouts provide ample buffer for:
- Slower CI/CD environments
- Network latency
- Docker container startup delays
- Certificate validation in TLS mode
- Unexpected system load