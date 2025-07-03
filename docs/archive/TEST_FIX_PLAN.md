# Comprehensive Test Fix Plan

## Overview
This plan outlines the systematic approach to fix all failing tests, implement missing features, and ensure end-to-end testing with running services.

## Phase 1: Missing Script Implementation

### 1.1 Create diagnose.sh
- System health check script
- Check Docker services status
- Verify MQTT connectivity
- Test camera connections
- Report GPIO pin states
- Check disk space and resources

### 1.2 Create collect_debug.sh
- Collect logs from all services
- Gather system information
- Package debug information
- Include configuration files

## Phase 2: Enhanced Consensus Features

### 2.1 Implement Growing Fire Detection
- Fix fire size tracking in consensus.py
- Ensure proper moving average calculations
- Add object tracking persistence
- Update tests to match implementation

### 2.2 Add Zone-Based Activation
- Add zone configuration to Config
- Implement zone filtering in consensus
- Update environment variables
- Add zone tests

### 2.3 Emergency Bypass Mode
- Add emergency override GPIO pin
- Implement bypass logic in trigger.py
- Add safety tests

## Phase 3: Docker Service Management

### 3.1 Create docker-compose.test.yml
- Lightweight test configuration
- Mock cameras for testing
- Fast startup times
- Resource limits for CI

### 3.2 Service Startup Script
- Check Docker daemon
- Start required services
- Wait for health checks
- Verify connectivity

## Phase 4: Python Version Management

### 4.1 Document Python Requirements
- Update README with Python 3.8 for Coral
- Create Python version detection
- Add compatibility layer
- Update Dockerfiles

### 4.2 Create Version Wrapper
- Detect Coral TPU usage
- Switch Python versions as needed
- Maintain compatibility

## Phase 5: Fix Integration Tests

### 5.1 Service Dependencies
- Start MQTT broker first
- Wait for broker readiness
- Start dependent services
- Verify all connections

### 5.2 Test Environment Setup
- Create test certificates
- Configure test cameras
- Set up test MQTT topics
- Initialize test data

## Phase 6: Systematic Testing

### 6.1 Test Execution Order
1. Unit tests (no services needed)
2. Start Docker services
3. Integration tests
4. End-to-end tests
5. Cleanup

### 6.2 Debugging Protocol
- Capture all logs
- Check service health
- Verify network connectivity
- Monitor resource usage

## Implementation Order

1. **Immediate fixes** (30 min)
   - Create missing scripts
   - Fix Python version documentation

2. **Feature implementation** (2 hours)
   - Growing fire detection
   - Zone-based activation
   - Emergency bypass

3. **Docker setup** (1 hour)
   - Create test compose file
   - Service startup script
   - Health check verification

4. **Test fixes** (2 hours)
   - Fix enhanced consensus tests
   - Fix integration tests
   - Remove all skips

5. **Verification** (30 min)
   - Run full test suite
   - Document results
   - Create CI configuration

## Success Criteria

- All tests pass without skips
- Services start reliably
- Features match documentation
- Python versions handled correctly
- CI/CD ready

## Rollback Plan

If issues arise:
1. Revert to working state
2. Fix one component at a time
3. Test incrementally
4. Document issues found