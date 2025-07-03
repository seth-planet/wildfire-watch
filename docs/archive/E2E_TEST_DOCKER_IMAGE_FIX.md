# E2E Test Docker Image Fix

## Problem
The E2E integration tests are failing because Docker containers are not respecting the `MQTT_TOPIC_PREFIX` environment variable. Services are publishing to non-namespaced topics instead of the expected namespaced topics.

## Root Cause
The Docker images (`wildfire-watch/camera_detector:latest`, etc.) are outdated and don't include the latest code that supports `MQTT_TOPIC_PREFIX`.

## Quick Fix
Rebuild the Docker images before running tests:

```bash
# Rebuild all service images
docker build -t wildfire-watch/camera_detector:latest ./camera_detector
docker build -t wildfire-watch/fire_consensus:latest ./fire_consensus
docker build -t wildfire-watch/gpio_trigger:latest ./gpio_trigger

# Then run the test
python3.12 -m pytest tests/test_integration_e2e_improved.py::TestE2EServiceIntegration::test_health_monitoring -xvs
```

## Permanent Fix
Add image building to the test setup to ensure tests always use the latest code.