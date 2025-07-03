# Comprehensive Test Fix Plan

## Overview

This document outlines the plan to fix all failing tests in the Wildfire Watch project. The goal is to have a fully passing test suite across all supported Python versions (3.8, 3.10, 3.12), with no skipped tests (except for hardware-specific tests when the hardware is not present).

## Phase 1: Initial Triage and Critical Fixes

### 1.1. Fix `ModuleNotFoundError: No module named 'hailo_platform'`

-   **File:** `tests/test_hailo_api_inspection.py`
-   **Problem:** The test unconditionally imports `hailo_platform`, causing a `ModuleNotFoundError` and crashing the test runner when Hailo is not installed.
-   **Fix:** Add a `pytest.mark.skipif` to the test to skip it if the `hailo_platform` module cannot be imported. This will prevent the test runner from crashing and allow other tests to run.
-   **Status:** ⏳ PENDING

### 1.2. Resolve "no tests ran" for Python 3.10 and 3.8

-   **Files:** `scripts/run_tests_by_python_version.sh`, `pytest-python310.ini`, `pytest-python38.ini`
-   **Problem:** The test script reports that no tests were run for Python 3.10 and 3.8. This is likely due to a misconfiguration in the `pytest` ini files or the test discovery mechanism.
-   **Fix:**
    1.  Investigate the `testpaths` and `python_files` configuration in the `pytest-python*.ini` files.
    2.  Ensure that the `run_tests_by_python_version.sh` script is correctly identifying and running the tests for each Python version.
    3.  Verify that the tests intended for these Python versions have the correct markers.
-   **Status:** ⏳ PENDING

### 1.3. Address `INTERNAL_MOCKING_VIOLATIONS_REPORT.md`

-   **Files:** `tests/test_new_features.py`, `tests/test_core_logic.py`
-   **Problem:** These files mock `paho.mqtt.client`, which violates the project's testing guidelines.
-   **Fix:**
    1.  Remove the `patch('consensus.mqtt.Client')` and `patch('trigger.mqtt.Client')` from the tests.
    2.  Use the `test_mqtt_broker` fixture to get a real MQTT broker for testing.
    3.  Update the code to use the real MQTT client provided by the fixture.
-   **Status:** ⏳ PENDING

## Phase 2: Restoring Test Functionality

### 2.1. Fix `test_camera_detector.py`

-   **Problem:** These tests are failing due to a variety of issues, including race conditions and incorrect mocking.
-   **Fix:**
    1.  Review the tests and the `camera_detector` service to identify the root causes of the failures.
    2.  Replace any incorrect mocks with real implementations or appropriate test doubles.
    3.  Add synchronization mechanisms (e.g., `Event`, `Queue`) to handle race conditions.
-   **Status:** ⏳ PENDING

### 2.2. Fix `test_consensus.py`

-   **Problem:** The consensus tests are failing due to timing issues and incorrect state management.
-   **Fix:**
    1.  Refactor the tests to be more deterministic.
    2.  Use the `mqtt_topic_factory` to ensure topic isolation.
    3.  Ensure that the `FireConsensus` service is properly initialized and cleaned up in each test.
-   **Status:** ⏳ PENDING

### 2.3. Fix `test_trigger.py`

-   **Problem:** The trigger tests are failing due to issues with the GPIO simulation and state machine logic.
-   **Fix:**
    1.  Improve the GPIO simulation to be more realistic.
    2.  Add tests for all possible state transitions in the trigger's state machine.
    3.  Ensure that the `max_runtime` and other safety features are working correctly.
-   **Status:** ⏳ PENDING

## Phase 3: Final Verification

### 3.1. Run all tests

-   **Problem:** Ensure all tests pass on all supported Python versions.
-   **Fix:**
    1.  Run `./scripts/run_tests_by_python_version.sh --all --timeout 1800`.
    2.  Analyze the results and fix any remaining failures.
-   **Status:** ⏳ PENDING

### 3.2. Final Report

-   **Problem:** Document the fixes and the final test status.
-   **Fix:**
    1.  Create a `TEST_FIX_FINAL_SUMMARY.md` file with a summary of the changes.
    2.  Update this plan to mark all items as complete.
-   **Status:** ⏳ PENDING