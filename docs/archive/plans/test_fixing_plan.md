# Comprehensive Test Fixing Plan

## Overview
This plan outlines the process for identifying, analyzing, and fixing all failing tests in the `wildfire-watch` repository. The goal is to achieve a fully passing test suite with no skipped or timed-out tests, adhering to the project's best practices.

## Phases

### Phase 1: Initial Test Run & Analysis - ⏳ PENDING
- [ ] Run all tests using `scripts/run_tests_by_python_version.sh --all --timeout 1800` to establish a baseline of failing tests.
- [ ] Analyze the test results to identify all failures, errors, and timeouts.
- [ ] Populate the "Failing Tests" section below with the initial list.

### Phase 2: Iterative Test Fixing - ⏳ PENDING
- This phase will be a series of steps, one for each failing test identified in Phase 1. Each step will be documented below.

### Phase 3: Final Verification - ⏳ PENDING
- [ ] Run the full test suite one last time with `scripts/run_tests_by_python_version.sh --all --timeout 1800`.
- [ ] Confirm that all tests pass and there are no failures, errors, or timeouts.
- [ ] Provide a final summary report.

## Failing Tests
- [ ] `tests/test_trigger.py`

