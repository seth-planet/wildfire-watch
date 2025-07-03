# YOLO-NAS Pipeline Fix Plan

## Overview
This plan addresses the remaining issues in the YOLO-NAS training pipeline, focusing on:
1. API discrepancies between tests and actual super-gradients implementation
2. Class numbering issues causing dataset incompatibility
3. Making unified_yolo_trainer.py run to completion

## Phase 1: Verify Current API with Web Search - ✅ COMPLETE
**Objective**: Use web search to verify the correct super-gradients API

### Tasks:
1. Search for current super-gradients documentation
2. Verify correct parameters for:
   - Trainer.train() method
   - Dataset creation APIs
   - Model initialization
   - Loss functions and metrics
3. Document the correct API signatures
4. Create API reference file

**Success Criteria**:
- Accurate API documentation gathered
- Clear understanding of current vs deprecated APIs

## Phase 2: Update Tests to Match Correct API - ⏳ PENDING
**Objective**: Fix all test files to use the verified API

### Tasks:
1. Update test_api_usage.py with correct API calls
2. Fix test_yolo_nas_training_updated.py 
3. Update test_qat_functionality.py
4. Ensure all mocks match actual API signatures
5. Remove any assumptions about API behavior

**Success Criteria**:
- All tests use correct API
- No hardcoded/guessed values
- Tests accurately reflect real usage

## Phase 3: Implement Robust Class Filtering - ✅ COMPLETE
**Objective**: Handle dataset with classes exceeding model capacity

### Tasks:
1. Create dataset preprocessor that filters invalid classes
2. Log statistics about filtered images/labels
3. Implement safe fallback for edge cases
4. Add configuration option to control filtering behavior
5. Create validation report showing filtering results

**Success Criteria**:
- Training can proceed with full dataset
- Clear logging of filtered data
- No CUDA assertion errors

## Phase 4: Fix unified_yolo_trainer.py - ⏳ PENDING
**Objective**: Make the training script run to completion

### Tasks:
1. Run script and document all failures
2. Fix each failure systematically
3. Add error handling for common issues
4. Implement proper cleanup on failure
5. Add progress logging

**Success Criteria**:
- Script runs end-to-end without errors
- Produces valid trained model
- Clear error messages for any issues

## Phase 5: Add Comprehensive Tests - ⏳ PENDING
**Objective**: Ensure all issues are caught by tests

### Tasks:
1. Add tests for class filtering logic
2. Create integration test for full pipeline
3. Add tests for error conditions
4. Test model export and validation
5. Add performance benchmarks

**Success Criteria**:
- 100% test coverage for new code
- All edge cases tested
- Clear test documentation

## Phase 6: Run Full Test Suite - ⏳ PENDING
**Objective**: Verify all fixes work correctly

### Tasks:
1. Run all unit tests
2. Run integration tests
3. Document any remaining failures
4. Fix final issues
5. Create summary report

**Success Criteria**:
- All tests pass
- No skipped tests without documentation
- Pipeline ready for production

## Timeline
- Phase 1: 1 hour (API research)
- Phase 2: 2 hours (Test updates)
- Phase 3: 2 hours (Class filtering)
- Phase 4: 3 hours (Script fixes)
- Phase 5: 2 hours (New tests)
- Phase 6: 1 hour (Final validation)
- Total: ~11 hours

## Progress Notes

### Phase 1 Completion (COMPLETE)
- Verified super-gradients API through web search
- Found that coco_detection_yolo_format_train uses 'images_dir' not 'train_images_dir'
- Discovered cache_annotations parameter causing slow dataset indexing
- Updated unified_yolo_trainer.py to set cache_annotations=False for training
- API is already correct in our implementation

### Phase 3 Completion (COMPLETE)
- Created dataset_preprocessor.py to handle invalid class indices
- Supports two modes: 'filter' (remove invalid labels) and 'remap' (remap to valid range)
- Provides detailed statistics about filtered data
- Can create a clean dataset copy with only valid class indices
- Analysis of actual dataset shows NO invalid class indices (max class = 31)
- CUDA errors must be coming from a different source

### Phase 4 Started (IN PROGRESS)
- Running unified_yolo_trainer.py reveals CUDA assertion errors during training
- Dataset has valid class indices (0-31), so error is elsewhere
- Created test scripts to isolate the issue:
  - Model initialization works fine
  - Forward pass works fine
  - Issue appears to be in loss computation or data augmentation
- SafeDataLoaderWrapper import is failing (missing __init__.py fixed)
- Need to investigate why CUDA errors occur despite valid data