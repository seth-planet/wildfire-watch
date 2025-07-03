# SafeDataLoaderWrapper Production Configuration

## Overview
The SafeDataLoaderWrapper has been enhanced with safety checks to prevent dataset corruption during YOLO-NAS training. It now enforces a maximum invalid class index ratio to catch dataset configuration errors early.

## Production Threshold: 0.1%
For production use, we expect less than 0.1% (0.001) of class indices to be invalid. This stringent threshold ensures:
- High-quality datasets without systematic labeling errors
- Early detection of dataset configuration mismatches
- Prevention of silent data corruption during training

## Configuration

### Default Behavior
```python
# Default: 0.1% threshold
wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32)
```

### Custom Threshold
```python
# For datasets with known issues, can temporarily increase threshold
wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.01)  # 1%
```

### Configuration in unified_yolo_trainer.py
```yaml
dataset:
  max_invalid_class_ratio: 0.001  # Default 0.1% for production
```

## Error Handling
When the threshold is exceeded, SafeDataLoaderWrapper raises a detailed ValueError:
```
ValueError: Too many invalid class indices detected! 
156/10000 (1.56%) exceeds threshold of 0.10%. 
This suggests a dataset configuration error. 
Please check that num_classes=32 matches your dataset. 
Found class indices: [33, 45, 67, 89, 144, ...]
```

## Statistics Reporting
After training, the wrapper reports statistics:
```python
stats = wrapper.get_statistics()
# Returns:
# {
#   'batches_processed': 1000,
#   'total_indices_seen': 16000,
#   'total_invalid_indices': 12,
#   'invalid_ratio': 0.00075,  # 0.075%
#   'max_invalid_ratio': 0.001  # 0.1% threshold
# }
```

## Best Practices

### 1. Clean Datasets First
Before training, validate and clean your dataset:
```bash
python3.10 converted_models/class_index_fixer.py
```

### 2. Monitor Invalid Ratios
Check logs during training for warnings:
```
[WARNING] Batch 42: Found 2/16 invalid class indices. Invalid classes: [144, 255]. Total invalid ratio: 0.0008
```

### 3. Investigate Threshold Violations
If training fails due to threshold:
1. Check dataset.yaml for correct num_classes
2. Validate label files for out-of-range indices
3. Use dataset_preprocessor.py to filter problematic labels
4. Consider if model num_classes matches dataset

### 4. Temporary Workarounds
For legacy datasets with known issues:
```python
# In training config
config['dataset']['max_invalid_class_ratio'] = 0.05  # Allow up to 5% temporarily
```

## Integration with CI/CD
Add dataset validation to your CI pipeline:
```bash
# Validate dataset before training
python3.10 scripts/validate_dataset_classes.py --max-invalid-ratio 0.001
```

## Migration Guide
For existing projects:
1. Run dataset validation to check current invalid ratio
2. Clean dataset if ratio > 0.1%
3. Update training configs to specify threshold explicitly
4. Monitor first training runs for threshold violations

## Performance Impact
- Minimal overhead: ~1-2% slower iteration due to validation
- Memory usage: Negligible (statistics tracking only)
- Worth it for production safety and early error detection