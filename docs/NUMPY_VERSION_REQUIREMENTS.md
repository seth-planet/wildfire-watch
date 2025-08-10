# NumPy Version Requirements

This document outlines the NumPy version requirements across different Python environments in the Wildfire Watch project.

## Overview

The project uses different Python environments with specific NumPy versions to maintain compatibility with various libraries and hardware:

| Python Version | NumPy Version | Usage | Reason |
|----------------|---------------|-------|---------|
| Python 3.12 | 2.0.2 | Production services | Latest stable version for production |
| Python 3.10 | 1.23.5 | YOLO-NAS training | super-gradients requires numpy<=1.23 |
| Python 3.8 | 1.24.3 | Coral TPU inference | tflite_runtime compatibility |

## Production Environment (Python 3.12)

- **NumPy Version**: 2.0.2
- **Services Using NumPy**:
  - `fire_consensus`: Uses `np.median()` for robust area calculations
- **Key Features Used**:
  - Basic array operations (median, mean, std)
  - Data type support (float32, uint8)
  - No deprecated APIs

## Training Environment (Python 3.10)

- **NumPy Version**: 1.23.5 (newest version compatible with super-gradients)
- **Usage**: YOLO-NAS model training only
- **Constraint**: super-gradients uses deprecated NumPy type aliases (np.int, np.float) removed in NumPy 1.24
- **Isolation**: Completely separate from production environment

## Coral TPU Environment (Python 3.8)

- **NumPy Version**: 1.24.3
- **Usage**: Coral TPU model inference
- **Constraint**: tflite_runtime requires Python 3.8
- **Isolation**: Only used for Coral TPU operations

## Version Management

### Checking NumPy Versions

```bash
# Check production NumPy
python3.12 -c "import numpy; print(numpy.__version__)"  # Should show 2.0.2

# Check training NumPy
python3.10 -c "import numpy; print(numpy.__version__)"  # Should show 1.23.5

# Check Coral TPU NumPy
python3.8 -c "import numpy; print(numpy.__version__)"   # Should show 1.24.3
```

### Fixing Version Issues

If NumPy versions get out of sync:

1. **Production (Python 3.12)**:
   ```bash
   python3.12 -m pip install numpy==2.0.2
   ```

2. **Training (Python 3.10)**:
   ```bash
   ./scripts/fix_numpy_python310.py
   # Or manually:
   python3.10 -m pip install numpy==1.23.5
   ```

3. **Coral TPU (Python 3.8)**:
   ```bash
   python3.8 -m pip install numpy==1.24.3
   ```

## Requirements Files

- `requirements-base.txt`: Specifies `numpy==2.0.2` for production
- `fire_consensus/requirements.txt`: Specifies `numpy>=2.0.2,<2.1.0`
- `converted_models/requirements-yolo-nas.txt`: Handled by super-gradients dependency

## Testing

Run tests with the appropriate Python version:

```bash
# Production tests (Python 3.12)
python3.12 -m pytest tests/test_consensus.py

# YOLO-NAS training tests (Python 3.10)
python3.10 -m pytest tests/test_yolo_nas_training.py

# Coral TPU tests (Python 3.8)
python3.8 -m pytest tests/test_coral_tpu.py
```

## Migration Notes

### NumPy 2.0 Upgrade (Applied)

The production environment was successfully upgraded from NumPy 1.26.4 to 2.0.2:

- **Compatibility**: All production code is compatible with NumPy 2.0
- **Performance**: No performance regressions observed
- **Breaking Changes**: None affecting our codebase
- **Benefits**: Latest bug fixes and performance improvements

### Future Considerations

1. **super-gradients**: Monitor for updates that support NumPy 2.0
2. **tflite_runtime**: Check for Python 3.10+ support
3. **Consolidation**: Eventually move all environments to NumPy 2.x when dependencies allow

## Troubleshooting

### Common Issues

1. **Import Error in Production**:
   - Ensure Python 3.12 is using NumPy 2.0.2
   - Check: `python3.12 -m pip show numpy`

2. **YOLO-NAS Training Failures**:
   - Verify Python 3.10 has NumPy 1.23.5
   - Run: `./scripts/fix_numpy_python310.py`

3. **Test Isolation**:
   - Use pytest configuration files for each Python version
   - Run: `./scripts/run_tests_by_python_version.sh`