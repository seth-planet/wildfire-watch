# Coverage Analysis Setup

## Overview
The test runner script has been upgraded to support comprehensive coverage analysis for the entire Wildfire Watch project.

## New Features

### Coverage Options
- `--coverage` - Enable coverage reporting for the entire project
- `--coverage-html` - Generate HTML coverage reports in addition to terminal output

### Usage Examples

```bash
# Run all tests with coverage analysis
CAMERA_CREDENTIALS=admin:S3thrule ./scripts/run_tests_by_python_version.sh --all --coverage

# Run specific Python version with coverage
./scripts/run_tests_by_python_version.sh --python312 --coverage

# Generate HTML coverage reports
./scripts/run_tests_by_python_version.sh --all --coverage-html

# Run with custom timeout and coverage
./scripts/run_tests_by_python_version.sh --all --coverage --timeout 1800
```

### Coverage Configuration
A `.coveragerc` file has been created with:
- **Full project coverage** - Covers all source files automatically
- **Intelligent exclusions** - Excludes test files, temporary directories, virtual environments
- **Parallel support** - Works correctly with pytest-xdist parallel execution
- **Detailed reporting** - Shows missing lines and precise coverage percentages

### HTML Reports
When using `--coverage-html`, reports are generated in:
- `htmlcov_python312/` - Coverage for Python 3.12 tests
- `htmlcov_python310/` - Coverage for Python 3.10 tests  
- `htmlcov_python38/` - Coverage for Python 3.8 tests

### Benefits
1. **Automatic inclusion** - New modules are automatically included in coverage
2. **Version-specific reports** - See coverage per Python version
3. **Missing line detection** - Identify exactly which lines need tests
4. **Parallel-safe** - Works with multi-worker test execution

## Next Steps
1. Run coverage analysis to identify untested code
2. Review HTML reports for detailed line-by-line coverage
3. Create tests for uncovered functions and edge cases
4. Monitor coverage trends over time