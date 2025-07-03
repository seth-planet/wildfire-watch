#!/bin/bash
# Intelligent test runner for Wildfire Watch with automatic Python version selection
# This script runs tests with the correct Python version based on test markers and content

# Note: Don't use 'set -e' here as we need to handle command failures gracefully

# Script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
RUN_ALL=false
RUN_312=false
RUN_310=false
RUN_38=false
SPECIFIC_TESTS=""
EXTRA_ARGS=""
DRY_RUN=false
VERBOSE=false
PARALLEL=true
COVERAGE=false
COVERAGE_REPORT="term"
COVERAGE_HTML=false

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if Python version is available
check_python_version() {
    local version=$1
    local python_cmd="python${version}"
    
    if command -v "$python_cmd" &> /dev/null; then
        local actual_version=$($python_cmd --version 2>&1)
        print_status "$GREEN" "✅ $python_cmd available: $actual_version"
        return 0
    else
        print_status "$RED" "❌ $python_cmd not found"
        return 1
    fi
}

# Function to run tests for a specific Python version
run_python_tests() {
    local version=$1
    local config_file="pytest-python${version//.}.ini"
    local python_cmd="python${version}"
    
    print_status "$BLUE" "Running Python $version tests..."
    
    if ! check_python_version "$version"; then
        print_status "$YELLOW" "Skipping Python $version tests - not available"
        return 0
    fi
    
    if [[ ! -f "$config_file" ]]; then
        print_status "$RED" "Configuration file $config_file not found"
        return 1
    fi
    
    local cmd="$python_cmd -m pytest -c $config_file"
    
    # Add coverage options if enabled
    if [[ "$COVERAGE" == "true" ]]; then
        # Cover the entire project, excluding test files and common non-source directories
        cmd="$cmd --cov=. --cov-config=.coveragerc"
        
        # Add coverage report format
        cmd="$cmd --cov-report=$COVERAGE_REPORT"
        
        # Add HTML report if requested
        if [[ "$COVERAGE_HTML" == "true" ]]; then
            cmd="$cmd --cov-report=html:htmlcov_python${version//.}"
        fi
        
        # Add missing lines in terminal report
        if [[ "$COVERAGE_REPORT" == "term" || "$COVERAGE_REPORT" == "term-missing" ]]; then
            cmd="$cmd --cov-report=term-missing"
        fi
    fi
    
    # Add specific tests if provided
    if [[ -n "$SPECIFIC_TESTS" ]]; then
        cmd="$cmd $SPECIFIC_TESTS"
    fi
    
    # Add extra arguments
    if [[ "$PARALLEL" == "true" ]]; then
        cmd="$cmd -n auto"
    fi

    if [[ -n "$EXTRA_ARGS" ]]; then
        cmd="$cmd $EXTRA_ARGS"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        print_status "$YELLOW" "DRY RUN: $cmd"
        return 0
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        print_status "$BLUE" "Command: $cmd"
    fi
    
    # Execute the command
    if eval "$cmd"; then
        print_status "$GREEN" "✅ Python $version tests passed"
        return 0
    else
        print_status "$RED" "❌ Python $version tests failed"
        return 1
    fi
}

# Function to validate environment
validate_environment() {
    print_status "$BLUE" "Validating Python environment..."
    
    local versions=("3.12" "3.10" "3.8")
    local available_count=0
    
    for version in "${versions[@]}"; do
        if check_python_version "$version"; then
            ((available_count++))
        fi
    done
    
    if [[ $available_count -eq 0 ]]; then
        print_status "$RED" "No required Python versions found!"
        print_status "$RED" "Please install at least one of: python3.12, python3.10, python3.8"
        exit 1
    fi
    
    if [[ $available_count -lt 3 ]]; then
        print_status "$YELLOW" "Note: Some Python versions are missing. Some tests may be skipped."
    fi
    
    # Check for camera credentials
    if [[ -z "${CAMERA_CREDENTIALS:-}" ]]; then
        print_status "$YELLOW" "⚠️  WARNING: CAMERA_CREDENTIALS environment variable not set"
        print_status "$YELLOW" "   E2E camera tests will fail without camera credentials"
        print_status "$YELLOW" "   Set CAMERA_CREDENTIALS=username:password to enable camera tests"
        print_status "$YELLOW" ""
    else
        print_status "$GREEN" "✅ CAMERA_CREDENTIALS configured for E2E camera tests"
    fi
    
    print_status "$GREEN" "Environment validation complete ($available_count/3 versions available)"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run Wildfire Watch tests with correct Python versions automatically.

OPTIONS:
    --all              Run all tests with appropriate Python versions
    --python312        Run Python 3.12 tests only
    --python310        Run Python 3.10 tests only (YOLO-NAS)
    --python38         Run Python 3.8 tests only (Coral TPU)
    --test PATTERN     Run specific test file or pattern
    --validate         Validate Python environment only
    --dry-run          Show what would be run without executing
    --verbose          Enable verbose output
    --timeout N        Set pytest timeout (0 = no timeout)
    --coverage         Enable coverage reporting (covers entire project)
    --coverage-html    Generate HTML coverage report in addition to terminal
    --no-parallel      Disable parallel test execution
    --help             Show this help message

EXAMPLES:
    $0 --all                           # Run all tests with correct Python versions
    $0 --all --coverage                # Run all tests with coverage analysis
    $0 --python312 --coverage-html     # Run Python 3.12 tests with HTML coverage
    $0 --test tests/test_detect.py     # Run specific test file
    $0 --test test_yolo_nas --verbose  # Run YOLO-NAS tests with verbose output
    $0 --validate                      # Check Python environment
    $0 --dry-run --all --coverage      # Show what would be run with coverage

PYTHON VERSION MAPPING:
    Python 3.12: camera_detector, fire_consensus, gpio_trigger, telemetry, MQTT, integration
    Python 3.10: YOLO-NAS training, super-gradients, API usage, QAT functionality  
    Python 3.8:  Coral TPU, TensorFlow Lite, model conversion, hardware, deployment
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --python312)
            RUN_312=true
            shift
            ;;
        --python310)
            RUN_310=true
            shift
            ;;
        --python38)
            RUN_38=true
            shift
            ;;
        --test)
            SPECIFIC_TESTS="$2"
            shift 2
            ;;
        --timeout)
            EXTRA_ARGS="$EXTRA_ARGS --timeout $2"
            shift 2
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --coverage-html)
            COVERAGE=true
            COVERAGE_HTML=true
            shift
            ;;
        --no-parallel)
            PARALLEL=false
            shift
            ;;
        --validate)
            validate_environment
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_status "$RED" "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "$BLUE" "Wildfire Watch Test Runner with Python Version Selection"
print_status "$BLUE" "======================================================="

# Validate environment first
validate_environment

# Track results
declare -a RESULTS
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

# Execute based on options
if [[ "$RUN_ALL" == "true" ]]; then
    print_status "$BLUE" "Running all tests with appropriate Python versions..."
    
    # Run Python 3.12 tests (most tests)
    if run_python_tests "3.12"; then
        RESULTS+=("Python 3.12: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.12: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
    # Run Python 3.10 tests (YOLO-NAS)
    if run_python_tests "3.10"; then
        RESULTS+=("Python 3.10: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.10: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
    # Run Python 3.8 tests (Coral TPU)
    if run_python_tests "3.8"; then
        RESULTS+=("Python 3.8: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.8: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
elif [[ "$RUN_312" == "true" ]]; then
    if run_python_tests "3.12"; then
        RESULTS+=("Python 3.12: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.12: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
elif [[ "$RUN_310" == "true" ]]; then
    if run_python_tests "3.10"; then
        RESULTS+=("Python 3.10: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.10: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
elif [[ "$RUN_38" == "true" ]]; then
    if run_python_tests "3.8"; then
        RESULTS+=("Python 3.8: ✅ PASSED")
        ((SUCCESSFUL_RUNS++))
    else
        RESULTS+=("Python 3.8: ❌ FAILED")
    fi
    ((TOTAL_RUNS++))
    
else
    # Default: run all tests
    print_status "$YELLOW" "No specific version selected, running all tests..."
    RUN_ALL=true
    exec "$0" --all "${EXTRA_ARGS[@]}"
fi

# Report results
print_status "$BLUE" "======================================================="
print_status "$BLUE" "TEST EXECUTION SUMMARY"
print_status "$BLUE" "======================================================="

for result in "${RESULTS[@]}"; do
    if [[ $result == *"✅ PASSED"* ]]; then
        print_status "$GREEN" "$result"
    else
        print_status "$RED" "$result"
    fi
done

print_status "$BLUE" "Overall: $SUCCESSFUL_RUNS/$TOTAL_RUNS test suites passed"

if [[ $SUCCESSFUL_RUNS -eq $TOTAL_RUNS ]]; then
    print_status "$GREEN" "🎉 All test suites passed!"
    exit 0
else
    print_status "$RED" "Some test suites failed - check output above"
    exit 1
fi