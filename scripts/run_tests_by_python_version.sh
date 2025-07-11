#!/bin/bash
# Intelligent test runner for Wildfire Watch with automatic Python version selection
# This script runs tests with the correct Python version based on test markers and content
# Includes comprehensive cleanup of zombie processes and test artifacts

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
CLEANUP_ONLY=false
SKIP_CLEANUP=false

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to perform comprehensive pre-test cleanup
perform_cleanup() {
    print_status "$BLUE" "🧹 Performing comprehensive test cleanup..."
    
    local total_cleaned=0
    local cleaned_items=""
    
    # 1. Kill lingering mosquitto test processes
    print_status "$YELLOW" "Cleaning up mosquitto test processes..."
    local mosquitto_count=0
    
    # Find mosquitto processes with test patterns
    local mosquitto_pids=$(pgrep -f 'mosquitto.*mqtt_test_' 2>/dev/null || true)
    if [[ -n "$mosquitto_pids" ]]; then
        for pid in $mosquitto_pids; do
            # Get process info before killing
            local proc_info=$(ps -p $pid -o pid,etime,cmd --no-headers 2>/dev/null || true)
            if [[ -n "$proc_info" ]]; then
                # Extract process age (etime format can be [DD-]HH:MM:SS or MM:SS)
                local etime=$(echo "$proc_info" | awk '{print $2}')
                local age_minutes=0
                
                # Parse etime to get age in minutes
                if [[ "$etime" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                    # DD-HH:MM:SS format
                    age_minutes=$((${BASH_REMATCH[1]} * 1440 + ${BASH_REMATCH[2]} * 60 + ${BASH_REMATCH[3]}))
                elif [[ "$etime" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                    # HH:MM:SS format
                    age_minutes=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
                elif [[ "$etime" =~ ^([0-9]+):([0-9]+)$ ]]; then
                    # MM:SS format
                    age_minutes=${BASH_REMATCH[1]}
                fi
                
                # Only kill if older than 5 minutes
                if [[ $age_minutes -gt 5 ]]; then
                    if [[ "$VERBOSE" == "true" ]]; then
                        print_status "$YELLOW" "  Killing mosquitto process $pid (age: $etime)"
                    fi
                    kill -TERM $pid 2>/dev/null || true
                    sleep 0.2
                    kill -KILL $pid 2>/dev/null || true
                    ((mosquitto_count++))
                fi
            fi
        done
    fi
    
    if [[ $mosquitto_count -gt 0 ]]; then
        cleaned_items="${cleaned_items}  - Mosquitto processes: $mosquitto_count\n"
        ((total_cleaned += mosquitto_count))
    fi
    
    # 2. Clean up Python test processes (but protect pytest infrastructure)
    print_status "$YELLOW" "Cleaning up stale Python test processes..."
    local python_count=0
    
    # Protected patterns - never kill these
    local protected_patterns=(
        "pytest"
        "py.test"
        "xdist"
        "execnet"
        "gateway_base"
        "gw[0-9]+"
        "run_tests_by_python_version.sh"
    )
    
    # Find python processes that might be test-related
    for python_cmd in "python" "python3" "python3.12" "python3.10" "python3.8"; do
        local python_pids=$(pgrep -f "^$python_cmd " 2>/dev/null || true)
        if [[ -n "$python_pids" ]]; then
            for pid in $python_pids; do
                # Skip our own process
                [[ $pid -eq $$ ]] && continue
                
                # Get process info
                local proc_info=$(ps -p $pid -o pid,ppid,etime,cmd --no-headers 2>/dev/null || true)
                if [[ -n "$proc_info" ]]; then
                    local cmd=$(echo "$proc_info" | cut -d' ' -f4-)
                    local etime=$(echo "$proc_info" | awk '{print $3}')
                    
                    # Check if it's a protected process
                    local is_protected=false
                    for pattern in "${protected_patterns[@]}"; do
                        if [[ "$cmd" =~ $pattern ]]; then
                            is_protected=true
                            break
                        fi
                    done
                    
                    if [[ "$is_protected" == "false" ]]; then
                        # Check if it's a test-related process
                        if [[ "$cmd" =~ test_ ]] || [[ "$cmd" =~ /tests/ ]] || [[ "$cmd" =~ "import sys;exec" ]]; then
                            # Parse age
                            local age_minutes=0
                            if [[ "$etime" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                                age_minutes=$((${BASH_REMATCH[1]} * 1440 + ${BASH_REMATCH[2]} * 60 + ${BASH_REMATCH[3]}))
                            elif [[ "$etime" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                                age_minutes=$((${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}))
                            elif [[ "$etime" =~ ^([0-9]+):([0-9]+)$ ]]; then
                                age_minutes=${BASH_REMATCH[1]}
                            fi
                            
                            # Only kill if older than 10 minutes
                            if [[ $age_minutes -gt 10 ]]; then
                                if [[ "$VERBOSE" == "true" ]]; then
                                    print_status "$YELLOW" "  Killing Python process $pid (age: $etime)"
                                fi
                                kill -TERM $pid 2>/dev/null || true
                                sleep 0.2
                                kill -KILL $pid 2>/dev/null || true
                                ((python_count++))
                            fi
                        fi
                    fi
                fi
            done
        fi
    done
    
    if [[ $python_count -gt 0 ]]; then
        cleaned_items="${cleaned_items}  - Python processes: $python_count\n"
        ((total_cleaned += python_count))
    fi
    
    # 3. Clean up Docker containers with test labels
    print_status "$YELLOW" "Cleaning up test Docker containers..."
    local container_count=0
    
    if command -v docker &> /dev/null; then
        # Find containers with test labels or patterns
        local test_containers=$(docker ps -aq --filter "label=com.wildfire.test=true" 2>/dev/null || true)
        if [[ -n "$test_containers" ]]; then
            for container in $test_containers; do
                if [[ "$VERBOSE" == "true" ]]; then
                    local name=$(docker inspect --format '{{.Name}}' $container 2>/dev/null || echo "unknown")
                    print_status "$YELLOW" "  Removing container: $name"
                fi
                docker rm -f $container &>/dev/null || true
                ((container_count++))
            done
        fi
        
        # Also clean containers by name patterns
        local patterns=("test-" "e2e-" "mqtt_test_" "wf-gw" "wf-master")
        for pattern in "${patterns[@]}"; do
            local matching=$(docker ps -aq --filter "name=$pattern" 2>/dev/null || true)
            if [[ -n "$matching" ]]; then
                for container in $matching; do
                    if [[ "$VERBOSE" == "true" ]]; then
                        local name=$(docker inspect --format '{{.Name}}' $container 2>/dev/null || echo "unknown")
                        print_status "$YELLOW" "  Removing container: $name"
                    fi
                    docker rm -f $container &>/dev/null || true
                    ((container_count++))
                done
            fi
        done
        
        # Clean up test networks
        local test_networks=$(docker network ls --filter "name=wf-" --format "{{.Name}}" 2>/dev/null || true)
        if [[ -n "$test_networks" ]]; then
            for network in $test_networks; do
                if [[ "$network" =~ ^wf-(gw|master) ]]; then
                    docker network rm $network &>/dev/null || true
                fi
            done
        fi
    fi
    
    if [[ $container_count -gt 0 ]]; then
        cleaned_items="${cleaned_items}  - Docker containers: $container_count\n"
        ((total_cleaned += container_count))
    fi
    
    # 4. Clean up temporary test directories
    print_status "$YELLOW" "Cleaning up temporary test directories..."
    local dir_count=0
    
    # Clean old mqtt_test_ directories
    if [[ -d /tmp ]]; then
        find /tmp -maxdepth 1 -type d -name "mqtt_test_*" -mmin +30 -exec rm -rf {} \; 2>/dev/null || true
        dir_count=$(find /tmp -maxdepth 1 -type d -name "mqtt_test_*" -mmin +30 2>/dev/null | wc -l || echo 0)
    fi
    
    if [[ $dir_count -gt 0 ]]; then
        cleaned_items="${cleaned_items}  - Temp directories: $dir_count\n"
        ((total_cleaned += dir_count))
    fi
    
    # 5. Run Python-based enhanced cleanup if available
    if [[ -f "$PROJECT_ROOT/tests/enhanced_process_cleanup.py" ]]; then
        print_status "$YELLOW" "Running enhanced Python cleanup..."
        python3.12 "$PROJECT_ROOT/tests/enhanced_process_cleanup.py" 2>/dev/null || true
    fi
    
    # Report results
    if [[ $total_cleaned -gt 0 ]]; then
        print_status "$GREEN" "✅ Cleanup complete! Removed $total_cleaned items:"
        echo -e "$cleaned_items"
    else
        print_status "$GREEN" "✅ No cleanup needed - environment is clean"
    fi
    
    return 0
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
        # Use WORKER_COUNT environment variable if set, otherwise default to auto
        WORKER_COUNT="${WORKER_COUNT:-auto}"
        cmd="$cmd -n $WORKER_COUNT"
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
Includes automatic cleanup of zombie processes and test artifacts.

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
    --cleanup-only     Only perform cleanup without running tests
    --skip-cleanup     Skip automatic cleanup before tests
    --help             Show this help message

EXAMPLES:
    $0 --all                           # Run all tests with correct Python versions
    $0 --all --coverage                # Run all tests with coverage analysis
    $0 --python312 --coverage-html     # Run Python 3.12 tests with HTML coverage
    $0 --test tests/test_detect.py     # Run specific test file
    $0 --test test_yolo_nas --verbose  # Run YOLO-NAS tests with verbose output
    $0 --validate                      # Check Python environment
    $0 --cleanup-only                  # Clean up zombie processes and containers
    $0 --dry-run --all --coverage      # Show what would be run with coverage

CLEANUP FEATURES:
    - Automatically kills mosquitto test processes older than 5 minutes
    - Removes Python test processes older than 10 minutes
    - Cleans up Docker containers with test labels
    - Removes temporary test directories older than 30 minutes
    - Protects active pytest worker processes (gw0, gw1, etc.)

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
        --cleanup-only)
            CLEANUP_ONLY=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
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

# Handle cleanup-only mode
if [[ "$CLEANUP_ONLY" == "true" ]]; then
    perform_cleanup
    exit 0
fi

# Validate environment first
validate_environment

# Perform cleanup before tests unless skipped
if [[ "$SKIP_CLEANUP" != "true" ]]; then
    perform_cleanup
    print_status "$BLUE" "======================================================="
fi

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