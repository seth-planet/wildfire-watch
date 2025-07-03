#!/bin/bash
# Comprehensive Hailo Integration Validation Script

set -e

echo "========================================"
echo "Hailo Integration Validation Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Function to check status
check_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $2${NC}"
        ((FAILED++))
    fi
}

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓ Found: $1${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ Missing: $1${NC}"
        ((FAILED++))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓ Found: $1${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ Missing: $1${NC}"
        ((FAILED++))
    fi
}

# Function for warnings
warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

echo "1. Checking System Requirements"
echo "================================"

# Check Hailo device
if [ -e /dev/hailo0 ]; then
    check_status 0 "Hailo device found at /dev/hailo0"
    
    # Check permissions
    if [ -r /dev/hailo0 ] && [ -w /dev/hailo0 ]; then
        check_status 0 "Hailo device has proper permissions"
    else
        check_status 1 "Hailo device permissions issue"
        echo "  Fix: sudo chmod 666 /dev/hailo0"
    fi
else
    check_status 1 "Hailo device not found"
    warning "Hailo hardware may not be installed"
fi

# Check Python version
if command -v python3.10 &> /dev/null; then
    check_status 0 "Python 3.10 installed"
else
    check_status 1 "Python 3.10 not found"
fi

# Check Docker
if command -v docker &> /dev/null; then
    check_status 0 "Docker installed"
    
    # Check Docker daemon
    if docker info &> /dev/null; then
        check_status 0 "Docker daemon running"
    else
        check_status 1 "Docker daemon not running"
    fi
else
    check_status 1 "Docker not installed"
fi

echo ""
echo "2. Checking Model Files"
echo "======================="

# Check HEF models
check_file "converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef"
check_file "converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8_qat.hef"

# Check ONNX baseline
check_file "converted_models/output/yolo8l_fire_640x640.onnx"

# Check calibration data
if [ -d "converted_models/calibration_data" ]; then
    check_status 0 "Calibration data directory exists"
    FILE_COUNT=$(find converted_models/calibration_data -name "*.npy" | wc -l)
    if [ $FILE_COUNT -gt 0 ]; then
        check_status 0 "Found $FILE_COUNT calibration files"
    else
        check_status 1 "No calibration files found"
    fi
else
    check_status 1 "Calibration data directory missing"
fi

echo ""
echo "3. Checking Conversion Tools"
echo "============================"

# Check conversion scripts
check_file "converted_models/convert_hailo_qat.py"
check_file "converted_models/hailo_utils/inspect_hef.py"
check_file "converted_models/hailo_utils/validate_hef.py"
check_file "converted_models/convert_model.py"

echo ""
echo "4. Checking Test Infrastructure"
echo "==============================="

# Check test files
check_file "tests/test_hailo_accuracy.py"
check_file "tests/test_hailo_e2e_fire_detection.py"
check_file "tests/test_performance_benchmarks.py"
check_file "tests/test_stability_temperature.py"
check_file "tests/hailo_test_utils.py"

# Check test videos
if [ -d "/tmp/wildfire_test_videos" ]; then
    check_status 0 "Test videos directory exists"
    VIDEO_COUNT=$(find /tmp/wildfire_test_videos -name "*.mo*" -o -name "*.mp4" | wc -l)
    if [ $VIDEO_COUNT -gt 0 ]; then
        check_status 0 "Found $VIDEO_COUNT test videos"
    else
        warning "No test videos found - run test to download"
    fi
else
    warning "Test videos not downloaded yet"
fi

echo ""
echo "5. Checking Docker Configuration"
echo "================================"

# Check Docker files
check_file "docker-compose.yml"
check_file "security_nvr/Dockerfile"
check_file ".env.hailo"

# Check if .env is configured for Hailo
if [ -f ".env" ]; then
    if grep -q "FRIGATE_DETECTOR=hailo8l" .env 2>/dev/null; then
        check_status 0 "Environment configured for Hailo"
    else
        warning "Environment not configured for Hailo (FRIGATE_DETECTOR not set)"
        echo "  Fix: cp .env.hailo .env"
    fi
fi

echo ""
echo "6. Checking Documentation"
echo "========================"

# Check documentation files
check_file "docs/hailo_integration_summary.md"
check_file "docs/hailo_deployment_guide.md"
check_file "docs/hailo_model_validation_guide.md"
check_file "docs/hailo_troubleshooting_guide.md"
check_file "docs/hailo_quick_start.md"
check_file "HAILO_INTEGRATION_COMPLETE.md"

echo ""
echo "7. Checking Python Dependencies"
echo "==============================="

# Check if we can import hailo_platform
if python3.10 -c "import hailo_platform" 2>/dev/null; then
    check_status 0 "hailo_platform Python module available"
    
    # Get version
    VERSION=$(python3.10 -c "import hailo_platform; print(hailo_platform.__version__)" 2>/dev/null || echo "unknown")
    echo "  Version: $VERSION"
else
    check_status 1 "hailo_platform Python module not installed"
    echo "  Fix: pip install hailort"
fi

# Check other dependencies
for pkg in numpy opencv-python paho-mqtt; do
    if python3.10 -c "import ${pkg//-/_}" 2>/dev/null; then
        check_status 0 "$pkg available"
    else
        warning "$pkg not installed"
    fi
done

echo ""
echo "8. Quick Functionality Test"
echo "==========================="

# Test HEF inspection
if [ -f "converted_models/hailo_utils/inspect_hef.py" ] && [ -f "converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef" ]; then
    echo "Testing HEF inspection..."
    if python3.10 converted_models/hailo_utils/inspect_hef.py \
        --hef converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef 2>/dev/null | grep -q "Model name"; then
        check_status 0 "HEF inspection tool works"
    else
        check_status 1 "HEF inspection tool failed"
    fi
else
    warning "Cannot test HEF inspection - files missing"
fi

echo ""
echo "9. Service Status Check"
echo "======================="

# Check if services are running
if docker ps --format "table {{.Names}}" | grep -q "mqtt-broker"; then
    check_status 0 "MQTT broker is running"
else
    warning "MQTT broker not running"
fi

if docker ps --format "table {{.Names}}" | grep -q "security-nvr"; then
    check_status 0 "Security NVR (Frigate) is running"
    
    # Check if it's using Hailo
    if docker logs security-nvr 2>&1 | tail -100 | grep -q "hailo"; then
        check_status 0 "Frigate is configured for Hailo"
    else
        warning "Frigate may not be using Hailo detector"
    fi
else
    warning "Security NVR not running"
fi

echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Hailo integration is properly installed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start services: docker-compose up -d"
    echo "2. Monitor detections: docker exec -it mqtt-broker mosquitto_sub -t 'frigate/+/fire' -v"
    echo "3. Check performance: docker logs security-nvr | grep hailo"
else
    echo -e "${RED}❌ Some components are missing or misconfigured.${NC}"
    echo ""
    echo "Please check the failed items above and:"
    echo "1. Review docs/hailo_troubleshooting_guide.md"
    echo "2. Ensure Hailo hardware is properly installed"
    echo "3. Run the setup steps in docs/hailo_deployment_guide.md"
fi

if [ $WARNINGS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Some warnings were found - system may work but check these items.${NC}"
fi

echo ""
echo "For detailed setup instructions, see:"
echo "- docs/hailo_quick_start.md (5-minute guide)"
echo "- docs/hailo_deployment_guide.md (full deployment)"
echo "- docs/hailo_troubleshooting_guide.md (fixing issues)"

exit $FAILED