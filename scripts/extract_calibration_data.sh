#!/bin/bash
# Extract calibration dataset for model converter tests

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONVERTED_MODELS_DIR="$PROJECT_ROOT/converted_models"
ARCHIVE_PATH="$CONVERTED_MODELS_DIR/wildfire_calibration_data.tar.gz"

# Check if archive exists
if [ ! -f "$ARCHIVE_PATH" ]; then
    echo "Error: Calibration archive not found at $ARCHIVE_PATH"
    exit 1
fi

# Check if already extracted
if [ -d "$CONVERTED_MODELS_DIR/calibration_data" ] && [ -d "$CONVERTED_MODELS_DIR/calibration_data_fire" ]; then
    echo "Calibration data already extracted"
    exit 0
fi

# Extract archive
echo "Extracting calibration data..."
cd "$CONVERTED_MODELS_DIR"
tar -xzf wildfire_calibration_data.tar.gz

if [ $? -eq 0 ]; then
    echo "Calibration data extracted successfully"
    echo "  - General dataset: $CONVERTED_MODELS_DIR/calibration_data"
    echo "  - Fire dataset: $CONVERTED_MODELS_DIR/calibration_data_fire"
else
    echo "Error: Failed to extract calibration data"
    exit 1
fi