# Wildfire Watch Calibration Dataset

This archive contains balanced calibration images for quantizing Wildfire Watch models.

## Contents

- `calibration_data/` - Balanced general calibration dataset (500 images)
- `calibration_data_fire/` - Fire-focused calibration dataset (300 images)

## Usage

```bash
# Extract the archive
tar -xzf wildfire_calibration_data.tar.gz

# Use with model converter
python3.12 convert_model.py model.pt --calibration-data calibration_data/

# Use fire-specific calibration for fire detection models
python3.12 convert_model.py fire_model.pt --calibration-data calibration_data_fire/
```

## Dataset Information

### General Calibration Dataset
- Source: COCO 2017 train + validation sets
- Images: 500 balanced across all 32 classes
- Fire images: 77 (15.4%)
- Format: JPEG
- Strategy: Balanced sampling with 5x emphasis on fire class
- Purpose: General object detection model calibration with fire awareness

### Fire-Specific Dataset
- Source: COCO 2017 train + validation sets  
- Images: 300 balanced across all classes
- Fire images: 92 (30.7%)
- Format: JPEG
- Strategy: Balanced sampling with 10x emphasis on fire class
- Purpose: Fire detection model calibration

### Class Distribution

#### General Dataset (Top 10 classes):
- Person: 128 images
- Fire: 77 images
- Human face: 46 images
- Car: 38 images
- Child: 32 images
- Bicycle: 22 images
- Truck: 21 images
- Dog: 20 images
- Boat: 19 images
- Weapon: 19 images

#### Fire-Specific Dataset (Top 10 classes):
- Fire: 92 images
- Person: 64 images
- Car: 26 images
- Human face: 25 images
- Child: 16 images
- Vehicle registration plate: 11 images
- Truck: 10 images
- Sheep: 9 images
- Bus: 8 images
- Horse: 8 images

## Key Features

1. **Balanced Representation**: Unlike random sampling, this dataset ensures all 32 COCO classes are represented, preventing bias toward common classes.

2. **Fire Emphasis**: Both datasets prioritize fire images while maintaining diversity:
   - General dataset: 5x more fire images than other classes
   - Fire-specific: 10x more fire images than other classes

3. **Real Fire Images**: Found 10,063 fire images in the full COCO dataset, ensuring authentic fire examples for calibration.

4. **Diverse Non-Fire Images**: Includes all COCO classes to ensure the model can distinguish fire from various other objects.

## Reproducibility

Both datasets were created with random seed 42 for reproducibility:

```bash
# Recreate general dataset
python3.12 create_balanced_calibration.py calibration_data \
    --num-images 500 --fire-emphasis 5 --seed 42

# Recreate fire-specific dataset  
python3.12 create_balanced_calibration.py calibration_data_fire \
    --num-images 300 --fire-emphasis 10 --seed 42
```

## Notes

- Total fire images available in COCO: 10,063
- This provides sufficient real fire examples for effective calibration
- The balanced approach ensures the model learns to distinguish fire from all types of objects, not just common ones

## License

The images are from COCO dataset and follow COCO's licensing terms (Creative Commons Attribution 4.0).