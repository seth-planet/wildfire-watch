#!/usr/bin/env python3
"""
Create balanced calibration dataset from COCO train+validation images
Balances across all classes with emphasis on fire
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
import argparse
from collections import defaultdict
from typing import List, Tuple, Set, Optional


# All COCO classes from dataset.yaml
COCO_CLASSES = {
    0: 'Person',
    1: 'Bicycle',
    2: 'Car',
    3: 'Motorcycle',
    4: 'Bus',
    5: 'Truck',
    6: 'Bird',
    7: 'Cat',
    8: 'Dog',
    9: 'Horse',
    10: 'Sheep',
    11: 'Cattle',
    12: 'Bear',
    13: 'Deer',
    14: 'Rabbit',
    15: 'Raccoon',
    16: 'Fox',
    17: 'Skunk',
    18: 'Squirrel',
    19: 'Pig',
    20: 'Chicken',
    21: 'Boat',
    22: 'Vehicle registration plate',
    23: 'Snowmobile',
    24: 'Human face',
    25: 'Armadillo',
    26: 'Fire',
    27: 'Package',
    28: 'Rodent',
    29: 'Child',
    30: 'Weapon',
    31: 'Backpack'
}


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse YOLO format label file"""
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    labels.append((class_id, *bbox))
    except Exception:
        pass
    return labels


def find_image_for_label(label_path: Path, images_dir: Path) -> Optional[Path]:
    """Find corresponding image for a label file"""
    # Try different naming patterns
    label_name = label_path.stem
    
    # Pattern 1: imagename_jpg.rf.hash.txt -> imagename.jpg
    base_name = label_name.split('_jpg')[0] + '.jpg'
    image_path = images_dir / base_name
    if image_path.exists():
        return image_path
    
    # Pattern 2: Full label name with .jpg
    image_path = images_dir / (label_name.replace('_jpg', '.jpg'))
    if image_path.exists():
        return image_path
    
    # Pattern 3: Direct stem + .jpg
    image_path = images_dir / (label_name + '.jpg')
    if image_path.exists():
        return image_path
    
    return None


def collect_images_by_class(
    train_images: Path,
    train_labels: Path,
    val_images: Path,
    val_labels: Path
) -> dict:
    """Collect all images organized by class"""
    
    images_by_class = defaultdict(list)
    total_processed = 0
    
    # Process both train and validation sets
    datasets = [
        ("train", train_images, train_labels),
        ("validation", val_images, val_labels)
    ]
    
    for dataset_name, images_dir, labels_dir in datasets:
        print(f"\nProcessing {dataset_name} dataset...")
        label_files = list(labels_dir.glob("*.txt"))
        print(f"Found {len(label_files)} label files")
        
        for i, label_path in enumerate(label_files):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i}/{len(label_files)} labels...")
            
            labels = parse_yolo_label(label_path)
            if not labels:
                continue
            
            image_path = find_image_for_label(label_path, images_dir)
            if not image_path:
                continue
            
            # Get unique classes in this image
            classes_in_image = set(label[0] for label in labels)
            
            # Add to each class list
            for class_id in classes_in_image:
                if class_id in COCO_CLASSES:
                    images_by_class[class_id].append((image_path, label_path, classes_in_image))
            
            total_processed += 1
    
    print(f"\nTotal images processed: {total_processed}")
    print(f"\nImages per class:")
    for class_id, class_name in sorted(COCO_CLASSES.items()):
        count = len(images_by_class[class_id])
        if count > 0:
            print(f"  {class_id:2d}. {class_name:25s}: {count:6d} images")
    
    return images_by_class


def create_balanced_calibration_dataset(
    images_by_class: dict,
    output_dir: str,
    num_images: int = 500,
    fire_emphasis_factor: int = 10,
    seed: int = 42
) -> Path:
    """Create balanced calibration dataset with fire emphasis"""
    
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate target images per class
    num_classes = len([c for c in images_by_class if len(images_by_class[c]) > 0])
    base_per_class = num_images // num_classes
    
    # Give fire class more representation
    fire_class_id = 26
    fire_target = base_per_class * fire_emphasis_factor
    other_per_class = (num_images - fire_target) // (num_classes - 1) if num_classes > 1 else num_images
    
    print(f"\nTarget distribution:")
    print(f"  Fire class: {fire_target} images")
    print(f"  Other classes: {other_per_class} images each")
    
    selected_images = []
    selected_paths = set()
    class_counts = defaultdict(int)
    
    # First, get all fire images (up to target)
    if fire_class_id in images_by_class:
        fire_images = images_by_class[fire_class_id]
        num_fire = min(fire_target, len(fire_images))
        if num_fire > 0:
            fire_selection = random.sample(fire_images, num_fire)
            for img_data in fire_selection:
                selected_images.append(img_data)
                selected_paths.add(img_data[0])
                for class_id in img_data[2]:
                    class_counts[class_id] += 1
            print(f"\nSelected {num_fire} fire images")
    
    # Then, balance other classes
    remaining_slots = num_images - len(selected_images)
    non_fire_classes = [c for c in images_by_class if c != fire_class_id and len(images_by_class[c]) > 0]
    
    if non_fire_classes and remaining_slots > 0:
        per_class = remaining_slots // len(non_fire_classes)
        
        for class_id in non_fire_classes:
            available = [img for img in images_by_class[class_id] if img[0] not in selected_paths]
            if not available:
                continue
            
            num_to_select = min(per_class, len(available))
            if num_to_select > 0:
                selection = random.sample(available, num_to_select)
                for img_data in selection:
                    selected_images.append(img_data)
                    selected_paths.add(img_data[0])
                    for cid in img_data[2]:
                        class_counts[cid] += 1
    
    # If we still need more images, add randomly
    if len(selected_images) < num_images:
        all_available = []
        for class_images in images_by_class.values():
            all_available.extend([img for img in class_images if img[0] not in selected_paths])
        
        if all_available:
            additional_needed = num_images - len(selected_images)
            additional = random.sample(all_available, min(additional_needed, len(all_available)))
            for img_data in additional:
                selected_images.append(img_data)
                for class_id in img_data[2]:
                    class_counts[class_id] += 1
    
    # Shuffle final selection
    random.shuffle(selected_images)
    
    # Copy selected images
    print(f"\nCreating balanced calibration dataset with {len(selected_images)} images...")
    
    for i, (img_path, label_path, classes) in enumerate(selected_images):
        # Copy image
        ext = img_path.suffix.lower()
        dst_name = f"calib_{i:04d}{ext}"
        dst_path = output_path / dst_name
        shutil.copy2(img_path, dst_path)
        
        if (i + 1) % 100 == 0:
            print(f"Copied {i + 1}/{len(selected_images)} images")
    
    # Create detailed metadata
    class_stats = {}
    for class_id, count in sorted(class_counts.items()):
        if class_id in COCO_CLASSES:
            class_stats[COCO_CLASSES[class_id]] = count
    
    metadata = {
        "dataset": "coco-balanced-calibration",
        "source": "coco-train-and-validation",
        "num_images": len(selected_images),
        "fire_emphasis_factor": fire_emphasis_factor,
        "seed": seed,
        "format": "mixed (jpg/png)",
        "description": "Balanced COCO dataset for model calibration with fire emphasis",
        "class_statistics": class_stats,
        "total_classes_represented": len(class_counts),
        "selection_strategy": "Balanced across all classes with extra emphasis on fire"
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nBalanced calibration dataset created!")
    print(f"Location: {output_path}")
    print(f"Total images: {len(selected_images)}")
    print(f"\nClass distribution (top 10):")
    sorted_stats = sorted(class_stats.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_stats[:10]:
        print(f"  {class_name:25s}: {count:4d}")
    if len(sorted_stats) > 10:
        print(f"  ... and {len(sorted_stats) - 10} more classes")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced calibration dataset from COCO"
    )
    
    parser.add_argument(
        "output_dir",
        help="Output directory for calibration dataset"
    )
    parser.add_argument(
        "--train-images",
        default="~/fiftyone/train_yolo/images/train",
        help="Training images directory"
    )
    parser.add_argument(
        "--train-labels",
        default="~/fiftyone/train_yolo/labels/train",
        help="Training labels directory"
    )
    parser.add_argument(
        "--val-images",
        default="~/fiftyone/train_yolo/images/validation",
        help="Validation images directory"
    )
    parser.add_argument(
        "--val-labels",
        default="~/fiftyone/train_yolo/labels/validation",
        help="Validation labels directory"
    )
    parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=500,
        help="Total number of calibration images (default: 500)"
    )
    parser.add_argument(
        "--fire-emphasis",
        type=int,
        default=10,
        help="How many times more fire images than other classes (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    train_images = Path(os.path.expanduser(args.train_images))
    train_labels = Path(os.path.expanduser(args.train_labels))
    val_images = Path(os.path.expanduser(args.val_images))
    val_labels = Path(os.path.expanduser(args.val_labels))
    
    try:
        # Collect all images by class
        print("Collecting images by class from train and validation sets...")
        images_by_class = collect_images_by_class(
            train_images, train_labels,
            val_images, val_labels
        )
        
        # Create balanced dataset
        create_balanced_calibration_dataset(
            images_by_class,
            args.output_dir,
            args.num_images,
            args.fire_emphasis,
            args.seed
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()