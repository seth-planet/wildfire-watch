#!/usr/bin/env python3.10
"""
Dataset Label Validation Script
Validates YOLO dataset labels to ensure class indices are within valid range
Logs statistics about invalid labels and provides filtering options
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def validate_label_file(label_file: Path, num_classes: int) -> Tuple[bool, List[Dict]]:
    """
    Validate a single label file
    
    Args:
        label_file: Path to label file
        num_classes: Number of valid classes (0 to num_classes-1)
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    is_valid = True
    
    if not label_file.exists():
        return False, [{'error': 'File does not exist'}]
    
    try:
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    issues.append({
                        'line': line_num,
                        'error': f'Invalid format: expected 5 values, got {len(parts)}',
                        'content': line
                    })
                    is_valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID range
                    if class_id < 0 or class_id >= num_classes:
                        issues.append({
                            'line': line_num,
                            'error': f'Class ID {class_id} out of range 0-{num_classes-1}',
                            'class_id': class_id,
                            'content': line
                        })
                        is_valid = False
                    
                    # Validate coordinate ranges
                    if not (0.0 <= x_center <= 1.0):
                        issues.append({
                            'line': line_num,
                            'error': f'x_center {x_center} out of range 0.0-1.0',
                            'content': line
                        })
                        is_valid = False
                    
                    if not (0.0 <= y_center <= 1.0):
                        issues.append({
                            'line': line_num,
                            'error': f'y_center {y_center} out of range 0.0-1.0',
                            'content': line
                        })
                        is_valid = False
                    
                    if not (0.0 < width <= 1.0):
                        issues.append({
                            'line': line_num,
                            'error': f'width {width} out of range 0.0-1.0',
                            'content': line
                        })
                        is_valid = False
                    
                    if not (0.0 < height <= 1.0):
                        issues.append({
                            'line': line_num,
                            'error': f'height {height} out of range 0.0-1.0',
                            'content': line
                        })
                        is_valid = False
                        
                except ValueError as e:
                    issues.append({
                        'line': line_num,
                        'error': f'Value parsing error: {e}',
                        'content': line
                    })
                    is_valid = False
    
    except IOError as e:
        return False, [{'error': f'File read error: {e}'}]
    
    return is_valid, issues


def validate_dataset(dataset_dir: Path, num_classes: int, split: str = 'train') -> Dict:
    """
    Validate entire dataset split
    
    Args:
        dataset_dir: Root dataset directory
        num_classes: Number of valid classes
        split: Dataset split ('train', 'val', 'test')
    
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating {split} dataset in {dataset_dir}")
    
    images_dir = dataset_dir / f'images/{split}'
    labels_dir = dataset_dir / f'labels/{split}'
    
    if not images_dir.exists():
        return {'error': f'Images directory not found: {images_dir}'}
    
    if not labels_dir.exists():
        return {'error': f'Labels directory not found: {labels_dir}'}
    
    results = {
        'total_images': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'missing_labels': 0,
        'valid_files': [],
        'invalid_files': [],
        'class_distribution': {},
        'error_summary': {}
    }
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    results['total_images'] = len(image_files)
    logger.info(f"Found {results['total_images']} images in {images_dir}")
    
    for image_file in image_files:
        label_file = labels_dir / (image_file.stem + '.txt')
        
        if not label_file.exists():
            results['missing_labels'] += 1
            results['invalid_files'].append({
                'image': str(image_file),
                'error': 'No corresponding label file'
            })
            continue
        
        is_valid, issues = validate_label_file(label_file, num_classes)
        
        if is_valid:
            results['valid_images'] += 1
            results['valid_files'].append({
                'image': str(image_file),
                'label': str(label_file)
            })
            
            # Count class distribution for valid files
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if 0 <= class_id < num_classes:
                                    results['class_distribution'][class_id] = results['class_distribution'].get(class_id, 0) + 1
            except (ValueError, IOError):
                pass
        else:
            results['invalid_images'] += 1
            results['invalid_files'].append({
                'image': str(image_file),
                'label': str(label_file),
                'issues': issues
            })
            
            # Count error types
            for issue in issues:
                error_type = issue.get('error', 'Unknown error')
                # Categorize errors
                if 'out of range' in error_type and 'Class ID' in error_type:
                    error_category = 'invalid_class_id'
                elif 'out of range' in error_type:
                    error_category = 'invalid_coordinates'
                elif 'Invalid format' in error_type:
                    error_category = 'format_error'
                else:
                    error_category = 'other_error'
                
                results['error_summary'][error_category] = results['error_summary'].get(error_category, 0) + 1
    
    # Calculate percentages
    if results['total_images'] > 0:
        results['valid_percentage'] = (results['valid_images'] / results['total_images']) * 100
        results['invalid_percentage'] = (results['invalid_images'] / results['total_images']) * 100
    else:
        results['valid_percentage'] = 0
        results['invalid_percentage'] = 0
    
    return results


def print_validation_summary(results: Dict, split: str):
    """Print validation summary"""
    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION SUMMARY - {split.upper()}")
    print(f"{'='*60}")
    
    if 'error' in results:
        print(f"❌ Validation failed: {results['error']}")
        return
    
    print(f"Total images: {results['total_images']}")
    print(f"Valid images: {results['valid_images']} ({results['valid_percentage']:.1f}%)")
    print(f"Invalid images: {results['invalid_images']} ({results['invalid_percentage']:.1f}%)")
    print(f"Missing labels: {results['missing_labels']}")
    
    if results['error_summary']:
        print(f"\nError breakdown:")
        for error_type, count in results['error_summary'].items():
            print(f"  {error_type}: {count}")
    
    if results['class_distribution']:
        print(f"\nClass distribution (valid labels only):")
        for class_id in sorted(results['class_distribution'].keys()):
            count = results['class_distribution'][class_id]
            print(f"  Class {class_id}: {count} instances")
    
    # Show examples of invalid files
    if results['invalid_files'] and len(results['invalid_files']) <= 10:
        print(f"\nInvalid files details:")
        for invalid in results['invalid_files'][:10]:
            print(f"  File: {Path(invalid['image']).name}")
            if 'issues' in invalid:
                for issue in invalid['issues'][:3]:  # Show first 3 issues
                    print(f"    Line {issue.get('line', '?')}: {issue['error']}")
            else:
                print(f"    Error: {invalid.get('error', 'Unknown')}")
    elif len(results['invalid_files']) > 10:
        print(f"\n{len(results['invalid_files'])} invalid files found (showing first few issues)")
        for invalid in results['invalid_files'][:3]:
            print(f"  File: {Path(invalid['image']).name}")
            if 'issues' in invalid:
                for issue in invalid['issues'][:2]:
                    print(f"    Line {issue.get('line', '?')}: {issue['error']}")


def filter_dataset(dataset_dir: Path, num_classes: int, output_dir: Path, split: str = 'train'):
    """
    Create filtered dataset with only valid images/labels
    
    Args:
        dataset_dir: Source dataset directory
        num_classes: Number of valid classes
        output_dir: Output directory for filtered dataset
        split: Dataset split to filter
    """
    logger.info(f"Filtering {split} dataset from {dataset_dir} to {output_dir}")
    
    # Validate first
    results = validate_dataset(dataset_dir, num_classes, split)
    
    if 'error' in results:
        logger.error(f"Validation failed: {results['error']}")
        return
    
    # Create output directories
    output_images = output_dir / f'images/{split}'
    output_labels = output_dir / f'labels/{split}'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy valid files
    import shutil
    copied_count = 0
    
    for valid_file in results['valid_files']:
        try:
            image_src = Path(valid_file['image'])
            label_src = Path(valid_file['label'])
            
            image_dst = output_images / image_src.name
            label_dst = output_labels / label_src.name
            
            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)
            copied_count += 1
            
        except IOError as e:
            logger.error(f"Failed to copy {image_src}: {e}")
    
    logger.info(f"Filtered dataset created: {copied_count} valid images copied")
    logger.info(f"Skipped {results['invalid_images']} invalid images")
    
    # Create summary file
    summary_file = output_dir / f'{split}_filtering_summary.json'
    import json
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Filtering summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate and filter YOLO dataset labels')
    parser.add_argument('dataset_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of classes (default: 32)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to validate (default: train)')
    parser.add_argument('--filter', action='store_true', help='Create filtered dataset with valid images only')
    parser.add_argument('--output-dir', type=str, help='Output directory for filtered dataset')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Try to read num_classes from dataset.yaml if available
    dataset_yaml = dataset_dir / 'dataset.yaml'
    if dataset_yaml.exists():
        try:
            with open(dataset_yaml, 'r') as f:
                config = yaml.safe_load(f)
                if 'nc' in config:
                    detected_classes = config['nc']
                    logger.info(f"Detected {detected_classes} classes from dataset.yaml")
                    if args.num_classes == 32:  # Default value
                        args.num_classes = detected_classes
                        logger.info(f"Using {args.num_classes} classes from dataset.yaml")
        except Exception as e:
            logger.warning(f"Could not read dataset.yaml: {e}")
    
    # Validate dataset
    results = validate_dataset(dataset_dir, args.num_classes, args.split)
    print_validation_summary(results, args.split)
    
    # Filter if requested
    if args.filter:
        if not args.output_dir:
            logger.error("--output-dir required when using --filter")
            sys.exit(1)
        
        output_dir = Path(args.output_dir)
        filter_dataset(dataset_dir, args.num_classes, output_dir, args.split)
    
    # Exit with error code if there are invalid images
    if 'invalid_images' in results and results['invalid_images'] > 0:
        logger.warning(f"Found {results['invalid_images']} invalid images")
        if not args.filter:
            logger.info("Use --filter to create a cleaned dataset")
        sys.exit(1)
    else:
        logger.info("All images have valid labels!")
        sys.exit(0)


if __name__ == '__main__':
    main()