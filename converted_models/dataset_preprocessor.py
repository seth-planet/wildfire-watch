#!/usr/bin/env python3.10
"""
Dataset Preprocessor for Filtering Invalid Class Indices

This module provides functionality to preprocess YOLO format datasets
to handle class indices that exceed the model's capacity.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class DatasetPreprocessor:
    """Preprocess dataset to filter out invalid class indices"""
    
    def __init__(self, dataset_dir: str, num_classes: int, logger: Optional[logging.Logger] = None):
        """
        Initialize dataset preprocessor
        
        Args:
            dataset_dir: Path to dataset directory
            num_classes: Number of classes the model supports
            logger: Optional logger instance
        """
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes
        self.logger = logger or logging.getLogger(__name__)
        
    def analyze_dataset(self) -> Dict[str, any]:
        """Analyze dataset for class distribution and invalid indices"""
        results = {
            'total_images': 0,
            'total_labels': 0,
            'invalid_labels': 0,
            'class_distribution': {},
            'invalid_class_ids': set(),
            'affected_files': []
        }
        
        # Analyze train and val splits
        for split in ['train', 'val']:
            labels_dir = self.dataset_dir / 'labels' / split
            if not labels_dir.exists():
                continue
                
            label_files = list(labels_dir.glob('*.txt'))
            results['total_images'] += len(label_files)
            
            for label_file in tqdm(label_files, desc=f"Analyzing {split} labels"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # class_id + 4 bbox coords
                            class_id = int(parts[0])
                            results['total_labels'] += 1
                            
                            # Track class distribution
                            results['class_distribution'][class_id] = \
                                results['class_distribution'].get(class_id, 0) + 1
                            
                            # Check if class ID is invalid
                            if class_id >= self.num_classes:
                                results['invalid_labels'] += 1
                                results['invalid_class_ids'].add(class_id)
                                if str(label_file) not in results['affected_files']:
                                    results['affected_files'].append(str(label_file))
                                    
                except Exception as e:
                    self.logger.warning(f"Error reading {label_file}: {e}")
                    
        results['invalid_class_ids'] = sorted(list(results['invalid_class_ids']))
        return results
    
    def create_filtered_dataset(self, output_dir: str, mode: str = 'filter') -> Dict[str, any]:
        """
        Create a filtered version of the dataset
        
        Args:
            output_dir: Directory to save filtered dataset
            mode: 'filter' to remove invalid labels, 'remap' to remap them to valid range
            
        Returns:
            Statistics about the filtering process
        """
        output_path = Path(output_dir)
        stats = {
            'total_images_processed': 0,
            'images_with_valid_labels': 0,
            'images_skipped': 0,
            'labels_filtered': 0,
            'labels_remapped': 0,
            'empty_label_files': 0
        }
        
        # Create output directory structure
        for split in ['train', 'val']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
            
        # Copy dataset.yaml
        src_yaml = self.dataset_dir / 'dataset.yaml'
        if src_yaml.exists():
            shutil.copy(src_yaml, output_path / 'dataset.yaml')
            
        # Process each split
        for split in ['train', 'val']:
            src_labels_dir = self.dataset_dir / 'labels' / split
            src_images_dir = self.dataset_dir / 'images' / split
            
            if not src_labels_dir.exists():
                continue
                
            label_files = list(src_labels_dir.glob('*.txt'))
            
            for label_file in tqdm(label_files, desc=f"Processing {split} split"):
                stats['total_images_processed'] += 1
                
                # Find corresponding image
                image_name = label_file.stem
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_file = None
                
                for ext in image_extensions:
                    candidate = src_images_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                        
                if image_file is None:
                    self.logger.warning(f"No image found for {label_file}")
                    continue
                    
                # Process label file
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    filtered_lines = []
                    has_valid_labels = False
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            
                            if class_id < self.num_classes:
                                # Valid class ID
                                filtered_lines.append(line)
                                has_valid_labels = True
                            elif mode == 'remap':
                                # Remap to last valid class
                                parts[0] = str(self.num_classes - 1)
                                filtered_lines.append(' '.join(parts) + '\n')
                                has_valid_labels = True
                                stats['labels_remapped'] += 1
                            else:
                                # Filter mode - skip this label
                                stats['labels_filtered'] += 1
                                
                    if has_valid_labels:
                        # Copy image and save filtered labels
                        dst_image = output_path / 'images' / split / image_file.name
                        dst_label = output_path / 'labels' / split / label_file.name
                        
                        shutil.copy(image_file, dst_image)
                        with open(dst_label, 'w') as f:
                            f.writelines(filtered_lines)
                            
                        stats['images_with_valid_labels'] += 1
                    else:
                        # No valid labels, skip this image
                        stats['images_skipped'] += 1
                        stats['empty_label_files'] += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing {label_file}: {e}")
                    stats['images_skipped'] += 1
                    
        return stats
    
    def validate_filtered_dataset(self, dataset_dir: str) -> Dict[str, any]:
        """Validate that filtered dataset has no invalid class indices"""
        validator = DatasetPreprocessor(dataset_dir, self.num_classes, self.logger)
        analysis = validator.analyze_dataset()
        
        return {
            'is_valid': len(analysis['invalid_class_ids']) == 0,
            'total_labels': analysis['total_labels'],
            'class_distribution': analysis['class_distribution'],
            'max_class_id': max(analysis['class_distribution'].keys()) if analysis['class_distribution'] else -1
        }


def preprocess_dataset(dataset_dir: str, num_classes: int, output_dir: Optional[str] = None,
                      mode: str = 'filter') -> Dict[str, any]:
    """
    Preprocess dataset to handle invalid class indices
    
    Args:
        dataset_dir: Input dataset directory
        num_classes: Number of classes supported by model
        output_dir: Output directory (defaults to dataset_dir + '_filtered')
        mode: 'filter' or 'remap'
        
    Returns:
        Dictionary with preprocessing results
    """
    logger = logging.getLogger(__name__)
    
    # Set default output directory
    if output_dir is None:
        output_dir = str(Path(dataset_dir).parent / f"{Path(dataset_dir).name}_filtered")
    
    preprocessor = DatasetPreprocessor(dataset_dir, num_classes, logger)
    
    # Analyze original dataset
    logger.info("Analyzing original dataset...")
    analysis = preprocessor.analyze_dataset()
    
    logger.info(f"Dataset analysis:")
    logger.info(f"  Total images: {analysis['total_images']}")
    logger.info(f"  Total labels: {analysis['total_labels']}")
    logger.info(f"  Invalid labels: {analysis['invalid_labels']}")
    logger.info(f"  Invalid class IDs: {analysis['invalid_class_ids']}")
    
    result = {
        'analysis': analysis,
        'filter_stats': None,
        'validation': None,
        'output_dir': output_dir
    }
    
    if analysis['invalid_labels'] > 0:
        # Create filtered dataset
        logger.info(f"Creating filtered dataset in {output_dir}...")
        filter_stats = preprocessor.create_filtered_dataset(output_dir, mode)
        result['filter_stats'] = filter_stats
        
        logger.info(f"Filtering complete:")
        logger.info(f"  Images processed: {filter_stats['total_images_processed']}")
        logger.info(f"  Images with valid labels: {filter_stats['images_with_valid_labels']}")
        logger.info(f"  Images skipped: {filter_stats['images_skipped']}")
        logger.info(f"  Labels filtered: {filter_stats['labels_filtered']}")
        logger.info(f"  Labels remapped: {filter_stats['labels_remapped']}")
        
        # Validate filtered dataset
        logger.info("Validating filtered dataset...")
        validation = preprocessor.validate_filtered_dataset(output_dir)
        result['validation'] = validation
        
        if validation['is_valid']:
            logger.info("✓ Filtered dataset is valid!")
        else:
            logger.error("✗ Filtered dataset still contains invalid class indices!")
    else:
        logger.info("✓ Dataset is already valid, no filtering needed")
        result['output_dir'] = dataset_dir
        
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess YOLO dataset for invalid class indices")
    parser.add_argument("dataset_dir", help="Path to dataset directory")
    parser.add_argument("--num-classes", type=int, default=32, help="Number of classes (default: 32)")
    parser.add_argument("--output-dir", help="Output directory for filtered dataset")
    parser.add_argument("--mode", choices=['filter', 'remap'], default='filter',
                      help="Filter mode: 'filter' removes invalid labels, 'remap' remaps to valid range")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run preprocessing
    result = preprocess_dataset(
        args.dataset_dir, 
        args.num_classes,
        args.output_dir,
        args.mode
    )
    
    # Print summary
    print("\nPreprocessing Summary:")
    print(f"Output directory: {result['output_dir']}")
    if result['filter_stats']:
        print(f"Filtering mode: {args.mode}")
        print(f"Images retained: {result['filter_stats']['images_with_valid_labels']}/{result['analysis']['total_images']}")
        print(f"Labels filtered: {result['filter_stats']['labels_filtered']}")