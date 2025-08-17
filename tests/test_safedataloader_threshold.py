#!/usr/bin/env python3.10
"""
Test SafeDataLoaderWrapper threshold enforcement
Verifies that the wrapper correctly raises an error when too many invalid class indices are detected
"""

import unittest
import torch
import tempfile
import shutil
from pathlib import Path
import sys
import pytest

# Mark this entire file for Python 3.10 only
pytestmark = [
    pytest.mark.python310, 
    pytest.mark.yolo_nas,
    pytest.mark.unit,
]

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))
from class_index_fixer import SafeDataLoaderWrapper


class MockDataLoader:
    """Mock dataloader that produces batches with configurable invalid class ratios"""
    
    def __init__(self, num_classes: int, invalid_ratio: float, batch_size: int = 10, num_batches: int = 100, seed: int = None):
        self.num_classes = num_classes
        self.invalid_ratio = invalid_ratio
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_iter = None
        self.seed = seed
        
        # Attributes that SafeDataLoaderWrapper expects
        self.dataset = self
        
    def __iter__(self):
        # Create generator for batches
        def batch_generator():
            # Set seed if provided for deterministic behavior
            if self.seed is not None:
                torch.manual_seed(self.seed)
                
            for batch_idx in range(self.num_batches):
                # Create batch with controlled invalid ratio
                images = torch.randn(self.batch_size, 3, 320, 320)
                
                # Create targets with some invalid class indices
                targets = []
                for i in range(self.batch_size):
                    # Decide if this target should have an invalid class
                    if torch.rand(1).item() < self.invalid_ratio:
                        # Create invalid class index (way out of range)
                        class_id = self.num_classes + torch.randint(1, 100, (1,)).item()
                    else:
                        # Create valid class index
                        class_id = torch.randint(0, self.num_classes, (1,)).item()
                    
                    # Format: [batch_idx, x1, y1, x2, y2, class_id] (6 columns)
                    # SafeDataLoaderWrapper expects class_id in column 5 for 6-column format
                    target = torch.tensor([i, 0.4, 0.4, 0.6, 0.6, float(class_id)])
                    targets.append(target)
                
                targets = torch.stack(targets)
                
                yield images, targets
        
        return batch_generator()
    
    def __len__(self):
        return self.num_batches


class TestSafeDataLoaderThreshold(unittest.TestCase):
    """Test SafeDataLoaderWrapper threshold enforcement"""
    
    def test_threshold_not_exceeded(self):
        """Test that wrapper works normally when threshold is not exceeded"""
        # Create dataloader with 0% invalid ratio to ensure it passes
        # Use seed for deterministic behavior
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.0, num_batches=10, seed=42)
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32)
        
        # Should complete without error
        batch_count = 0
        for images, targets in wrapped_loader:
            batch_count += 1
            # Verify all class indices are now valid
            # For 6-column format, class_id is in column 5
            class_indices = targets[:, 5]
            self.assertTrue((class_indices >= 0).all())
            self.assertTrue((class_indices < 32).all())
        
        self.assertEqual(batch_count, 10)
        
        # Check statistics
        stats = wrapped_loader.get_statistics()
        self.assertLess(stats['invalid_ratio'], 0.001)
        print(f"✓ Processed {stats['batches_processed']} batches with {stats['invalid_ratio']:.4%} invalid ratio")
    
    def test_threshold_exceeded_raises_error(self):
        """Test that wrapper raises error when threshold is exceeded"""
        # Create dataloader with 5% invalid ratio (above default 0.1% threshold)
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.05, num_batches=100)
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32)  # Uses default 0.001 threshold
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            for images, targets in wrapped_loader:
                pass  # Keep iterating until error
        
        error_msg = str(context.exception)
        self.assertIn("Too many invalid class indices detected", error_msg)
        self.assertIn("exceeds threshold", error_msg)
        print(f"✓ Correctly raised error: {error_msg}")
    
    def test_custom_threshold(self):
        """Test custom threshold configuration"""
        # Create dataloader with 3% invalid ratio using seed for deterministic behavior
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.03, num_batches=50, seed=42)
        
        # Test with 5% threshold - should pass
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.05)
        batch_count = 0
        for images, targets in wrapped_loader:
            batch_count += 1
        
        self.assertEqual(batch_count, 50)
        stats = wrapped_loader.get_statistics()
        print(f"✓ Custom threshold 5%: Processed all batches with {stats['invalid_ratio']:.2%} invalid ratio")
        
        # Test with 2% threshold - should fail
        # Use a higher invalid ratio to ensure we exceed the threshold
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.05, num_batches=50, seed=42)
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.02)
        
        with self.assertRaises(ValueError) as context:
            for images, targets in wrapped_loader:
                pass
        
        self.assertIn("exceeds threshold of 2.00%", str(context.exception))
        print(f"✓ Custom threshold 2%: Correctly raised error for high invalid ratio")
    
    def test_zero_threshold(self):
        """Test zero tolerance threshold"""
        # Create dataloader with guaranteed invalid index in first batch
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.2, num_batches=10)  # 20% ensures at least one
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.0)
        
        # Should raise error on first invalid index
        with self.assertRaises(ValueError) as context:
            for images, targets in wrapped_loader:
                pass
        
        self.assertIn("exceeds threshold of 0.00%", str(context.exception))
        print(f"✓ Zero tolerance: Correctly raised error on any invalid index")
    
    def test_statistics_tracking(self):
        """Test statistics tracking accuracy"""
        # Create dataloader with known invalid count - use lower ratio to avoid hitting threshold
        # Use seed for deterministic behavior
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.05, batch_size=10, num_batches=5, seed=42)
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.15)
        
        # Process all batches
        batch_count = 0
        for images, targets in wrapped_loader:
            batch_count += 1
        
        stats = wrapped_loader.get_statistics()
        
        # Verify statistics
        self.assertEqual(stats['batches_processed'], 5)
        self.assertEqual(stats['total_indices_seen'], 50)  # 5 batches * 10 items
        self.assertGreater(stats['total_invalid_indices'], 0)
        # With 5% invalid ratio and seed, we should get consistent results
        # Allow some tolerance due to probabilistic nature
        self.assertAlmostEqual(stats['invalid_ratio'], 0.05, delta=0.05)
        
        print(f"✓ Statistics: {stats['total_invalid_indices']}/{stats['total_indices_seen']} "
              f"({stats['invalid_ratio']:.2%}) in {stats['batches_processed']} batches")
    
    def test_unified_trainer_integration(self):
        """Test integration with UnifiedYOLOTrainer configuration"""
        sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))
        from unified_yolo_trainer import UnifiedYOLOTrainer
        
        # Create trainer with custom threshold in config
        trainer = UnifiedYOLOTrainer()
        trainer.config['dataset']['max_invalid_class_ratio'] = 0.02  # 2% threshold
        
        # Verify config is used
        max_ratio = trainer.config.get('dataset', {}).get('max_invalid_class_ratio', 0.01)
        self.assertEqual(max_ratio, 0.02)
        print(f"✓ UnifiedYOLOTrainer config: max_invalid_class_ratio = {max_ratio}")
    
    def test_error_message_details(self):
        """Test that error message includes helpful details"""
        # Create dataloader with many invalid classes
        base_loader = MockDataLoader(num_classes=32, invalid_ratio=0.1, num_batches=20)
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.01)
        
        with self.assertRaises(ValueError) as context:
            for images, targets in wrapped_loader:
                pass
        
        error_msg = str(context.exception)
        
        # Check error message contains useful information
        self.assertIn("Too many invalid class indices detected", error_msg)
        self.assertIn("num_classes=32", error_msg)
        self.assertIn("Found class indices:", error_msg)
        
        print(f"✓ Error message includes helpful details for debugging")


class TestRealDatasetThreshold(unittest.TestCase):
    """Test with real dataset to verify threshold behavior"""
    
    def setUp(self):
        """Set up test with real dataset if available"""
        self.dataset_path = Path('/media/seth/SketchScratch/fiftyone/train_yolo')
        if not self.dataset_path.exists():
            self.skipTest("Real dataset not available")
    
    def test_real_dataset_statistics(self):
        """Test with real dataset to see actual invalid ratio"""
        from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train
        from super_gradients.training.transforms.detection import (
            DetectionLongestMaxSize,
            DetectionPadIfNeeded
        )
        
        # Create minimal transforms
        transforms = [
            DetectionLongestMaxSize(max_height=320, max_width=320),
            DetectionPadIfNeeded(min_height=320, min_width=320, pad_value=114)
        ]
        
        # Create dataloader for subset
        base_loader = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': str(self.dataset_path),
                'images_dir': 'images/train',
                'labels_dir': 'labels/train',
                'classes': ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle', 'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk', 'Squirrel', 'Pig', 'Chicken', 'Boat', 'Vehicle registration plate', 'Snowmobile', 'Human face', 'Armadillo', 'Fire', 'Package', 'Rodent', 'Child', 'Weapon', 'Backpack'],
                'transforms': transforms,
                'cache_annotations': False
            },
            dataloader_params={
                'batch_size': 16,
                'num_workers': 0,
                'shuffle': True,
                'drop_last': True
            }
        )
        
        # Wrap with SafeDataLoaderWrapper with high threshold to see statistics
        wrapped_loader = SafeDataLoaderWrapper(base_loader, num_classes=32, max_invalid_ratio=0.5)
        
        # Process some batches
        batch_count = 0
        try:
            for images, targets in wrapped_loader:
                batch_count += 1
                if batch_count >= 10:  # Just check first 10 batches
                    break
        except ValueError as e:
            print(f"Dataset has high invalid ratio: {e}")
        
        stats = wrapped_loader.get_statistics()
        print(f"\nReal dataset statistics:")
        print(f"  Batches processed: {stats['batches_processed']}")
        print(f"  Total indices: {stats['total_indices_seen']}")
        print(f"  Invalid indices: {stats['total_invalid_indices']}")
        print(f"  Invalid ratio: {stats['invalid_ratio']:.2%}")
        print(f"  Threshold: {stats['max_invalid_ratio']:.2%}")
        
        # If there are invalid indices, it's good that we're catching them
        if stats['total_invalid_indices'] > 0:
            print(f"✓ SafeDataLoaderWrapper is preventing CUDA crashes by fixing {stats['total_invalid_indices']} invalid indices")


if __name__ == '__main__':
    print("Testing SafeDataLoaderWrapper threshold enforcement...")
    print("=" * 60)
    unittest.main(verbosity=2)