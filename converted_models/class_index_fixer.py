#!/usr/bin/env python3.10
"""
Class Index Fixer for YOLO-NAS Training
Fixes the CUDA assertion error by ensuring class indices are within valid range
"""
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

logger = logging.getLogger(__name__)

class ClassIndexValidator:
    """Validates and fixes class indices in YOLO datasets"""
    
    def __init__(self, dataset_dir: str, num_classes: int):
        self.dataset_dir = Path(dataset_dir)
        self.num_classes = num_classes
        self.class_mapping = {}
        
    def analyze_dataset_classes(self) -> Dict[str, Any]:
        """Analyze all class indices in the dataset"""
        logger.info("Analyzing dataset class indices...")
        
        found_classes = set()
        invalid_files = []
        
        for split in ['train', 'validation']:
            labels_dir = self.dataset_dir / 'labels' / split
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                found_classes.add(class_id)
                                
                                if class_id >= self.num_classes:
                                    invalid_files.append({
                                        'file': str(label_file),
                                        'line': line_num,
                                        'class_id': class_id,
                                        'split': split
                                    })
                except Exception as e:
                    logger.error(f"Error reading {label_file}: {e}")
        
        analysis = {
            'found_classes': sorted(found_classes),
            'max_class_id': max(found_classes) if found_classes else -1,
            'invalid_files': invalid_files,
            'total_classes_found': len(found_classes),
            'configured_classes': self.num_classes,
            'is_valid': max(found_classes) < self.num_classes if found_classes else True
        }
        
        logger.info(f"Found {len(found_classes)} unique classes: {sorted(found_classes)[:10]}...")
        logger.info(f"Max class ID: {analysis['max_class_id']}, Configured: {self.num_classes}")
        logger.info(f"Invalid files: {len(invalid_files)}")
        
        return analysis
    
    def create_class_mapping(self, analysis: Dict[str, Any]) -> Dict[int, int]:
        """Create mapping from invalid class IDs to valid ones"""
        if analysis['is_valid']:
            logger.info("Dataset is already valid - no mapping needed")
            return {}
        
        found_classes = analysis['found_classes']
        mapping = {}
        
        # Strategy 1: Direct remapping for out-of-range classes
        for class_id in found_classes:
            if class_id >= self.num_classes:
                # Map to the highest valid class (background/other)
                mapped_id = self.num_classes - 1
                mapping[class_id] = mapped_id
                logger.info(f"Mapping class {class_id} -> {mapped_id}")
        
        return mapping
    
    def fix_dataset_labels(self, class_mapping: Dict[int, int]) -> int:
        """Fix all label files with invalid class indices"""
        if not class_mapping:
            logger.info("No class mapping needed")
            return 0
        
        fixed_count = 0
        
        for split in ['train', 'validation']:
            labels_dir = self.dataset_dir / 'labels' / split
            if not labels_dir.exists():
                continue
                
            for label_file in labels_dir.glob('*.txt'):
                try:
                    # Read original content
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Fix class indices
                    fixed_lines = []
                    file_modified = False
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            fixed_lines.append(line + '\n')
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id in class_mapping:
                                parts[0] = str(class_mapping[class_id])
                                file_modified = True
                            
                            fixed_lines.append(' '.join(parts) + '\n')
                        else:
                            fixed_lines.append(line + '\n')
                    
                    # Write back if modified
                    if file_modified:
                        with open(label_file, 'w') as f:
                            f.writelines(fixed_lines)
                        fixed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error fixing {label_file}: {e}")
        
        logger.info(f"Fixed {fixed_count} label files")
        return fixed_count

class YOLONASDataLoaderFixer:
    """Fixes YOLO-NAS dataloader to prevent CUDA assertion errors"""
    
    @staticmethod
    def create_safe_dataloader(dataset_params: Dict[str, Any], dataloader_params: Dict[str, Any], num_classes: int):
        """Create a dataloader with class index validation"""
        from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train
        
        # Add validation wrapper
        original_dataset_params = dataset_params.copy()
        
        # Create base dataloader
        dataloader = coco_detection_yolo_format_train(
            dataset_params=dataset_params,
            dataloader_params=dataloader_params
        )
        
        # Wrap with validation
        wrapped_dataloader = SafeDataLoaderWrapper(dataloader, num_classes)
        return wrapped_dataloader

class SafeDataLoaderWrapper:
    """Wrapper that validates class indices in dataloader output"""
    
    def __init__(self, base_dataloader, num_classes: int):
        self.base_dataloader = base_dataloader
        self.num_classes = num_classes
        self.dataset = base_dataloader.dataset
        
    def __iter__(self):
        for batch in self.base_dataloader:
            # Validate and fix batch
            fixed_batch = self._validate_batch(batch)
            yield fixed_batch
    
    def __len__(self):
        return len(self.base_dataloader)
    
    def _validate_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and fix class indices in batch"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
            
            if isinstance(targets, torch.Tensor) and targets.numel() > 0:
                # Check if targets have class column (usually column 1)
                if targets.dim() == 2 and targets.shape[1] > 1:
                    class_indices = targets[:, 1]
                    
                    # Find invalid class indices
                    invalid_mask = class_indices >= self.num_classes
                    
                    if invalid_mask.any():
                        logger.warning(f"Found {invalid_mask.sum()} invalid class indices, fixing...")
                        # Clamp to valid range
                        targets[:, 1] = torch.clamp(targets[:, 1], 0, self.num_classes - 1)
            
            return images, targets
        
        return batch

def fix_yolo_nas_class_issues(dataset_dir: str, num_classes: int) -> Dict[str, Any]:
    """Main function to fix all YOLO-NAS class-related issues"""
    logger.info(f"Fixing YOLO-NAS class issues for {num_classes} classes...")
    
    # Step 1: Validate dataset
    validator = ClassIndexValidator(dataset_dir, num_classes)
    analysis = validator.analyze_dataset_classes()
    
    # Step 2: Create mapping for invalid classes
    class_mapping = validator.create_class_mapping(analysis)
    
    # Step 3: Fix dataset files if needed
    fixed_files = 0
    if class_mapping:
        fixed_files = validator.fix_dataset_labels(class_mapping)
    
    # Step 4: Re-analyze to confirm fix
    final_analysis = validator.analyze_dataset_classes()
    
    result = {
        'initial_analysis': analysis,
        'class_mapping': class_mapping,
        'fixed_files': fixed_files,
        'final_analysis': final_analysis,
        'is_fixed': final_analysis['is_valid']
    }
    
    if result['is_fixed']:
        logger.info("✓ All class issues fixed successfully")
    else:
        logger.error("✗ Some class issues remain")
    
    return result

# Integration with unified trainer
def create_safe_yolo_nas_trainer_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create YOLO-NAS trainer with class validation"""
    
    # Fix dataset first
    dataset_dir = config['dataset']['data_dir']
    num_classes = config['model']['num_classes']
    
    fix_result = fix_yolo_nas_class_issues(dataset_dir, num_classes)
    
    if not fix_result['is_fixed']:
        raise ValueError("Failed to fix class index issues in dataset")
    
    # Now create trainer components normally
    from super_gradients import Trainer
    from super_gradients.training import models
    from super_gradients.common.object_names import Models
    from super_gradients.training.transforms.detection import DetectionLongestMaxSize, DetectionPadIfNeeded
    
    # Create trainer
    trainer = Trainer(
        experiment_name=config['experiment_name'],
        ckpt_root_dir=str(Path(config['output_dir']) / "checkpoints")
    )
    
    # Create model
    arch_mapping = {
        'yolo_nas_s': Models.YOLO_NAS_S,
        'yolo_nas_m': Models.YOLO_NAS_M,
        'yolo_nas_l': Models.YOLO_NAS_L
    }
    
    model = models.get(
        arch_mapping[config['model']['architecture']],
        num_classes=num_classes,
        pretrained_weights=config['model']['pretrained_weights']
    )
    
    # Create safe dataloaders
    input_size = config['model']['input_size']
    transforms = [
        DetectionLongestMaxSize(max_height=input_size[0], max_width=input_size[1]),
        DetectionPadIfNeeded(min_height=input_size[0], min_width=input_size[1], pad_value=114)
    ]
    
    dataset_params = {
        "data_dir": dataset_dir,
        "images_dir": f"images/{config['dataset']['train_split']}",
        "labels_dir": f"labels/{config['dataset']['train_split']}",
        "classes": config['dataset']['class_names'],
        "input_dim": input_size,
        "ignore_empty_annotations": True,
        "transforms": transforms
    }
    
    dataloader_params = {
        "batch_size": config['training']['batch_size'],
        "num_workers": config['training']['workers'],
        "shuffle": True,
        "drop_last": False,
        "pin_memory": True
    }
    
    # Create safe dataloaders
    train_loader = YOLONASDataLoaderFixer.create_safe_dataloader(
        dataset_params, dataloader_params, num_classes
    )
    
    # Validation dataloader
    val_dataset_params = dataset_params.copy()
    val_dataset_params.update({
        "images_dir": f"images/{config['dataset']['val_split']}",
        "labels_dir": f"labels/{config['dataset']['val_split']}"
    })
    val_dataloader_params = dataloader_params.copy()
    val_dataloader_params["shuffle"] = False
    
    val_loader = YOLONASDataLoaderFixer.create_safe_dataloader(
        val_dataset_params, val_dataloader_params, num_classes
    )
    
    return {
        'trainer': trainer,
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'fix_result': fix_result
    }

if __name__ == "__main__":
    # Test the fix
    dataset_dir = str(Path.home() / "fiftyone" / "train_yolo")
    result = fix_yolo_nas_class_issues(dataset_dir, 32)
    print(f"Fix result: {result['is_fixed']}")