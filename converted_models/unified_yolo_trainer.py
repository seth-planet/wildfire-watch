#!/usr/bin/env python3.10
"""
Unified YOLO Training Script for Wildfire Watch
Consolidated training script that handles YOLO-NAS, YOLOv8, and YOLOv9 with robust error handling

IMPORTANT: This script requires Python 3.10 for super-gradients compatibility
Run with: python3.10 unified_yolo_trainer.py [config_file]
"""
import os
import sys
import logging
import yaml
import torch
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmp.fix_class_index_cuda import create_safe_dataloader

# Setup logging
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup centralized logging configuration"""
    output_dir.mkdir(exist_ok=True)
    
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "unified_training.log")
        ]
    )
    return logging.getLogger(__name__)

class UnifiedYOLOTrainer:
    """Unified trainer for multiple YOLO architectures"""
    
    SUPPORTED_ARCHITECTURES = ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l", "yolov8n", "yolov8s", "yolov8m"]
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get('output_dir', '../output'))
        self.logger = setup_logging(self.output_dir, self.config.get('log_level', 'INFO'))
        self.device_info = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load and validate training configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._get_default_config()
            
        return self._validate_config(config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'model': {
                'architecture': 'yolo_nas_s',
                'num_classes': None,  # Auto-detect from dataset
                'input_size': [640, 640],
                'pretrained_weights': None
            },
            'dataset': {
                'data_dir': '/media/seth/SketchScratch/fiftyone/train_yolo',
                'format': 'coco',  # coco, yolo, custom
                'train_split': 'train',
                'val_split': 'validation',
                'class_names': None,  # Auto-detect from dataset.yaml
                'validate_labels': True
            },
            'training': {
                'epochs': 200,
                'batch_size': 8,
                'learning_rate': 0.001,
                'warmup_epochs': 5,
                'lr_scheduler': 'cosine',  # cosine, step, polynomial
                'lr_decay_factor': 0.1,
                'workers': 4,
                'mixed_precision': False,  # Disabled to avoid dtype issues
                'gradient_accumulation': 1,
                'early_stopping': True,
                'patience': 50
            },
            'qat': {
                'enabled': True,
                'start_epoch': 150,
                'calibration_batches': 100
            },
            'validation': {
                'interval': 10,
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_predictions': 300
            },
            'output_dir': '../output',
            'experiment_name': 'wildfire_detection',
            'log_level': 'INFO'
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix configuration"""
        # Validate architecture
        arch = config.get('model', {}).get('architecture', 'yolo_nas_s')
        if arch not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {arch}. Supported: {self.SUPPORTED_ARCHITECTURES}")
        
        # Validate dataset path
        data_dir = Path(config.get('dataset', {}).get('data_dir', ''))
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        return config
    
    def check_environment(self) -> Dict[str, Any]:
        """Check training environment and requirements"""
        self.logger.info("Checking training environment...")
        
        env_info = {
            'python_version': sys.version,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'torch_version': torch.__version__
        }
        
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            env_info.update({
                'gpu_name': gpu_info.name,
                'gpu_memory': f"{gpu_info.total_memory / 1e9:.1f} GB",
                'cuda_version': torch.version.cuda
            })
            
        self.logger.info(f"Environment: {json.dumps(env_info, indent=2)}")
        
        # Check required packages
        required_packages = self._get_required_packages()
        missing_packages = self._check_packages(required_packages)
        
        if missing_packages:
            self.logger.warning(f"Missing packages: {missing_packages}")
            self._install_packages(missing_packages)
        
        self.device_info = env_info
        return env_info
    
    def _get_required_packages(self) -> List[str]:
        """Get required packages based on architecture"""
        base_packages = ["torch", "torchvision", "opencv-python", "pyyaml", "tensorboard"]
        
        arch = self.config['model']['architecture']
        if arch.startswith('yolo_nas'):
            base_packages.append("super-gradients")
        elif arch.startswith('yolov'):
            base_packages.append("ultralytics")
            
        return base_packages
    
    def _check_packages(self, packages: List[str]) -> List[str]:
        """Check which packages are missing"""
        missing = []
        for package in packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        return missing
    
    def _install_packages(self, packages: List[str]):
        """Install missing packages"""
        for package in packages:
            self.logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    def auto_detect_classes(self) -> Dict[str, Any]:
        """Auto-detect number of classes and class names from dataset"""
        self.logger.info("Auto-detecting dataset classes...")
        
        dataset_dir = Path(self.config['dataset']['data_dir'])
        dataset_yaml = dataset_dir / 'dataset.yaml'
        
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml}")
        
        with open(dataset_yaml, 'r') as f:
            dataset_info = yaml.safe_load(f)
        
        # Handle different dataset.yaml formats
        num_classes = dataset_info.get('nc')
        class_names = dataset_info.get('names', {})
        
        # If nc is not present, infer from names dict
        if num_classes is None and isinstance(class_names, dict):
            num_classes = len(class_names)
            self.logger.info(f"Inferred {num_classes} classes from names dict")
        
        if num_classes is None:
            raise ValueError("Cannot determine number of classes from dataset.yaml")
        
        # Convert names to list if it's a dict
        if isinstance(class_names, dict):
            # Ensure we have all indices from 0 to num_classes-1
            class_names = [class_names.get(i, f"class_{i}") for i in range(num_classes)]
        
        self.logger.info(f"Detected {num_classes} classes: {class_names[:5]}...")
        
        # Update config
        self.config['model']['num_classes'] = num_classes
        self.config['dataset']['class_names'] = class_names
        
        # Find fire class index
        fire_class_index = None
        for i, name in enumerate(class_names):
            if name.lower() == 'fire':
                fire_class_index = i
                self.logger.info(f"Found 'Fire' class at index {i}")
                break
        
        return {
            'num_classes': num_classes,
            'class_names': class_names,
            'fire_class_index': fire_class_index
        }
    
    def validate_dataset_labels(self) -> Dict[str, Any]:
        """Validate dataset labels for class index consistency"""
        if not self.config['dataset'].get('validate_labels', True):
            self.logger.info("Label validation disabled")
            return {'validation_skipped': True}
        
        self.logger.info("Validating dataset labels...")
        
        dataset_dir = Path(self.config['dataset']['data_dir'])
        num_classes = self.config['model']['num_classes']
        
        validation_results = {}
        
        for split in ['train', 'validation']:
            split_dir = self.config['dataset'].get(f'{split}_split', split)
            labels_dir = dataset_dir / 'labels' / split_dir
            
            if not labels_dir.exists():
                self.logger.warning(f"Labels directory not found: {labels_dir}")
                continue
                
            valid_count = 0
            invalid_count = 0
            class_distribution = {}
            
            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file, 'r') as f:
                        file_valid = True
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) < 5:
                                self.logger.warning(f"Invalid format in {label_file.name}:{line_num}")
                                file_valid = False
                                break
                            
                            try:
                                class_id = int(parts[0])
                                if class_id < 0 or class_id >= num_classes:
                                    self.logger.warning(f"Invalid class {class_id} in {label_file.name}:{line_num} (valid: 0-{num_classes-1})")
                                    file_valid = False
                                    break
                                else:
                                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                            except ValueError:
                                self.logger.warning(f"Cannot parse class ID in {label_file.name}:{line_num}")
                                file_valid = False
                                break
                        
                        if file_valid:
                            valid_count += 1
                        else:
                            invalid_count += 1
                            
                except IOError as e:
                    self.logger.error(f"Cannot read {label_file}: {e}")
                    invalid_count += 1
            
            validation_results[split] = {
                'valid_files': valid_count,
                'invalid_files': invalid_count,
                'success_rate': valid_count / (valid_count + invalid_count) * 100 if (valid_count + invalid_count) > 0 else 0,
                'class_distribution': class_distribution
            }
            
            self.logger.info(f"{split}: {valid_count} valid, {invalid_count} invalid ({validation_results[split]['success_rate']:.1f}% success)")
        
        return validation_results
    
    def create_trainer(self):
        """Create trainer based on architecture"""
        arch = self.config['model']['architecture']
        
        if arch.startswith('yolo_nas'):
            return self._create_yolo_nas_trainer()
        elif arch.startswith('yolov'):
            return self._create_ultralytics_trainer()
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
    
    def _create_yolo_nas_trainer(self):
        """Create YOLO-NAS trainer using super-gradients"""
        try:
            from super_gradients import Trainer
            from super_gradients.training import models
            from super_gradients.training.dataloaders.dataloaders import (
                coco_detection_yolo_format_train, 
                coco_detection_yolo_format_val
            )
            from super_gradients.training.losses import PPYoloELoss
            from super_gradients.training.metrics import DetectionMetrics_050
            from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
            from super_gradients.training.utils.distributed_training_utils import setup_device
            from super_gradients.common.object_names import Models
            from super_gradients.training.transforms.detection import (
                DetectionLongestMaxSize, 
                DetectionPadIfNeeded
            )
            from super_gradients.training.transforms.transforms import (
                DetectionStandardize,
                DetectionTargetsFormatTransform
            )
        except ImportError as e:
            raise ImportError(f"super-gradients not available: {e}")
        
        # Setup device
        device = setup_device(multi_gpu="auto", num_gpus=None)
        self.logger.info(f"Using device: {device}")
        
        # Create trainer
        trainer = Trainer(
            experiment_name=self.config['experiment_name'],
            ckpt_root_dir=str(self.output_dir / "checkpoints")
        )
        
        # Create model
        arch_mapping = {
            'yolo_nas_s': Models.YOLO_NAS_S,
            'yolo_nas_m': Models.YOLO_NAS_M,
            'yolo_nas_l': Models.YOLO_NAS_L
        }
        
        model = models.get(
            arch_mapping[self.config['model']['architecture']],
            num_classes=self.config['model']['num_classes'],
            pretrained_weights=self.config['model']['pretrained_weights']
        )
        
        # Create dataloaders
        dataloaders = self._create_yolo_nas_dataloaders()
        
        # Create training parameters
        training_params = self._create_yolo_nas_training_params()
        
        return {
            'trainer': trainer,
            'model': model,
            'train_loader': dataloaders['train'],
            'val_loader': dataloaders['val'],
            'training_params': training_params
        }
    
    def _create_yolo_nas_dataloaders(self):
        """Create YOLO-NAS compatible dataloaders with class index validation"""
        from super_gradients.training.dataloaders.dataloaders import (
            coco_detection_yolo_format_train, 
            coco_detection_yolo_format_val
        )
        from super_gradients.training.transforms.detection import (
            DetectionLongestMaxSize, 
            DetectionPadIfNeeded
        )
        from super_gradients.training.transforms.transforms import (
            DetectionStandardize,
            DetectionTargetsFormatTransform
        )
        
        # Fix class index issues first
        dataset_dir = self.config['dataset']['data_dir']
        num_classes = self.config['model']['num_classes']
        original_dataset_dir = dataset_dir
        
        # Check if dataset needs preprocessing
        try:
            from dataset_preprocessor import preprocess_dataset
            
            # First analyze the dataset
            self.logger.info("Checking dataset for invalid class indices...")
            preprocess_result = preprocess_dataset(
                dataset_dir, 
                num_classes,
                mode='filter'  # Remove images with invalid class indices
            )
            
            if preprocess_result['analysis']['invalid_labels'] > 0:
                self.logger.warning(f"Found {preprocess_result['analysis']['invalid_labels']} invalid labels")
                self.logger.info(f"Using filtered dataset from: {preprocess_result['output_dir']}")
                dataset_dir = preprocess_result['output_dir']
                
                # Log filtering statistics
                stats = preprocess_result['filter_stats']
                self.logger.info(f"Dataset filtering complete:")
                self.logger.info(f"  - Original images: {stats['total_images_processed']}")
                self.logger.info(f"  - Images retained: {stats['images_with_valid_labels']}")
                self.logger.info(f"  - Images skipped: {stats['images_skipped']}")
                self.logger.info(f"  - Labels filtered: {stats['labels_filtered']}")
            else:
                self.logger.info("✓ Dataset has no invalid class indices")
                
        except ImportError:
            self.logger.warning("Dataset preprocessor not available, attempting class index fixer")
            # Fallback to class index fixer
            try:
                from class_index_fixer import fix_yolo_nas_class_issues
                fix_result = fix_yolo_nas_class_issues(dataset_dir, num_classes)
                
                if not fix_result['is_fixed']:
                    raise ValueError("Failed to fix class index issues in dataset")
                    
                self.logger.info("✓ Dataset class indices validated and fixed")
            except ImportError:
                self.logger.warning("No dataset validation available, proceeding with caution")
        
        # Setup transforms to prevent tensor size mismatches and convert to correct format
        input_size = self.config['model']['input_size']
        transforms = [
            DetectionLongestMaxSize(max_height=input_size[0], max_width=input_size[1]),
            DetectionPadIfNeeded(min_height=input_size[0], min_width=input_size[1], pad_value=114),
            DetectionStandardize(max_value=255.),
            DetectionTargetsFormatTransform(
                input_dim=input_size,
                output_format='LABEL_CXCYWH'
            )
        ]
        
        # Training dataloader
        train_dataset_params = {
            "data_dir": dataset_dir,
            "images_dir": f"images/{self.config['dataset']['train_split']}",
            "labels_dir": f"labels/{self.config['dataset']['train_split']}",
            "classes": self.config['dataset']['class_names'],
            "input_dim": input_size,
            "ignore_empty_annotations": True,
            "transforms": transforms,
            "cache_annotations": False  # Disable caching to speed up initialization for large datasets
        }
        
        train_dataloader_params = {
            "batch_size": self.config['training']['batch_size'],
            "num_workers": self.config['training']['workers'],
            "shuffle": True,
            "drop_last": False,
            "pin_memory": True
        }
        
        # Create base dataloaders
        train_loader = coco_detection_yolo_format_train(
            dataset_params=train_dataset_params,
            dataloader_params=train_dataloader_params
        )
        
        # Validation dataloader
        val_dataset_params = train_dataset_params.copy()
        val_dataset_params.update({
            "images_dir": f"images/{self.config['dataset']['val_split']}",
            "labels_dir": f"labels/{self.config['dataset']['val_split']}",
            "cache_annotations": True  # Enable caching for validation since it's smaller
        })
        
        val_dataloader_params = train_dataloader_params.copy()
        val_dataloader_params["shuffle"] = False
        
        val_loader = coco_detection_yolo_format_val(
            dataset_params=val_dataset_params,
            dataloader_params=val_dataloader_params
        )
        
        # Wrap with safety validation
        try:
            # Use absolute import for class_index_fixer
            import sys
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            
            from class_index_fixer import SafeDataLoaderWrapper
            
            # Get max invalid ratio from config or use default (0.1% for production)
            max_invalid_ratio = self.config.get('dataset', {}).get('max_invalid_class_ratio', 0.001)
            
            train_loader = SafeDataLoaderWrapper(train_loader, num_classes, max_invalid_ratio=max_invalid_ratio)
            val_loader = SafeDataLoaderWrapper(val_loader, num_classes, max_invalid_ratio=max_invalid_ratio)
            self.logger.info(f"✓ Dataloaders wrapped with class index validation (max_invalid_ratio={max_invalid_ratio})")
            
            # Store references for statistics reporting
            self._train_wrapper = train_loader
            self._val_wrapper = val_loader
            
        except ImportError as e:
            self.logger.warning(f"SafeDataLoaderWrapper not available: {e}, using standard dataloaders")
            self._train_wrapper = None
            self._val_wrapper = None
        
        self.logger.info(f"Dataloaders created: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
        
        # Wrap dataloaders with fixed collate function to handle target format
        try:
            import sys
            sys.path.append(str(Path(__file__).parent.parent / "tmp"))
            from fixed_yolo_nas_collate import wrap_dataloader_with_fixed_collate
            
            train_loader = wrap_dataloader_with_fixed_collate(train_loader, num_classes)
            val_loader = wrap_dataloader_with_fixed_collate(val_loader, num_classes)
            self.logger.info("✓ Dataloaders wrapped with fixed collate function for target format")
        except ImportError as e:
            self.logger.warning(f"Could not wrap dataloaders with fixed collate: {e}")
        
        return {'train': train_loader, 'val': val_loader}
    
    def _create_yolo_nas_training_params(self):
        """Create YOLO-NAS training parameters dict"""
        from super_gradients.training.losses import PPYoloELoss
        from super_gradients.training.metrics import DetectionMetrics_050
        from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
        
        config = self.config['training']
        num_classes = self.config['model']['num_classes']
        
        training_params = {
            # Core parameters
            "max_epochs": config['epochs'],
            "lr_mode": "CosineLRScheduler",  # Use correct scheduler name
            "initial_lr": config['learning_rate'],
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=num_classes,
                reg_max=16
            ),
            
            # Learning rate schedule
            "lr_warmup_epochs": config['warmup_epochs'],
            "warmup_mode": "LinearEpochLRWarmup",
            "warmup_initial_lr": 1e-6,
            "cosine_final_lr_ratio": 0.1,
            
            # Optimizer
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "warmup_bias_lr": 0.0,
            "warmup_momentum": 0.9,
            
            # Training settings
            "mixed_precision": config['mixed_precision'],
            "average_best_models": config['epochs'] >= 50,  # Only average if we have enough epochs
            "ema": config['epochs'] >= 10,  # Only use EMA if we have enough epochs
            "ema_params": {
                "decay": 0.9999,
                "decay_type": "exp",
                "beta": 15
            },
            
            # Validation metrics
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=self.config['validation']['conf_threshold'],
                    top_k_predictions=self.config['validation']['max_predictions'],
                    num_cls=num_classes,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=self.config['validation']['max_predictions'],
                        nms_threshold=self.config['validation']['iou_threshold']
                    )
                )
            ],
            "metric_to_watch": "mAP@0.50",  # Use correct metric name
            "greater_metric_to_watch_is_better": True,
            
            # Checkpointing
            "save_ckpt_epoch_list": list(range(50, config['epochs'] + 1, 50)) if config['epochs'] >= 50 else [],
            "resume": False,
            "run_validation_freq": min(10, config['epochs']),  # Validate at reasonable intervals
            
            # Logging
            "silent_mode": False,
            "sg_logger": "base_sg_logger",
            "sg_logger_params": {
                "tb_files_user_prompt": False,
                "launch_tensorboard": False,
                "tensorboard_port": None,
                "save_checkpoints_remote": False,
                "save_tensorboard_remote": False,
                "save_logs_remote": False
            },
            
            # Other settings
            "seed": 42,
            "launch_tensorboard": False
        }
        
        # Add scheduler-specific parameters
        if config['lr_scheduler'] == 'step':
            training_params.update({
                "lr_decay_factor": config['lr_decay_factor'],
                "lr_updates": [int(config['epochs'] * 0.7), int(config['epochs'] * 0.9)]
            })
        elif config['lr_scheduler'] == 'cosine':
            training_params["cosine_final_lr_ratio"] = 0.1
        
        # Add max train batches if specified (for testing)
        if config.get('max_train_batches'):
            training_params['max_train_batches'] = config['max_train_batches']
        
        return training_params
    
    def _create_ultralytics_trainer(self):
        """Create Ultralytics YOLO trainer"""
        # TODO: Implement ultralytics trainer for YOLOv8/v9
        raise NotImplementedError("Ultralytics trainer not yet implemented")
    

    def _report_cuda_fix_stats(self):
        """Report statistics from CUDA index fixing"""
        # Report SafeDataLoaderWrapper statistics
        if hasattr(self, '_train_wrapper') and self._train_wrapper is not None:
            train_stats = self._train_wrapper.get_statistics()
            if train_stats['total_invalid_indices'] > 0:
                self.logger.warning(
                    f"SafeDataLoaderWrapper - Training: Fixed {train_stats['total_invalid_indices']} "
                    f"invalid indices out of {train_stats['total_indices_seen']} total "
                    f"({train_stats['invalid_ratio']:.2%}) in {train_stats['batches_processed']} batches"
                )
            else:
                self.logger.info("SafeDataLoaderWrapper - Training: No invalid indices found")
                
        if hasattr(self, '_val_wrapper') and self._val_wrapper is not None:
            val_stats = self._val_wrapper.get_statistics()
            if val_stats['total_invalid_indices'] > 0:
                self.logger.warning(
                    f"SafeDataLoaderWrapper - Validation: Fixed {val_stats['total_invalid_indices']} "
                    f"invalid indices out of {val_stats['total_indices_seen']} total "
                    f"({val_stats['invalid_ratio']:.2%}) in {val_stats['batches_processed']} batches"
                )
            else:
                self.logger.info("SafeDataLoaderWrapper - Validation: No invalid indices found")
        
        # Report old collate function statistics if available
        if hasattr(self, '_train_collate'):
            train_stats = self._train_collate.get_stats()
            if train_stats['invalid_batches'] > 0:
                self.logger.warning(f"CUDA Fix Stats - Training: Fixed {train_stats['fixed_indices']} "
                                  f"invalid indices in {train_stats['invalid_batches']} batches")
                
        if hasattr(self, '_val_collate'):
            val_stats = self._val_collate.get_stats()
            if val_stats['invalid_batches'] > 0:
                self.logger.warning(f"CUDA Fix Stats - Validation: Fixed {val_stats['fixed_indices']} "
                                  f"invalid indices in {val_stats['invalid_batches']} batches")
    

    def train(self):
        """Execute training pipeline"""
        self.logger.info("Starting unified YOLO training pipeline...")
        
        try:
            # Phase 1: Environment setup
            self.check_environment()
            
            # Phase 2: Dataset analysis
            class_info = self.auto_detect_classes()
            validation_results = self.validate_dataset_labels()
            
            # Phase 3: Create trainer
            trainer_components = self.create_trainer()
            
            # Phase 4: Execute training
            self.logger.info(f"Starting training for {self.config['training']['epochs']} epochs...")
            start_time = time.time()
            
            # Execute training with fixed dataloaders
            trainer_components['trainer'].train(
                model=trainer_components['model'],
                training_params=trainer_components['training_params'],
                train_loader=trainer_components['train_loader'],
                valid_loader=trainer_components['val_loader']
            )
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time/3600:.1f} hours")
            
            # Report CUDA fix statistics
            self._report_cuda_fix_stats()
            
            # Phase 5: Save final model
            final_model_path = self._save_final_model(trainer_components['trainer'])
            
            # Phase 6: Generate training report
            report = self._generate_training_report(class_info, validation_results, training_time, final_model_path)
            
            return report
            
        except Exception as e:
            import traceback
            self.logger.error(f"Training failed: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
    
    def _save_final_model(self, trainer) -> str:
        """Save the best trained model"""
        experiment_name = self.config['experiment_name']
        checkpoints_dir = self.output_dir / "checkpoints" / experiment_name
        
        # Find the best model
        best_model_candidates = list(checkpoints_dir.glob("**/average_model.pth"))
        if not best_model_candidates:
            best_model_candidates = list(checkpoints_dir.glob("**/ckpt_best.pth"))
        
        if best_model_candidates:
            best_model_path = best_model_candidates[0]
            final_model_path = self.output_dir / f"{experiment_name}_final.pth"
            
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            self.logger.info(f"Final model saved: {final_model_path}")
            return str(final_model_path)
        else:
            self.logger.warning("No best model found in checkpoints")
            return None
    
    def _generate_training_report(self, class_info, validation_results, training_time, model_path) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'experiment_name': self.config['experiment_name'],
            'architecture': self.config['model']['architecture'],
            'training_completed': True,
            'training_time_hours': training_time / 3600,
            'final_model_path': model_path,
            'dataset_info': {
                'num_classes': class_info['num_classes'],
                'fire_class_index': class_info.get('fire_class_index'),
                'validation_results': validation_results
            },
            'configuration': self.config,
            'device_info': self.device_info,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save report
        report_path = self.output_dir / f"{self.config['experiment_name']}_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Training report saved: {report_path}")
        return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Unified YOLO Training Script')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--architecture', type=str, choices=UnifiedYOLOTrainer.SUPPORTED_ARCHITECTURES,
                       help='Model architecture to train')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--qat', action='store_true', help='Enable Quantization-Aware Training')
    parser.add_argument('--validate-labels', action='store_true', default=True, help='Validate dataset labels')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UnifiedYOLOTrainer(args.config)
    
    # Override config with command line arguments
    if args.architecture:
        trainer.config['model']['architecture'] = args.architecture
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.lr:
        trainer.config['training']['learning_rate'] = args.lr
    if args.qat:
        trainer.config['qat']['enabled'] = True
    
    # Execute training
    try:
        report = trainer.train()
        print(f"✓ Training completed successfully!")
        print(f"Model: {report['final_model_path']}")
        print(f"Time: {report['training_time_hours']:.1f} hours")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()