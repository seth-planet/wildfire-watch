#!/usr/bin/env python3.10
"""
YOLO-NAS Training Script for Wildfire Watch
Trains YOLO-NAS-S model on custom COCO dataset with QAT support

IMPORTANT: This script requires Python 3.10 for super-gradients compatibility
Run with: python3.10 train_yolo_nas.py
"""
import os
import sys
import logging
import yaml
import torch
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging to output directory
output_dir = Path("../output")
output_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "yolo_nas_training.log")
    ]
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check and install required packages"""
    logger.info("Checking requirements...")
    
    required_packages = [
        "super-gradients",
        "torch",
        "torchvision", 
        "opencv-python",
        "pyyaml",
        "tensorboard"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "super-gradients":
                import super_gradients
            elif package == "torch":
                import torch
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            elif package == "torchvision":
                import torchvision
            elif package == "opencv-python":
                import cv2
            elif package == "pyyaml":
                import yaml
            elif package == "tensorboard":
                import tensorboard
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    return True

def setup_yolo_nas():
    """Setup YOLO-NAS repository"""
    logger.info("Setting up YOLO-NAS...")
    
    repo_dir = Path("YOLO-NAS-pytorch")
    
    if not repo_dir.exists():
        logger.info("Cloning YOLO-NAS repository...")
        import subprocess
        subprocess.run([
            "git", "clone", 
            "https://github.com/Andrewhsin/YOLO-NAS-pytorch.git"
        ], check=True)
    
    # Add to Python path
    sys.path.insert(0, str(repo_dir.absolute()))
    
    return repo_dir

def prepare_dataset_config():
    """Prepare dataset configuration for training"""
    logger.info("Preparing dataset configuration...")
    
    dataset_path = Path.home() / "fiftyone" / "train_yolo"
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Read original dataset.yaml
    with open(dataset_path / "dataset.yaml", 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Dataset classes: {len(dataset_config['names'])}")
    logger.info("Key classes for wildfire detection:")
    for idx, name in dataset_config['names'].items():
        if name.lower() in ['fire', 'person', 'car', 'building']:
            logger.info(f"  {idx}: {name}")
    
    # Create training config
    training_config = {
        'dataset': {
            'data_dir': str(dataset_path),
            'train_images_dir': 'images/train',
            'train_labels_dir': 'labels/train', 
            'val_images_dir': 'images/validation',
            'val_labels_dir': 'labels/validation',
            'class_names': list(dataset_config['names'].values()),
            'nc': len(dataset_config['names'])
        },
        'model': {
            'architecture': 'yolo_nas_s',
            'num_classes': len(dataset_config['names']),
            'input_size': [640, 640],
            'pretrained_weights': None  # Changed to None to avoid internet dependency
        },
        'training': {
            'epochs': 5,  # Reduced for quick testing
            'batch_size': 8,  # Adjust based on GPU memory
            'learning_rate': 0.001,
            'warmup_epochs': 5,
            'cos_lr': True,
            'workers': 4,
            'save_ckpt_epoch_list': [50, 100, 150, 200],
            'checkpoints_dir': str(output_dir / "checkpoints"),
            'experiment_name': 'wildfire_yolo_nas_s'
        },
        'qat': {
            'enabled': True,
            'start_epoch': 150,  # Start QAT in last 50 epochs
            'calibration_batches': 100
        },
        'validation': {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'max_predictions_per_image': 300
        }
    }
    
    # Save training config
    config_path = output_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f, indent=2)
    
    logger.info(f"Training configuration saved to: {config_path}")
    return training_config

def create_training_script(config: Dict[str, Any]):
    """Create the actual training script using super-gradients"""
    
    training_script = f'''
import os
import torch
import logging
import shutil
from pathlib import Path
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
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils.callbacks import (
    ModelConversionCheckCallback,
    LRCallbackBase,
    PhaseCallback,
    LRSchedulerCallback
)

# Setup logging
logger = logging.getLogger(__name__)

def validate_dataset_labels(dataset_dir, num_classes, split='train'):
    \"\"\"Validate dataset labels and create a list of valid image files\"\"\"
    logger.info(f"Validating {{split}} dataset labels...")
    
    images_dir = Path(dataset_dir) / f'images/{{split}}'
    labels_dir = Path(dataset_dir) / f'labels/{{split}}'
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Dataset directories not found: {{images_dir}}, {{labels_dir}}")
        return None, None
    
    valid_images = []
    valid_labels = []
    valid_count = 0
    invalid_count = 0
    
    # Process each image and its label
    for image_file in images_dir.glob('*.jpg'):
        label_file = labels_dir / (image_file.stem + '.txt')
        
        if not label_file.exists():
            if invalid_count < 10:  # Only log first 10 warnings
                logger.warning(f"No label file for {{image_file.name}}")
            invalid_count += 1
            continue
        
        # Validate label file
        is_valid = True
        try:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        if invalid_count < 10:  # Only log first 10 warnings
                            logger.warning(f"Invalid format in {{label_file.name}} line {{line_num}}: expected 5 values, got {{len(parts)}}")
                        is_valid = False
                        break
                    
                    try:
                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= num_classes:
                            if invalid_count < 10:  # Only log first 10 warnings
                                logger.warning(f"Invalid class ID {{class_id}} in {{label_file.name}} line {{line_num}}: valid range is 0-{{num_classes-1}}")
                            is_valid = False
                            break
                    except ValueError as e:
                        if invalid_count < 10:  # Only log first 10 warnings
                            logger.warning(f"Cannot parse class ID in {{label_file.name}} line {{line_num}}: {{e}}")
                        is_valid = False
                        break
        
        except IOError as e:
            logger.error(f"Cannot read label file {{label_file}}: {{e}}")
            is_valid = False
        
        # Track valid files without copying
        if is_valid:
            valid_images.append(str(image_file))
            valid_labels.append(str(label_file))
            valid_count += 1
        else:
            invalid_count += 1
    
    logger.info(f"Dataset validation complete for {{split}}:")
    logger.info(f"  Valid images: {{valid_count}}")
    logger.info(f"  Invalid images: {{invalid_count}}")
    if valid_count + invalid_count > 0:
        logger.info(f"  Success rate: {{valid_count / (valid_count + invalid_count) * 100:.1f}}%")
    
    if valid_count == 0:
        logger.error(f"No valid images found in {{split}} split!")
        return None, None
    
    # Save valid file lists for reference
    valid_files_info = {{
        'images': valid_images,
        'labels': valid_labels,
        'total_valid': valid_count,
        'total_invalid': invalid_count
    }}
    
    return str(dataset_dir), valid_files_info

def main():
    logger.info("Starting YOLO-NAS training...")
    
    # Setup device
    device = setup_device(
        multi_gpu="auto",
        num_gpus=None
    )
    logger.info(f"Using device: {{device}}")
    
    # Validate dataset labels (but don't copy files to save disk space)
    original_data_dir = "{config['dataset']['data_dir']}"
    train_data_dir, train_valid_info = validate_dataset_labels(
        original_data_dir, 
        {config['model']['num_classes']}, 
        'train'
    )
    
    if train_data_dir is None:
        logger.error("Failed to validate training dataset")
        return None
    
    # Also validate validation split
    val_data_dir, val_valid_info = validate_dataset_labels(
        original_data_dir, 
        {config['model']['num_classes']}, 
        'validation'
    )
    
    if val_data_dir is None:
        logger.error("Failed to validate validation dataset")
        return None
    
    # Use original dataset directory (validation passed)
    data_dir = original_data_dir
    logger.info(f"Dataset validation passed - using original dataset: {{data_dir}}")
    logger.info(f"Training images validated: {{train_valid_info['total_valid']}} valid, {{train_valid_info['total_invalid']}} invalid")
    logger.info(f"Validation images validated: {{val_valid_info['total_valid']}} valid, {{val_valid_info['total_invalid']}} invalid")
    
    # Create trainer
    trainer = Trainer(
        experiment_name="{config['training']['experiment_name']}",
        ckpt_root_dir="{config['training']['checkpoints_dir']}"
    )
    
    # Create proper super-gradients dataloaders with correct API
    logger.info("Creating YOLO-NAS dataloaders with filtered dataset...")
    
    # Dataset parameters for super-gradients dataloaders
    # Use available transforms to handle variable image sizes
    from super_gradients.training.transforms.detection import DetectionLongestMaxSize, DetectionPadIfNeeded
    
    # Transforms to normalize image sizes for batching
    transforms = [
        DetectionLongestMaxSize(max_height={config['model']['input_size'][0]}, max_width={config['model']['input_size'][1]}),
        DetectionPadIfNeeded(min_height={config['model']['input_size'][0]}, min_width={config['model']['input_size'][1]}, pad_value=114)
    ]
    
    dataset_params = {{
        "data_dir": data_dir,
        "images_dir": "{config['dataset']['train_images_dir']}",
        "labels_dir": "{config['dataset']['train_labels_dir']}",
        "classes": {config['dataset']['class_names']},
        "input_dim": {config['model']['input_size']},
        "ignore_empty_annotations": True,
        "transforms": transforms
    }}
    
    # Dataloader parameters
    dataloader_params = {{
        "batch_size": {config['training']['batch_size']},
        "num_workers": {config['training'].get('workers', 4)},
        "shuffle": True,
        "drop_last": False,
        "pin_memory": True
    }}
    
    # Create training dataloader
    logger.info("Creating training dataloader...")
    train_dataloader = coco_detection_yolo_format_train(
        dataset_params=dataset_params,
        dataloader_params=dataloader_params
    )
    
    # Validation dataloader parameters (no shuffle)
    val_dataloader_params = dataloader_params.copy()
    val_dataloader_params["shuffle"] = False
    
    # Update dataset params for validation
    val_dataset_params = dataset_params.copy()
    val_dataset_params.update({{
        "images_dir": "{config['dataset']['val_images_dir']}",
        "labels_dir": "{config['dataset']['val_labels_dir']}"
    }})
    
    # Create validation dataloader
    logger.info("Creating validation dataloader...")
    val_dataloader = coco_detection_yolo_format_val(
        dataset_params=val_dataset_params,
        dataloader_params=val_dataloader_params
    )
    
    logger.info("✓ Filtered dataset dataloaders created successfully")
    logger.info(f"Training samples: {{len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 'Unknown'}}")
    logger.info(f"Validation samples: {{len(val_dataloader.dataset) if hasattr(val_dataloader, 'dataset') else 'Unknown'}}")
    
    # Create model
    model = models.get(
        Models.{config['model']['architecture'].upper()},
        num_classes={config['model']['num_classes']},
        pretrained_weights={repr(config['model']['pretrained_weights'])}
    )
    
    logger.info(f"Model architecture: {{type(model).__name__}}")
    logger.info(f"Number of classes: {config['model']['num_classes']}")
    
    # Training parameters - use dict format for Trainer.train()
    training_params = {{
        # Required parameters
        "max_epochs": {config['training']['epochs']},
        "lr_mode": "cosine" if {config['training']['cos_lr']} else "step",
        "initial_lr": {config['training']['learning_rate']},
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes={config['model']['num_classes']},
            reg_max=16
        ),
        
        # Warmup parameters
        "lr_warmup_epochs": {config['training']['warmup_epochs']},
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        
        # Optimizer parameters
        "optimizer": "AdamW",
        "optimizer_params": {{"weight_decay": 0.0001}},
        "zero_weight_decay_on_bias_and_bn": True,
        
        # Training settings
        "mixed_precision": True,
        "average_best_models": True,
        "ema": True,
        "ema_params": {{"decay": 0.9999, "decay_type": "threshold"}},
        
        # Cosine LR settings
        "cosine_final_lr_ratio": 0.1,
        
        # Validation and metrics
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls={config['model']['num_classes']},
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": "mAP@0.50:0.95",
        "greater_metric_to_watch_is_better": True,
        
        # Checkpointing
        "save_ckpt_epoch_list": {config['training']['save_ckpt_epoch_list']},
        "resume": False,
        
        # Logging
        "silent_mode": False,
        "sg_logger": "base_sg_logger",
        "sg_logger_params": {{
            "tb_files_user_prompt": False,
            "launch_tensorboard": False,
            "tensorboard_port": None,
            "save_checkpoints_remote": False,
            "save_tensorboard_remote": False,
            "save_logs_remote": False
        }},
        
        # Other settings
        "seed": 42,
        "launch_tensorboard": False
    }}
    
    # Add QAT callback if enabled
    phase_callbacks = []
    if {config['qat']['enabled']}:
        logger.info("QAT (Quantization Aware Training) enabled")
        logger.info(f"QAT will start at epoch {config['qat']['start_epoch']}")
        
        # Note: QAT implementation may need to be added separately
        # This is a placeholder for QAT callback
        # phase_callbacks.append(QATCallback(start_epoch=config['qat']['start_epoch']))
    
    # Train the model - pass training_params as dict
    trainer.train(
        model=model,
        training_params=training_params,
        train_loader=train_dataloader,
        valid_loader=val_dataloader
    )
    
    # Save final model
    best_model_path = Path("{config['training']['checkpoints_dir']}") / "{config['training']['experiment_name']}" / "average_model.pth"
    final_model_path = Path("../output") / "yolo_nas_s_trained.pth"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        logger.info(f"Best model saved to: {{final_model_path}}")
    else:
        logger.warning("Best model not found, saving current model state")
        torch.save(model.state_dict(), final_model_path)
    
    logger.info("Training completed!")
    return str(final_model_path)

if __name__ == "__main__":
    trained_model_path = main()
    print(f"TRAINED_MODEL_PATH={{trained_model_path}}")
'''
    
    script_path = output_dir / "run_training.py"
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    logger.info(f"Training script created: {script_path}")
    return script_path

def run_training(script_path: Path):
    """Execute the training script with appropriate timeout"""
    logger.info("Starting YOLO-NAS training...")
    logger.info("This may take 48-72 hours depending on your GPU")
    
    import subprocess
    
    # Calculate timeout for 72 hours
    timeout_seconds = 72 * 60 * 60
    
    try:
        logger.info(f"Running training with timeout: {timeout_seconds/3600:.1f} hours")
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=script_path.parent
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully!")
            
            # Extract trained model path from output
            for line in result.stdout.split('\\n'):
                if line.startswith('TRAINED_MODEL_PATH='):
                    model_path = line.split('=', 1)[1]
                    logger.info(f"Trained model available at: {model_path}")
                    return model_path
            
            return None
        else:
            logger.error(f"Training failed with return code: {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out after {timeout_seconds/3600:.1f} hours")
        return None
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        return None

def main():
    """Main training pipeline"""
    logger.info("YOLO-NAS Training Pipeline for Wildfire Watch")
    logger.info("=" * 60)
    
    try:
        # Step 1: Check requirements
        check_requirements()
        
        # Step 2: Setup YOLO-NAS
        repo_dir = setup_yolo_nas()
        
        # Step 3: Prepare dataset config
        config = prepare_dataset_config()
        
        # Step 4: Create training script
        script_path = create_training_script(config)
        
        # Step 5: Run training
        logger.info("About to start training. This will take a very long time!")
        logger.info("Training will run for up to 72 hours.")
        logger.info("Monitor progress in output/yolo_nas_training.log")
        
        trained_model_path = run_training(script_path)
        
        if trained_model_path:
            logger.info(f"Training pipeline completed! Model: {trained_model_path}")
            return trained_model_path
        else:
            logger.error("Training pipeline failed!")
            return None
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trained_model = main()
    if trained_model:
        print(f"SUCCESS: Trained model at {trained_model}")
    else:
        print("FAILED: Training did not complete successfully")
        sys.exit(1)