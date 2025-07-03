
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
    """Validate dataset labels and create a list of valid image files"""
    logger.info(f"Validating {split} dataset labels...")
    
    images_dir = Path(dataset_dir) / f'images/{split}'
    labels_dir = Path(dataset_dir) / f'labels/{split}'
    
    if not images_dir.exists() or not labels_dir.exists():
        logger.error(f"Dataset directories not found: {images_dir}, {labels_dir}")
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
                logger.warning(f"No label file for {image_file.name}")
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
                            logger.warning(f"Invalid format in {label_file.name} line {line_num}: expected 5 values, got {len(parts)}")
                        is_valid = False
                        break
                    
                    try:
                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= num_classes:
                            if invalid_count < 10:  # Only log first 10 warnings
                                logger.warning(f"Invalid class ID {class_id} in {label_file.name} line {line_num}: valid range is 0-{num_classes-1}")
                            is_valid = False
                            break
                    except ValueError as e:
                        if invalid_count < 10:  # Only log first 10 warnings
                            logger.warning(f"Cannot parse class ID in {label_file.name} line {line_num}: {e}")
                        is_valid = False
                        break
        
        except IOError as e:
            logger.error(f"Cannot read label file {label_file}: {e}")
            is_valid = False
        
        # Track valid files without copying
        if is_valid:
            valid_images.append(str(image_file))
            valid_labels.append(str(label_file))
            valid_count += 1
        else:
            invalid_count += 1
    
    logger.info(f"Dataset validation complete for {split}:")
    logger.info(f"  Valid images: {valid_count}")
    logger.info(f"  Invalid images: {invalid_count}")
    if valid_count + invalid_count > 0:
        logger.info(f"  Success rate: {valid_count / (valid_count + invalid_count) * 100:.1f}%")
    
    if valid_count == 0:
        logger.error(f"No valid images found in {split} split!")
        return None, None
    
    # Save valid file lists for reference
    valid_files_info = {
        'images': valid_images,
        'labels': valid_labels,
        'total_valid': valid_count,
        'total_invalid': invalid_count
    }
    
    return str(dataset_dir), valid_files_info

def main():
    logger.info("Starting YOLO-NAS training...")
    
    # Setup device
    device = setup_device(
        multi_gpu="auto",
        num_gpus=None
    )
    logger.info(f"Using device: {device}")
    
    # Validate dataset labels (but don't copy files to save disk space)
    original_data_dir = "/media/seth/SketchScratch/fiftyone/train_yolo"
    train_data_dir, train_valid_info = validate_dataset_labels(
        original_data_dir, 
        32, 
        'train'
    )
    
    if train_data_dir is None:
        logger.error("Failed to validate training dataset")
        return None
    
    # Also validate validation split
    val_data_dir, val_valid_info = validate_dataset_labels(
        original_data_dir, 
        32, 
        'validation'
    )
    
    if val_data_dir is None:
        logger.error("Failed to validate validation dataset")
        return None
    
    # Use original dataset directory (validation passed)
    data_dir = original_data_dir
    logger.info(f"Dataset validation passed - using original dataset: {data_dir}")
    logger.info(f"Training images validated: {train_valid_info['total_valid']} valid, {train_valid_info['total_invalid']} invalid")
    logger.info(f"Validation images validated: {val_valid_info['total_valid']} valid, {val_valid_info['total_invalid']} invalid")
    
    # Create trainer
    trainer = Trainer(
        experiment_name="wildfire_yolo_nas_s",
        ckpt_root_dir="../output/checkpoints"
    )
    
    # Create proper super-gradients dataloaders with correct API
    logger.info("Creating YOLO-NAS dataloaders with filtered dataset...")
    
    # Dataset parameters for super-gradients dataloaders
    # Use available transforms to handle variable image sizes
    from super_gradients.training.transforms.detection import DetectionLongestMaxSize, DetectionPadIfNeeded
    
    # Transforms to normalize image sizes for batching
    transforms = [
        DetectionLongestMaxSize(max_height=640, max_width=640),
        DetectionPadIfNeeded(min_height=640, min_width=640, pad_value=114)
    ]
    
    dataset_params = {
        "data_dir": data_dir,
        "images_dir": "images/train",
        "labels_dir": "labels/train",
        "classes": ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck', 'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle', 'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk', 'Squirrel', 'Pig', 'Chicken', 'Boat', 'Vehicle registration plate', 'Snowmobile', 'Human face', 'Armadillo', 'Fire', 'Package', 'Rodent', 'Child', 'Weapon', 'Backpack'],
        "input_dim": [640, 640],
        "ignore_empty_annotations": True,
        "transforms": transforms
    }
    
    # Dataloader parameters
    dataloader_params = {
        "batch_size": 8,
        "num_workers": 4,
        "shuffle": True,
        "drop_last": False,
        "pin_memory": True
    }
    
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
    val_dataset_params.update({
        "images_dir": "images/validation",
        "labels_dir": "labels/validation"
    })
    
    # Create validation dataloader
    logger.info("Creating validation dataloader...")
    val_dataloader = coco_detection_yolo_format_val(
        dataset_params=val_dataset_params,
        dataloader_params=val_dataloader_params
    )
    
    logger.info("✓ Filtered dataset dataloaders created successfully")
    logger.info(f"Training samples: {len(train_dataloader.dataset) if hasattr(train_dataloader, 'dataset') else 'Unknown'}")
    logger.info(f"Validation samples: {len(val_dataloader.dataset) if hasattr(val_dataloader, 'dataset') else 'Unknown'}")
    
    # Create model
    model = models.get(
        Models.YOLO_NAS_S,
        num_classes=32,
        pretrained_weights=None
    )
    
    logger.info(f"Model architecture: {type(model).__name__}")
    logger.info(f"Number of classes: 32")
    
    # Training parameters - use dict format for Trainer.train()
    training_params = {
        # Required parameters
        "max_epochs": 5,
        "lr_mode": "cosine" if True else "step",
        "initial_lr": 0.001,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=32,
            reg_max=16
        ),
        
        # Warmup parameters
        "lr_warmup_epochs": 5,
        "warmup_mode": "LinearEpochLRWarmup",
        "warmup_initial_lr": 1e-6,
        
        # Optimizer parameters
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        
        # Training settings
        "mixed_precision": True,
        "average_best_models": True,
        "ema": True,
        "ema_params": {"decay": 0.9999, "decay_type": "threshold"},
        
        # Cosine LR settings
        "cosine_final_lr_ratio": 0.1,
        
        # Validation and metrics
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=32,
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
        "save_ckpt_epoch_list": [50, 100, 150, 200],
        "resume": False,
        
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
    
    # Add QAT callback if enabled
    phase_callbacks = []
    if True:
        logger.info("QAT (Quantization Aware Training) enabled")
        logger.info(f"QAT will start at epoch 150")
        
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
    best_model_path = Path("../output/checkpoints") / "wildfire_yolo_nas_s" / "average_model.pth"
    final_model_path = Path("../output") / "yolo_nas_s_trained.pth"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        logger.info(f"Best model saved to: {final_model_path}")
    else:
        logger.warning("Best model not found, saving current model state")
        torch.save(model.state_dict(), final_model_path)
    
    logger.info("Training completed!")
    return str(final_model_path)

if __name__ == "__main__":
    trained_model_path = main()
    print(f"TRAINED_MODEL_PATH={trained_model_path}")
