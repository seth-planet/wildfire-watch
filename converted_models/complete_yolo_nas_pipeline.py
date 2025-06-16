#!/usr/bin/env python3.10
"""
Complete YOLO-NAS Pipeline for Wildfire Watch
Orchestrates training, conversion, and deployment

IMPORTANT: This script requires Python 3.10 for super-gradients compatibility
Run with: python3.10 complete_yolo_nas_pipeline.py
"""
import os
import sys
import logging
import time
import subprocess
from pathlib import Path
from typing import Optional

# Setup logging
output_dir = Path("../output")
output_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "complete_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check all prerequisites for the pipeline"""
    logger.info("Checking prerequisites...")
    
    # Check dataset
    dataset_path = Path.home() / "fiftyone" / "train_yolo"
    if not dataset_path.exists():
        logger.error(f"Dataset not found at: {dataset_path}")
        return False
    
    if not (dataset_path / "dataset.yaml").exists():
        logger.error("dataset.yaml not found")
        return False
    
    # Check images and labels
    train_images = dataset_path / "images" / "train"
    train_labels = dataset_path / "labels" / "train"
    
    if not train_images.exists() or not train_labels.exists():
        logger.error("Training images or labels directory not found")
        return False
    
    # Count files
    image_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
    label_count = len(list(train_labels.glob("*.txt")))
    
    logger.info(f"Found {image_count} training images and {label_count} labels")
    
    if image_count < 100:
        logger.warning(f"Low image count ({image_count}). Training may not be effective.")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if gpu_memory < 8:
                logger.warning(f"Low GPU memory ({gpu_memory:.1f} GB). Consider reducing batch size.")
        else:
            logger.error("CUDA not available. GPU training required for reasonable training times.")
            return False
    except ImportError:
        logger.warning("PyTorch not installed. Will be installed during setup.")
    
    # Check disk space
    import shutil
    free_space = shutil.disk_usage(output_dir).free / 1e9
    logger.info(f"Available disk space: {free_space:.1f} GB")
    
    if free_space < 50:
        logger.warning(f"Low disk space ({free_space:.1f} GB). Training may require 20-50 GB.")
    
    return True

def run_training_phase() -> Optional[str]:
    """Run the training phase"""
    logger.info("=" * 60)
    logger.info("PHASE 1: MODEL TRAINING")
    logger.info("=" * 60)
    
    current_dir = Path(__file__).parent
    training_script = current_dir / "unified_yolo_trainer.py"
    
    if not training_script.exists():
        logger.error(f"Training script not found: {training_script}")
        return None
    
    logger.info("Starting YOLO-NAS training...")
    logger.info("This phase will take 48-72 hours depending on your GPU")
    logger.info("Training progress will be logged to output/yolo_nas_training.log")
    
    start_time = time.time()
    
    try:
        # Create config file for training
        config_file = current_dir / "pipeline_config.yaml"
        with open(config_file, 'w') as f:
            f.write("""model:
  architecture: yolo_nas_s
  num_classes: 32
  input_size: [640, 640]
  pretrained_weights: null

dataset:
  data_dir: /home/seth/fiftyone/train_yolo
  train_split: train
  val_split: validation
  class_names: []  # Will be auto-detected
  validate_labels: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  warmup_epochs: 3
  lr_scheduler: cosine
  lr_decay_factor: 0.1
  workers: 4
  mixed_precision: false
  gradient_accumulation: 1
  early_stopping: true
  patience: 50
  weight_decay: 0.0001
  optimizer: Adam

qat:
  enabled: true
  start_epoch: 75
  calibration_batches: 100

validation:
  interval: 10
  conf_threshold: 0.25
  iou_threshold: 0.45
  max_predictions: 300

output_dir: ../output
experiment_name: yolo_nas_wildfire
log_level: INFO""")
        
        # Run training with very long timeout (72 hours)
        result = subprocess.run(
            [sys.executable, str(training_script), "--config", str(config_file)],
            capture_output=True,
            text=True,
            timeout=72 * 60 * 60  # 72 hours
        )
        
        elapsed_time = (time.time() - start_time) / 3600  # hours
        
        if result.returncode == 0:
            logger.info(f"Training completed successfully in {elapsed_time:.1f} hours!")
            
            # Check for trained model
            trained_model = output_dir / "yolo_nas_s_trained.pth"
            if trained_model.exists():
                logger.info(f"Trained model available: {trained_model}")
                return str(trained_model)
            else:
                logger.error("Training completed but model file not found")
                return None
        else:
            logger.error(f"Training failed after {elapsed_time:.1f} hours")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        elapsed_time = (time.time() - start_time) / 3600
        logger.error(f"Training timed out after {elapsed_time:.1f} hours")
        return None
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None
    except Exception as e:
        elapsed_time = (time.time() - start_time) / 3600
        logger.error(f"Training failed after {elapsed_time:.1f} hours with exception: {e}")
        return None

def run_conversion_phase(trained_model_path: str) -> Optional[str]:
    """Run the conversion phase"""
    logger.info("=" * 60)
    logger.info("PHASE 2: MODEL CONVERSION")
    logger.info("=" * 60)
    
    deployment_script = Path("deploy_trained_yolo_nas.py")
    
    if not deployment_script.exists():
        logger.error(f"Deployment script not found: {deployment_script}")
        return None
    
    logger.info("Converting trained model to TensorRT INT8 QAT...")
    logger.info("This phase will take 2-4 hours")
    
    start_time = time.time()
    
    try:
        # Run conversion with 4 hour timeout
        result = subprocess.run(
            [sys.executable, str(deployment_script)],
            capture_output=True,
            text=True,
            timeout=4 * 60 * 60  # 4 hours
        )
        
        elapsed_time = (time.time() - start_time) / 3600  # hours
        
        if result.returncode == 0:
            logger.info(f"Conversion completed successfully in {elapsed_time:.1f} hours!")
            
            # Look for TensorRT engine
            import glob
            engine_pattern = str(output_dir / "converted_yolo_nas" / "640x640" / "*tensorrt*.engine")
            engine_files = glob.glob(engine_pattern)
            
            if engine_files:
                engine_path = engine_files[0]
                logger.info(f"TensorRT engine created: {engine_path}")
                return engine_path
            else:
                logger.error("Conversion completed but TensorRT engine not found")
                return None
        else:
            logger.error(f"Conversion failed after {elapsed_time:.1f} hours")
            logger.error(f"Error output: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        elapsed_time = (time.time() - start_time) / 3600
        logger.error(f"Conversion timed out after {elapsed_time:.1f} hours")
        return None
    except Exception as e:
        elapsed_time = (time.time() - start_time) / 3600
        logger.error(f"Conversion failed after {elapsed_time:.1f} hours with exception: {e}")
        return None

def run_deployment_phase(tensorrt_engine_path: str) -> bool:
    """Run the deployment phase"""
    logger.info("=" * 60)
    logger.info("PHASE 3: FRIGATE DEPLOYMENT")
    logger.info("=" * 60)
    
    # Check if deployment script exists
    deploy_script = output_dir / "deploy_to_frigate.sh"
    
    if not deploy_script.exists():
        logger.error("Deployment script not found. Conversion phase may have failed.")
        return False
    
    logger.info(f"TensorRT engine ready: {tensorrt_engine_path}")
    logger.info(f"Deployment script: {deploy_script}")
    
    # Auto-deploy for automated execution
    logger.info("\\n" + "=" * 60)
    logger.info("DEPLOYMENT READY")
    logger.info("=" * 60)
    logger.info(f"Trained YOLO-NAS model converted to TensorRT INT8 QAT")
    logger.info(f"Engine: {tensorrt_engine_path}")
    logger.info(f"Size: {Path(tensorrt_engine_path).stat().st_size / 1e6:.1f} MB")
    logger.info("")
    logger.info("To deploy to Frigate:")
    logger.info(f"1. Run: {deploy_script}")
    logger.info("2. Update your Frigate configuration")
    logger.info("3. Restart the security_nvr service")
    logger.info("")
    
    # Auto-deploy for automation
    deploy_now = True
    
    if deploy_now:
        logger.info("Deploying to Frigate...")
        
        try:
            result = subprocess.run(
                [str(deploy_script)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )
            
            if result.returncode == 0:
                logger.info("Deployment completed successfully!")
                logger.info("Check Frigate logs: docker-compose logs -f security_nvr")
                return True
            else:
                logger.error("Deployment failed:")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            return False
    else:
        logger.info("Deployment skipped. You can deploy later using the provided script.")
        return True

def main():
    """Run the complete pipeline"""
    logger.info("YOLO-NAS Complete Pipeline for Wildfire Watch")
    logger.info("=" * 60)
    logger.info("This pipeline will:")
    logger.info("1. Train YOLO-NAS-S on your custom dataset (48-72 hours)")
    logger.info("2. Convert to TensorRT INT8 QAT (2-4 hours)")
    logger.info("3. Deploy to Frigate NVR service")
    logger.info("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please fix issues and try again.")
        return False
    
    # Auto-confirm for non-interactive execution
    logger.info("\\nStarting training automatically...")
    logger.info("This will take a very long time!")
    
    start_time = time.time()
    
    try:
        # Phase 1: Training
        trained_model = run_training_phase()
        if not trained_model:
            logger.error("Training phase failed!")
            return False
        
        # Phase 2: Conversion
        tensorrt_engine = run_conversion_phase(trained_model)
        if not tensorrt_engine:
            logger.error("Conversion phase failed!")
            return False
        
        # Phase 3: Deployment
        deployment_success = run_deployment_phase(tensorrt_engine)
        
        total_time = (time.time() - start_time) / 3600  # hours
        
        if deployment_success:
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time:.1f} hours")
            logger.info(f"Trained model: {trained_model}")
            logger.info(f"TensorRT engine: {tensorrt_engine}")
            logger.info("Your custom YOLO-NAS model is ready for wildfire detection!")
            return True
        else:
            logger.warning("Pipeline completed but deployment had issues")
            logger.info(f"You can manually deploy using the generated scripts")
            return True
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return False
    except Exception as e:
        total_time = (time.time() - start_time) / 3600
        logger.error(f"Pipeline failed after {total_time:.1f} hours with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\\nSUCCESS: Complete YOLO-NAS pipeline finished!")
    else:
        print("\\nFAILED: Pipeline did not complete successfully")
        sys.exit(1)