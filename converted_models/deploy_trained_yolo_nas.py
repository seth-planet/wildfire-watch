#!/usr/bin/env python3.10
"""
YOLO-NAS Deployment Script
Converts trained YOLO-NAS model to INT8 QAT TensorRT and deploys to Frigate

IMPORTANT: This script requires Python 3.10 for super-gradients compatibility
Run with: python3.10 deploy_trained_yolo_nas.py
"""
import os
import sys
import logging
import shutil
import yaml
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
        logging.FileHandler(output_dir / "yolo_nas_deployment.log")
    ]
)
logger = logging.getLogger(__name__)

def convert_model_to_tensorrt(trained_model_path: str) -> Optional[str]:
    """Convert trained YOLO-NAS model to INT8 QAT TensorRT"""
    logger.info("Converting trained YOLO-NAS to TensorRT INT8 QAT...")
    
    from convert_model import EnhancedModelConverter
    
    # Setup conversion parameters
    conversion_params = {
        'model_path': trained_model_path,
        'output_dir': str(output_dir / "converted_yolo_nas"),
        'model_name': 'wildfire_yolo_nas_s',
        'model_size': [(640, 640)],  # Standard Frigate resolution
        'formats': ['onnx', 'tensorrt'],
        'qat_enabled': True,
        'calibration_data': 'wildfire_calibration_data.tar.gz',
        'validate': True,
        'benchmark': True
    }
    
    try:
        # Create converter
        converter = EnhancedModelConverter(
            model_path=conversion_params['model_path'],
            output_dir=conversion_params['output_dir'],
            model_name=conversion_params['model_name'],
            model_size=conversion_params['model_size'],
            calibration_data=conversion_params['calibration_data'],
            qat_enabled=conversion_params['qat_enabled'],
            debug=True
        )
        
        # Run conversion
        logger.info("Starting model conversion (this may take 2-4 hours)...")
        results = converter.convert_all(
            formats=conversion_params['formats'],
            validate=conversion_params['validate'],
            benchmark=conversion_params['benchmark']
        )
        
        if results.get('status') == 'success':
            # Find the TensorRT engine
            engine_pattern = f"{conversion_params['output_dir']}/640x640/*tensorrt*.engine"
            import glob
            engine_files = glob.glob(engine_pattern)
            
            if engine_files:
                tensorrt_engine = engine_files[0]
                logger.info(f"TensorRT engine created: {tensorrt_engine}")
                return tensorrt_engine
            else:
                logger.error("TensorRT engine not found after conversion")
                return None
        else:
            logger.error(f"Model conversion failed: {results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Model conversion failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_frigate_config(tensorrt_engine_path: str, num_classes: int = 32) -> str:
    """Create Frigate configuration for the trained model"""
    logger.info("Creating Frigate configuration...")
    
    # Read the original dataset config to get class names
    dataset_path = Path.home() / "fiftyone" / "train_yolo" / "dataset.yaml"
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    class_names = list(dataset_config['names'].values())
    
    # Create labels file
    labels_content = "\\n".join(class_names)
    labels_path = output_dir / "wildfire_yolo_nas_labels.txt"
    with open(labels_path, 'w') as f:
        f.write(labels_content)
    
    # Create Frigate model configuration
    frigate_config = {
        'model': {
            'path': '/models/wildfire_yolo_nas_s_tensorrt.engine',
            'input_tensor': 'nchw',
            'input_pixel_format': 'rgb',
            'width': 640,
            'height': 640,
            'labelmap_path': '/models/wildfire_yolo_nas_labels.txt',
            'model_type': 'yolov8'  # YOLO-NAS compatible with YOLOv8 format
        },
        'detectors': {
            'tensorrt': {
                'type': 'tensorrt',
                'device': 0  # GPU 0
            },
            'cpu': {
                'type': 'cpu',
                'num_threads': 4
            }
        },
        'objects': {
            'track': ['fire', 'person', 'car', 'truck'],  # Key objects for wildfire detection
            'filters': {
                'fire': {
                    'min_area': 100,
                    'max_area': 100000,
                    'threshold': 0.7
                },
                'person': {
                    'min_area': 500,
                    'max_area': 100000,
                    'threshold': 0.5
                },
                'car': {
                    'min_area': 1000,
                    'max_area': 100000,
                    'threshold': 0.5
                },
                'truck': {
                    'min_area': 2000,
                    'max_area': 100000,
                    'threshold': 0.5
                }
            }
        },
        'motion': {
            'threshold': 20,
            'contour_area': 100,
            'delta_alpha': 0.2,
            'frame_alpha': 0.2,
            'frame_height': 180,
            'mask': ''
        }
    }
    
    # Save Frigate config
    frigate_config_path = output_dir / "wildfire_yolo_nas_frigate_config.yml"
    with open(frigate_config_path, 'w') as f:
        yaml.dump(frigate_config, f, indent=2)
    
    logger.info(f"Frigate configuration saved: {frigate_config_path}")
    logger.info(f"Labels file saved: {labels_path}")
    
    return str(frigate_config_path)

def deploy_to_frigate(tensorrt_engine_path: str, frigate_config_path: str, labels_path: str):
    """Deploy the trained model to Frigate NVR service"""
    logger.info("Deploying trained YOLO-NAS to Frigate...")
    
    # Check if Frigate is running
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=security_nvr", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        if 'security_nvr' not in result.stdout:
            logger.warning("Frigate (security_nvr) container not running")
            logger.info("You may need to start it with: docker-compose up security_nvr")
        
        # Create deployment script
        deployment_script = f'''#!/bin/bash
# Deployment script for trained YOLO-NAS model

set -e

echo "Deploying trained YOLO-NAS to Frigate..."

# Model and config paths
TENSORRT_ENGINE="{tensorrt_engine_path}"
FRIGATE_CONFIG="{frigate_config_path}"
LABELS_FILE="{labels_path}"

# Frigate model directory (adjust path as needed)
FRIGATE_MODEL_DIR="/opt/frigate/models"
FRIGATE_CONFIG_DIR="/opt/frigate/config"

# Copy files to Frigate directories
echo "Copying TensorRT engine..."
docker cp "$TENSORRT_ENGINE" security_nvr:"$FRIGATE_MODEL_DIR/wildfire_yolo_nas_s_tensorrt.engine"

echo "Copying labels file..."
docker cp "$LABELS_FILE" security_nvr:"$FRIGATE_MODEL_DIR/wildfire_yolo_nas_labels.txt"

echo "Copying Frigate config..."
docker cp "$FRIGATE_CONFIG" security_nvr:"$FRIGATE_CONFIG_DIR/wildfire_model_config.yml"

# Restart Frigate to load new model
echo "Restarting Frigate to load new model..."
docker-compose restart security_nvr

echo "Deployment complete!"
echo ""
echo "To use the trained model, update your Frigate configuration to include:"
echo "model:"
echo "  path: /models/wildfire_yolo_nas_s_tensorrt.engine"
echo "  labelmap_path: /models/wildfire_yolo_nas_labels.txt"
echo ""
echo "Monitor Frigate logs: docker-compose logs -f security_nvr"
'''
        
        deploy_script_path = output_dir / "deploy_to_frigate.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deployment_script)
        
        os.chmod(deploy_script_path, 0o755)
        
        logger.info(f"Deployment script created: {deploy_script_path}")
        logger.info("Run the deployment script to install the model in Frigate")
        
        # Optionally run the deployment automatically
        run_deployment = input("Run deployment script now? (y/N): ").lower().strip()
        if run_deployment == 'y':
            logger.info("Running deployment script...")
            result = subprocess.run([str(deploy_script_path)], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Deployment completed successfully!")
                logger.info(result.stdout)
            else:
                logger.error("Deployment failed:")
                logger.error(result.stderr)
        else:
            logger.info(f"To deploy later, run: {deploy_script_path}")
        
        return str(deploy_script_path)
        
    except Exception as e:
        logger.error(f"Deployment preparation failed: {e}")
        return None

def main():
    """Main deployment pipeline"""
    logger.info("YOLO-NAS Deployment Pipeline for Wildfire Watch")
    logger.info("=" * 60)
    
    # Check for trained model
    trained_model_path = output_dir / "yolo_nas_s_trained.pth"
    
    if not trained_model_path.exists():
        logger.error(f"Trained model not found at: {trained_model_path}")
        logger.info("Please run train_yolo_nas.py first to train the model")
        return False
    
    logger.info(f"Found trained model: {trained_model_path}")
    
    try:
        # Step 1: Convert to TensorRT
        tensorrt_engine = convert_model_to_tensorrt(str(trained_model_path))
        
        if not tensorrt_engine:
            logger.error("Model conversion failed!")
            return False
        
        # Step 2: Create Frigate configuration
        frigate_config = create_frigate_config(tensorrt_engine)
        labels_path = output_dir / "wildfire_yolo_nas_labels.txt"
        
        # Step 3: Deploy to Frigate
        deployment_script = deploy_to_frigate(tensorrt_engine, frigate_config, str(labels_path))
        
        if deployment_script:
            logger.info("Deployment pipeline completed successfully!")
            logger.info(f"TensorRT Engine: {tensorrt_engine}")
            logger.info(f"Frigate Config: {frigate_config}")
            logger.info(f"Deployment Script: {deployment_script}")
            return True
        else:
            logger.error("Deployment preparation failed!")
            return False
            
    except Exception as e:
        logger.error(f"Deployment pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("SUCCESS: YOLO-NAS deployment pipeline completed")
    else:
        print("FAILED: Deployment pipeline did not complete successfully")
        sys.exit(1)