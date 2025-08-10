#!/usr/bin/env python3.10
"""
Frigate Integrator for YOLO-NAS Models
Handles integration of YOLO-NAS models with Frigate NVR
"""
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil
import subprocess

logger = logging.getLogger(__name__)


class FrigateIntegrator:
    """Integrates YOLO-NAS models with Frigate NVR"""
    
    # Default Frigate configuration values
    FRIGATE_DEFAULTS = {
        'input_tensor': 'nhwc',  # YOLO models use NHWC format
        'input_pixel_format': 'rgb',
        'width': 640,
        'height': 640,
        'model_type': 'yolov8',  # YOLO-NAS uses YOLOv8 output format
        'fire_class_id': 26,  # Standard fire class ID
        'fire_min_score': 0.5,
        'fire_threshold': 0.7
    }
    
    def __init__(self, model_name: str = "yolo_nas"):
        """Initialize Frigate integrator
        
        Args:
            model_name: Name prefix for generated files
        """
        self.model_name = model_name
        self.logger = logger
    
    def generate_labelmap(self, class_names: List[str], 
                         output_path: Optional[Path] = None) -> Path:
        """Generate labelmap file for Frigate.
        
        Args:
            class_names: List of class names (index = class ID)
            output_path: Optional path to save labelmap
            
        Returns:
            Path to generated labelmap file
        """
        if output_path is None:
            output_path = Path(f"{self.model_name}_labels.txt")
        
        self.logger.info(f"Generating labelmap with {len(class_names)} classes")
        
        # Frigate expects one class name per line
        with open(output_path, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        self.logger.info(f"Labelmap saved to: {output_path}")
        return output_path
    
    def generate_frigate_config(
        self,
        model_path: Path,
        labelmap_path: Path,
        output_path: Optional[Path] = None,
        detector_type: str = 'tensorrt',
        custom_objects: Optional[List[str]] = None,
        model_size: Tuple[int, int] = (640, 640)
    ) -> Path:
        """Generate Frigate configuration for YOLO-NAS model.
        
        Args:
            model_path: Path to model file (engine, hef, etc.)
            labelmap_path: Path to labelmap file
            output_path: Optional path to save config
            detector_type: Detector type ('tensorrt', 'hailo', 'openvino', 'cpu')
            custom_objects: List of object names to track (defaults to ['fire'])
            model_size: Model input size (width, height)
            
        Returns:
            Path to generated config file
        """
        if output_path is None:
            output_path = Path(f"{self.model_name}_frigate_config.yml")
        
        if custom_objects is None:
            custom_objects = ['fire']
        
        self.logger.info(f"Generating Frigate config for {detector_type} detector")
        
        # Base configuration
        config = {
            'detectors': self._get_detector_config(detector_type),
            'model': {
                'path': str(model_path.absolute()),
                'input_tensor': self.FRIGATE_DEFAULTS['input_tensor'],
                'input_pixel_format': self.FRIGATE_DEFAULTS['input_pixel_format'],
                'width': model_size[0],
                'height': model_size[1],
                'labelmap_path': str(labelmap_path.absolute()),
                'model_type': self.FRIGATE_DEFAULTS['model_type']
            },
            'objects': {
                'track': custom_objects,
                'filters': self._get_object_filters(custom_objects)
            }
        }
        
        # Detector-specific model configuration
        if detector_type == 'hailo':
            # Hailo models need specific input configuration
            config['model']['input_tensor'] = 'nhwc'
            config['model']['input_pixel_format'] = 'rgb'
        elif detector_type == 'openvino':
            # OpenVINO might need different settings
            config['model']['input_tensor'] = 'nchw'
        
        # Save configuration
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"Frigate config saved to: {output_path}")
        return output_path
    
    def _get_detector_config(self, detector_type: str) -> Dict[str, Any]:
        """Get detector configuration based on type"""
        detector_configs = {
            'tensorrt': {
                'tensorrt': {
                    'type': 'tensorrt',
                    'device': 0  # GPU device ID
                }
            },
            'hailo': {
                'hailo': {
                    'type': 'hailo',
                    'device': 'PCIe',  # or 'USB' for Hailo-8 USB
                    'num_threads': 3
                }
            },
            'openvino': {
                'openvino': {
                    'type': 'openvino',
                    'device': 'AUTO'  # AUTO, CPU, GPU, etc.
                }
            },
            'cpu': {
                'cpu': {
                    'type': 'cpu',
                    'num_threads': 4
                }
            }
        }
        
        return detector_configs.get(detector_type, detector_configs['cpu'])
    
    def _get_object_filters(self, object_names: List[str]) -> Dict[str, Dict]:
        """Get object filter configuration"""
        filters = {}
        
        for obj_name in object_names:
            if obj_name.lower() == 'fire':
                # Special configuration for fire detection
                filters[obj_name] = {
                    'min_score': self.FRIGATE_DEFAULTS['fire_min_score'],
                    'threshold': self.FRIGATE_DEFAULTS['fire_threshold'],
                    'min_area': 100,  # Minimum detection area
                    'max_area': 500000,  # Maximum detection area
                }
            else:
                # Default configuration for other objects
                filters[obj_name] = {
                    'min_score': 0.5,
                    'threshold': 0.6
                }
        
        return filters
    
    def create_deployment_package(
        self,
        model_path: Path,
        output_dir: Path,
        class_names: List[str],
        detector_type: str = 'tensorrt',
        include_test_config: bool = True
    ) -> Dict[str, Path]:
        """Create complete deployment package for Frigate.
        
        Args:
            model_path: Path to model file
            output_dir: Directory to create deployment package
            class_names: List of class names
            detector_type: Detector type for Frigate
            include_test_config: Whether to include test configuration
            
        Returns:
            Dictionary of created file paths
        """
        self.logger.info(f"Creating Frigate deployment package in {output_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = {}
        
        # Copy model file
        model_dest = output_dir / model_path.name
        if model_path != model_dest:
            shutil.copy2(model_path, model_dest)
        created_files['model'] = model_dest
        
        # Generate labelmap
        labelmap_path = self.generate_labelmap(
            class_names, 
            output_dir / f"{self.model_name}_labels.txt"
        )
        created_files['labelmap'] = labelmap_path
        
        # Generate Frigate config
        config_path = self.generate_frigate_config(
            model_dest,
            labelmap_path,
            output_dir / f"{self.model_name}_frigate.yml",
            detector_type
        )
        created_files['config'] = config_path
        
        # Create deployment script
        deploy_script = self._create_deployment_script(
            model_dest, labelmap_path, config_path, detector_type
        )
        deploy_script_path = output_dir / "deploy_to_frigate.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        deploy_script_path.chmod(0o755)
        created_files['deploy_script'] = deploy_script_path
        
        # Create test configuration if requested
        if include_test_config:
            test_config = self._create_test_config(detector_type)
            test_config_path = output_dir / "test_config.yml"
            with open(test_config_path, 'w') as f:
                yaml.dump(test_config, f)
            created_files['test_config'] = test_config_path
        
        # Create README
        readme_content = self._create_readme(
            model_path.name, detector_type, class_names
        )
        readme_path = output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        created_files['readme'] = readme_path
        
        self.logger.info(f"Deployment package created with {len(created_files)} files")
        return created_files
    
    def _create_deployment_script(self, model_path: Path, labelmap_path: Path,
                                 config_path: Path, detector_type: str) -> str:
        """Create deployment script for Frigate"""
        script = f"""#!/bin/bash
# Frigate deployment script for {self.model_name}
# Generated by Wildfire Watch

set -e

echo "Deploying {self.model_name} to Frigate..."

# Check if Frigate directory exists
FRIGATE_DIR="${{FRIGATE_DIR:-/opt/frigate}}"
if [ ! -d "$FRIGATE_DIR" ]; then
    echo "Error: Frigate directory not found at $FRIGATE_DIR"
    echo "Set FRIGATE_DIR environment variable to correct path"
    exit 1
fi

# Create models directory if it doesn't exist
MODELS_DIR="$FRIGATE_DIR/models"
mkdir -p "$MODELS_DIR"

# Copy model files
echo "Copying model files..."
cp "{model_path.name}" "$MODELS_DIR/"
cp "{labelmap_path.name}" "$MODELS_DIR/"

# Backup existing config
if [ -f "$FRIGATE_DIR/config.yml" ]; then
    echo "Backing up existing config..."
    cp "$FRIGATE_DIR/config.yml" "$FRIGATE_DIR/config.yml.backup"
fi

# Merge configuration
echo "Updating Frigate configuration..."
echo ""
echo "Add the following to your Frigate config.yml:"
echo "----------------------------------------"
cat "{config_path.name}"
echo "----------------------------------------"

# Restart Frigate if running
if command -v docker &> /dev/null; then
    if docker ps | grep -q frigate; then
        echo "Restarting Frigate container..."
        docker restart frigate
    fi
fi

echo ""
echo "Deployment complete!"
echo "Model: $MODELS_DIR/{model_path.name}"
echo "Labels: $MODELS_DIR/{labelmap_path.name}"
echo ""
echo "Remember to:"
echo "1. Update your Frigate config.yml with the model configuration"
echo "2. Restart Frigate to load the new model"
echo "3. Configure your cameras to use the '{detector_type}' detector"
"""
        return script
    
    def _create_test_config(self, detector_type: str) -> Dict[str, Any]:
        """Create test configuration for validating deployment"""
        return {
            'mqtt': {
                'host': 'mqtt_broker',
                'port': 1883,
                'topic_prefix': 'frigate'
            },
            'detectors': self._get_detector_config(detector_type),
            'cameras': {
                'test_camera': {
                    'ffmpeg': {
                        'inputs': [{
                            'path': 'rtsp://admin:password@192.168.1.100:554/stream',
                            'roles': ['detect']
                        }]
                    },
                    'detect': {
                        'width': 640,
                        'height': 640,
                        'fps': 5
                    }
                }
            },
            'objects': {
                'track': ['fire', 'person'],
                'filters': self._get_object_filters(['fire', 'person'])
            }
        }
    
    def _create_readme(self, model_name: str, detector_type: str, 
                      class_names: List[str]) -> str:
        """Create README for deployment package"""
        fire_idx = class_names.index('fire') if 'fire' in class_names else 26
        
        return f"""# Frigate Deployment Package for {self.model_name}

## Contents
- `{model_name}` - The converted model file
- `{self.model_name}_labels.txt` - Class labels (labelmap)
- `{self.model_name}_frigate.yml` - Frigate configuration
- `deploy_to_frigate.sh` - Deployment script
- `test_config.yml` - Test configuration

## Deployment Instructions

### Automatic Deployment
```bash
./deploy_to_frigate.sh
```

### Manual Deployment
1. Copy model files to Frigate models directory:
   ```bash
   cp {model_name} /opt/frigate/models/
   cp {self.model_name}_labels.txt /opt/frigate/models/
   ```

2. Update your Frigate config.yml with the configuration from `{self.model_name}_frigate.yml`

3. Restart Frigate:
   ```bash
   docker restart frigate
   ```

## Model Information
- **Model Type**: YOLO-NAS
- **Detector**: {detector_type}
- **Input Size**: 640x640
- **Classes**: {len(class_names)}
- **Fire Class ID**: {fire_idx}

## Configuration
The model is configured to detect the following objects:
{', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}

Fire detection is configured with:
- Minimum score: {self.FRIGATE_DEFAULTS['fire_min_score']}
- Threshold: {self.FRIGATE_DEFAULTS['fire_threshold']}

## Testing
1. Configure a camera in Frigate
2. Monitor MQTT for fire detection events:
   ```bash
   mosquitto_sub -h localhost -t frigate/+/fire
   ```

## Troubleshooting
- Check Frigate logs: `docker logs frigate`
- Verify model loaded: Look for "Model loaded successfully" in logs
- Test inference: Use Frigate's debug UI

## Performance
Expected performance on typical hardware:
- RTX 4090: 5-8ms per frame
- RTX 3070: 8-12ms per frame
- Hailo-8: 10-15ms per frame
- CPU: 50-100ms per frame
"""
    
    def validate_deployment(self, frigate_url: str = "http://localhost:5000",
                          timeout: int = 30) -> bool:
        """Validate Frigate deployment by checking API.
        
        Args:
            frigate_url: Frigate web UI URL
            timeout: Timeout for validation
            
        Returns:
            True if deployment is valid
        """
        import requests
        from time import sleep
        
        self.logger.info(f"Validating Frigate deployment at {frigate_url}")
        
        # Check if Frigate is running
        try:
            response = requests.get(f"{frigate_url}/api/config", timeout=5)
            if response.status_code == 200:
                config = response.json()
                self.logger.info("Frigate is running and accessible")
                
                # Check if model is loaded
                if 'model' in config:
                    self.logger.info(f"Model configured: {config['model'].get('path', 'Unknown')}")
                    return True
                else:
                    self.logger.warning("No model configured in Frigate")
                    return False
            else:
                self.logger.error(f"Frigate API returned status {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Frigate: {e}")
            return False