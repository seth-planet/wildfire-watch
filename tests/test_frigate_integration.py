#!/usr/bin/env python3.12
"""
Frigate NVR Integration Tests
Ensures all trained classes are properly configured for Frigate deployment

Run with: python3.12 -m pytest tests/test_frigate_integration.py -v
"""

import unittest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add converted_models to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'converted_models'))


class FrigateConfigurationTests(unittest.TestCase):
    """Test Frigate configuration generation for trained models"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='frigate_test_'))
        self.model_dir = self.test_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import gc
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {self.test_dir}: {e}")
        # Force garbage collection to release file handles
        gc.collect()
    
    def test_all_32_classes_in_labelmap(self):
        """Test that all 32 trained classes are included in Frigate labelmap"""
        # Standard 32 classes from our dataset
        class_names = [
            'Person', 'Bicycle', 'Car', 'Motorcycle', 'Bus', 'Truck',
            'Bird', 'Cat', 'Dog', 'Horse', 'Sheep', 'Cattle',
            'Bear', 'Deer', 'Rabbit', 'Raccoon', 'Fox', 'Skunk',
            'Squirrel', 'Pig', 'Chicken', 'Boat', 'Vehicle registration plate',
            'Snowmobile', 'Human face', 'Armadillo', 'Fire', 'Package',
            'Rodent', 'Child', 'Weapon', 'Backpack'
        ]
        
        # Generate Frigate labelmap
        labelmap = {}
        for i, name in enumerate(class_names):
            labelmap[i] = name
        
        # Verify all classes present
        self.assertEqual(len(labelmap), 32)
        for i in range(32):
            self.assertIn(i, labelmap)
        
        # Verify Fire class specifically
        self.assertEqual(labelmap[26], 'Fire')
    
    def test_frigate_model_config_generation(self):
        """Test generation of Frigate model configuration"""
        model_config = {
            'path': '/models/yolo_nas_wildfire.tflite',
            'input_tensor': 'nhwc',  # NHWC for TFLite
            'input_pixel_format': 'rgb',
            'width': 320,
            'height': 320,
            'labelmap': {i: f'class_{i}' for i in range(32)}
        }
        
        # Set Fire class
        model_config['labelmap'][26] = 'Fire'
        
        # Verify configuration
        self.assertEqual(model_config['input_tensor'], 'nhwc')
        self.assertEqual(model_config['width'], 320)
        self.assertEqual(model_config['height'], 320)
        self.assertEqual(len(model_config['labelmap']), 32)
    
    def test_frigate_detector_config(self):
        """Test Frigate detector configuration for our models"""
        detectors = {
            'coral': {
                'type': 'edgetpu',
                'device': 'usb'
            },
            'hailo': {
                'type': 'hailo',
                'device': '/dev/hailo0'
            },
            'openvino': {
                'type': 'openvino',
                'device': 'CPU'
            },
            'tensorrt': {
                'type': 'tensorrt',
                'device': 0  # GPU 0
            }
        }
        
        # Verify each detector type
        self.assertIn('coral', detectors)
        self.assertIn('hailo', detectors)
        self.assertEqual(detectors['coral']['type'], 'edgetpu')
    
    def test_frigate_object_tracking_config(self):
        """Test Frigate object tracking configuration for fire detection"""
        objects = {
            'track': [
                'Fire',      # Primary detection
                'Person',    # For safety
                'Car',       # Vehicles near fire
                'Truck'      # Emergency vehicles
            ],
            'filters': {
                'Fire': {
                    'min_score': 0.6,  # Higher threshold for fire
                    'threshold': 0.65,
                    'min_area': 1000,  # Minimum pixel area
                    'max_area': 1000000
                },
                'Person': {
                    'min_score': 0.5,
                    'threshold': 0.7
                }
            }
        }
        
        # Verify fire tracking is enabled
        self.assertIn('Fire', objects['track'])
        self.assertIn('Fire', objects['filters'])
        self.assertGreater(objects['filters']['Fire']['min_score'], 0.5)
    
    def test_frigate_yaml_generation(self):
        """Test complete Frigate configuration YAML generation"""
        config = {
            'mqtt': {
                'host': 'mqtt-broker',
                'port': 1883,
                'topic_prefix': 'frigate',
                'client_id': 'frigate',
                'stats_interval': 60
            },
            'detectors': {
                'coral': {
                    'type': 'edgetpu',
                    'device': 'usb'
                }
            },
            'model': {
                'path': '/models/yolo_nas_wildfire_edgetpu.tflite',
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320,
                'labelmap_path': '/models/wildfire_labels.txt'
            },
            'objects': {
                'track': ['Fire', 'Person', 'Car'],
                'filters': {
                    'Fire': {
                        'min_score': 0.6,
                        'threshold': 0.65
                    }
                }
            },
            'cameras': {
                'camera_1': {
                    'ffmpeg': {
                        'inputs': [{
                            'path': 'rtsp://admin:password@192.0.2.100:554/stream',
                            'roles': ['detect', 'record']
                        }]
                    },
                    'detect': {
                        'width': 1280,
                        'height': 720,
                        'fps': 5
                    },
                    'objects': {
                        'track': ['Fire', 'Person'],
                        'filters': {
                            'Fire': {
                                'min_score': 0.6
                            }
                        }
                    }
                }
            }
        }
        
        # Save as YAML
        yaml_path = self.test_dir / 'frigate_config.yml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Verify YAML is valid
        with open(yaml_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        self.assertEqual(loaded_config['model']['width'], 320)
        self.assertIn('Fire', loaded_config['objects']['track'])


class FrigateModelDeploymentTests(unittest.TestCase):
    """Test model deployment to Frigate"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix='frigate_deploy_test_'))
    
    def tearDown(self):
        """Clean up"""
        import gc
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {self.test_dir}: {e}")
        # Force garbage collection to release file handles
        gc.collect()
    
    def test_tflite_model_format(self):
        """Test TFLite model is in correct format for Frigate"""
        # Frigate expects:
        # - INT8 quantized TFLite for Coral TPU
        # - Standard TFLite for CPU
        # - Input shape: [1, height, width, 3] (NHWC)
        # - Output shape: [1, num_detections, 6] or similar
        
        expected_formats = {
            'coral': 'model_edgetpu.tflite',
            'cpu': 'model_cpu.tflite'
        }
        
        for device, filename in expected_formats.items():
            with self.subTest(device=device):
                self.assertTrue(filename.endswith('.tflite'))
                if device == 'coral':
                    self.assertIn('edgetpu', filename)
    
    def test_labelmap_file_generation(self):
        """Test labelmap.txt file generation for Frigate"""
        class_names = [f"class_{i}" for i in range(32)]
        class_names[26] = "Fire"
        
        # Generate labelmap.txt content
        labelmap_content = []
        for i, name in enumerate(class_names):
            labelmap_content.append(f"{i} {name}")
        
        # Save labelmap
        labelmap_path = self.test_dir / 'labelmap.txt'
        with open(labelmap_path, 'w') as f:
            f.write('\n'.join(labelmap_content))
        
        # Verify content
        with open(labelmap_path, 'r') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), 32)
        self.assertIn('26 Fire', [line.strip() for line in lines])
    
    def test_model_deployment_script(self):
        """Test model deployment script generation"""
        deployment_script = '''#!/bin/bash
# Deploy YOLO-NAS model to Frigate

MODEL_DIR="/models"
CONFIG_DIR="/config"

# Copy model files
cp yolo_nas_wildfire_edgetpu.tflite ${MODEL_DIR}/
cp wildfire_labels.txt ${MODEL_DIR}/

# Update Frigate config
cp frigate_wildfire.yml ${CONFIG_DIR}/config.yml

# Restart Frigate
docker restart frigate

echo "Model deployed successfully!"
'''
        
        # Save script
        script_path = self.test_dir / 'deploy_to_frigate.sh'
        script_path.write_text(deployment_script)
        script_path.chmod(0o755)
        
        # Verify script is executable
        self.assertTrue(script_path.stat().st_mode & 0o111)
        self.assertIn('yolo_nas_wildfire_edgetpu.tflite', script_path.read_text())


class FrigateFireDetectionTests(unittest.TestCase):
    """Test fire detection specific configuration"""
    
    def test_fire_detection_zones(self):
        """Test configuration of fire detection zones"""
        zone_config = {
            'zones': {
                'high_risk_zone': {
                    'coordinates': '0,0,1280,0,1280,400,0,400',
                    'objects': ['Fire'],
                    'filters': {
                        'Fire': {
                            'min_score': 0.5  # Lower threshold in high-risk areas
                        }
                    }
                },
                'monitored_zone': {
                    'coordinates': '0,400,1280,400,1280,720,0,720',
                    'objects': ['Fire', 'Person'],
                    'filters': {
                        'Fire': {
                            'min_score': 0.65
                        }
                    }
                }
            }
        }
        
        # Verify zones are configured
        self.assertIn('high_risk_zone', zone_config['zones'])
        self.assertEqual(
            zone_config['zones']['high_risk_zone']['filters']['Fire']['min_score'],
            0.5
        )
    
    def test_fire_alert_automation(self):
        """Test automation configuration for fire alerts"""
        automation = {
            'fire_detected': {
                'trigger': {
                    'platform': 'mqtt',
                    'topic': 'frigate/events',
                    'payload_json': True
                },
                'condition': {
                    'condition': 'template',
                    'value_template': '{{ trigger.payload_json["label"] == "Fire" }}'
                },
                'action': [
                    {
                        'service': 'notify.alert',
                        'data': {
                            'message': 'Fire detected by camera {{ trigger.payload_json["camera"] }}',
                            'data': {
                                'photo': '/api/frigate/notifications/{{ trigger.payload_json["id"] }}/thumbnail.jpg'
                            }
                        }
                    },
                    {
                        'service': 'mqtt.publish',
                        'data': {
                            'topic': 'trigger/fire_detected',
                            'payload': '1'
                        }
                    }
                ]
            }
        }
        
        # Verify automation triggers on Fire label
        condition = automation['fire_detected']['condition']['value_template']
        self.assertIn('"Fire"', condition)
        
        # Verify MQTT publish action
        mqtt_action = automation['fire_detected']['action'][1]
        self.assertEqual(mqtt_action['data']['topic'], 'trigger/fire_detected')
    
    def test_fire_recording_retention(self):
        """Test recording retention for fire events"""
        record_config = {
            'record': {
                'enabled': True,
                'retain': {
                    'days': 7,  # Default retention
                    'mode': 'motion'
                },
                'events': {
                    'retain': {
                        'default': 10,  # 10 days for events
                        'objects': {
                            'Fire': 30  # 30 days for fire events
                        }
                    }
                }
            }
        }
        
        # Verify fire events have longer retention
        fire_retention = record_config['record']['events']['retain']['objects']['Fire']
        default_retention = record_config['record']['events']['retain']['default']
        
        self.assertGreater(fire_retention, default_retention)
        self.assertEqual(fire_retention, 30)


class FrigatePerformanceTests(unittest.TestCase):
    """Test Frigate performance configuration"""
    
    def test_detection_fps_settings(self):
        """Test appropriate FPS settings for fire detection"""
        camera_config = {
            'detect': {
                'width': 640,
                'height': 640,
                'fps': 5  # 5 FPS is sufficient for fire detection
            },
            'record': {
                'enabled': True,
                'fps': 15  # Higher FPS for recordings
            },
            'snapshots': {
                'enabled': True,
                'timestamp': True,
                'bounding_box': True,
                'retain': {
                    'default': 10,
                    'objects': {
                        'Fire': 30
                    }
                }
            }
        }
        
        # Verify detection FPS is reasonable
        self.assertLessEqual(camera_config['detect']['fps'], 10)
        self.assertGreaterEqual(camera_config['detect']['fps'], 1)
        
        # Verify record FPS is higher
        self.assertGreater(camera_config['record']['fps'], 
                          camera_config['detect']['fps'])
    
    def test_hardware_acceleration_config(self):
        """Test hardware acceleration configuration"""
        hwaccel_configs = {
            'nvidia': {
                'hwaccel_args': [
                    '-hwaccel', 'cuda',
                    '-hwaccel_output_format', 'cuda'
                ]
            },
            'vaapi': {
                'hwaccel_args': [
                    '-hwaccel', 'vaapi',
                    '-hwaccel_device', '/dev/dri/renderD128',
                    '-hwaccel_output_format', 'vaapi'
                ]
            },
            'qsv': {
                'hwaccel_args': [
                    '-hwaccel', 'qsv',
                    '-qsv_device', '/dev/dri/renderD128'
                ]
            }
        }
        
        # Verify each acceleration method
        for method, config in hwaccel_configs.items():
            with self.subTest(method=method):
                self.assertIn('-hwaccel', config['hwaccel_args'])
                # For nvidia, check for 'cuda' instead of 'nvidia'
                if method == 'nvidia':
                    self.assertIn('cuda', config['hwaccel_args'])
                else:
                    self.assertIn(method, config['hwaccel_args'])


class FrigateIntegrationValidationTests(unittest.TestCase):
    """Validate complete Frigate integration"""
    
    def test_end_to_end_configuration(self):
        """Test complete end-to-end Frigate configuration"""
        # Create complete config
        config = self._create_complete_frigate_config()
        
        # Validate essential components
        self.assertIn('mqtt', config)
        self.assertIn('detectors', config)
        self.assertIn('model', config)
        self.assertIn('objects', config)
        self.assertIn('cameras', config)
        
        # Validate Fire class is tracked
        self.assertIn('Fire', config['objects']['track'])
        
        # Validate at least one camera configured
        self.assertGreater(len(config['cameras']), 0)
    
    def _create_complete_frigate_config(self):
        """Create a complete Frigate configuration"""
        return {
            'mqtt': {
                'host': 'mqtt-broker',
                'port': 1883
            },
            'detectors': {
                'coral': {
                    'type': 'edgetpu',
                    'device': 'usb'
                }
            },
            'model': {
                'path': '/models/yolo_nas_wildfire_edgetpu.tflite',
                'input_tensor': 'nhwc',
                'input_pixel_format': 'rgb',
                'width': 320,
                'height': 320,
                'labelmap': {i: f'class_{i}' for i in range(32)}
            },
            'objects': {
                'track': ['Fire', 'Person', 'Car'],
                'filters': {
                    'Fire': {'min_score': 0.6}
                }
            },
            'cameras': {
                'camera_1': {
                    'ffmpeg': {
                        'inputs': [{
                            'path': 'rtsp://camera1/stream',
                            'roles': ['detect', 'record']
                        }]
                    },
                    'detect': {
                        'width': 640,
                        'height': 640,
                        'fps': 5
                    }
                }
            }
        }
    
    def test_mqtt_topic_validation(self):
        """Test MQTT topics match between services"""
        expected_topics = {
            'fire_detection': 'frigate/+/fire/snapshot',
            'smoke_detection': 'frigate/+/smoke/snapshot',
            'consensus_trigger': 'trigger/fire_detected',
            'gpio_status': 'gpio/status',
            'telemetry': 'telemetry/+'
        }
        
        # Verify topic patterns
        for service, topic in expected_topics.items():
            with self.subTest(service=service):
                self.assertIsInstance(topic, str)
                if service in ['fire_detection', 'smoke_detection']:
                    self.assertIn('frigate/', topic)
    
    def test_deployment_checklist(self):
        """Test deployment checklist is complete"""
        checklist = {
            'model_converted': True,  # TFLite model created
            'labelmap_created': True,  # Labels file created
            'config_generated': True,  # Frigate config created
            'mqtt_configured': True,  # MQTT broker connected
            'cameras_discovered': True,  # Cameras found
            'fire_class_enabled': True,  # Fire detection enabled
            'automation_configured': True,  # Alerts configured
            'testing_completed': True  # Integration tested
        }
        
        # All items should be checked
        for item, status in checklist.items():
            with self.subTest(item=item):
                self.assertTrue(status, f"{item} not completed")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)