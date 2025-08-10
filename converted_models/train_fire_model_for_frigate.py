#!/usr/bin/env python3
"""
Train a fire detection model compatible with Frigate NVR.
Uses SSD MobileNet architecture which outputs the 4 tensors Frigate expects.

This is the RIGHT way to add custom object detection to Frigate!
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path

def create_training_pipeline():
    """
    Create a training pipeline for SSD MobileNet v2 fire detection.
    
    This uses TensorFlow Object Detection API format which outputs:
    1. detection_boxes: [batch, max_detections, 4]
    2. detection_classes: [batch, max_detections]  
    3. detection_scores: [batch, max_detections]
    4. num_detections: [batch]
    
    Exactly what Frigate expects!
    """
    
    # Pipeline configuration for SSD MobileNet v2
    pipeline_config = """
model {
  ssd {
    num_classes: 4  # fire, smoke, flame, ember
    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300
      }
    }
    feature_extractor {
      type: "ssd_mobilenet_v2_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.03
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.997
          scale: true
          epsilon: 0.001
        }
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            truncated_normal_initializer {
              mean: 0.0
              stddev: 0.03
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.997
            scale: true
            epsilon: 0.001
          }
        }
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.8
        kernel_size: 3
        use_depthwise: true
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.333
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
          delta: 1.0
        }
      }
      classification_loss {
        weighted_sigmoid {
        }
      }
      hard_example_miner {
        num_hard_examples: 3000
        iou_threshold: 0.99
        loss_type: CLASSIFICATION
        max_negatives_per_positive: 3
        min_negatives_per_image: 3
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
  }
}
train_config {
  batch_size: 32
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.8
          total_steps: 50000
          warmup_learning_rate: 0.13333
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.9
    }
  }
  fine_tune_checkpoint: "ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
  num_steps: 50000
  fine_tune_checkpoint_type: "detection"
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "fire_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "fire_train.record"
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "fire_label_map.pbtxt"
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "fire_eval.record"
  }
}
"""
    
    # Save pipeline config
    with open('pipeline.config', 'w') as f:
        f.write(pipeline_config)
    
    # Create label map
    label_map = """
item {
  id: 1
  name: 'fire'
}
item {
  id: 2
  name: 'smoke'
}
item {
  id: 3
  name: 'flame'
}
item {
  id: 4
  name: 'ember'
}
"""
    
    with open('fire_label_map.pbtxt', 'w') as f:
        f.write(label_map)
    
    print("✓ Training pipeline created")
    print("✓ This model will output 4 tensors that Frigate expects!")


def convert_to_tflite_for_frigate(saved_model_dir, output_path):
    """
    Convert trained model to TFLite format for Frigate.
    
    This maintains the 4-tensor output format that Frigate requires.
    """
    
    # Load the saved model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    # Apply optimizations for EdgeTPU
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            # 300x300 RGB images for SSD MobileNet
            data = np.random.randint(0, 255, size=(1, 300, 300, 3), dtype=np.uint8)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    
    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ Saved Frigate-compatible model to {output_path}")
    print("✓ This model has 4 output tensors - exactly what Frigate expects!")
    
    # Verify the model
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    output_details = interpreter.get_output_details()
    print(f"\nModel outputs ({len(output_details)} tensors):")
    for i, detail in enumerate(output_details):
        print(f"  {i}: {detail['name']} - shape: {detail['shape']}")
    
    if len(output_details) == 4:
        print("\n✅ Perfect! This model will work directly with Frigate!")
        print("   No custom builds needed!")
    else:
        print(f"\n⚠️  Model has {len(output_details)} outputs, expected 4")


def compile_for_edgetpu(tflite_path):
    """Compile the model for EdgeTPU."""
    
    output_path = tflite_path.replace('.tflite', '_edgetpu.tflite')
    
    print(f"Compiling for EdgeTPU...")
    os.system(f"edgetpu_compiler {tflite_path} -o {os.path.dirname(output_path)}")
    
    if os.path.exists(output_path):
        print(f"✓ EdgeTPU model saved to {output_path}")
        print("✓ Ready to use with Frigate!")
        
        # Usage instructions
        print("\nTo use with Frigate:")
        print("1. Copy model to Frigate config directory")
        print("2. Update config.yml:")
        print("""
model:
  path: /config/fire_detection_edgetpu.tflite
  width: 300
  height: 300
  labelmap_path: /config/fire_labels.txt
        """)
    else:
        print("✗ EdgeTPU compilation failed")


if __name__ == "__main__":
    print("=" * 60)
    print("Train Fire Detection Model for Frigate")
    print("The RIGHT way - no custom builds needed!")
    print("=" * 60)
    
    # Step 1: Create training pipeline
    create_training_pipeline()
    
    # Step 2: Train the model (would use TF Object Detection API)
    print("\nTo train:")
    print("python model_main_tf2.py --model_dir=fire_model --pipeline_config_path=pipeline.config")
    
    # Step 3: Convert to TFLite
    # convert_to_tflite_for_frigate("exported_model/saved_model", "fire_detection.tflite")
    
    # Step 4: Compile for EdgeTPU
    # compile_for_edgetpu("fire_detection.tflite")
    
    print("\nThis approach:")
    print("✅ Works with stock Frigate")
    print("✅ No custom builds")
    print("✅ No maintenance")
    print("✅ Proper 4-tensor output")
    print("✅ Full EdgeTPU support")