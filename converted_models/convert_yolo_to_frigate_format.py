#!/usr/bin/env python3.12
"""
Convert YOLO models to Frigate-compatible format by adding post-processing layers.

This converter adds TensorFlow Object Detection API compatible output layers
to YOLO models, making them work directly with Frigate.
"""

import os
import sys
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import subprocess
import shutil
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOToFrigateConverter:
    """Convert YOLO models to output Frigate-compatible format."""
    
    def __init__(self, max_detections: int = 100):
        self.max_detections = max_detections
        
    def convert_onnx_to_frigate_tf(self, onnx_path: str, output_dir: str, 
                                   num_classes: int = 4, input_size: int = 640):
        """
        Convert ONNX YOLO model to TensorFlow SavedModel with Frigate-compatible outputs.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting {onnx_path} to Frigate-compatible TensorFlow model...")
        
        # First, convert ONNX to TensorFlow using onnx-tf
        saved_model_dir = output_dir / "saved_model_temp"
        
        logger.info("Step 1: Converting ONNX to TensorFlow SavedModel...")
        try:
            from onnx_tf.backend import prepare
            import onnx
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(saved_model_dir))
            
            logger.info(f"✓ Converted to SavedModel: {saved_model_dir}")
        except Exception as e:
            logger.error(f"Failed to convert ONNX to TF: {e}")
            return None
        
        # Load the converted model and add post-processing
        logger.info("Step 2: Adding Frigate-compatible post-processing layers...")
        
        # Load the base model
        base_model = tf.saved_model.load(str(saved_model_dir))
        
        # Create a new model with post-processing
        class FrigateYOLOModel(tf.Module):
            def __init__(self, base_model, num_classes, max_detections, input_size):
                super().__init__()
                self.base_model = base_model
                self.num_classes = num_classes
                self.max_detections = max_detections
                self.input_size = input_size
                
            @tf.function
            def preprocess(self, images):
                """Preprocess input images."""
                # Convert uint8 to float32 and normalize
                images = tf.cast(images, tf.float32)
                images = images / 255.0
                return images
            
            @tf.function
            def parse_yolo_output(self, raw_output):
                """
                Parse YOLO output and convert to TF Object Detection API format.
                
                YOLO output: [batch, num_predictions, 4 + 1 + num_classes]
                Frigate expects:
                - boxes: [batch, max_detections, 4] as [ymin, xmin, ymax, xmax]
                - classes: [batch, max_detections] as float32
                - scores: [batch, max_detections]
                - num_detections: [batch] as float32
                """
                # Handle different YOLO output shapes
                output_shape = tf.shape(raw_output)
                
                if len(raw_output.shape) == 3:
                    # [batch, predictions, features]
                    batch_size = output_shape[0]
                    predictions = raw_output
                elif len(raw_output.shape) == 2:
                    # [predictions, features] - add batch dimension
                    batch_size = 1
                    predictions = tf.expand_dims(raw_output, 0)
                else:
                    # Flatten and reshape
                    batch_size = 1
                    num_features = 4 + 1 + self.num_classes
                    num_predictions = tf.shape(raw_output)[-1] // num_features
                    predictions = tf.reshape(raw_output, [batch_size, num_predictions, num_features])
                
                # Extract components
                xy = predictions[..., 0:2]  # Center x, y
                wh = predictions[..., 2:4]  # Width, height
                objectness = predictions[..., 4:5]  # Objectness score
                class_probs = predictions[..., 5:5+self.num_classes]  # Class probabilities
                
                # Calculate confidence scores
                class_conf = objectness * class_probs
                max_class_conf = tf.reduce_max(class_conf, axis=-1)
                class_ids = tf.cast(tf.argmax(class_probs, axis=-1), tf.float32)
                
                # Convert center format to corner format
                half_wh = wh / 2.0
                x1 = xy[..., 0:1] - half_wh[..., 0:1]
                y1 = xy[..., 1:2] - half_wh[..., 1:2]
                x2 = xy[..., 0:1] + half_wh[..., 0:1]
                y2 = xy[..., 1:2] + half_wh[..., 1:2]
                
                # Normalize coordinates if they're in pixel space
                x1 = tf.clip_by_value(x1 / self.input_size, 0.0, 1.0)
                y1 = tf.clip_by_value(y1 / self.input_size, 0.0, 1.0)
                x2 = tf.clip_by_value(x2 / self.input_size, 0.0, 1.0)
                y2 = tf.clip_by_value(y2 / self.input_size, 0.0, 1.0)
                
                # Format as [ymin, xmin, ymax, xmax] for TF Object Detection API
                boxes = tf.concat([y1, x1, y2, x2], axis=-1)
                
                # Process each batch item
                batch_boxes = []
                batch_scores = []
                batch_classes = []
                batch_num_detections = []
                
                for i in range(batch_size):
                    # Apply confidence threshold
                    conf_mask = max_class_conf[i] > 0.25
                    filtered_boxes = tf.boolean_mask(boxes[i], conf_mask)
                    filtered_scores = tf.boolean_mask(max_class_conf[i], conf_mask)
                    filtered_classes = tf.boolean_mask(class_ids[i], conf_mask)
                    
                    # Apply NMS
                    if tf.shape(filtered_boxes)[0] > 0:
                        selected_indices = tf.image.non_max_suppression(
                            filtered_boxes,
                            filtered_scores,
                            max_output_size=self.max_detections,
                            iou_threshold=0.45
                        )
                        
                        selected_boxes = tf.gather(filtered_boxes, selected_indices)
                        selected_scores = tf.gather(filtered_scores, selected_indices)
                        selected_classes = tf.gather(filtered_classes, selected_indices)
                        num_valid = tf.shape(selected_indices)[0]
                    else:
                        selected_boxes = tf.zeros([0, 4], dtype=tf.float32)
                        selected_scores = tf.zeros([0], dtype=tf.float32)
                        selected_classes = tf.zeros([0], dtype=tf.float32)
                        num_valid = 0
                    
                    # Pad to max_detections
                    pad_size = self.max_detections - num_valid
                    padded_boxes = tf.pad(selected_boxes, [[0, pad_size], [0, 0]])
                    padded_scores = tf.pad(selected_scores, [[0, pad_size]])
                    padded_classes = tf.pad(selected_classes, [[0, pad_size]])
                    
                    batch_boxes.append(padded_boxes)
                    batch_scores.append(padded_scores)
                    batch_classes.append(padded_classes)
                    batch_num_detections.append(tf.cast(num_valid, tf.float32))
                
                # Stack batch results
                detection_boxes = tf.stack(batch_boxes)
                detection_scores = tf.stack(batch_scores)
                detection_classes = tf.stack(batch_classes)
                num_detections = tf.stack(batch_num_detections)
                
                return detection_boxes, detection_classes, detection_scores, num_detections
            
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[None, input_size, input_size, 3], dtype=tf.uint8, name='images')
            ])
            def __call__(self, images):
                """Main inference function with Frigate-compatible outputs."""
                # Preprocess
                preprocessed = self.preprocess(images)
                
                # Run base model
                # Note: This assumes the model expects NHWC format
                # If it expects NCHW, we need to transpose
                if hasattr(self.base_model, 'signatures'):
                    # SavedModel format
                    infer = self.base_model.signatures['serving_default']
                    outputs = infer(preprocessed)
                    # Get the output tensor (key might vary)
                    output_keys = list(outputs.keys())
                    raw_output = outputs[output_keys[0]]
                else:
                    # Direct call
                    raw_output = self.base_model(preprocessed)
                
                # Parse YOLO output to Frigate format
                boxes, classes, scores, num_det = self.parse_yolo_output(raw_output)
                
                return {
                    'detection_boxes': boxes,
                    'detection_classes': classes,
                    'detection_scores': scores,
                    'num_detections': num_det
                }
        
        # Create the wrapped model
        wrapped_model = FrigateYOLOModel(
            base_model, num_classes, self.max_detections, input_size
        )
        
        # Save the new model
        output_saved_model = output_dir / "yolo_frigate_format"
        tf.saved_model.save(
            wrapped_model,
            str(output_saved_model),
            signatures=wrapped_model.__call__
        )
        
        logger.info(f"✓ Saved Frigate-compatible model: {output_saved_model}")
        
        # Clean up temp directory
        shutil.rmtree(saved_model_dir)
        
        return output_saved_model
    
    def convert_to_tflite(self, saved_model_path: str, output_path: str,
                         quantize: bool = True, use_uint8: bool = True):
        """Convert SavedModel to TFLite with optional quantization."""
        logger.info("Step 3: Converting to TFLite...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if use_uint8:
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.float32  # Frigate expects float outputs
                
                # Representative dataset
                def representative_dataset():
                    for _ in range(100):
                        data = np.random.randint(0, 255, size=(1, 640, 640, 3), dtype=np.uint8)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Additional converter settings
        converter.allow_custom_ops = False
        converter.experimental_new_converter = True
        
        # Convert
        try:
            tflite_model = converter.convert()
            
            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"✓ Saved TFLite model: {output_path}")
            
            # Verify the model
            self.verify_tflite_model(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"TFLite conversion failed: {e}")
            return None
    
    def verify_tflite_model(self, model_path: str):
        """Verify the converted TFLite model has correct format."""
        logger.info("Step 4: Verifying TFLite model...")
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"Input: {input_details[0]['name']} - shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
        
        logger.info(f"Number of outputs: {len(output_details)}")
        for i, output in enumerate(output_details):
            logger.info(f"Output {i}: {output['name']} - shape: {output['shape']}, dtype: {output['dtype']}")
        
        # Check if outputs match Frigate's expectations
        expected_outputs = ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections']
        actual_outputs = [out['name'] for out in output_details]
        
        missing = set(expected_outputs) - set(actual_outputs)
        if missing:
            logger.warning(f"Missing expected outputs: {missing}")
        else:
            logger.info("✓ All expected outputs present!")
        
        # Test inference
        try:
            test_input = np.random.randint(0, 255, size=input_details[0]['shape'], dtype=np.uint8)
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            
            for output in output_details:
                result = interpreter.get_tensor(output['index'])
                logger.info(f"  {output['name']}: {result.shape}")
            
            logger.info("✓ Model inference test passed!")
        except Exception as e:
            logger.error(f"Model inference test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO models to Frigate format")
    parser.add_argument('--onnx', required=True, help='Path to YOLO ONNX model')
    parser.add_argument('--output-dir', default='frigate_models', help='Output directory')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--input-size', type=int, default=640, help='Model input size')
    parser.add_argument('--max-detections', type=int, default=100, help='Maximum detections')
    parser.add_argument('--skip-edgetpu', action='store_true', help='Skip EdgeTPU compilation')
    
    args = parser.parse_args()
    
    converter = YOLOToFrigateConverter(max_detections=args.max_detections)
    
    # Convert ONNX to Frigate-compatible SavedModel
    saved_model = converter.convert_onnx_to_frigate_tf(
        args.onnx, args.output_dir, args.num_classes, args.input_size
    )
    
    if saved_model:
        # Convert to TFLite
        tflite_path = Path(args.output_dir) / f"{Path(args.onnx).stem}_frigate.tflite"
        tflite_model = converter.convert_to_tflite(saved_model, tflite_path)
        
        if tflite_model and not args.skip_edgetpu:
            # Compile for EdgeTPU
            logger.info("Step 5: Compiling for EdgeTPU...")
            edgetpu_path = Path(args.output_dir) / f"{Path(args.onnx).stem}_frigate_edgetpu.tflite"
            
            result = subprocess.run([
                'edgetpu_compiler',
                '-s',
                '-o', str(Path(args.output_dir)),
                str(tflite_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ EdgeTPU compilation successful!")
                logger.info(result.stdout)
            else:
                logger.error(f"EdgeTPU compilation failed: {result.stderr}")


if __name__ == "__main__":
    main()