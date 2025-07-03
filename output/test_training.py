
import torch
import logging
from super_gradients import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train
from super_gradients.common.object_names import Models

logger = logging.getLogger(__name__)

def quick_test():
    try:
        logger.info("Creating trainer...")
        trainer = Trainer(experiment_name="test_run", ckpt_root_dir="../output/test_checkpoints")
        
        logger.info("Creating model...")
        model = models.get(Models.YOLO_NAS_S, num_classes=32)
        
        logger.info("Model created successfully!")
        logger.info(f"Model type: {type(model).__name__}")
        
        # Quick forward pass test
        test_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(test_input)
        
        # Handle different output formats
        if isinstance(output, (list, tuple)):
            output_info = []
            for i, o in enumerate(output):
                if hasattr(o, 'shape'):
                    output_info.append(f"output[{i}]: {o.shape}")
                else:
                    output_info.append(f"output[{i}]: {type(o)}")
            logger.info(f"✅ Forward pass successful, outputs: {output_info}")
        else:
            logger.info(f"✅ Forward pass successful, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"TEST_RESULT: {success}")
