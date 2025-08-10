# Model Conversion Fix for Frigate Compatibility

## Issue
Fire detection models have INT8 input tensors, but Frigate's EdgeTPU detector requires UINT8 input tensors. This causes the error:
```
ValueError: Cannot set tensor: Got value of type UINT8 but expected type INT8 for input 0
```

## Root Cause Analysis

1. **Current Conversion Code** (lines 1431-1432 in convert_model.py):
   ```python
   converter.inference_input_type = tf.uint8
   converter.inference_output_type = tf.uint8
   ```
   This SHOULD produce UINT8 models, but doesn't always work.

2. **Why It Fails**:
   - The `TFLITE_BUILTINS_INT8` ops set forces INT8 quantization
   - Some model architectures resist UINT8 conversion
   - The quantization process may override the inference type setting

3. **Verified Models**:
   - ✓ `ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite` - Has UINT8 input (works)
   - ✗ `yolo8l_fire_640x640_full_integer_quant_edgetpu.tflite` - Has INT8 input (fails)
   - ✗ `yolov8n_416_edgetpu.tflite` - Has FLOAT32 input (fails)

## Proposed Fix

### Option 1: Update convert_model.py (Recommended)

Add a dedicated Frigate conversion method:

```python
def _convert_tflite_frigate_uint8(self, saved_model_path: Path) -> Optional[Path]:
    """Convert to TFLite with UINT8 input specifically for Frigate EdgeTPU."""
    output_path = self.output_dir / f"{self.model_name}_frigate.tflite"
    
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    
    # Frigate-specific settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = self._get_representative_dataset()
    
    # Critical: Don't use INT8 ops, use regular ops with quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS  # Not TFLITE_BUILTINS_INT8
    ]
    
    # Force UINT8 for input/output
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    # Disable per-channel quantization if causing issues
    converter._experimental_disable_per_channel = True
    
    tflite_model = converter.convert()
    
    # Verify UINT8 input before saving
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    if input_details[0]['dtype'] != np.uint8:
        logger.error(f"Failed to create UINT8 model, got {input_details[0]['dtype']}")
        return None
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    return output_path
```

### Option 2: Post-process Existing Models

Create a wrapper that handles the INT8 to UINT8 conversion at runtime, but this adds complexity.

### Option 3: Use Different Base Models

Start with models that are known to quantize well to UINT8, like MobileNet architectures.

## Implementation Steps

1. **Update convert_model.py**:
   - Add the Frigate-specific conversion method
   - Ensure it's called for all EdgeTPU conversions
   - Verify UINT8 input before EdgeTPU compilation

2. **Update the test**:
   - Once fixed, revert test_e2e_coral_frigate.py to use fire models
   - Remove the SSD MobileNet workaround

3. **Validation**:
   - Test converted models with pycoral to verify UINT8 input
   - Run Frigate container to ensure compatibility
   - Verify detection accuracy is maintained

## Testing the Fix

```python
# Verify model has UINT8 input
from pycoral.utils.edgetpu import make_interpreter
interpreter = make_interpreter('model_edgetpu.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
assert input_details[0]['dtype'] == np.uint8
```

## Alternative Quick Fix

Until the conversion is fixed, use the provided script:
```bash
python3.12 converted_models/convert_to_frigate_uint8.py
```

This will create Frigate-compatible versions of existing models.