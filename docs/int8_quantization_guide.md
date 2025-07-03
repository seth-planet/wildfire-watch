# INT8 Quantization Guide for YOLO-NAS Edge Deployment

This guide clarifies the INT8 quantization options for deploying YOLO-NAS models on edge devices like Coral TPU.

## Key Concepts

### 1. YOLO-NAS is Inherently Quantization-Friendly

YOLO-NAS was designed from the ground up to be quantization-friendly:
- Uses simple activation functions (ReLU) instead of complex ones
- Contains quantization-aware blocks in its architecture
- Can achieve <1 mAP drop with post-training quantization
- No special training flags needed - it's always ready for INT8

### 2. QAT vs PTQ - What's the Difference?

**Post-Training Quantization (PTQ)** - Recommended for most users:
- Quantizes the model after training is complete
- Simpler process, excellent results with YOLO-NAS
- Typical accuracy drop: 0.5-1.0 mAP

**Quantization-Aware Training (QAT)** - For maximum accuracy:
- Simulates INT8 quantization during training
- More complex, requires special training setup
- Typical accuracy drop: 0.2-0.5 mAP
- Only worth it if you need every bit of accuracy

## Recommended Workflow

### Step 1: Train YOLO-NAS Normally

```bash
python3.10 converted_models/train_custom_yolo_nas.py \
  --dataset_path /path/to/dataset \
  --no_pretrained \
  --epochs 100 \
  --batch_size 16 \
  --experiment_name wildfire_model
```

### Step 2: Convert to INT8 with Post-Training Quantization

```bash
# Download calibration data if needed
wget https://huggingface.co/datasets/mailseth/wildfire-watch/resolve/main/wildfire_calibration_data.tar.gz
tar -xzf wildfire_calibration_data.tar.gz

# Convert with INT8 quantization
python3.10 converted_models/convert_model.py \
  --model_path output/checkpoints/wildfire_model/ckpt_best.pth \
  --model_type yolo_nas \
  --output_formats tflite \
  --calibration_dir wildfire_calibration_data/
```

This produces several TFLite variants:
- `model_cpu.tflite` - FP16 for CPU (best accuracy, larger)
- `model_dynamic.tflite` - Dynamic range quantization (good balance)
- `model_quant.tflite` - Full INT8 quantization (for edge TPU)
- `model_edgetpu.tflite` - Compiled for Coral TPU (if compiler available)

### Step 3: Deploy on Edge Device

For Coral TPU:
```bash
# If not already compiled for Edge TPU
edgetpu_compiler model_quant.tflite

# Deploy
cp model_edgetpu.tflite /path/to/edge/device/
```

## Common Misconceptions

### "I need to enable QAT during training"
**False for YOLO-NAS.** The architecture is already quantization-friendly. Post-training quantization works excellently.

### "I need special flags for quantization-friendly training"
**False.** YOLO-NAS doesn't need any special flags. It's always ready for INT8.

### "QAT is always better than PTQ"
**Not necessarily.** With YOLO-NAS, PTQ often gives results within 0.5 mAP of QAT, making the added complexity rarely worth it.

## When to Use Each Approach

### Use Standard Training + PTQ When:
- You want simplicity (most users)
- 0.5-1.0 mAP drop is acceptable
- You're deploying to Coral TPU or similar edge devices
- You're starting a new project

### Consider QAT Only When:
- You need absolute maximum accuracy
- Even 0.5 mAP matters for your use case
- You have experience with quantization workflows
- You're fine with the added complexity

## Troubleshooting

### "Quantized model accuracy is poor"
1. Ensure you're using enough calibration images (300+ recommended)
2. Use representative calibration data (similar to deployment conditions)
3. Try different quantization options in convert_model.py

### "Model won't compile for Edge TPU"
1. Ensure all operations are Edge TPU compatible
2. Check model size fits in Edge TPU memory
3. Use the latest edgetpu_compiler

### "Training takes too long"
- YOLO-NAS doesn't need special quantization training
- Use standard training - it's already optimized for INT8

## Performance Expectations

With proper INT8 quantization on YOLO-NAS:
- Model size: ~4x reduction
- Inference speed: 2-4x faster on edge devices
- Accuracy: Typically <1 mAP drop with PTQ
- Power consumption: Significantly reduced

## Summary

For 99% of users deploying YOLO-NAS to edge devices:
1. Train normally (no special flags needed)
2. Use post-training quantization during conversion
3. Deploy the INT8 model

The architecture is already optimized for this workflow!