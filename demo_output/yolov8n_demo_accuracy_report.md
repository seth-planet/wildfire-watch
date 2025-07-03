# Model Accuracy Validation Report

## Baseline (PyTorch)
- mAP@50: 0.625
- mAP@50-95: 0.495
- Precision: 0.706
- Recall: 0.488
- F1 Score: 0.538
- Inference Time: 12.1ms
- Model Size: 6.2MB

## Converted Models

### ONNX_640X640
- mAP@50: 0.622
- mAP@50-95: 0.490
- Precision: 0.703
- Recall: 0.485
- F1 Score: 0.536
- Inference Time: 9.7ms
- Model Size: 12.2MB

  **Comparison to baseline:**
  - mAP@50 degradation: 0.50%
  - mAP@50-95 degradation: 1.00%
  - Speed improvement: 1.2x
  - Size reduction: -96.0%
  - **Status:** ❌ FAILED

### ONNX_416X416
- mAP@50: 0.622
- mAP@50-95: 0.490
- Precision: 0.703
- Recall: 0.485
- F1 Score: 0.536
- Inference Time: 9.7ms
- Model Size: 12.1MB

  **Comparison to baseline:**
  - mAP@50 degradation: 0.50%
  - mAP@50-95 degradation: 1.00%
  - Speed improvement: 1.2x
  - Size reduction: -94.5%
  - **Status:** ❌ FAILED