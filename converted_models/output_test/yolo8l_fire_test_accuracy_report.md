# Model Accuracy Validation Report

## Baseline (PyTorch)
- mAP@50: 0.850
- mAP@50-95: 0.650
- Precision: 0.900
- Recall: 0.850
- F1 Score: 0.870
- Inference Time: 50.0ms
- Model Size: 83.6MB

## Converted Models

### TENSORRT_FP16_640X640
- mAP@50: 0.846
- mAP@50-95: 0.643
- Precision: 0.895
- Recall: 0.846
- F1 Score: 0.866
- Inference Time: 15.0ms
- Model Size: 87.8MB

  **Comparison to baseline:**
  - mAP@50 degradation: 0.50%
  - mAP@50-95 degradation: 1.00%
  - Speed improvement: 3.3x
  - Size reduction: -5.0%
  - **Status:** ❌ FAILED