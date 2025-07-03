# Hailo Model Validation Guide

## Overview
This guide provides procedures for validating Hailo model performance, accuracy, and reliability for the Wildfire Watch fire detection system.

## Validation Stages

### 1. Model Structure Validation

```bash
# Inspect HEF model structure
python3.10 converted_models/hailo_utils/inspect_hef.py \
  --hef converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef

# Expected output:
# ✓ Model name: yolo8l_fire_640x640
# ✓ Target device: HAILO8L
# ✓ Input shape: [1, 640, 640, 3]
# ✓ Output shape: [1, 16032]
# ✓ Quantization: INT8
```

### 2. Inference Validation

#### Quick Test
```bash
# Run single inference test
python3.10 converted_models/hailo_utils/validate_hef.py \
  --hef converted_models/hailo_qat_output/yolo8l_fire_640x640_hailo8l_qat.hef \
  --image test_data/fire_sample.jpg \
  --visualize

# Expected results:
# - Inference time: <25ms
# - Detections: Fire/smoke objects with >0.5 confidence
# - Output visualization saved
```

#### Batch Processing Test
```python
# test_batch_inference.py
import numpy as np
from hailo_test_utils import HailoDevice

device = HailoDevice()
batch_sizes = [1, 4, 8, 16]

for batch_size in batch_sizes:
    # Create dummy batch
    batch = np.random.rand(batch_size, 640, 640, 3).astype(np.float32)
    
    # Time inference
    start = time.time()
    # Run inference (implementation depends on working API)
    elapsed = time.time() - start
    
    fps = batch_size / elapsed
    print(f"Batch {batch_size}: {fps:.1f} FPS")
```

### 3. Accuracy Validation

#### Prepare Test Dataset
```bash
# Download and extract test videos
cd tests
python3.10 -c "
from hailo_test_utils import VideoDownloader
downloader = VideoDownloader()
videos = downloader.download_all_videos()
print(f'Downloaded {len(videos)} test videos')
"
```

#### Compare with ONNX Baseline
```bash
# Run accuracy comparison test
python3.10 tests/test_hailo_accuracy.py

# Expected metrics:
# - Precision: >0.98 (vs ONNX)
# - Recall: >0.98 (vs ONNX)
# - F1 Score: >0.98 (vs ONNX)
# - Inference speedup: >10x
```

### 4. Performance Benchmarking

#### Latency Testing
```bash
# Run performance benchmarks
python3.10 tests/test_performance_benchmarks.py

# Key metrics to track:
# - P50 latency: <20ms
# - P95 latency: <25ms
# - P99 latency: <30ms
# - Throughput: >40 FPS
```

#### Resource Monitoring
```bash
# Monitor during inference
watch -n 1 '
echo "=== Hailo Metrics ==="
echo "Temperature: $(cat /sys/class/hwmon/hwmon*/temp1_input)°C"
echo "Power: $(cat /sys/class/hwmon/hwmon*/power1_average)W"
echo ""
echo "=== System Metrics ==="
top -bn1 | grep "Cpu\|Mem" | head -2
'
```

### 5. Stability Testing

#### Long-Running Test
```bash
# Run 1-hour stability test
python3.10 tests/test_stability_temperature.py \
  --duration 3600 \
  --fps 30

# Monitor for:
# - Temperature stability (<85°C)
# - No memory leaks
# - Consistent latency
# - Zero crashes
```

#### Stress Testing
```python
# stress_test.py
import threading
import time

def inference_thread(thread_id):
    """Run continuous inference"""
    while not stop_event.is_set():
        # Run inference
        result = model.infer(test_image)
        latencies[thread_id].append(result.latency)

# Launch multiple threads
threads = []
for i in range(4):  # 4 concurrent streams
    t = threading.Thread(target=inference_thread, args=(i,))
    threads.append(t)
    t.start()

# Run for 10 minutes
time.sleep(600)
stop_event.set()
```

## Validation Metrics

### Accuracy Metrics
| Metric | Target | Critical |
|--------|--------|----------|
| mAP@0.5 | >0.85 | >0.80 |
| Precision | >0.90 | >0.85 |
| Recall | >0.90 | >0.85 |
| F1 Score | >0.90 | >0.85 |

### Performance Metrics
| Metric | Target | Maximum |
|--------|--------|---------|
| Latency (P95) | <25ms | 50ms |
| Throughput | >40 FPS | - |
| Temperature | <80°C | 85°C |
| Power | <5W | 7W |

### Reliability Metrics
| Metric | Target |
|--------|--------|
| Uptime | >99.9% |
| MTBF | >720 hours |
| Memory Stability | <1% growth/hour |
| Error Rate | <0.01% |

## Automated Validation Pipeline

```bash
#!/bin/bash
# validate_hailo_model.sh

echo "=== Hailo Model Validation Pipeline ==="

# 1. Structure validation
echo -e "\n[1/5] Validating model structure..."
python3.10 converted_models/hailo_utils/inspect_hef.py \
  --hef $HEF_PATH || exit 1

# 2. Quick inference test
echo -e "\n[2/5] Running inference test..."
python3.10 converted_models/hailo_utils/validate_hef.py \
  --hef $HEF_PATH \
  --image test_data/fire_sample.jpg || exit 1

# 3. Accuracy validation
echo -e "\n[3/5] Validating accuracy..."
python3.10 tests/test_hailo_accuracy.py || exit 1

# 4. Performance benchmarks
echo -e "\n[4/5] Running performance benchmarks..."
python3.10 tests/test_performance_benchmarks.py || exit 1

# 5. Short stability test
echo -e "\n[5/5] Running stability test (5 min)..."
python3.10 tests/test_stability_temperature.py \
  --duration 300 || exit 1

echo -e "\n✅ All validation tests PASSED!"
```

## Troubleshooting Validation Failures

### Low Accuracy
1. Check quantization calibration dataset
2. Verify input preprocessing matches training
3. Compare with FP32 ONNX model
4. Adjust confidence thresholds

### Poor Performance
1. Check batch size configuration
2. Verify PCIe lane configuration
3. Monitor thermal throttling
4. Update Hailo drivers

### Instability
1. Check power supply adequacy
2. Improve cooling solution
3. Reduce concurrent streams
4. Update firmware

## Continuous Validation

### Daily Automated Tests
```yaml
# .github/workflows/hailo_validation.yml
name: Hailo Model Validation
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily
jobs:
  validate:
    runs-on: [self-hosted, hailo]
    steps:
      - uses: actions/checkout@v3
      - name: Run validation pipeline
        run: ./scripts/validate_hailo_model.sh
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: validation_report.json
```

### Metrics Dashboard
```python
# metrics_dashboard.py
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Load historical metrics
with open('validation_history.json') as f:
    history = json.load(f)

# Plot trends
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Latency trend
axes[0, 0].plot([h['date'] for h in history], 
                [h['p95_latency'] for h in history])
axes[0, 0].set_title('P95 Latency Trend')
axes[0, 0].axhline(y=25, color='r', linestyle='--', label='Target')

# Accuracy trend  
axes[0, 1].plot([h['date'] for h in history],
                [h['f1_score'] for h in history])
axes[0, 1].set_title('F1 Score Trend')
axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='Target')

# Temperature trend
axes[1, 0].plot([h['date'] for h in history],
                [h['max_temp'] for h in history])
axes[1, 0].set_title('Max Temperature Trend')
axes[1, 0].axhline(y=85, color='r', linestyle='--', label='Limit')

# Throughput trend
axes[1, 1].plot([h['date'] for h in history],
                [h['avg_fps'] for h in history])
axes[1, 1].set_title('Average FPS Trend')
axes[1, 1].axhline(y=40, color='g', linestyle='--', label='Target')

plt.tight_layout()
plt.savefig('validation_trends.png')
```

## Report Generation

```python
# generate_validation_report.py
import json
from datetime import datetime

def generate_report(results):
    """Generate validation report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': 'yolo8l_fire_640x640_hailo8l_qat.hef',
        'status': 'PASS' if all_tests_passed(results) else 'FAIL',
        'summary': {
            'accuracy': {
                'f1_score': results['accuracy']['f1_score'],
                'vs_baseline': results['accuracy']['vs_onnx']
            },
            'performance': {
                'p95_latency_ms': results['performance']['p95_latency'],
                'throughput_fps': results['performance']['avg_fps']
            },
            'stability': {
                'max_temperature_c': results['stability']['max_temp'],
                'uptime_hours': results['stability']['uptime']
            }
        },
        'recommendations': generate_recommendations(results)
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
```

## Best Practices

1. **Regular Validation**: Run full validation weekly
2. **Version Control**: Tag validated model versions
3. **Documentation**: Document any threshold adjustments
4. **Monitoring**: Set up alerts for degraded performance
5. **Backup**: Keep last 3 validated models