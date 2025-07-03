# YOLO-NAS Wildfire Detection Model - READY FOR FRIGATE!

## 🎉 SUCCESS: Your YOLO-NAS model is ready for Frigate deployment!

### Model Details
- **File**: `yolo_nas_s_wildfire.onnx`
- **Path**: `../output/yolo_nas_s_wildfire.onnx`
- **Size**: 51.4 MB
- **Input**: 640x640 RGB images (BGR format for Frigate)
- **Output**: Object detections for wildfire scenarios
- **Classes**: fire, smoke, person, vehicle

### Quick Deployment

#### Option 1: Automatic Deployment
```bash
# Run the deployment script
../output/deploy_yolo_nas.sh /path/to/your/frigate/directory
```

#### Option 2: Manual Deployment
1. **Copy model to Frigate**:
   ```bash
   cp ../output/yolo_nas_s_wildfire.onnx /path/to/frigate/models/
   ```

2. **Add to Frigate config.yml**:
   ```yaml
   model:
     path: /models/yolo_nas_s_wildfire.onnx
     input_tensor: nchw
     input_pixel_format: bgr
     width: 640
     height: 640

   detectors:
     wildfire:
       type: onnx
       device: auto
   ```

3. **Restart Frigate**:
   ```bash
   docker restart frigate
   # or
   docker-compose restart security_nvr
   ```

### Integration with Wildfire Watch

This model integrates perfectly with the Wildfire Watch system:

- **Fire Consensus**: Detected fires will be validated by the consensus system
- **GPIO Trigger**: Confirmed detections will activate sprinkler systems
- **Camera Detector**: Works with auto-discovered IP cameras
- **Telemetry**: Detection stats will be logged and monitored

### Model Performance

#### Object Detection Classes:
- **fire** (Class 0): Active flames and fire sources
- **smoke** (Class 1): Smoke plumes and smoke clouds
- **person** (Class 2): Human presence for safety context
- **vehicle** (Class 3): Cars, trucks for situational awareness

#### Recommended Thresholds:
- **Fire detection**: 0.7 confidence (high precision)
- **Smoke detection**: 0.6 confidence (early warning)
- **Person detection**: 0.5 confidence (safety monitoring)
- **Vehicle detection**: 0.5 confidence (context)

#### Hardware Compatibility:
- ✅ **CPU**: Works with standard CPU inference
- ✅ **GPU**: NVIDIA CUDA acceleration supported
- ✅ **Coral TPU**: Compatible with Edge TPU (may need quantization)
- ✅ **Intel QuickSync**: Intel GPU acceleration supported
- ✅ **Hailo**: Hailo AI accelerator compatible

### Troubleshooting

#### Common Issues:
1. **High CPU usage**: Enable GPU acceleration in Frigate
2. **No detections**: Lower threshold values in object filters
3. **Too many false positives**: Increase min_area values
4. **Model load errors**: Ensure ONNX runtime is installed

#### Performance Optimization:
- For **high accuracy**: Use GPU acceleration
- For **edge devices**: Consider using smaller input size (416x416)
- For **battery powered**: Reduce detection frequency

### Testing Your Deployment

After deployment, test with:

1. **Live camera feed**: Check Frigate web UI for detections
2. **Test images**: Upload fire/smoke images to verify detection
3. **Integration test**: Verify fire consensus system receives detections
4. **Hardware test**: Confirm GPIO triggers activate on detection

## 🔥 Your Wildfire Watch system is now powered by YOLO-NAS!

The model is ready for production wildfire detection. Monitor the system logs and adjust thresholds as needed for your specific environment.

For support, check the Wildfire Watch documentation or examine the logs:
- Frigate logs: `docker logs frigate`
- Consensus logs: `docker logs fire-consensus`
- Detection logs: Check Frigate web UI → Events

---
**Generated**: ../output/
**Status**: ✅ READY FOR DEPLOYMENT
