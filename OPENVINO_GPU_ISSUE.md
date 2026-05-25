# OpenVINO Intel GPU Detection Issue - Root Cause Analysis

## Problem Summary
Model OpenVINO tidak dapat mendeteksi objek saat menggunakan `device="intel:gpu"`, tetapi berfungsi normal dengan `device="cpu"` atau model PyTorch.

## Root Cause
**Intel GPU menghasilkan NaN (Not a Number) values pada output inference**, menyebabkan semua deteksi gagal.

## Evidence

### Test Results
```
[CPU Inference]
Output range: [-70.630951, 716.540955]
Detections (conf > 0.25): 1

[GPU Inference]
Output range: [nan, nan]
Detections (conf > 0.25): 0
```

### Comparison Tests
| Device | Model Type | Detections | Status |
|--------|-----------|------------|--------|
| CPU | PyTorch (.pt) | 3 | ✅ OK |
| CPU | OpenVINO | 4 | ✅ OK |
| intel:gpu | OpenVINO (FP16) | 0 | ❌ FAIL (NaN output) |
| intel:gpu | OpenVINO (FP32) | 0 | ❌ FAIL (NaN output) |

## System Information
- **GPU**: Intel(R) Iris(R) Xe Graphics (iGPU)
- **CPU**: 11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz
- **OpenVINO**: 2026.1.0-21367
- **Ultralytics**: 8.4.53
- **Model**: YOLO26m (640x640)

## GPU Capabilities (Detected)
```
Optimization capabilities: ['FP32', 'BIN', 'FP16', 'INT8', 'GPU_USM_MEMORY', 'EXPORT_IMPORT']
```

## Possible Causes
1. **Intel GPU Driver Bug**: Numerical instability di driver Iris Xe Graphics
2. **OpenVINO Runtime Issue**: Bug di OpenVINO 2026.1.0 untuk Intel GPU
3. **Model Architecture Incompatibility**: Operasi tertentu di YOLO26m trigger bug di GPU path
4. **Memory/Precision Issue**: GPU memory management atau precision handling bermasalah

## Solutions

### ✅ Solution 1: Use CPU (RECOMMENDED - IMPLEMENTED)
```python
# config.py
INFER_DEVICE = "cpu"
```
**Pros**: Reliable, works consistently
**Cons**: Slower inference (~350ms vs ~85ms)

### 🔧 Solution 2: Update Intel GPU Driver
1. Download latest Intel Graphics Driver dari intel.com
2. Install dan restart
3. Test kembali dengan `device="intel:gpu"`

### 🔧 Solution 3: Try Different OpenVINO Version
```bash
pip uninstall openvino
pip install openvino==2024.4.0  # atau versi stable lainnya
```

### 🔧 Solution 4: Export dengan Dynamic Shapes
```bash
python export_openvino.py 4th-trainingyolo26m640.pt --imgsz 640 --dynamic
```

### 🔧 Solution 5: Use ONNX Runtime dengan DirectML
Alternative inference engine untuk Intel GPU:
```bash
pip install onnxruntime-directml
```
Export model ke ONNX dan gunakan DirectML backend.

## Verification Steps
Untuk memverifikasi fix:
```bash
python test_openvino_direct.py
```

Expected output untuk GPU yang berfungsi:
```
[GPU Inference]
Output range: [-70.xxx, 716.xxx]  # Bukan NaN!
Detections (conf > 0.25): 1+
```

## Current Status
✅ **RESOLVED** - Menggunakan CPU untuk inference OpenVINO
- Model: `4th-trainingyolo26m640_openvino_model`
- Device: `cpu`
- Performance: ~350ms per frame (acceptable untuk production)

## Files Modified
- `config.py`: Changed `INFER_DEVICE` from `"intel:gpu"` to `"cpu"`

## Test Scripts Created
- `test_openvino_vs_pytorch.py`: Compare PyTorch vs OpenVINO
- `test_openvino_detailed.py`: Test OpenVINO dengan berbagai device
- `test_openvino_direct.py`: Direct OpenVINO API test (bypass Ultralytics)
- `debug_openvino_gpu.py`: Check GPU detection dan capabilities

## References
- OpenVINO Documentation: https://docs.openvino.ai/
- Intel GPU Driver: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
- Ultralytics OpenVINO Export: https://docs.ultralytics.com/integrations/openvino/
