"""
Test INT8 quantized model on Intel GPU
Berdasarkan dokumentasi, INT8 kadang work saat FP16/FP32 menghasilkan NaN
"""
import cv2
import numpy as np
try:
    from openvino.runtime import Core
except ImportError:
    from openvino import Core

# Load frame
video_path = "uploaded_sources/VIGI_C340_BA-DC_20260421135123-20260421135245_0_1776816060449_3ddeb6f4.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Tidak bisa membaca video")
    exit(1)

print(f"Original frame shape: {frame.shape}")

# Preprocess frame
input_h, input_w = 640, 640
resized = cv2.resize(frame, (input_w, input_h))
input_image = resized.astype(np.float32) / 255.0
input_image = np.transpose(input_image, (2, 0, 1))
input_image = np.expand_dims(input_image, axis=0)

print(f"Preprocessed input shape: {input_image.shape}")

# Load model
model_path = "4th-trainingyolo26m640_openvino_model/4th-trainingyolo26m640.xml"
core = Core()

print("\n" + "=" * 80)
print("Testing with different GPU configurations")
print("=" * 80)

# Test 1: Default GPU config
print("\n[Test 1] GPU with default config")
try:
    compiled_model = core.compile_model(model_path, "GPU")
    output = compiled_model([input_image])[0]
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")

    if output.shape[-1] == 6:
        confidences = output[0, :, 4]
        valid = confidences > 0.25
        num_dets = np.sum(valid)
        print(f"Detections (conf > 0.25): {num_dets}")
        if num_dets > 0:
            print(f"Confidences: {confidences[valid][:5]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: GPU with INFERENCE_PRECISION_HINT
print("\n[Test 2] GPU with INFERENCE_PRECISION_HINT=f32")
try:
    compiled_model = core.compile_model(
        model_path,
        "GPU",
        {"INFERENCE_PRECISION_HINT": "f32"}
    )
    output = compiled_model([input_image])[0]
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")

    if output.shape[-1] == 6:
        confidences = output[0, :, 4]
        valid = confidences > 0.25
        num_dets = np.sum(valid)
        print(f"Detections (conf > 0.25): {num_dets}")
        if num_dets > 0:
            print(f"Confidences: {confidences[valid][:5]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: GPU with PERFORMANCE_HINT
print("\n[Test 3] GPU with PERFORMANCE_HINT=THROUGHPUT")
try:
    compiled_model = core.compile_model(
        model_path,
        "GPU",
        {"PERFORMANCE_HINT": "THROUGHPUT"}
    )
    output = compiled_model([input_image])[0]
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")

    if output.shape[-1] == 6:
        confidences = output[0, :, 4]
        valid = confidences > 0.25
        num_dets = np.sum(valid)
        print(f"Detections (conf > 0.25): {num_dets}")
        if num_dets > 0:
            print(f"Confidences: {confidences[valid][:5]}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: GPU with NUM_STREAMS
print("\n[Test 4] GPU with NUM_STREAMS=1")
try:
    compiled_model = core.compile_model(
        model_path,
        "GPU",
        {"NUM_STREAMS": "1"}
    )
    output = compiled_model([input_image])[0]
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")

    if output.shape[-1] == 6:
        confidences = output[0, :, 4]
        valid = confidences > 0.25
        num_dets = np.sum(valid)
        print(f"Detections (conf > 0.25): {num_dets}")
        if num_dets > 0:
            print(f"Confidences: {confidences[valid][:5]}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
