"""
Test inference langsung dengan OpenVINO API (bypass Ultralytics)
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

# Preprocess frame (resize to 640x640, normalize, transpose to NCHW)
input_h, input_w = 640, 640
resized = cv2.resize(frame, (input_w, input_h))
input_image = resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension -> NCHW

print(f"Preprocessed input shape: {input_image.shape}")
print(f"Input dtype: {input_image.dtype}")
print(f"Input range: [{input_image.min():.3f}, {input_image.max():.3f}]")

# Load model
model_path = "4th-trainingyolo26m640_openvino_model/4th-trainingyolo26m640.xml"
core = Core()

print("\n" + "=" * 80)

# Test on CPU
print("\n[CPU Inference]")
compiled_model_cpu = core.compile_model(model_path, "CPU")
output_cpu = compiled_model_cpu([input_image])[0]
print(f"Output shape: {output_cpu.shape}")
print(f"Output dtype: {output_cpu.dtype}")
print(f"Output range: [{output_cpu.min():.6f}, {output_cpu.max():.6f}]")

# Count detections (assuming output format is [batch, num_dets, 6] where 6 = [x1,y1,x2,y2,conf,cls])
# Filter by confidence > 0.25
if output_cpu.shape[-1] == 6:
    confidences_cpu = output_cpu[0, :, 4]
    valid_cpu = confidences_cpu > 0.25
    num_dets_cpu = np.sum(valid_cpu)
    print(f"Detections (conf > 0.25): {num_dets_cpu}")
    if num_dets_cpu > 0:
        print(f"Top 5 confidences: {confidences_cpu[valid_cpu][:5]}")

print("\n" + "=" * 80)

# Test on GPU
print("\n[GPU Inference]")
compiled_model_gpu = core.compile_model(model_path, "GPU")
output_gpu = compiled_model_gpu([input_image])[0]
print(f"Output shape: {output_gpu.shape}")
print(f"Output dtype: {output_gpu.dtype}")
print(f"Output range: [{output_gpu.min():.6f}, {output_gpu.max():.6f}]")

# Count detections
if output_gpu.shape[-1] == 6:
    confidences_gpu = output_gpu[0, :, 4]
    valid_gpu = confidences_gpu > 0.25
    num_dets_gpu = np.sum(valid_gpu)
    print(f"Detections (conf > 0.25): {num_dets_gpu}")
    if num_dets_gpu > 0:
        print(f"Top 5 confidences: {confidences_gpu[valid_gpu][:5]}")

print("\n" + "=" * 80)
print("\n[COMPARISON]")
print(f"CPU detections: {num_dets_cpu if 'num_dets_cpu' in locals() else 'N/A'}")
print(f"GPU detections: {num_dets_gpu if 'num_dets_gpu' in locals() else 'N/A'}")

# Check if outputs are similar
if output_cpu.shape == output_gpu.shape:
    diff = np.abs(output_cpu - output_gpu)
    print(f"\nOutput difference (CPU vs GPU):")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max absolute diff: {diff.max():.6f}")
