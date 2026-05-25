"""
Test OpenVINO model dengan berbagai konfigurasi
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Path ke model
OPENVINO_MODEL = "4th-trainingyolo26m640_openvino_model"

# Ambil frame pertama dari video
video_path = "uploaded_sources/VIGI_C340_BA-DC_20260421135123-20260421135245_0_1776816060449_3ddeb6f4.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Tidak bisa membaca video")
    exit(1)

print(f"Original frame shape: {frame.shape}")
print("=" * 80)

# Load model
print(f"\nLoading OpenVINO model: {OPENVINO_MODEL}")
model = YOLO(OPENVINO_MODEL)

# Test 1: intel:gpu (sesuai config)
print("\n[Test 1] Device: intel:gpu, conf=0.25")
results = model.predict(
    source=frame,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="intel:gpu",
    verbose=True,
)
print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")

# Test 2: CPU
print("\n[Test 2] Device: cpu, conf=0.25")
results = model.predict(
    source=frame,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="cpu",
    verbose=True,
)
print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")

# Test 3: Lower confidence threshold
print("\n[Test 3] Device: intel:gpu, conf=0.01 (very low)")
results = model.predict(
    source=frame,
    conf=0.01,
    iou=0.7,
    imgsz=640,
    device="intel:gpu",
    verbose=True,
)
print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")
if results[0].boxes and len(results[0].boxes) > 0:
    print("Confidence scores:", results[0].boxes.conf.cpu().numpy())

# Test 4: Cek raw output sebelum NMS
print("\n[Test 4] Checking model output details...")
results = model.predict(
    source=frame,
    conf=0.01,
    iou=0.7,
    imgsz=640,
    device="intel:gpu",
    verbose=False,
)
result = results[0]
print(f"Result type: {type(result)}")
print(f"Has boxes: {result.boxes is not None}")
if result.boxes is not None:
    print(f"Boxes shape: {result.boxes.xyxy.shape}")
    print(f"Boxes data: {result.boxes.data}")
