"""
Test script untuk membandingkan deteksi PyTorch vs OpenVINO
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Path ke model
PYTORCH_MODEL = "4th-trainingyolo26m640.pt"
OPENVINO_MODEL = "4th-trainingyolo26m640_openvino_model"

# Ambil frame pertama dari video
video_path = "uploaded_sources/VIGI_C340_BA-DC_20260421135123-20260421135245_0_1776816060449_3ddeb6f4.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Tidak bisa membaca video")
    exit(1)

print(f"Frame shape: {frame.shape}")
print("=" * 80)

# Test PyTorch model
print("\n[1] Testing PyTorch model...")
print(f"Model: {PYTORCH_MODEL}")
model_pt = YOLO(PYTORCH_MODEL)

t0 = time.perf_counter()
results_pt = model_pt.predict(
    source=frame,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="cpu",  # Gunakan CPU untuk fair comparison
    verbose=True,
)
elapsed_pt = (time.perf_counter() - t0) * 1000

print(f"\nPyTorch Results:")
print(f"  Inference time: {elapsed_pt:.1f}ms")
if results_pt and results_pt[0].boxes is not None:
    boxes_pt = results_pt[0].boxes
    print(f"  Detections: {len(boxes_pt)}")
    print(f"  boxes.xyxy shape: {boxes_pt.xyxy.shape if boxes_pt.xyxy is not None else 'None'}")
    print(f"  boxes.conf shape: {boxes_pt.conf.shape if boxes_pt.conf is not None else 'None'}")
    print(f"  boxes.cls shape: {boxes_pt.cls.shape if boxes_pt.cls is not None else 'None'}")
    if len(boxes_pt) > 0:
        print(f"  First detection: conf={boxes_pt.conf[0]:.3f}, cls={int(boxes_pt.cls[0])}")
else:
    print("  Detections: 0")

print("=" * 80)

# Test OpenVINO model
print("\n[2] Testing OpenVINO model...")
print(f"Model: {OPENVINO_MODEL}")
model_ov = YOLO(OPENVINO_MODEL)

t0 = time.perf_counter()
results_ov = model_ov.predict(
    source=frame,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="intel:gpu",  # Sesuai config
    verbose=True,
)
elapsed_ov = (time.perf_counter() - t0) * 1000

print(f"\nOpenVINO Results:")
print(f"  Inference time: {elapsed_ov:.1f}ms")
if results_ov and results_ov[0].boxes is not None:
    boxes_ov = results_ov[0].boxes
    print(f"  Detections: {len(boxes_ov)}")
    print(f"  boxes.xyxy shape: {boxes_ov.xyxy.shape if boxes_ov.xyxy is not None else 'None'}")
    print(f"  boxes.conf shape: {boxes_ov.conf.shape if boxes_ov.conf is not None else 'None'}")
    print(f"  boxes.cls shape: {boxes_ov.cls.shape if boxes_ov.cls is not None else 'None'}")
    if len(boxes_ov) > 0:
        print(f"  First detection: conf={boxes_ov.conf[0]:.3f}, cls={int(boxes_ov.cls[0])}")
else:
    print("  Detections: 0")

print("=" * 80)
print("\n[SUMMARY]")
print(f"PyTorch detections: {len(results_pt[0].boxes) if results_pt and results_pt[0].boxes else 0}")
print(f"OpenVINO detections: {len(results_ov[0].boxes) if results_ov and results_ov[0].boxes else 0}")
