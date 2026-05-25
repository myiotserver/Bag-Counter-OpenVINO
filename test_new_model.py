"""
Test model OpenVINO yang baru (FP32) vs yang lama (FP16)
"""
import cv2
from ultralytics import YOLO

# Ambil frame test
video_path = "uploaded_sources/VIGI_C340_BA-DC_20260421135123-20260421135245_0_1776816060449_3ddeb6f4.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Tidak bisa membaca video")
    exit(1)

print(f"Frame shape: {frame.shape}")
print("=" * 80)

# Test model baru (FP32)
print("\n[NEW MODEL - FP32] Testing on intel:gpu")
model_new = YOLO("4th-trainingyolo26m640_openvino_model")
results = model_new.predict(
    source=frame,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="intel:gpu",
    verbose=True,
)
print(f"Detections: {len(results[0].boxes) if results[0].boxes else 0}")
if results[0].boxes and len(results[0].boxes) > 0:
    print("Confidence scores:", results[0].boxes.conf.cpu().numpy())
    print("Classes:", results[0].boxes.cls.cpu().numpy())

print("\n" + "=" * 80)
print("\nSUMMARY:")
print(f"Model baru (FP32) di intel:gpu: {len(results[0].boxes) if results[0].boxes else 0} detections")
