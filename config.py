# =============================================================================
# config.py - Konfigurasi terpusat Bag Counter
# =============================================================================

# --- Local Model Inference ---
# Bisa berupa file .pt, .onnx, atau folder OpenVINO hasil export.
MODEL_PATH = "1st-trainingyolo26s512_int8_openvino_model"

# Untuk Intel GPU dengan OpenVINO gunakan "intel:gpu".
# Opsi lain yang umum: "cpu", "intel:cpu".
# bag_counter.py akan memakai OpenVINO Runtime langsung untuk model folder
# agar bisa menerapkan GPU config yang direkomendasikan dokumentasi Intel.
INFER_DEVICE = "intel:gpu"

# --- Parameter Inferensi ---
INFER_CONF = 0.25       # confidence threshold
INFER_IOU = 0.7         # IoU threshold untuk NMS
INFER_IMGSZ = 512       # ukuran gambar input model

# --- Source Video / Kamera ---
# Bisa dioverride lewat argumen CLI --source
# 0           -> webcam default
# "video.mp4" -> file video
# "foto.jpg"  -> gambar tunggal (diproses sekali)
# "rtsp://..." -> RTSP stream
DEFAULT_SOURCE = 0

# --- Frame Sampling ---
# Jalankan inferensi tiap FRAME_SKIP frame yang dibaca
# Naikkan nilai ini untuk mengurangi beban komputasi
FRAME_SKIP = 3

# --- Preview ---
# Tampilkan jendela cv2.imshow() dengan bounding box
SHOW_PREVIEW = False  # set True untuk debugging visual; butuh akses display (X11/Wayland)

# --- Node-RED Bridge ---
BRIDGE_HOST = "0.0.0.0"   # listen semua interface; ganti "127.0.0.1" untuk lokal saja
BRIDGE_PORT = 5000

# --- ByteTrack Tracker ---
TRACK_THRESH = 0.25     # threshold minimum confidence untuk create track baru
TRACK_BUFFER = 30       # jumlah frame sebelum track dihapus (missed)
MATCH_THRESH = 0.8      # threshold IoU untuk matching deteksi ke track
TRACK_FPS = 30          # frame rate estimasi (untuk Kalman filter prediction)

# --- Virtual Line (Crossing) ---
# Posisi garis virtual sebagai rasio dari tinggi frame (0.0 = atas, 1.0 = bawah)
LINE_Y_RATIO = 0.65      # default: tengah frame
