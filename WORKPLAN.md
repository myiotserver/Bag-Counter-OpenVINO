# WORKPLAN — Bag Counter (Ultralytics Cloud Deployment + Node-RED)

## Tujuan
Membangun ulang pipeline deteksi tas dari arsitektur WebRTC/Roboflow menjadi arsitektur berbasis
**Ultralytics Cloud Deployment** dengan output yang dapat dikonsumsi oleh **Node-RED**.

---

## Arsitektur Baru

```
┌─────────────────────────────────────────────────────────────────────┐
│                         bag_counter.py                              │
│                                                                     │
│  ┌───────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │ Source Input  │───▶│ Frame Split  │───▶│ Ultralytics Cloud    │ │
│  │               │    │ (cv2)        │    │ Deployment API       │ │
│  │ • Video file  │    │              │    │ /predict (REST)      │ │
│  │ • Gambar/foto │    │ Encode JPEG  │    │                      │ │
│  │ • Webcam      │    │ in-memory    │    │ conf=0.25, iou=0.7   │ │
│  │ • RTSP stream │    │              │    │ imgsz=640            │ │
│  └───────────────┘    └──────────────┘    └──────────┬───────────┘ │
│                                                       │             │
│                                           ┌───────────▼───────────┐│
│                                           │  Parse JSON Response  ││
│                                           │  (boxes, labels,      ││
│                                           │   confidence, count)  ││
│                                           └───────────┬───────────┘│
└───────────────────────────────────────────────────────┼────────────┘
                                                        │
                              ┌─────────────────────────▼──────────────────────────┐
                              │              node_red_bridge.py                     │
                              │         HTTP Server (Flask)  port 5000              │
                              │                                                     │
                              │  GET  /results      → JSON hasil deteksi terbaru   │
                              │  GET  /results/stream → SSE live stream             │
                              │  GET  /health       → status pipeline               │
                              └─────────────────────────────────────────────────────┘
                                                        │
                              ┌─────────────────────────▼──────────────────────────┐
                              │                   Node-RED                          │
                              │                                                     │
                              │  [http request node] → GET http://localhost:5000/  │
                              │  [function node]     → parse JSON payload           │
                              │  [dashboard / MQTT / database node]                 │
                              └─────────────────────────────────────────────────────┘
```

---

## Struktur File

```
python_bag_counter/
├── bag_counter.py          ← Pipeline utama (REWRITE)
├── node_red_bridge.py      ← HTTP server untuk Node-RED (BARU)
├── config.py               ← Konfigurasi terpusat (BARU)
├── requirements.txt        ← Dependency (UPDATE)
└── WORKPLAN.md             ← Dokumen ini
```

---

## Detail Implementasi

### 1. `config.py` — Konfigurasi Terpusat
| Parameter       | Nilai Default                                                           |
|-----------------|-------------------------------------------------------------------------|
| `API_URL`       | `https://predict-69e50cc8d7d9d51b7eea-dproatj77a-et.a.run.app/predict` |
| `API_KEY`       | `ul_ca3174e11f46973bb50d042fb7866b7977aba35c`                           |
| `CONF`          | `0.25`                                                                  |
| `IOU`           | `0.7`                                                                   |
| `IMGSZ`         | `640`                                                                   |
| `FRAME_SKIP`    | `5` (kirim 1 frame tiap N frame untuk hemat bandwidth)                  |
| `BRIDGE_PORT`   | `5000`                                                                  |
| `SOURCE`        | `0` (webcam) / path file / RTSP URL                                     |

### 2. `bag_counter.py` — Pipeline Utama
- Baca source dengan `cv2.VideoCapture` (mendukung int/path/RTSP)
- Loop frame, skip setiap N frame (`FRAME_SKIP`)
- Encode frame ke JPEG in-memory (`cv2.imencode`)
- POST ke Ultralytics `/predict` dengan header `Authorization: Bearer <key>`
- Parse response JSON → ekstrak `boxes`, `labels`, `confidence`, hitung jumlah deteksi
- Simpan hasil ke **shared state** yang dibaca oleh bridge
- Tampilkan preview dengan bounding box (opsional, bisa di-disable)

### 3. `node_red_bridge.py` — HTTP Bridge
- Flask app jalan di thread terpisah
- Endpoint `GET /results` → return JSON:
  ```json
  {
    "timestamp": "2026-04-20T10:30:00",
    "frame_id": 142,
    "source": "rtsp://...",
    "total_detections": 5,
    "detections": [
      {"label": "bag", "confidence": 0.91, "box": [x1,y1,x2,y2]},
      ...
    ]
  }
  ```
- Endpoint `GET /results/stream` → Server-Sent Events (SSE) untuk live push ke Node-RED
- Endpoint `GET /health` → `{"status": "ok", "running": true}`

### 4. `requirements.txt`
```
opencv-python
requests
flask
```

---

## Cara Pakai

### Jalankan pipeline
```bash
# Webcam (default)
python bag_counter.py

# Video file
python bag_counter.py --source video.mp4

# RTSP
python bag_counter.py --source rtsp://user:pass@192.168.1.100/stream

# Gambar tunggal
python bag_counter.py --source foto.jpg
```

### Integrasi Node-RED
1. Tambahkan node **`http request`** dengan URL `http://localhost:5000/results`
2. Set method `GET`
3. Sambungkan ke node **`json`** → node **`function`** untuk parsing
4. Output ke dashboard, MQTT, database, dsb.

Atau gunakan **SSE** live stream:
1. Node **`http request`** → `http://localhost:5000/results/stream`
2. Data push otomatis setiap ada frame baru

---

## Alur Data Detail

```
cv2.VideoCapture(source)
        │
        ▼ setiap FRAME_SKIP frame
cv2.imencode('.jpg', frame)
        │
        ▼
requests.post(API_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    data={"conf":0.25, "iou":0.7, "imgsz":640},
    files={"file": jpeg_bytes}
)
        │
        ▼
response.json()  →  {"predictions": [...], "image": {...}}
        │
        ▼
parse → {"total": N, "detections": [...]}
        │
        ▼
shared_state (thread-safe dict)
        │
        ├──▶  cv2.imshow() preview (opsional)
        │
        └──▶  Flask /results  ◀──── Node-RED HTTP request
```

---

## Dependency Baru vs Lama

| Library               | Sebelum       | Sesudah   | Keterangan                          |
|-----------------------|---------------|-----------|-------------------------------------|
| `opencv-python`       | ✅ ada        | ✅ tetap  | Capture & encode frame              |
| `inference-sdk`       | ✅ ada        | ❌ hapus  | Diganti direct HTTP ke Ultralytics  |
| `aiortc`              | ✅ ada        | ❌ hapus  | WebRTC tidak dipakai lagi           |
| `requests`            | ❌ tidak ada  | ✅ tambah | HTTP POST ke Ultralytics API        |
| `flask`               | ❌ tidak ada  | ✅ tambah | Bridge server untuk Node-RED        |
