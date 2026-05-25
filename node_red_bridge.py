# =============================================================================
# node_red_bridge.py  -  HTTP server: Web UI + Node-RED API
# =============================================================================

import json
import queue
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, Response, jsonify, render_template, request

import config

# ---------------------------------------------------------------------------
# Import shared state dari bag_counter (injected at runtime)
# ---------------------------------------------------------------------------
try:
    import bag_counter as _bc
    _get_state      = lambda: dict(_bc.shared_state)
    _sse_subs       = _bc.sse_subscribers
    _sse_lock       = _bc.sse_lock
    _get_frame      = lambda: _bc.latest_frame_jpeg
    _frame_lock     = _bc.latest_frame_lock
    _start_pipeline = _bc.start_pipeline
    _stop_pipeline  = _bc.stop_pipeline
except ImportError:
    _shared = {"running": False, "error": "bag_counter tidak di-import"}
    _get_state      = lambda: dict(_shared)
    _sse_subs       = []
    _sse_lock       = threading.Lock()
    _get_frame      = lambda: None
    _frame_lock     = threading.Lock()
    _start_pipeline = lambda source=None: False
    _stop_pipeline  = lambda: None

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates")
app.config["JSON_SORT_KEYS"] = False

UPLOAD_DIR = Path("uploaded_sources")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_UPLOAD_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".jpg", ".jpeg", ".png", ".bmp"
}


# ============================================================
# Web UI
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Halaman utama Web UI."""
    return render_template("index.html", bridge_port=config.BRIDGE_PORT)


@app.route("/video_feed", methods=["GET"])
def video_feed():
    """
    MJPEG stream — ditampilkan langsung di tag <img> di browser.
    Tidak perlu plugin apapun; semua browser modern mendukung.
    """
    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        placeholder = _make_placeholder()
        while True:
            with _frame_lock:
                frame = _get_frame()
            if frame is None:
                frame = placeholder
            yield boundary + frame + b"\r\n"
            time.sleep(0.05)   # ~20 FPS display

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload_source", methods=["POST"])
def upload_source():
    """Terima file lokal dari browser dan simpan agar bisa dibuka OpenCV."""
    if "source_file" not in request.files:
        return jsonify({"ok": False, "message": "File upload tidak ditemukan"}), 400

    file = request.files["source_file"]
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()

    if not original_name:
        return jsonify({"ok": False, "message": "Nama file kosong"}), 400
    if suffix not in ALLOWED_UPLOAD_EXTS:
        return jsonify({"ok": False, "message": f"Format file tidak didukung: {suffix}"}), 400

    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(original_name).stem).strip("_")
    if not safe_stem:
        safe_stem = "source"

    target = UPLOAD_DIR / f"{safe_stem}_{uuid.uuid4().hex[:8]}{suffix}"
    file.save(target)

    return jsonify({
        "ok": True,
        "message": "File berhasil diupload",
        "name": original_name,
        "path": str(target.resolve()),
    })


def _make_placeholder() -> bytes:
    """Buat gambar placeholder hitam 640×360 untuk saat pipeline belum jalan."""
    import cv2, numpy as np
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Pipeline belum berjalan",
                (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
    cv2.putText(img, "Tekan  Start  untuk memulai",
                (140, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 60), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ============================================================
# Kontrol Pipeline
# ============================================================

@app.route("/control", methods=["POST"])
def control():
    """
    Kontrol pipeline dari Web UI.

    Body JSON:
      {"action": "start", "source": "0"}   ← mulai pipeline
      {"action": "stop"}                    ← hentikan pipeline
    """
    data = request.get_json(silent=True) or {}
    action = data.get("action", "").lower()

    if action == "start":
        source = data.get("source", None)
        ok = _start_pipeline(source)
        return jsonify({"ok": ok, "message": "Pipeline dimulai" if ok else "Sudah berjalan"})

    if action == "stop":
        _stop_pipeline()
        return jsonify({"ok": True, "message": "Pipeline dihentikan"})

    return jsonify({"ok": False, "message": f"Action tidak dikenal: {action}"}), 400


@app.route("/config", methods=["POST"])
def update_config():
    """
    Update parameter inferensi lokal secara live (tanpa restart pipeline).

    Body JSON (semua opsional):
      {
        "conf": 0.25,
        "iou":  0.7,
        "imgsz": 640,
        "frame_skip": 5
      }
    """
    data = request.get_json(silent=True) or {}
    changed = []

    if "conf" in data:
        config.INFER_CONF = float(data["conf"])
        changed.append(f"conf={config.INFER_CONF}")
    if "iou" in data:
        config.INFER_IOU = float(data["iou"])
        changed.append(f"iou={config.INFER_IOU}")
    if "imgsz" in data:
        config.INFER_IMGSZ = int(data["imgsz"])
        changed.append(f"imgsz={config.INFER_IMGSZ}")
    if "frame_skip" in data:
        config.FRAME_SKIP = int(data["frame_skip"])
        changed.append(f"frame_skip={config.FRAME_SKIP}")
    if "model_path" in data:
        config.MODEL_PATH = str(data["model_path"]).strip()
        changed.append(f"model_path={config.MODEL_PATH}")
    if "infer_device" in data:
        config.INFER_DEVICE = str(data["infer_device"]).strip()
        changed.append(f"infer_device={config.INFER_DEVICE}")

    return jsonify({"ok": True, "changed": changed})


def _scan_available_models():
    """
    Scan folder kerja untuk model YOLO yang tersedia.
    Kembalikan list dict dengan structure:
      {
        "name": "model_name",
        "path": "relative/path/to/model",
        "type": "openvino" | "pytorch" | "onnx",
        "is_current": True/False
      }
    """
    models = []
    base_path = Path(".")
    
    # Scan .pt files (PyTorch)
    for pt_file in base_path.glob("*.pt"):
        models.append({
            "name": pt_file.stem,
            "path": str(pt_file),
            "type": "pytorch",
            "is_current": config.MODEL_PATH == str(pt_file)
        })
    
    # Scan OpenVINO folders (folders dengan .xml inside)
    for ov_folder in base_path.glob("*_openvino_model"):
        if (ov_folder / f"{ov_folder.stem.replace('_openvino_model', '')}.xml").exists() or \
           list(ov_folder.glob("*.xml")):
            models.append({
                "name": ov_folder.name,
                "path": str(ov_folder),
                "type": "openvino",
                "is_current": config.MODEL_PATH == str(ov_folder)
            })
    
    # Scan .onnx files
    for onnx_file in base_path.glob("*.onnx"):
        models.append({
            "name": onnx_file.stem,
            "path": str(onnx_file),
            "type": "onnx",
            "is_current": config.MODEL_PATH == str(onnx_file)
        })
    
    # Sort by name
    models.sort(key=lambda x: x["name"])
    return models


@app.route("/models", methods=["GET"])
def get_models():
    """Kembalikan daftar model yang tersedia."""
    return jsonify({
        "ok": True,
        "models": _scan_available_models()
    })


@app.route("/config", methods=["GET"])
def get_config():
    """Kembalikan konfigurasi aktif saat ini."""
    return jsonify({
        "conf":       config.INFER_CONF,
        "iou":        config.INFER_IOU,
        "imgsz":      config.INFER_IMGSZ,
        "frame_skip": config.FRAME_SKIP,
        "source":     str(config.DEFAULT_SOURCE),
        "model_path": config.MODEL_PATH,
        "infer_device": config.INFER_DEVICE,
    })


# ============================================================
# Node-RED API
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    state = _get_state()
    return jsonify({
        "status": "ok",
        "running": state.get("running", False),
        "last_frame": state.get("frame_id", 0),
        "last_timestamp": state.get("timestamp"),
    })


@app.route("/results", methods=["GET"])
def results():
    return jsonify(_get_state())


@app.route("/results/stream", methods=["GET"])
def results_stream():
    """SSE live stream untuk Node-RED."""
    q: queue.Queue = queue.Queue(maxsize=100)
    with _sse_lock:
        _sse_subs.append(q)

    def generate():
        try:
            yield f"data: {json.dumps(_get_state())}\n\n"
        except Exception:
            pass
        try:
            while True:
                try:
                    yield q.get(timeout=30)
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_subs.remove(q)
                except ValueError:
                    pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Start
# ============================================================

def start_bridge():
    """Jalankan Flask server di daemon thread."""
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app.run(
        host=config.BRIDGE_HOST,
        port=config.BRIDGE_PORT,
        threaded=True,
        use_reloader=False,
        debug=False,
    )


if __name__ == "__main__":
    print(f"[BRIDGE] Standalone — http://localhost:{config.BRIDGE_PORT}")
    start_bridge()
