# =============================================================================
# bag_counter.py  —  Pipeline utama
#
# Workflow:
#   Source (video / gambar / webcam / RTSP)
#     → split frame-by-frame (OpenCV)
#     → encode JPEG in-memory
#     → inferensi model lokal (YOLO / OpenVINO)
#     → parse JSON response
#     → ByteTrack multi-object tracking
#     → line crossing detection (count IN / OUT)
#     → simpan ke shared_state (dibaca oleh node_red_bridge / Web UI)
#     → (opsional) tampilkan preview di browser via MJPEG
#
# Arsitektur multi-thread:
#   Thread 1 (Capture)   : baca frame dari source, simpan ke buffer
#   Thread 2 (Inference) : ambil frame → inferensi lokal → ByteTrack → update state
#   Thread 3 (Display)   : overlay detections ke frame terbaru → MJPEG
# =============================================================================

import argparse
import os
import threading
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

import config

# ByteTrack
from cjm_byte_track.core import BYTETracker, BaseTrack
from ultralytics import YOLO
from yolo_openvino_fixed import OpenVINOModel, is_openvino_model_dir

# ---------------------------------------------------------------------------
# Shared state — ditulis oleh pipeline, dibaca oleh node_red_bridge & Web UI
# ---------------------------------------------------------------------------
shared_state: dict = {
    "timestamp": None,
    "frame_id": 0,
    "source": None,
    "model_path": config.MODEL_PATH,
    "infer_device": config.INFER_DEVICE,
    "total_detections": 0,
    "detections": [],
    "error": None,
    "running": False,
    "count_in": 0,
    "count_out": 0,
    "inference_latency_ms": 0,
}

state_lock = threading.Lock()

# SSE subscribers: list of queue.Queue, diisi oleh bridge
sse_subscribers: list = []
sse_lock = threading.Lock()

# Buffer frame terakhir dengan bounding box (JPEG bytes) untuk MJPEG stream
latest_frame_lock = threading.Lock()
latest_frame_jpeg: bytes | None = None

# Event untuk menghentikan pipeline dari luar (Web UI tombol Stop)
stop_event = threading.Event()

# Thread pipeline yang sedang berjalan
pipeline_thread: threading.Thread | None = None

# Model cache agar tidak reload setiap frame
model_lock = threading.Lock()
loaded_model: object | None = None
loaded_model_path: str | None = None


# ---------------------------------------------------------------------------
# Frame Buffer — thread-safe buffer untuk frame terbaru
# ---------------------------------------------------------------------------
class FrameBuffer:
    """Thread-safe buffer yang selalu menyimpan frame terbaru saja."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0

    def put(self, frame):
        """Simpan frame terbaru (overwrite yang lama)."""
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get(self):
        """Ambil frame terbaru dan ID-nya. Returns (frame, frame_id) or (None, 0)."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_id
            return None, 0

    def clear(self):
        with self._lock:
            self._frame = None
            self._frame_id = 0


# ---------------------------------------------------------------------------
# Tracked Detections Buffer — menyimpan hasil tracking terbaru
# ---------------------------------------------------------------------------
class DetectionBuffer:
    """Thread-safe buffer untuk hasil deteksi + tracking + frame inferensi."""

    def __init__(self):
        self._lock = threading.Lock()
        self._detections: list = []
        self._frame = None  # frame yang dipakai untuk inferensi (sinkron dengan detections)

    def put(self, detections: list, frame=None):
        with self._lock:
            self._detections = detections
            if frame is not None:
                self._frame = frame

    def get(self) -> tuple[list, any]:
        """Returns (detections, frame) — frame yang sinkron dengan detections."""
        with self._lock:
            return list(self._detections), self._frame

    def clear(self):
        with self._lock:
            self._detections = []
            self._frame = None


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------

def update_state(frame_id: int, source: str, detections: list,
                 error: str | None = None, inference_latency_ms: float = 0):
    """Perbarui shared_state dan broadcast ke semua SSE subscriber."""
    with state_lock:
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "frame_id": frame_id,
            "source": str(source),
            "total_detections": len(detections),
            "detections": detections,
            "error": error,
            "running": True,
            "count_in": shared_state["count_in"],
            "count_out": shared_state["count_out"],
            "model_path": config.MODEL_PATH,
            "infer_device": config.INFER_DEVICE,
            "inference_latency_ms": round(inference_latency_ms),
        }
        shared_state.update(payload)

    import json
    event = f"data: {json.dumps(payload)}\n\n"
    with sse_lock:
        for q in sse_subscribers:
            q.put(event)


def get_inference_model() -> object:
    """Load model lokal sekali dan cache untuk reuse antar frame."""
    global loaded_model, loaded_model_path

    model_path = str(Path(config.MODEL_PATH))
    with model_lock:
        if loaded_model is not None and loaded_model_path == model_path:
            return loaded_model

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Model lokal tidak ditemukan: {path_obj}. "
                "Export model OpenVINO terlebih dahulu atau ubah config.MODEL_PATH."
            )

        print(f"[MODEL] Loading model: {path_obj}")
        if is_openvino_model_dir(path_obj):
            print("[MODEL] Using direct OpenVINO Runtime backend")
            loaded_model = OpenVINOModel(path_obj, device=config.INFER_DEVICE)
        else:
            loaded_model = YOLO(str(path_obj))
        loaded_model_path = model_path
        return loaded_model


def infer_local(model: object, frame: np.ndarray) -> tuple[list, float]:
    """
    Jalankan inferensi lokal via Ultralytics.
    Model dapat berupa .pt atau folder OpenVINO hasil export.
    """
    t0 = time.perf_counter()
    if isinstance(model, OpenVINOModel):
        detections = model.infer(
            frame,
            conf=config.INFER_CONF,
            iou=config.INFER_IOU,
            imgsz=config.INFER_IMGSZ,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return detections, elapsed_ms

    results = model.predict(
        source=frame,
        conf=config.INFER_CONF,
        iou=config.INFER_IOU,
        imgsz=config.INFER_IMGSZ,
        device=config.INFER_DEVICE,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []
    if not results:
        return detections, elapsed_ms

    result = results[0]
    names = result.names or {}
    boxes = result.boxes
    if boxes is None:
        return detections, elapsed_ms

    xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.empty((0,))
    classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.empty((0,), dtype=int)

    for box, conf, cls_idx in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append({
            "label": str(names.get(int(cls_idx), cls_idx)),
            "confidence": round(float(conf), 4),
            "box": [x1, y1, x2, y2],
        })

    return detections, elapsed_ms


# ---------------------------------------------------------------------------
# ByteTrack Integration
# ---------------------------------------------------------------------------

def update_tracking(tracker: BYTETracker, detections: list, frame_h: int, frame_w: int) -> list:
    """
    Update ByteTrack tracker dengan deteksi baru dan cek line crossing.

    Args:
        tracker: BYTETracker instance
        detections: list of {"label", "confidence", "box": [x1,y1,x2,y2]}
        frame_h: tinggi frame
        frame_w: lebar frame

    Returns:
        List of tracked detections dengan track_id
    """
    if not detections:
        # Tetap update tracker dengan array kosong supaya tracks bisa di-mark lost
        empty = np.empty((0, 5), dtype=np.float32)
        tracker.update(empty, (frame_h, frame_w), (frame_h, frame_w))
        return []

    # Convert detections ke numpy array [x1, y1, x2, y2, score]
    det_array = np.array(
        [[d["box"][0], d["box"][1], d["box"][2], d["box"][3], d["confidence"]]
         for d in detections],
        dtype=np.float32
    )

    # Update tracker — returns list of active STrack objects
    online_targets = tracker.update(
        output_results=det_array,
        img_info=(frame_h, frame_w),
        img_size=(frame_h, frame_w),
    )

    # Build tracked detections list
    tracked = []
    for track in online_targets:
        x1, y1, x2, y2 = [int(v) for v in track.tlbr]
        # Cari label dari deteksi terdekat (berdasarkan IoU terbaik)
        label = _find_best_label(detections, [x1, y1, x2, y2])
        tracked.append({
            "track_id": track.track_id,
            "label": label,
            "confidence": round(float(track.score), 4),
            "box": [x1, y1, x2, y2],
        })

    return tracked


def _find_best_label(detections: list, track_box: list) -> str:
    """Cari label dari deteksi yang paling overlap dengan track box."""
    best_iou = 0.0
    best_label = "object"
    for d in detections:
        iou = _iou(d["box"], track_box)
        if iou > best_iou:
            best_iou = iou
            best_label = d["label"]
    return best_label


def _iou(a, b):
    """Hitung Intersection over Union antara dua box [x1,y1,x2,y2]."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------
# Line Crossing Detection (menggunakan track ID)
# ---------------------------------------------------------------------------

# Track history: {track_id: {"prev_cy": int, "counted": bool}}
_track_history: dict = {}
_track_history_lock = threading.Lock()


def check_crossing_tracked(tracked_dets: list, frame_h: int):
    """
    Cek apakah tracked objects melewati garis virtual.
    Menggunakan track_id sehingga setiap objek hanya dihitung sekali.
    """
    line_y = int(frame_h * config.LINE_Y_RATIO)

    active_ids = set()

    with _track_history_lock:
        for det in tracked_dets:
            tid = det["track_id"]
            box = det["box"]
            cy = (box[1] + box[3]) // 2  # center-y of bounding box
            active_ids.add(tid)

            if tid not in _track_history:
                # Track baru — simpan posisi awal, belum dihitung
                _track_history[tid] = {"prev_cy": cy, "counted": False}
                continue

            prev_cy = _track_history[tid]["prev_cy"]
            counted = _track_history[tid]["counted"]

            if not counted:
                # Cek crossing: dari atas ke bawah (IN) atau bawah ke atas (OUT)
                with state_lock:
                    if prev_cy < line_y <= cy:
                        # Crossing ke bawah → IN
                        shared_state["count_in"] += 1
                        _track_history[tid]["counted"] = True
                    elif prev_cy >= line_y > cy:
                        # Crossing ke atas → OUT
                        shared_state["count_out"] += 1
                        _track_history[tid]["counted"] = True

            _track_history[tid]["prev_cy"] = cy

        # Hapus tracks yang sudah tidak aktif (sudah hilang dari tracker)
        stale = [tid for tid in _track_history if tid not in active_ids]
        for tid in stale:
            del _track_history[tid]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame, detections: list):
    """Gambar bounding box, garis crossing, track ID, dan label di atas frame."""
    h, w = frame.shape[:2]
    line_y = int(h * config.LINE_Y_RATIO)

    # Garis crossing
    cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)
    cv2.putText(frame, "OUT ^  | v IN", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det.get("confidence", 0)
        track_id = det.get("track_id")

        # Label: "#ID label conf" jika track_id ada, atau "label conf"
        if track_id is not None:
            label = f"#{track_id} {det['label']} {conf:.2f}"
            color = (0, 255, 0)  # hijau untuk tracked
        else:
            label = f"{det['label']} {conf:.2f}"
            color = (0, 180, 255)  # oranye untuk untracked

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    with state_lock:
        cin  = shared_state["count_in"]
        cout = shared_state["count_out"]
    cv2.putText(frame, f"IN:{cin}  OUT:{cout}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
    return frame


def set_latest_frame(frame):
    """Simpan frame (dengan bounding box) ke buffer untuk MJPEG stream."""
    global latest_frame_jpeg
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with latest_frame_lock:
            latest_frame_jpeg = buf.tobytes()


# ---------------------------------------------------------------------------
# Source Input
# ---------------------------------------------------------------------------

def is_image_file(source: str) -> bool:
    ext = os.path.splitext(str(source))[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Multi-Thread Pipeline
# ---------------------------------------------------------------------------

def _capture_loop(cap: cv2.VideoCapture, frame_buf: FrameBuffer, stop_ev: threading.Event,
                   frame_delay: float = 0.0):
    """
    Thread 1: Terus baca frame dari source dan simpan ke buffer.
    Hanya menyimpan frame terbaru (overwrite) sehingga tidak ada backlog.

    Args:
        frame_delay: Detik antara tiap frame (0 = secepat mungkin, cocok untuk webcam/RTSP).
                     Untuk file video, diisi 1/FPS agar playback sesuai kecepatan asli.
    """
    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[CAPTURE] Stream selesai atau koneksi terputus.")
            stop_ev.set()
            break
        frame_buf.put(frame)
        if frame_delay > 0:
            time.sleep(frame_delay)
    print("[CAPTURE] Thread selesai.")


def _inference_loop(
    frame_buf: FrameBuffer,
    det_buf: DetectionBuffer,
    tracker: BYTETracker,
    model: YOLO,
    source,
    stop_ev: threading.Event,
):
    """
    Thread 2: Ambil frame terbaru → inferensi lokal → ByteTrack → update state.
    Berjalan secara independen dari capture dan display.
    """
    inference_frame_id = 0
    last_processed_fid = 0
    capture_frame_count = 0

    while not stop_ev.is_set():
        frame, fid = frame_buf.get()
        if frame is None or fid == last_processed_fid:
            time.sleep(0.01)
            continue

        capture_frame_count += 1
        last_processed_fid = fid

        # Skip frame sesuai konfigurasi
        if capture_frame_count % config.FRAME_SKIP != 0:
            continue

        try:
            raw_detections, latency = infer_local(model, frame)
            frame_h, frame_w = frame.shape[:2]

            # ByteTrack update
            tracked_dets = update_tracking(tracker, raw_detections, frame_h, frame_w)

            # Cek line crossing
            check_crossing_tracked(tracked_dets, frame_h)

            # Simpan tracked detections + frame inferensi ke buffer
            det_buf.put(tracked_dets, frame)

            inference_frame_id += 1
            update_state(
                frame_id=inference_frame_id,
                source=source,
                detections=tracked_dets,
                inference_latency_ms=latency,
            )
            print(f"[FRAME {inference_frame_id}] {len(tracked_dets)} tracked | {latency:.0f}ms")
        except Exception as exc:
            inference_frame_id += 1
            update_state(
                frame_id=inference_frame_id,
                source=source,
                detections=det_buf.get()[0],
                error=f"Inferensi lokal gagal: {exc}",
                inference_latency_ms=0,
            )

    print("[INFERENCE] Thread selesai.")


def _display_loop(
    det_buf: DetectionBuffer,
    stop_ev: threading.Event,
):
    """
    Thread 3: Tampilkan frame inferensi + bounding box → MJPEG stream.
    Update hanya saat ada hasil inferensi baru (sinkron dengan detections).
    """
    last_dets = None

    while not stop_ev.is_set():
        current_dets, inf_frame = det_buf.get()
        if inf_frame is None:
            time.sleep(0.05)
            continue

        # Hanya update display jika ada deteksi baru (hindari re-render yang sama)
        if current_dets is not last_dets:
            last_dets = current_dets
            preview = draw_detections(inf_frame.copy(), current_dets)
            set_latest_frame(preview)

            # Optional: tampilkan jendela cv2 lokal
            if config.SHOW_PREVIEW:
                try:
                    cv2.imshow("Bag Counter", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[DISPLAY] Keluar via tombol 'q'.")
                        stop_ev.set()
                        break
                except cv2.error:
                    config.SHOW_PREVIEW = False

        time.sleep(0.05)  # check setiap 50ms

    print("[DISPLAY] Thread selesai.")


def run_pipeline(source):
    """Loop utama pipeline: baca frame → inferensi lokal → tracking → update state."""
    global latest_frame_jpeg, _track_history

    stop_event.clear()

    # Reset tracking state
    with _track_history_lock:
        _track_history = {}
    # Reset ByteTrack internal ID counter
    BaseTrack._count = 0

    with state_lock:
        shared_state["running"] = True
        shared_state["source"] = str(source)
        shared_state["model_path"] = config.MODEL_PATH
        shared_state["infer_device"] = config.INFER_DEVICE
        shared_state["error"] = None
        shared_state["count_in"] = 0
        shared_state["count_out"] = 0
        shared_state["inference_latency_ms"] = 0

    print(f"[PIPELINE] Memulai pipeline dari source: {source}")

    try:
        model = get_inference_model()
        print(f"[MODEL] Siap untuk inferensi lokal | path={config.MODEL_PATH} | device={config.INFER_DEVICE}")
    except Exception as exc:
        msg = str(exc)
        print(f"[ERROR] {msg}")
        with state_lock:
            shared_state["running"] = False
            shared_state["error"] = msg
        return

    # Inisialisasi ByteTrack tracker
    tracker = BYTETracker(
        track_thresh=config.TRACK_THRESH,
        track_buffer=config.TRACK_BUFFER,
        match_thresh=config.MATCH_THRESH,
        frame_rate=config.TRACK_FPS,
    )

    # ── Kasus khusus: gambar tunggal ──────────────────────────────────────
    if isinstance(source, str) and is_image_file(source):
        frame = cv2.imread(source)
        if frame is None:
            msg = f"Tidak bisa membaca file gambar: {source}"
            print(f"[ERROR] {msg}")
            with state_lock:
                shared_state["running"] = False
                shared_state["error"] = msg
            return

        try:
            raw_detections, latency = infer_local(model, frame)
            frame_h, frame_w = frame.shape[:2]
            tracked_dets = update_tracking(tracker, raw_detections, frame_h, frame_w)
            update_state(frame_id=0, source=source, detections=tracked_dets, inference_latency_ms=latency)
            print(f"[RESULT] {len(tracked_dets)} tracked objects | {latency:.0f}ms")
            preview = draw_detections(frame.copy(), tracked_dets)
            set_latest_frame(preview)
        except Exception as exc:
            update_state(
                frame_id=0,
                source=source,
                detections=[],
                error=f"Inferensi lokal gagal: {exc}",
                inference_latency_ms=0,
            )

        with state_lock:
            shared_state["running"] = False
        return

    # ── Video / Webcam / RTSP — Multi-Thread Pipeline ────────────────────
    if isinstance(source, str) and not source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        source_path = Path(source)
        if not source_path.exists():
            msg = f"Source file tidak ditemukan: {source_path}"
            print(f"[ERROR] {msg}")
            with state_lock:
                shared_state["running"] = False
                shared_state["error"] = msg
            return
        source = str(source_path.resolve())

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        msg = f"Tidak bisa membuka source: {source}"
        print(f"[ERROR] {msg}")
        with state_lock:
            shared_state["running"] = False
            shared_state["error"] = msg
        return

    # Hitung delay antar frame untuk file video agar tidak fast-forward.
    # Webcam dan RTSP sudah di-throttle oleh hardware/server.
    frame_delay = 0.0
    is_live = isinstance(source, int) or (isinstance(source, str) and source.startswith("rtsp"))
    if not is_live:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps and native_fps > 0:
            frame_delay = 1.0 / native_fps
            print(f"[PIPELINE] Video file: {native_fps:.1f} FPS, delay={frame_delay*1000:.1f}ms/frame")

    frame_buf = FrameBuffer()
    det_buf = DetectionBuffer()

    # Mulai 3 thread
    threads = [
        threading.Thread(
            target=_capture_loop,
            args=(cap, frame_buf, stop_event, frame_delay),
            daemon=True,
            name="CaptureThread",
        ),
        threading.Thread(
            target=_inference_loop,
            args=(frame_buf, det_buf, tracker, model, source, stop_event),
            daemon=True,
            name="InferenceThread",
        ),
        threading.Thread(
            target=_display_loop,
            args=(det_buf, stop_event),
            daemon=True,
            name="DisplayThread",
        ),
    ]

    try:
        for t in threads:
            t.start()
            print(f"[PIPELINE] Thread '{t.name}' dimulai.")

        # Tunggu hingga dihentikan
        for t in threads:
            t.join()

    finally:
        cap.release()
        if config.SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        with state_lock:
            shared_state["running"] = False
        # Broadcast stopped state ke semua SSE subscriber
        import json
        with state_lock:
            payload = dict(shared_state)
        event = f"data: {json.dumps(payload)}\n\n"
        with sse_lock:
            for q in sse_subscribers:
                q.put(event)
        with latest_frame_lock:
            latest_frame_jpeg = None
        print("[PIPELINE] Pipeline berhenti.")


def start_pipeline(source=None):
    """Mulai pipeline di thread baru. Dipanggil dari Web UI tombol Start."""
    global pipeline_thread

    if shared_state.get("running"):
        print("[PIPELINE] Sudah berjalan, abaikan start baru.")
        return False

    if source is None:
        source = config.DEFAULT_SOURCE
    else:
        try:
            source = int(source)
        except (ValueError, TypeError):
            source = str(source).strip().strip('"').strip("'")
            if source and not source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                source = str(Path(source).expanduser())

    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(source,),
        daemon=True,
        name="PipelineThread",
    )
    pipeline_thread.start()
    return True


def stop_pipeline():
    """Hentikan pipeline yang sedang berjalan. Dipanggil dari Web UI tombol Stop."""
    stop_event.set()
    print("[PIPELINE] Stop diminta.")


# ---------------------------------------------------------------------------
# Entry Point (CLI)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bag Counter - Local YOLO/OpenVINO + ByteTrack + Web UI + Node-RED Bridge"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Webcam index (0/1), path video/gambar, atau RTSP URL.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Nonaktifkan jendela preview cv2 lokal.",
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Jangan jalankan Web UI / Node-RED HTTP bridge.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path ke model lokal (.pt/.onnx/folder OpenVINO). Override config.MODEL_PATH.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device inferensi lokal, misalnya intel:gpu, intel:cpu, atau cpu.",
    )
    args = parser.parse_args()

    if args.no_preview:
        config.SHOW_PREVIEW = False
    if args.model:
        config.MODEL_PATH = args.model
    if args.device:
        config.INFER_DEVICE = args.device

    # ── Jalankan Web UI + Node-RED bridge ─────────────────────────────────
    if not args.no_bridge:
        try:
            from node_red_bridge import start_bridge
            bridge_thread = threading.Thread(
                target=start_bridge, daemon=True, name="NodeREDBridge"
            )
            bridge_thread.start()
            print(
                f"[BRIDGE] Web UI berjalan di "
                f"http://localhost:{config.BRIDGE_PORT}"
            )
        except ImportError:
            print("[WARNING] node_red_bridge.py tidak ditemukan.")

    # ── Auto-start pipeline jika source diberikan via CLI ─────────────────
    if args.source is not None:
        start_pipeline(args.source)
    else:
        # Tanpa --source: biarkan user menekan tombol Start di Web UI
        print("[INFO] Buka browser ke http://localhost:5000 dan tekan Start.")

    # Jaga proses tetap hidup
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_pipeline()
        print("[MAIN] Dihentikan.")


if __name__ == "__main__":
    main()
