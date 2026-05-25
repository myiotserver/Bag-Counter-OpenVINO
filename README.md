# Bag Counter (Local YOLO/OpenVINO + Node-RED Bridge)

Aplikasi Python untuk menghitung jumlah karung pada conveyor dengan inferensi lokal menggunakan model Ultralytics YOLO atau OpenVINO, dilengkapi Web UI dan bridge untuk Node-RED.

## Mode inferensi

Project ini sekarang memakai inferensi lokal saja. Alur utamanya:

`source video -> local model inference -> ByteTrack -> line crossing -> Web UI / Node-RED`

Jika ingin memakai Intel GPU, export model `.pt` ke OpenVINO lalu jalankan dengan `device="intel:gpu"`.

## Instalasi

```bash
pip install -r requirements.txt
```

## Export model ke OpenVINO

Repository ini sudah berisi beberapa file bobot `.pt`, misalnya `4th-trainingyolo26m640.pt`.

Contoh export FP16 untuk Intel GPU:

```bash
python export_openvino.py 4th-trainingyolo26m640.pt --imgsz 640 --half
```

Contoh export INT8:

```bash
python export_openvino.py 4th-trainingyolo26m640.pt --imgsz 640 --int8 --data data.yaml
```

Hasil export akan berupa folder seperti `4th-trainingyolo26m640_openvino_model/` yang berisi file `.xml` dan `.bin`.

## Konfigurasi

Edit [config.py](/C:/Users/ThinkPad%20Yoga%20L13/Nextcloud/Documents/Programming/python_bag_counter%20v1.0.1/config.py:1) untuk menyesuaikan:

- `MODEL_PATH`: path ke model lokal, misalnya folder OpenVINO hasil export.
- `INFER_DEVICE`: device inferensi, misalnya `intel:gpu`, `intel:cpu`, atau `cpu`.
- `INFER_CONF`, `INFER_IOU`, `INFER_IMGSZ`, `FRAME_SKIP`.
- `DEFAULT_SOURCE` dan `BRIDGE_PORT`.

Default saat ini diarahkan ke:

```python
MODEL_PATH = "4th-trainingyolo26m640_openvino_model"
INFER_DEVICE = "intel:gpu"
```

## Menjalankan aplikasi

Jalankan:

```bash
python bag_counter.py
```

Opsi CLI:

- `--source <path-or-index>` untuk webcam, RTSP, video, atau gambar.
- `--model <path>` untuk override `MODEL_PATH`.
- `--device <name>` untuk override `INFER_DEVICE`.
- `--no-preview` untuk menonaktifkan jendela `cv2.imshow`.
- `--no-bridge` untuk tidak menjalankan Web UI / bridge.

Contoh:

```bash
python bag_counter.py --model 4th-trainingyolo26m640_openvino_model --device intel:gpu --source 0
```

## Web UI

Setelah aplikasi berjalan, buka:

`http://localhost:5000`

UI tetap mendukung:

- Start/stop pipeline
- Live MJPEG preview
- Update `confidence`, `IoU`, `imgsz`, dan `frame_skip`
- Tampilan hasil tracking, count IN/OUT, dan latency inferensi

## Endpoint Node-RED

- `GET /health`
- `GET /results`
- `GET /results/stream`

## Struktur proyek

- `bag_counter.py`: pipeline utama inferensi lokal, tracking, counting.
- `export_openvino.py`: export `.pt` ke OpenVINO.
- `node_red_bridge.py`: server Flask untuk Web UI dan API.
- `config.py`: konfigurasi terpusat.
- `templates/`: dashboard web.

## Catatan

- Model OpenVINO perlu sudah tersedia sebelum pipeline dijalankan jika `MODEL_PATH` diarahkan ke folder export.
- Untuk Intel GPU, kompatibilitas akhir tetap bergantung pada driver Intel GPU dan instalasi OpenVINO di mesin komputer Anda.
