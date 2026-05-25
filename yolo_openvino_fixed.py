"""
Direct OpenVINO Runtime wrapper for exported YOLO models.

This bypasses the default Ultralytics OpenVINO execution path so we can apply
documented GPU compile properties for Intel integrated graphics on Windows.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import openvino.properties.hint as hints
import yaml


def is_openvino_model_dir(model_path: str | Path) -> bool:
    """Return True when the path looks like an exported OpenVINO model folder."""
    path = Path(model_path)
    return path.is_dir() and any(path.glob("*.xml")) and any(path.glob("*.bin"))


class OpenVINOModel:
    """Small inference wrapper for exported Ultralytics OpenVINO models."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_path)
        self.device = self._normalize_device(device)
        self.core = ov.Core()
        self.xml_path = self._resolve_xml_path()
        self.names = self._load_names()
        self.compiled_model = self._compile_model()
        self.output_layer = self.compiled_model.output(0)

    def infer(
        self,
        frame: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
    ) -> list[dict]:
        """Run one inference and return bag_counter-style detections."""
        input_tensor, gain, pad = self._preprocess(frame, imgsz)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape[:2], gain, pad, conf)
        return self._nms(detections, iou)

    def _resolve_xml_path(self) -> Path:
        xml_files = sorted(self.model_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"Tidak ada file .xml di model OpenVINO: {self.model_dir}")
        return xml_files[0]

    def _load_names(self) -> dict[int, str]:
        metadata_path = self.model_dir / "metadata.yaml"
        if not metadata_path.exists():
            return {}

        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        raw_names = data.get("names", {})
        return {int(k): str(v) for k, v in raw_names.items()}

    def _compile_model(self):
        model = self.core.read_model(str(self.xml_path))
        config = {}

        if self.device == "GPU":
            # Based on OpenVINO GPU-device docs:
            # - ACCURACY keeps floating-point inference on the safer path.
            # - disable_winograd_convolution is recommended for fine-tuned models
            #   that show GPU inaccuracies.
            config = {
                hints.execution_mode: hints.ExecutionMode.ACCURACY,
                hints.inference_precision: "f32",
                "GPU_DISABLE_WINOGRAD_CONVOLUTION": True,
            }

        return self.core.compile_model(model, self.device, config)

    @staticmethod
    def _normalize_device(device: str) -> str:
        normalized = str(device).strip().lower()
        if normalized in {"intel:gpu", "gpu", "gpu.0"}:
            return "GPU"
        if normalized in {"intel:cpu", "cpu"}:
            return "CPU"
        return str(device)

    @staticmethod
    def _preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
        h0, w0 = frame.shape[:2]
        gain = min(imgsz / h0, imgsz / w0)
        new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = imgsz - new_w
        pad_h = imgsz - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(np.ascontiguousarray(tensor), axis=0)
        return tensor, gain, (left, top)

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
        gain: float,
        pad: tuple[int, int],
        conf_threshold: float,
    ) -> list[dict]:
        frame_h, frame_w = original_shape
        pad_x, pad_y = pad
        detections: list[dict] = []

        predictions = np.asarray(output)[0]
        for pred in predictions:
            if pred.shape[0] < 6 or np.isnan(pred).any():
                continue

            x1, y1, x2, y2, conf, cls_idx = pred[:6].tolist()
            if conf < conf_threshold:
                continue

            x1 = int(round((x1 - pad_x) / gain))
            y1 = int(round((y1 - pad_y) / gain))
            x2 = int(round((x2 - pad_x) / gain))
            y2 = int(round((y2 - pad_y) / gain))

            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_idx = int(cls_idx)
            detections.append({
                "label": self.names.get(cls_idx, str(cls_idx)),
                "confidence": round(float(conf), 4),
                "box": [x1, y1, x2, y2],
            })

        return detections

    @staticmethod
    def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
        kept: list[dict] = []
        for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            if all(OpenVINOModel._iou(det["box"], kept_det["box"]) <= iou_threshold for kept_det in kept):
                kept.append(det)
        return kept

    @staticmethod
    def _iou(box_a: list[int], box_b: list[int]) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
