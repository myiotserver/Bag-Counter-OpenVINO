from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics YOLO .pt model to OpenVINO format."
    )
    parser.add_argument(
        "weights",
        type=Path,
        help="Path to the .pt model weights, for example 4th-trainingyolo26m640.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference/export image size. Use the same size as training/inference when possible.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device used by Ultralytics, usually 'cpu'.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 weights. Useful for Intel GPU inference.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8 model. Requires calibration data for good results.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes in the exported model.",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Embed NMS into the exported model when supported.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML used for INT8 calibration, for example data.yaml.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the calibration dataset to use for INT8 export.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.weights.exists():
        print(f"Model file not found: {args.weights}", file=sys.stderr)
        return 1

    if args.weights.suffix.lower() != ".pt":
        print("Expected a .pt model file.", file=sys.stderr)
        return 1

    if args.half and args.int8:
        print("Choose either --half or --int8, not both.", file=sys.stderr)
        return 1

    if args.int8 and not args.data:
        print(
            "INT8 export should be provided with --data <dataset.yaml> for calibration.",
            file=sys.stderr,
        )
        return 1

    model = YOLO(str(args.weights))
    export_kwargs = {
        "format": "openvino",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
        "nms": args.nms,
    }
    if args.data:
        export_kwargs["data"] = args.data
        export_kwargs["fraction"] = args.fraction

    print(f"Exporting {args.weights.name} to OpenVINO...")
    exported_path = Path(model.export(**export_kwargs))
    xml_files = list(exported_path.glob("*.xml"))
    bin_files = list(exported_path.glob("*.bin"))

    print(f"Export complete: {exported_path}")
    if xml_files:
        print(f"XML: {xml_files[0]}")
    if bin_files:
        print(f"BIN: {bin_files[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
