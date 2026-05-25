from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Ultralytics NDJSON dataset to local YOLO layout.")
    parser.add_argument("ndjson", type=Path, help="Path to Ultralytics NDJSON dataset export.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bag_counter_yolo_dataset"),
        help="Output directory for YOLO dataset.",
    )
    return parser.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label_file(path: Path, boxes: list[list[float]]) -> None:
    lines = [" ".join(str(v) for v in box) for box in boxes]
    path.write_text("\n".join(lines), encoding="utf-8")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def main() -> int:
    args = parse_args()
    if not args.ndjson.exists():
        raise FileNotFoundError(f"NDJSON not found: {args.ndjson}")

    output_root = args.output
    ensure_dirs(output_root)

    dataset_info: dict | None = None
    counts = {"train": 0, "val": 0, "test": 0}

    with args.ndjson.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            record_type = record.get("type")

            if record_type == "dataset":
                dataset_info = record
                continue

            if record_type != "image":
                continue

            split = record.get("split", "train")
            if split not in counts:
                split = "train"

            file_name = record["file"]
            image_path = output_root / "images" / split / file_name
            label_path = output_root / "labels" / split / f"{Path(file_name).stem}.txt"

            download_file(record["url"], image_path)

            boxes = record.get("annotations", {}).get("boxes", [])
            write_label_file(label_path, boxes)
            counts[split] += 1

    if dataset_info is None:
        raise ValueError("Dataset header record not found in NDJSON.")

    names = {int(k): v for k, v in dataset_info.get("class_names", {}).items()}
    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_data, fh, sort_keys=False, allow_unicode=False)

    print(f"Converted dataset to: {output_root}")
    print(f"Counts: {counts}")
    print(f"YAML: {output_root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
