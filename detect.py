"""Run image or directory inference and save annotated detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from datasets.transforms import letterbox
from models.head import decode_predictions, nms
from models.yolo import build_model
from utils.plots import draw_bounding_boxes

ROOT = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load(model, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


def _image_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@torch.no_grad()
def infer_image(model, image: np.ndarray, anchors, image_size: int, device: torch.device,
                conf: float, iou: float) -> np.ndarray:
    padded, ratio, (dw, dh) = letterbox(image, image_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    rows = nms(decode_predictions(model(tensor), anchors, image_size, conf)[0], iou).cpu().numpy()
    if len(rows):
        rows[:, [0, 2]] = (rows[:, [0, 2]] - dw) / ratio
        rows[:, [1, 3]] = (rows[:, [1, 3]] - dh) / ratio
        rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, image.shape[1] - 1)
        rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, image.shape[0] - 1)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect helmets and persons")
    parser.add_argument("--weights", default="weights/model_6_best.pth")
    parser.add_argument("--config", default="configs/model6.yaml")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--source", default="dataset/test")
    parser.add_argument("--output", default="results/detections")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def detect() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data_cfg = yaml.safe_load(_resolve(args.data).read_text(encoding="utf-8"))
    image_size = int(data_cfg.get("img_size", [608, 608])[0])
    model = build_model(yaml.safe_load(_resolve(args.config).read_text(encoding="utf-8")), int(data_cfg.get("num_classes", 2))).to(device).eval()
    weights = _resolve(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}. Train a model first or pass --weights.")
    _load(model, weights, device)
    files = _image_files(_resolve(args.source))
    if args.max_images is not None:
        files = files[:args.max_images]
    output = _resolve(args.output); output.mkdir(parents=True, exist_ok=True)
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections = infer_image(model, image, data_cfg["anchors"], image_size, device, args.conf, args.iou)
        annotated = draw_bounding_boxes(image, detections[:, :4], detections[:, 5], detections[:, 4], class_names=("hat", "person")) if len(detections) else image
        cv2.imwrite(str(output / path.name), annotated)
        print(f"{path.name}: {len(detections)} detections")
    return len(files)


if __name__ == "__main__":
    detect()
