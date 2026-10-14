"""Evaluate one model or benchmark all six ablation configurations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import yaml

from datasets.shwd import build_dataloader
from models.head import decode_predictions, nms
from models.yolo import build_model
from utils.metrics import calculate_map, measure_fps

ROOT = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _load_weights(model, path: str | Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def evaluate_model(model, loader, anchors, image_size: int, device: torch.device,
                   conf: float, iou: float) -> dict[str, float]:
    model.eval()
    detections, truths = [], []
    for images, targets, _ in loader:
        outputs = model(images.to(device, non_blocking=True))
        decoded = decode_predictions(outputs, anchors, image_size, conf)
        for batch_index, rows in enumerate(decoded):
            detections.append(nms(rows, iou).detach().cpu().numpy())
            target_rows = targets[targets[:, 0] == batch_index]
            if target_rows.numel():
                _, cls, cx, cy, w, h = target_rows.T
                truths.append(torch.stack([(cx - w / 2) * image_size, (cy - h / 2) * image_size,
                                            (cx + w / 2) * image_size, (cy + h / 2) * image_size, cls], dim=1).numpy())
            else:
                truths.append(torch.zeros((0, 5)).numpy())
    return calculate_map(detections, truths, num_classes=2)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv4 helmet detector")
    parser.add_argument("--weights")
    parser.add_argument("--config", default="configs/model6.yaml")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark-all", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--output", default="results")
    return parser.parse_args()


def evaluate() -> list[dict]:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data_path = _resolve(args.data)
    data_cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    image_size = int(data_cfg.get("img_size", [608, 608])[0])
    data_root = _resolve(data_cfg.get("data_dir", "dataset"))
    loader = build_dataloader(data_root, _resolve(data_cfg["test_split"]), image_size, 4, False, 0, args.limit)
    configs = []
    if args.benchmark_all:
        configs = [(i, _resolve(f"configs/model{i}.yaml"), _resolve(args.weights_dir) / f"model_{i}_best.pth") for i in range(1, 7)]
    else:
        configs = [(None, _resolve(args.config), _resolve(args.weights) if args.weights else None)]
    rows = []
    for model_id, config_path, weights_path in configs:
        if weights_path is None or not weights_path.exists():
            print(f"skip {config_path.name}: weights not found ({weights_path})")
            continue
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = build_model(cfg, int(data_cfg.get("num_classes", 2))).to(device)
        _load_weights(model, weights_path, device)
        metrics = evaluate_model(model, loader, data_cfg["anchors"], image_size, device, args.conf, args.iou)
        metrics.update({"model": model_id or cfg.get("model_id", config_path.stem), "size_mb": model.get_model_size_mb(),
                        "weights_mb": weights_path.stat().st_size / (1024 ** 2)})
        if not args.no_fps:
            metrics["fps"] = measure_fps(model, (1, 3, image_size, image_size), str(device), warmup=2, iterations=5)
        print(metrics)
        rows.append(metrics)
    _write_csv(_resolve(args.output) / "benchmark.csv", rows)
    return rows


if __name__ == "__main__":
    evaluate()
