"""Run Model-1..Model-6 sequentially and persist a comparable report.

Examples:
    python run_ablation.py --dry-run --limit 8 --no-fps
    python run_ablation.py --device cuda --output-root weights/ablation
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from datasets.shwd import build_dataloader
from eval import evaluate_model, _load_weights
from models.yolo import build_model
from train import train_model
from utils.metrics import measure_fps

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    1: {"map": 92.46, "size_mb": 243.92},
    2: {"map": 91.86, "size_mb": 136.13},
    3: {"map": 89.34, "size_mb": 38.75},
    4: {"map": 90.09, "size_mb": 38.99},
    5: {"map": 91.29, "size_mb": 41.88},
    6: {"map": 92.98, "size_mb": 41.88},
}


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# YOLOv4 Helmet Detection Ablation Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Device: `{args.device}`",
        f"- Epoch override: `{args.epochs or 'config (50 freeze + 150 unfreeze)'}`",
        f"- Sample limit: `{args.limit or (8 if args.dry_run else 'full split')}`",
        "",
        "| Model | Status | mAP (%) | Target (%) | Delta (pp) | Precision (%) | Recall (%) | F1 (%) | FPS | Size (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('model', '')} | {row.get('status', '')} | "
            f"{row.get('map_percent', '')} | {row.get('expected_map_percent', '')} | "
            f"{row.get('map_delta_pp', '')} | {row.get('precision_percent', '')} | "
            f"{row.get('recall_percent', '')} | {row.get('f1_percent', '')} | "
            f"{row.get('fps', '')} | {row.get('model_size_mb', '')} |"
        )
    lines += [
        "",
        "Values are measured by the current implementation. Paper targets are reference values only; they are not copied into measured results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Model-1..Model-6 ablation runner")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="weights/ablation")
    parser.add_argument("--results", default="results/ablation")
    parser.add_argument("--models", default="1,2,3,4,5,6", help="Comma-separated model IDs")
    parser.add_argument("--epochs", type=int, help="Override total epochs per model")
    parser.add_argument("--limit", type=int, help="Limit train and test samples")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Two epochs with a small sample limit")
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing best checkpoints")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run() -> list[dict]:
    args = parse_args()
    model_ids = [int(value.strip()) for value in args.models.split(",") if value.strip()]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    data_cfg = yaml.safe_load(resolve(args.data).read_text(encoding="utf-8"))
    image_size = int(data_cfg.get("img_size", [608, 608])[0])
    data_root = resolve(data_cfg.get("data_dir", "dataset"))
    sample_limit = args.limit or (8 if args.dry_run else None)
    test_loader = build_dataloader(
        data_root, resolve(data_cfg["test_split"]), image_size, 4, False,
        args.workers, sample_limit,
    )
    output_root, results_dir = resolve(args.output_root), resolve(args.results)
    rows: list[dict] = []
    for model_number in model_ids:
        if model_number not in EXPECTED:
            raise ValueError(f"Model ID must be 1..6, got {model_number}")
        model_id = f"model_{model_number}"
        config_path = resolve(f"configs/model{model_number}.yaml")
        model_output = output_root / model_id
        checkpoint = model_output / f"{model_id}_best.pth"
        record: dict = {"model": model_id, "status": "started", "config": str(config_path)}
        print(f"\n===== {model_id}: {config_path.name} =====")
        try:
            if not (args.skip_existing and checkpoint.exists()):
                training = train_model(
                    config_path=config_path, data_path=resolve(args.data),
                    device_name=str(device), output=model_output, epochs=args.epochs,
                    limit=args.limit, workers=args.workers, dry_run=args.dry_run,
                )
            else:
                training = {"checkpoint": str(checkpoint), "model_id": model_id, "status": "reused"}
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint after training: {checkpoint}")
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model = build_model(cfg, int(data_cfg.get("num_classes", 2))).to(device)
            _load_weights(model, checkpoint, device)
            measured = evaluate_model(
                model, test_loader, data_cfg["anchors"], image_size, device,
                conf=0.25, iou=0.5,
            )
            record.update(training)
            record.update(measured)
            record.update({
                "status": "ok", "weights": str(checkpoint),
                "weights_mb": checkpoint.stat().st_size / (1024 ** 2),
                "expected_map_percent": EXPECTED[model_number]["map"],
                "expected_size_mb": EXPECTED[model_number]["size_mb"],
                "map_percent": measured["map"] * 100,
                "precision_percent": measured["precision"] * 100,
                "recall_percent": measured["recall"] * 100,
                "f1_percent": measured["f1"] * 100,
                "map_delta_pp": measured["map"] * 100 - EXPECTED[model_number]["map"],
            })
            if not args.no_fps:
                record["fps"] = measure_fps(
                    model, (1, 3, image_size, image_size), str(device), warmup=5, iterations=20,
                )
            (results_dir / f"{model_id}.json").parent.mkdir(parents=True, exist_ok=True)
            (results_dir / f"{model_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print({key: record.get(key) for key in ("status", "map_percent", "precision_percent", "recall_percent", "f1_percent", "fps")})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as error:
            record.update({"status": "failed", "error": repr(error)})
            rows.append(record)
            (results_dir / f"{model_id}.json").parent.mkdir(parents=True, exist_ok=True)
            (results_dir / f"{model_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"{model_id} FAILED: {error}")
            if not args.continue_on_error:
                raise
        else:
            rows.append(record)
    write_csv(results_dir / "ablation_summary.csv", rows)
    write_report(results_dir / "ablation_report.md", rows, args)
    print(f"\nSaved summary: {results_dir / 'ablation_summary.csv'}")
    print(f"Saved report: {results_dir / 'ablation_report.md'}")
    return rows


if __name__ == "__main__":
    run()
