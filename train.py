"""Two-stage YOLOv4 training entry point."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch import nn

from datasets.shwd import build_dataloader
from models.yolo import build_model
from utils.colab_sync import load_checkpoint, save_checkpoint
from utils.loss import ComputeLoss

ROOT = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one YOLOv4 helmet detector")
    parser.add_argument("--config", default="configs/model6.yaml")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="weights")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for smoke/dry runs")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Run two short epochs")
    return parser.parse_args()


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.cuda.amp.autocast()


def run_epoch(model: nn.Module, loader, criterion: ComputeLoss, optimizer=None,
              scaler=None, device: torch.device | None = None, amp: bool = True) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    for images, targets, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp and device.type == "cuda"):
            outputs = model(images)
            loss, parts = criterion(outputs, targets)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        for key in totals:
            totals[key] += parts[key]
    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}


def train() -> dict:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config_path, data_path = _resolve(args.config), _resolve(args.data)
    with config_path.open(encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)
    with data_path.open(encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    image_size = int(data_cfg.get("img_size", [608, 608])[0])
    data_root = _resolve(data_cfg.get("data_dir", "dataset"))
    train_split = _resolve(data_cfg["train_split"])
    val_split = _resolve(data_cfg.get("val_split", data_cfg["train_split"]))
    total_epochs = 2 if args.dry_run else (args.epochs or int(model_cfg["training"]["epochs_freeze"] + model_cfg["training"]["epochs_unfreeze"]))
    limit = args.limit or (8 if args.dry_run else None)
    train_batch = 2 if args.dry_run else int(model_cfg["training"]["batch_size_freeze"])
    val_batch = min(train_batch, 4)
    train_loader = build_dataloader(data_root, train_split, image_size, train_batch, True, args.workers, limit)
    val_loader = build_dataloader(data_root, val_split, image_size, val_batch, False, args.workers, limit)
    model = build_model(model_cfg, num_classes=int(data_cfg.get("num_classes", 2))).to(device)
    criterion = ComputeLoss(
        data_cfg["anchors"], num_classes=int(data_cfg.get("num_classes", 2)), img_size=image_size,
        loss_type=model_cfg.get("architecture", model_cfg).get("loss", "ciou"),
    )
    output_dir = _resolve(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    freeze_epochs = min(int(model_cfg["training"].get("epochs_freeze", 50)), total_epochs)
    _set_backbone_trainable(model, False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(model_cfg["training"].get("lr_freeze", 1e-3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(freeze_epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_metric = 0, float("inf")
    if args.resume:
        start_epoch, best_metric = load_checkpoint(args.resume, model, optimizer, scheduler, device)
    history = []
    for epoch in range(start_epoch, total_epochs):
        if epoch == freeze_epochs and total_epochs > freeze_epochs:
            _set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["training"].get("lr_unfreeze", 1e-4)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - freeze_epochs, 1))
            train_loader = build_dataloader(data_root, train_split, image_size, 2 if args.dry_run else int(model_cfg["training"]["batch_size_unfreeze"]), True, args.workers, limit)
        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, criterion, None, None, device, amp=False)
        scheduler.step()
        metric = val_stats["total"]
        is_best = metric < best_metric
        best_metric = min(best_metric, metric)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_map": -best_metric,
                 "config": model_cfg, "data_config": data_cfg, "model_size_mb": model.get_model_size_mb()}
        if is_best or (epoch + 1) % int(model_cfg["training"].get("save_interval_epochs", 10)) == 0 or epoch == total_epochs - 1:
            save_checkpoint(state, is_best, output_dir, model_cfg.get("model_id", "model"))
        row = {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
        history.append(row)
        print(f"epoch {epoch + 1}/{total_epochs} train={train_stats['total']:.4f} val={metric:.4f}")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"epochs": total_epochs, "best_val_loss": best_metric, "model_size_mb": model.get_model_size_mb()}


if __name__ == "__main__":
    print(train())
