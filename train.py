"""Two-stage YOLOv4 training entry point."""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch import nn

from datasets.shwd import build_dataloader
from datasets.audit import require_paper_dataset
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
    parser.add_argument("--output", help="Defaults to the resume checkpoint directory, or weights")
    parser.add_argument("--resume")
    parser.add_argument("--backbone-weights", help="Compatible PyTorch backbone or full-model weights")
    parser.add_argument("--from-scratch", action="store_true", help="Development experiment without paper transfer learning")
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
    # BF16 has FP32's exponent range and avoids FP16 overflow in deep backbones.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_epoch(model: nn.Module, loader, criterion: ComputeLoss, optimizer=None,
              scaler=None, device: torch.device | None = None, amp: bool = True) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training and not any(p.requires_grad for p in model.backbone.parameters()):
        # Freezing includes BatchNorm running statistics, not just parameters.
        model.backbone.eval()
    totals = {"total": 0.0, "box": 0.0, "obj": 0.0, "cls": 0.0}
    updates = 0
    samples = 0
    for images, targets, _ in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp and device.type == "cuda"):
            outputs = model(images)
            loss, parts = criterion(outputs, targets)
        if training:
            if scaler is not None and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() >= old_scale:
                    updates += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                updates += 1
        for key in totals:
            totals[key] += parts[key] * images.shape[0]
        samples += images.shape[0]
    result = {key: value / max(samples, 1) for key, value in totals.items()}
    result["updates"] = updates
    return result


def _phase_optimizer(model: nn.Module, training: dict, frozen: bool,
                     freeze_epochs: int, total_epochs: int):
    _set_backbone_trainable(model, not frozen)
    phase = "freeze" if frozen else "unfreeze"
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(training.get(f"lr_{phase}", 1e-3 if frozen else 1e-4)),
    )
    duration = freeze_epochs if frozen else total_epochs - freeze_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(duration, 1))
    return optimizer, scheduler


def train_model(config_path: str | Path = "configs/model6.yaml",
                data_path: str | Path = "configs/dataset.yaml",
                device_name: str = "cuda",
                output: str | Path | None = None,
                resume: str | Path | None = None,
                epochs: int | None = None,
                limit: int | None = None,
                workers: int = 0,
                dry_run: bool = False,
                backbone_weights: str | Path | None = None,
                from_scratch: bool = False) -> dict:
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config_path, data_path = _resolve(config_path), _resolve(data_path)
    with config_path.open(encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)
    with data_path.open(encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    image_size = int(data_cfg.get("img_size", [608, 608])[0])
    data_root = _resolve(data_cfg.get("data_dir", "dataset"))
    train_split = _resolve(data_cfg["train_split"])
    val_split = _resolve(data_cfg.get("val_split", data_cfg["train_split"]))
    total_epochs = 2 if dry_run else (epochs if epochs is not None else int(model_cfg["training"]["epochs_freeze"] + model_cfg["training"]["epochs_unfreeze"]))
    limit = 8 if dry_run and limit is None else limit
    if total_epochs <= 0:
        raise ValueError("epochs must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    training = model_cfg["training"]
    arch = model_cfg.get("architecture", model_cfg)
    protocol = "smoke" if dry_run else "development" if from_scratch else training.get("protocol", "paper")
    if protocol not in {"paper", "development", "smoke"}:
        raise ValueError(f"Unsupported training protocol: {protocol}")
    pretrained = backbone_weights or training.get("backbone_weights")
    if from_scratch and pretrained:
        raise ValueError("Choose --from-scratch or pretrained backbone weights, not both")
    if protocol == "paper" and not resume and not pretrained:
        raise ValueError("Paper protocol requires --backbone-weights with a compatible pretrained backbone. "
                         "Use --dry-run to check execution or --from-scratch for a development experiment.")
    data_audit = None
    if protocol == "paper":
        explicit_settings = {"epochs_freeze": 50, "epochs_unfreeze": 150,
                             "batch_size_freeze": 16, "batch_size_unfreeze": 8,
                             "lr_freeze": .001, "optimizer": "Adam", "scheduler": "CosineAnnealingLR"}
        if total_epochs != 200 or any(training.get(k) != v for k, v in explicit_settings.items()):
            raise ValueError("Paper protocol requires the Section 4.2 schedule (50 freeze + 150 unfreeze, "
                             "batches 16/8, initial LR .001, Adam, cosine annealing). "
                             "Use a development protocol for custom schedules.")
        if data_cfg.get("img_size") != [608, 608] or data_cfg.get("class_names") != ["hat", "person"]:
            raise ValueError("Paper protocol requires 608x608 inputs and the hat/person classes")
        if limit is not None:
            raise ValueError("Paper protocol uses the full splits; use --dry-run/--from-scratch with --limit")
        data_audit = require_paper_dataset(data_root, [_resolve(data_cfg[k]) for k in
                                                     ("train_split", "val_split", "test_split")])
    model = build_model(model_cfg, num_classes=int(data_cfg.get("num_classes", 2))).to(device)
    criterion = ComputeLoss(
        data_cfg["anchors"], num_classes=int(data_cfg.get("num_classes", 2)), img_size=image_size,
        loss_type=arch.get("loss", "ciou"),
        siou_theta=float(arch.get("siou_theta", 4.0)),
        siou_eps=float(arch.get("siou_eps", 1e-7)),
        anchor_iou_threshold=float(arch.get("anchor_iou_threshold", 0.213)),
        ignore_iou_threshold=float(arch.get("ignore_iou_threshold", 0.7)),
    )
    resume_path = _resolve(resume) if resume else None
    payload = torch.load(resume_path, map_location=device) if resume_path else None
    if protocol == "paper" and payload is not None and payload.get("protocol") != "paper":
        raise ValueError("Cannot resume a development/smoke checkpoint as a paper experiment")
    output_dir = _resolve(output) if output is not None else (resume_path.parent if resume_path else ROOT / "weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_epochs = min(int(training.get("epochs_freeze", 50)), total_epochs)
    if freeze_epochs < 0:
        raise ValueError("epochs_freeze must be nonnegative")
    if payload is not None:
        # The saved phase describes the optimizer at the end of the saved epoch.
        freeze_epochs = int(payload.get("freeze_epochs", freeze_epochs))
        saved_epoch = int(payload.get("epoch", -1))
        frozen = payload.get("phase", "freeze" if saved_epoch < freeze_epochs else "unfreeze") == "freeze"
    else:
        if pretrained:
            model.backbone.load_pretrained(_resolve(pretrained))
        elif freeze_epochs:
            freeze_epochs = 0
            print("No pretrained backbone supplied: training all parameters from epoch 1.")
        frozen = freeze_epochs > 0
    optimizer, scheduler = _phase_optimizer(model, training, frozen, freeze_epochs, total_epochs)
    use_amp = bool(training.get("amp", protocol != "paper"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda" and not torch.cuda.is_bf16_supported())
    start_epoch, best_metric = 0, float("inf")
    if resume_path:
        start_epoch, best_metric = load_checkpoint(resume_path, model, optimizer, scheduler, device,
                                                   scaler=scaler, payload=payload)
    history = list(payload.get("history", [])) if payload is not None else []
    if resume_path and not history and (resume_path.parent / "history.json").exists():
        history = json.loads((resume_path.parent / "history.json").read_text(encoding="utf-8"))
    history = [row for row in history if row["epoch"] <= start_epoch]
    model_id = model_cfg.get("model_id", "model")
    checkpoint = output_dir / f"{model_id}_best.pth"
    if resume_path and output_dir.resolve() != resume_path.parent.resolve():
        previous_best = resume_path.parent / checkpoint.name
        if previous_best.exists() and not checkpoint.exists():
            shutil.copy2(previous_best, checkpoint)

    def make_train_loader(is_frozen: bool):
        phase = "freeze" if is_frozen else "unfreeze"
        batch = 2 if dry_run else int(training[f"batch_size_{phase}"])
        return build_dataloader(data_root, train_split, image_size, batch, True, workers, limit)

    train_loader = make_train_loader(frozen)
    val_loader = build_dataloader(data_root, val_split, image_size, 2 if dry_run else 4, False, workers, limit)
    for epoch in range(start_epoch, total_epochs):
        should_freeze = epoch < freeze_epochs
        if frozen != should_freeze:
            frozen = should_freeze
            optimizer, scheduler = _phase_optimizer(model, training, frozen, freeze_epochs, total_epochs)
            train_loader = make_train_loader(frozen)
        train_stats = run_epoch(model, train_loader, criterion, optimizer, scaler, device, amp=use_amp)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, criterion, None, None, device, amp=False)
        if train_stats.get("updates", 0) > 0:
            scheduler.step()
        train_stats.pop("updates", None)
        metric = val_stats["total"]
        is_best = metric < best_metric
        best_metric = min(best_metric, metric)
        row = {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
        history.append(row)
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "best_val_loss": best_metric, "phase": "freeze" if frozen else "unfreeze",
                 "freeze_epochs": freeze_epochs, "history": history,
                 "protocol": protocol, "dataset_audit": data_audit,
                 "amp": use_amp,
                 "config": model_cfg, "data_config": data_cfg, "model_size_mb": model.get_model_size_mb()}
        if is_best or (epoch + 1) % int(training.get("save_interval_epochs", 10)) == 0 or epoch == total_epochs - 1:
            save_checkpoint(state, is_best, output_dir, model_id)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch + 1}/{total_epochs} train={train_stats['total']:.4f} val={metric:.4f}")
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model_id": model_cfg.get("model_id", config_path.stem), "epochs": total_epochs,
            "best_val_loss": best_metric, "model_size_mb": model.get_model_size_mb(),
            "protocol": protocol, "freeze_epochs": freeze_epochs,
            "checkpoint": str(checkpoint), "history": str(history_path)}


def train() -> dict:
    args = parse_args()
    return train_model(
        config_path=args.config, data_path=args.data, device_name=args.device,
        output=args.output, resume=args.resume, epochs=args.epochs,
        limit=args.limit, workers=args.workers, dry_run=args.dry_run,
        backbone_weights=args.backbone_weights,
        from_scratch=args.from_scratch,
    )


if __name__ == "__main__":
    print(train())
