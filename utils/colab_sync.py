"""Atomic checkpoint helpers and optional Google Colab Drive mounting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        return False
    drive.mount(mount_point, force_remount=False)
    return True


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(state: dict[str, Any], is_best: bool, save_dir: str | Path,
                    model_name: str = "model", filename: str = "last.pth") -> Path:
    output_dir = Path(save_dir)
    last_path = output_dir / filename
    _atomic_torch_save(state, last_path)
    if is_best:
        _atomic_torch_save(state, output_dir / f"{model_name}_best.pth")
        _atomic_torch_save(state, output_dir / "best.pth")
    return last_path


def load_checkpoint(checkpoint_path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, device: str | torch.device = "cpu",
                    scaler: Any = None, payload: dict | None = None) -> tuple[int, float]:
    if payload is None:
        payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    if isinstance(payload, dict) and optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if isinstance(payload, dict) and scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if isinstance(payload, dict) and scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    epoch = int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1
    best_loss = float("inf")
    if isinstance(payload, dict):
        if "best_val_loss" in payload:
            best_loss = float(payload["best_val_loss"])
        elif "best_map" in payload:
            # Legacy training checkpoints stored negative validation loss here.
            best_loss = -float(payload["best_map"])
        elif "best_metric" in payload:
            best_loss = float(payload["best_metric"])
    return epoch + 1, best_loss


if __name__ == "__main__":
    print(f"colab_sync.py: Colab detected={mount_google_drive()}")
