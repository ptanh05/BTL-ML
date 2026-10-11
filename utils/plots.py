"""Plotting and visualisation helpers for M5 evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _curve(value: str | Path | list[float] | np.ndarray) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    path = Path(value)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("loss", data.get("values", next(iter(data.values()))))
        return np.asarray(data, dtype=float)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        key = "loss" if rows and "loss" in rows[0] else list(rows[0])[0]
        return np.asarray([float(row[key]) for row in rows], dtype=float)
    return np.loadtxt(path, dtype=float)


def plot_loss_comparison(log_model5: str | Path | list[float], log_model6: str | Path | list[float],
                         save_path: str | Path = "results/plots/loss_comparison.png") -> None:
    a, b = _curve(log_model5), _curve(log_model6)
    out = Path(save_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5)); plt.plot(a, label="Model-5 CIoU"); plt.plot(b, label="Model-6 SIoU")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def plot_pr_curve(pr_data: dict[str, Any], save_path: str | Path = "results/plots/pr_curve.png") -> None:
    out = Path(save_path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    for name, values in pr_data.items():
        if isinstance(values, dict):
            recall, precision = values["recall"], values["precision"]
        else:
            recall, precision = values
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(alpha=0.25); plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def draw_bounding_boxes(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray,
                        scores: np.ndarray, save_path: str | Path | None = None,
                        class_names: tuple[str, ...] = ("hat", "person")) -> np.ndarray:
    canvas = image.copy()
    colors = ((0, 200, 0), (0, 0, 220))
    for box, cls, score in zip(np.asarray(boxes), np.asarray(classes), np.asarray(scores)):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cls = int(cls); color = colors[cls % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls] if cls < len(class_names) else cls}: {float(score):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(th + base, y1)
        cv2.rectangle(canvas, (x1, y_text - th - base), (x1 + tw, y_text), color, -1)
        cv2.putText(canvas, label, (x1, y_text - base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if save_path is not None:
        output = Path(save_path); output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), canvas)
    return canvas


visualize_detections = draw_bounding_boxes
