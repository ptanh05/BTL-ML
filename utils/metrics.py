"""Detection metrics used by evaluation and benchmark scripts."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=-1)
    area1 = np.prod(np.maximum(boxes1[:, 2:] - boxes1[:, :2], 0), axis=-1)
    area2 = np.prod(np.maximum(boxes2[:, 2:] - boxes2[:, :2], 0), axis=-1)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], np.asarray(recall), [1.0]))
    precision = np.concatenate(([0.0], np.asarray(precision), [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1]))


def _class_stats(detections: list[np.ndarray], ground_truths: list[np.ndarray],
                 cls: int, iou_thresh: float) -> tuple[float, int, int, int, np.ndarray, np.ndarray]:
    gt_by_image = {}
    total_gt = 0
    for image_id, gt in enumerate(ground_truths):
        selected = gt[gt[:, 4].astype(int) == cls] if gt.size else np.empty((0, 5))
        gt_by_image[image_id] = selected
        total_gt += len(selected)
    records = []
    for image_id, det in enumerate(detections):
        selected = det[det[:, 5].astype(int) == cls] if det.size else np.empty((0, 6))
        records.extend((float(row[4]), image_id, row[:4]) for row in selected)
    records.sort(key=lambda item: item[0], reverse=True)
    matched = {idx: np.zeros(len(gt_by_image[idx]), dtype=bool) for idx in gt_by_image}
    tp, fp = [], []
    for _, image_id, box in records:
        gt = gt_by_image[image_id]
        if len(gt) == 0:
            tp.append(0); fp.append(1); continue
        ious = box_iou(box[None, :], gt[:, :4])[0]
        best = int(ious.argmax())
        if ious[best] >= iou_thresh and not matched[image_id][best]:
            matched[image_id][best] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp = np.asarray(tp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)
    if not len(tp):
        return 0.0, 0, 0, total_gt, np.zeros(0), np.zeros(0)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)
    return compute_ap(recall, precision), int(tp.sum()), int(fp.sum()), max(total_gt - int(tp.sum()), 0), recall, precision


def calculate_map(detections: list[np.ndarray], ground_truths: list[np.ndarray],
                  iou_thresh: float = 0.5, num_classes: int = 2) -> dict[str, float]:
    result: dict[str, float] = {}
    aps, total_tp, total_fp, total_fn = [], 0, 0, 0
    for cls in range(num_classes):
        ap, tp, fp, fn, _, _ = _class_stats(detections, ground_truths, cls, iou_thresh)
        result[f"ap_class_{cls}"] = ap
        aps.append(ap)
        total_tp += tp; total_fp += fp; total_fn += fn
    precision, recall, f1 = compute_prf1(total_tp, total_fp, total_fn)
    result.update({"map": float(np.mean(aps)) if aps else 0.0,
                   "precision": precision, "recall": recall, "f1": f1,
                   "tp": float(total_tp), "fp": float(total_fp), "fn": float(total_fn)})
    return result


def compute_prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


@torch.no_grad()
def measure_fps(model: torch.nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 608, 608),
                device: str = "cuda", warmup: int = 10, iterations: int = 50) -> float:
    device_obj = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device_obj).eval()
    x = torch.randn(input_size, device=device_obj)
    for _ in range(max(1, warmup)):
        model(x)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(max(1, iterations)):
        model(x)
    if device_obj.type == "cuda":
        torch.cuda.synchronize()
    return float(iterations / max(time.perf_counter() - start, 1e-9))


if __name__ == "__main__":
    detections = [np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32)]
    truths = [np.array([[0, 0, 10, 10, 0]], dtype=np.float32)]
    assert calculate_map(detections, truths)["map"] == 0.5
    print("metrics.py: PASS")
