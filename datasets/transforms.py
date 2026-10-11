"""Image/box transforms for the SHWD detection dataset."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
import torch


def letterbox(image: np.ndarray, size: int | tuple[int, int] = 608,
              color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[float, float]]:
    if isinstance(size, int):
        target_h, target_w = size, size
    else:
        target_h, target_w = size
    h, w = image.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (target_w - new_w) / 2, (target_h - new_h) / 2
    padded = cv2.copyMakeBorder(
        resized, int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)), cv2.BORDER_CONSTANT, value=color,
    )
    return padded, ratio, (dw, dh)


def xyxy_to_normalized_xywh(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    return np.stack(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                     (x2 - x1) / w, (y2 - y1) / h), axis=1).astype(np.float32)


def normalized_xywh_to_xyxy(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    cx, cy, bw, bh = boxes.T
    return np.stack(((cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h), axis=1).astype(np.float32)


class DetectionTransform:
    def __init__(self, image_size: int = 608, train: bool = False) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(self, image: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if self.train and np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            if boxes.size:
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        image, ratio, (dw, dh) = letterbox(image, self.image_size)
        if boxes.size:
            boxes = boxes.copy().astype(np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + dw
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + dh
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.image_size)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.image_size)
        boxes = xyxy_to_normalized_xywh(boxes, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return tensor, torch.from_numpy(boxes), torch.from_numpy(labels.astype(np.int64))


def build_transforms(image_size: int = 608, train: bool = False) -> DetectionTransform:
    return DetectionTransform(image_size=image_size, train=train)


if __name__ == "__main__":
    image = np.zeros((300, 500, 3), dtype=np.uint8)
    boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
    labels = np.array([0])
    tensor, transformed, out_labels = DetectionTransform(train=False)(image, boxes, labels)
    assert tensor.shape == (3, 608, 608) and transformed.shape == (1, 4)
    print("transforms.py: PASS")
