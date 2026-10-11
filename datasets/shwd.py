"""Pascal-VOC/YOLO loader for the Safety Helmet Wearing Dataset."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms

CLASS_NAMES = ("hat", "person")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _read_split(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class SHWDDataset(Dataset):
    def __init__(self, data_root: str | Path = "dataset", split_file: str | Path | None = None,
                 image_size: int = 608, train: bool = False,
                 transform: Callable | None = None, limit: int | None = None) -> None:
        self.data_root = Path(data_root).resolve()
        self.project_root = self.data_root.parent
        self.annotation_root = self.data_root / "annotations"
        self.transform = transform or build_transforms(image_size, train=train)
        self.train = train
        if split_file is None:
            split_file = self.project_root / "datasets" / "splits" / ("train.txt" if train else "val.txt")
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = self.project_root / split_path
        self.items = self._resolve_items(_read_split(split_path))
        if limit is not None:
            self.items = self.items[:limit]
        if not self.items:
            raise RuntimeError(f"No images found for split {split_path} under {self.data_root}")

    def _resolve_items(self, entries: list[str]) -> list[Path]:
        result: list[Path] = []
        for entry in entries:
            raw = Path(entry.replace("\\", os.sep))
            candidates = []
            if raw.is_absolute():
                candidates.append(raw)
            candidates.extend([
                self.project_root / raw,
                self.data_root / raw,
                self.data_root / "train" / raw.name,
                self.data_root / "val" / raw.name,
                self.data_root / "test" / raw.name,
            ])
            found = next((p for p in candidates if p.is_file()), None)
            if found is not None:
                result.append(found)
        return result

    def _annotation_path(self, image_path: Path) -> Path | None:
        candidates = [
            self.annotation_root / f"{image_path.stem}.xml",
            image_path.with_suffix(".xml"),
            image_path.with_suffix(".txt"),
            self.data_root / "labels" / f"{image_path.stem}.txt",
        ]
        return next((p for p in candidates if p.is_file()), None)

    @staticmethod
    def _parse_voc(path: Path) -> tuple[np.ndarray, np.ndarray]:
        root = ET.parse(path).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name not in CLASS_TO_ID:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            values = [float(box.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax")]
            if values[2] > values[0] and values[3] > values[1]:
                boxes.append(values)
                labels.append(CLASS_TO_ID[name])
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)

    @staticmethod
    def _parse_yolo(path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_shape
        boxes, labels = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) < 5:
                continue
            cls, cx, cy, bw, bh = map(float, values[:5])
            if int(cls) not in range(len(CLASS_NAMES)):
                continue
            if max(cx, cy, bw, bh) <= 1.5:
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            labels.append(int(cls))
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path = self.items[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Unable to read image: {path}")
        annotation = self._annotation_path(path)
        if annotation is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
        elif annotation.suffix.lower() == ".xml":
            boxes, labels = self._parse_voc(annotation)
        else:
            boxes, labels = self._parse_yolo(annotation, image.shape[:2])
        image_tensor, boxes_tensor, labels_tensor = self.transform(image, boxes, labels)
        targets = torch.cat([labels_tensor.float().unsqueeze(1), boxes_tensor], dim=1)
        return image_tensor, targets, str(path)


def detection_collate(batch):
    images, targets, paths = zip(*batch)
    images = torch.stack(images)
    merged = []
    for batch_index, target in enumerate(targets):
        if target.numel():
            batch_column = torch.full((target.shape[0], 1), batch_index, dtype=target.dtype)
            merged.append(torch.cat([batch_column, target], dim=1))
    merged_targets = torch.cat(merged, dim=0) if merged else torch.zeros((0, 6), dtype=torch.float32)
    return images, merged_targets, list(paths)


def build_dataloader(data_root: str | Path, split_file: str | Path, image_size: int,
                     batch_size: int, train: bool, workers: int = 0,
                     limit: int | None = None) -> DataLoader:
    dataset = SHWDDataset(data_root, split_file, image_size=image_size, train=train, limit=limit)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=workers,
        pin_memory=torch.cuda.is_available(), collate_fn=detection_collate,
        persistent_workers=workers > 0,
    )


if __name__ == "__main__":
    root = Path("dataset")
    split = Path("datasets/splits/train.txt")
    if root.exists() and split.exists():
        dataset = SHWDDataset(root, split, image_size=608, limit=1)
        image, target, path = dataset[0]
        print(f"shwd.py: PASS {path} {tuple(image.shape)} {tuple(target.shape)}")
