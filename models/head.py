"""YOLO detection head, decoding and class-aware NMS."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .common import conv3x3
except ImportError:  # direct execution: python models/head.py
    from common import conv3x3


class YOLOHead(nn.Module):
    def __init__(self, in_channels: Iterable[int], num_classes: int = 2,
                 num_anchors: int = 3, dsc: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)
        self.heads = nn.ModuleList([
            nn.Sequential(conv3x3(c, 2 * c, dsc=dsc), nn.Conv2d(2 * c, out_channels, 1))
            for c in in_channels
        ])

    def forward(self, *features: torch.Tensor) -> list[torch.Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def decode_predictions(preds: list[torch.Tensor], anchors: list, img_size: int = 608,
                       conf_thres: float = 0.25) -> list[torch.Tensor]:
    """Decode raw feature maps into one ``(N,6)`` tensor per image."""
    decoded: list[torch.Tensor] = []
    batch_size = preds[0].shape[0]
    per_image = [[] for _ in range(batch_size)]
    for level, pred in enumerate(preds):
        b, _, h, w = pred.shape
        n_anchors = len(anchors[level])
        n_values = pred.shape[1] // n_anchors
        stride_y, stride_x = img_size / h, img_size / w
        raw = pred.view(b, n_anchors, n_values, h, w).permute(0, 1, 3, 4, 2)
        gy, gx = torch.meshgrid(
            torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij"
        )
        grid = torch.stack((gx, gy), dim=-1).view(1, 1, h, w, 2).float()
        anchor = torch.as_tensor(anchors[level], device=pred.device, dtype=pred.dtype).view(1, n_anchors, 1, 1, 2)
        xy = (raw[..., :2].sigmoid() + grid) * raw.new_tensor([stride_x, stride_y])
        wh = raw[..., 2:4].clamp(-10, 10).exp() * anchor
        obj = raw[..., 4:5].sigmoid()
        cls_prob = raw[..., 5:].sigmoid()
        scores, labels = cls_prob.max(dim=-1, keepdim=True)
        scores = scores * obj
        boxes = _xywh_to_xyxy(torch.cat([xy, wh], dim=-1))
        rows = torch.cat([boxes, scores, labels.float()], dim=-1).view(b, -1, 6)
        for i in range(batch_size):
            per_image[i].append(rows[i][rows[i, :, 4] >= conf_thres])
    for rows in per_image:
        decoded.append(torch.cat(rows, dim=0) if rows else preds[0].new_zeros((0, 6)))
    return decoded


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(dim=-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(dim=-1)
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-7)


def nms(boxes: torch.Tensor, iou_thres: float = 0.45, max_det: int = 300) -> torch.Tensor:
    """Class-aware NMS for ``[x1,y1,x2,y2,score,class_id]`` rows."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 6)
    keep: list[torch.Tensor] = []
    for cls in boxes[:, 5].unique():
        subset = boxes[boxes[:, 5] == cls]
        order = subset[:, 4].argsort(descending=True)
        while order.numel():
            idx = order[0]
            keep.append(subset[idx])
            if order.numel() == 1:
                break
            overlaps = box_iou(subset[idx:idx + 1, :4], subset[order[1:], :4]).squeeze(0)
            order = order[1:][overlaps <= iou_thres]
    result = torch.stack(keep) if keep else boxes.new_zeros((0, 6))
    return result[result[:, 4].argsort(descending=True)[:max_det]]


if __name__ == "__main__":
    head = YOLOHead((128, 128, 128))
    outputs = head(torch.randn(1, 128, 76, 76), torch.randn(1, 128, 38, 38), torch.randn(1, 128, 19, 19))
    assert [tuple(o.shape) for o in outputs] == [(1, 21, 76, 76), (1, 21, 38, 38), (1, 21, 19, 19)]
    print("head.py: PASS")
