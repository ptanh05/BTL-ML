"""CIoU/SIoU losses and a small YOLO-style multi-scale training loss."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p, t = _xywh_to_xyxy(pred), _xywh_to_xyxy(target)
    lt = torch.maximum(p[..., :2], t[..., :2])
    rb = torch.minimum(p[..., 2:], t[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_p = (p[..., 2:] - p[..., :2]).clamp(min=0).prod(-1)
    area_t = (t[..., 2:] - t[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_p + area_t - inter + eps)


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU loss for ``[..., cx, cy, w, h]`` boxes."""
    iou = _iou(pred, target, eps)
    p, t = _xywh_to_xyxy(pred), _xywh_to_xyxy(target)
    center_dist = ((pred[..., :2] - target[..., :2]) ** 2).sum(-1)
    enclose_lt = torch.minimum(p[..., :2], t[..., :2])
    enclose_rb = torch.maximum(p[..., 2:], t[..., 2:])
    c2 = ((enclose_rb - enclose_lt) ** 2).sum(-1).clamp_min(eps)
    v = (4 / math.pi ** 2) * (
        torch.atan(target[..., 2] / target[..., 3].clamp_min(eps))
        - torch.atan(pred[..., 2] / pred[..., 3].clamp_min(eps))
    ) ** 2
    alpha = v / (1 - iou + v + eps)
    return 1 - iou + center_dist / c2 + alpha * v


def siou_loss(pred: torch.Tensor, target: torch.Tensor, theta: float = 4.0,
              eps: float = 1e-7) -> torch.Tensor:
    """SIoU loss with finite gradients when box centres coincide."""
    iou = _iou(pred, target, eps)
    px, py, pw, ph = pred.unbind(-1)
    tx, ty, tw, th = target.unbind(-1)
    dx, dy = tx - px, ty - py
    sigma = torch.sqrt(dx.square() + dy.square() + eps)
    sin_alpha = (dy.abs() / sigma).clamp(0.0, 0.9999)
    angle = 1 - 2 * torch.sin(torch.asin(sin_alpha) - math.pi / 4).square()
    gamma = 2 - angle
    p_xyxy, t_xyxy = _xywh_to_xyxy(pred), _xywh_to_xyxy(target)
    enc_w = (torch.maximum(p_xyxy[..., 2], t_xyxy[..., 2])
             - torch.minimum(p_xyxy[..., 0], t_xyxy[..., 0])).clamp_min(eps)
    enc_h = (torch.maximum(p_xyxy[..., 3], t_xyxy[..., 3])
             - torch.minimum(p_xyxy[..., 1], t_xyxy[..., 1])).clamp_min(eps)
    rho_x = (dx / enc_w).square()
    rho_y = (dy / enc_h).square()
    distance = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))
    omega_w = (pw - tw).abs() / torch.maximum(pw, tw).clamp_min(eps)
    omega_h = (ph - th).abs() / torch.maximum(ph, th).clamp_min(eps)
    shape = (1 - torch.exp(-omega_w)).pow(theta) + (1 - torch.exp(-omega_h)).pow(theta)
    return 1 - iou + 0.5 * (distance + shape)


class ComputeLoss:
    """Assign one anchor per target and train the three YOLO output scales."""

    def __init__(self, anchors: Iterable, num_classes: int = 2, img_size: int = 608,
                 loss_type: str = "ciou", box_gain: float = 5.0,
                 obj_gain: float = 1.0, cls_gain: float = 1.0) -> None:
        self.anchors = [torch.tensor(level, dtype=torch.float32) for level in anchors]
        self.num_classes = num_classes
        self.img_size = img_size
        self.loss_type = loss_type.lower()
        self.box_gain, self.obj_gain, self.cls_gain = box_gain, obj_gain, cls_gain
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, outputs: list[torch.Tensor], targets: torch.Tensor):
        device = outputs[0].device
        box_loss = outputs[0].new_zeros(())
        obj_loss = outputs[0].new_zeros(())
        cls_loss = outputs[0].new_zeros(())
        for level, output in enumerate(outputs):
            bsz, _, height, width = output.shape
            n_anchor = self.anchors[level].shape[0]
            values = 5 + self.num_classes
            raw = output.view(bsz, n_anchor, values, height, width).permute(0, 1, 3, 4, 2)
            obj_target = torch.zeros((bsz, n_anchor, height, width), device=device)
            anchors = self.anchors[level].to(device)
            positives = []
            if targets.numel():
                for row in targets:
                    batch, cls, cx, cy, bw, bh = row.tolist()
                    size = max(bw, bh) * self.img_size
                    expected_level = 0 if size < 64 else 1 if size < 192 else 2
                    if expected_level != level:
                        continue
                    ratios = torch.maximum(
                        torch.tensor([bw * self.img_size, bh * self.img_size], device=device) / anchors,
                        anchors / torch.tensor([bw * self.img_size, bh * self.img_size], device=device).clamp_min(1e-6),
                    ).amax(dim=1)
                    anchor_idx = int(ratios.argmin())
                    gx = min(width - 1, max(0, int(cx * width)))
                    gy = min(height - 1, max(0, int(cy * height)))
                    positives.append((int(batch), anchor_idx, gx, gy, int(cls), cx, cy, bw, bh))
                    obj_target[int(batch), anchor_idx, gy, gx] = 1.0
            obj_loss = obj_loss + self.bce(raw[..., 4], obj_target)
            for batch, anchor_idx, gx, gy, cls, cx, cy, bw, bh in positives:
                pred = raw[batch, anchor_idx, gy, gx]
                anchor = anchors[anchor_idx] / self.img_size
                pred_box = torch.stack([
                    (pred[0].sigmoid() + gx) / width,
                    (pred[1].sigmoid() + gy) / height,
                    pred[2].clamp(-10, 10).exp() * anchor[0],
                    pred[3].clamp(-10, 10).exp() * anchor[1],
                ])
                target_box = pred.new_tensor([cx, cy, bw, bh])
                loss_fn = siou_loss if self.loss_type == "siou" else ciou_loss
                box_loss = box_loss + loss_fn(pred_box.unsqueeze(0), target_box.unsqueeze(0)).mean()
                target_cls = torch.zeros(self.num_classes, device=device)
                target_cls[cls] = 1.0
                cls_loss = cls_loss + self.bce(pred[5:], target_cls)
        total = self.box_gain * box_loss + self.obj_gain * obj_loss + self.cls_gain * cls_loss
        parts = {"total": float(total.detach()), "box": float(box_loss.detach()),
                 "obj": float(obj_loss.detach()), "cls": float(cls_loss.detach())}
        return total, parts


if __name__ == "__main__":
    target = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    pred = target.clone().requires_grad_(True)
    loss = siou_loss(pred, target).mean()
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    print("loss.py: PASS")
