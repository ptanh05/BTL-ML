"""CIoU/SIoU with global anchor matching and ignored overlapping negatives."""

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
    """Best global anchor plus matching anchors, with YOLO-style ignore masks.

    The Sensors paper specifies the box loss but not target assignment/gains.
    The anchor and ignore thresholds below are explicit YOLOv4 conventions,
    not experimentally recovered hyperparameters of the authors.
    """

    def __init__(self, anchors: Iterable, num_classes: int = 2, img_size: int = 608,
                 loss_type: str = "ciou", box_gain: float = 5.0,
                 obj_gain: float = 1.0, cls_gain: float = 1.0,
                 siou_theta: float = 4.0, siou_eps: float = 1e-7,
                 anchor_iou_threshold: float = 0.213,
                 ignore_iou_threshold: float = 0.7) -> None:
        self.anchors = [torch.tensor(level, dtype=torch.float32) for level in anchors]
        self.num_classes = num_classes
        self.img_size = img_size
        self.loss_type = loss_type.lower()
        if self.loss_type not in {"ciou", "siou"}:
            raise ValueError(f"Unsupported box loss: {loss_type}")
        if siou_theta <= 0 or siou_eps <= 0:
            raise ValueError("SIoU theta and eps must be positive")
        self.siou_theta, self.siou_eps = siou_theta, siou_eps
        if not 0 <= anchor_iou_threshold <= 1 or not 0 <= ignore_iou_threshold <= 1:
            raise ValueError("Anchor/ignore IoU thresholds must be between 0 and 1")
        self.anchor_iou_threshold = anchor_iou_threshold
        self.ignore_iou_threshold = ignore_iou_threshold
        self.box_gain, self.obj_gain, self.cls_gain = box_gain, obj_gain, cls_gain
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def match_anchors(self, targets: torch.Tensor) -> torch.Tensor:
        """Return target-by-global-anchor assignments, independent of image scale."""
        anchors = torch.cat(self.anchors).to(targets.device) / self.img_size
        wh = targets[:, 4:6].float()
        inter = torch.minimum(wh[:, None], anchors[None]).prod(-1)
        overlaps = inter / (wh.prod(-1)[:, None] + anchors.prod(-1)[None] - inter).clamp_min(1e-9)
        matches = overlaps > self.anchor_iou_threshold
        if len(targets):
            matches[torch.arange(len(targets), device=targets.device), overlaps.argmax(1)] = True
        return matches

    @staticmethod
    def _decode_boxes(raw: torch.Tensor, anchors: torch.Tensor, height: int, width: int) -> torch.Tensor:
        gy, gx = torch.meshgrid(torch.arange(height, device=raw.device),
                                torch.arange(width, device=raw.device), indexing="ij")
        grid = torch.stack([gx, gy], -1).float()
        xy = (raw[..., :2].float().sigmoid() + grid) / raw.new_tensor([width, height]).float()
        wh = raw[..., 2:4].float().clamp(-10, 10).exp() * anchors.view(1, -1, 1, 1, 2)
        return torch.cat([xy, wh], -1)

    @torch.no_grad()
    def _objectness_mask(self, boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(boxes.shape[:-1], dtype=torch.bool, device=boxes.device)
        for batch in range(boxes.shape[0]):
            truths = targets[targets[:, 0] == batch, 2:6]
            if not len(truths):
                continue
            pred = boxes[batch].detach().reshape(-1, 4)
            flat_mask = mask[batch].view(-1)
            for start in range(0, len(pred), 2048):
                # Bounding-box IoU, not the anchor width/height IoU.
                overlaps = _iou(pred[start:start + 2048, None], truths[None])
                flat_mask[start:start + 2048] = overlaps.amax(1) < self.ignore_iou_threshold
        return mask

    def __call__(self, outputs: list[torch.Tensor], targets: torch.Tensor):
        device = outputs[0].device
        targets = targets.to(device=device, dtype=torch.float32)
        targets = targets[(targets[:, 4:6] > 0).all(1)]
        matches = self.match_anchors(targets)
        box_loss = torch.zeros((), device=device)
        obj_loss = torch.zeros((), device=device)
        cls_loss = torch.zeros((), device=device)
        positive_count = 0
        anchor_offset = 0
        for level, output in enumerate(outputs):
            bsz, _, height, width = output.shape
            n_anchor = self.anchors[level].shape[0]
            values = 5 + self.num_classes
            raw = output.view(bsz, n_anchor, values, height, width).permute(0, 1, 3, 4, 2)
            obj_target = torch.zeros((bsz, n_anchor, height, width), device=device)
            anchors = self.anchors[level].to(device) / self.img_size
            pred_boxes = self._decode_boxes(raw, anchors, height, width)
            obj_mask = self._objectness_mask(pred_boxes, targets)
            target_ids, anchor_ids = matches[:, anchor_offset:anchor_offset + n_anchor].nonzero(as_tuple=True)
            anchor_offset += n_anchor
            if len(target_ids):
                rows = targets[target_ids]
                batch = rows[:, 0].long()
                gx = (rows[:, 2] * width).long().clamp(0, width - 1)
                gy = (rows[:, 3] * height).long().clamp(0, height - 1)
                # Each slot must regress one box; resolve collisions by anchor IoU.
                wh = rows[:, 4:6]
                inter = torch.minimum(wh, anchors[anchor_ids]).prod(-1)
                scores = inter / (wh.prod(-1) + anchors[anchor_ids].prod(-1) - inter).clamp_min(1e-9)
                keys = ((batch * n_anchor + anchor_ids) * height + gy) * width + gx
                priority = scores.argsort(descending=True, stable=True)
                ordered = priority[keys[priority].argsort(stable=True)]
                ordered_keys = keys[ordered]
                unique = torch.cat([torch.ones(1, dtype=torch.bool, device=device),
                                    ordered_keys[1:] != ordered_keys[:-1]])
                selected = ordered[unique]
                batch, anchor_ids, gx, gy = [v[selected] for v in (batch, anchor_ids, gx, gy)]
                rows = rows[selected]
                pred = raw[batch, anchor_ids, gy, gx].float()
                pred_box = pred_boxes[batch, anchor_ids, gy, gx]
                target_box = rows[:, 2:6]
                obj_target[batch, anchor_ids, gy, gx] = 1.0
                obj_mask[batch, anchor_ids, gy, gx] = True
                if self.loss_type == "siou":
                    box_term = siou_loss(pred_box, target_box,
                                         theta=self.siou_theta, eps=self.siou_eps)
                else:
                    box_term = ciou_loss(pred_box, target_box)
                box_loss = box_loss + box_term.sum()
                target_cls = F.one_hot(rows[:, 1].long(), self.num_classes).float()
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(pred[:, 5:], target_cls, reduction="sum") / self.num_classes
                positive_count += len(rows)
            obj_terms = F.binary_cross_entropy_with_logits(raw[..., 4].float(), obj_target, reduction="none")
            obj_loss = obj_loss + (obj_terms * obj_mask).sum() / obj_mask.sum().clamp_min(1)
        box_loss = box_loss / max(positive_count, 1)
        cls_loss = cls_loss / max(positive_count, 1)
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
