"""
=============================================================
 OliveVision — Loss Functions
 Hungarian matching + GIoU + Focal-CE + IoU-aware loss
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────
# Box utilities
# ─────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2]."""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU) between two sets of boxes.
    Both inputs: (N, 4) in [x1, y1, x2, y2] format.
    Returns: (N, M) GIoU matrix.
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]

    b1 = boxes1.unsqueeze(1).expand(N, M, 4)
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter + 1e-6

    iou = inter / union

    # Enclosing box
    enc_x1 = torch.min(b1[..., 0], b2[..., 0])
    enc_y1 = torch.min(b1[..., 1], b2[..., 1])
    enc_x2 = torch.max(b1[..., 2], b2[..., 2])
    enc_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0) + 1e-6

    giou = iou - (enc_area - union) / enc_area
    return giou


# ─────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance (background >> olives).
    α modulates easy/hard example weighting.
    γ focuses learning on hard-to-classify examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (N,) logits
            target: (N,) binary {0, 1}
        """
        bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        p_t  = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────
# Hungarian Matcher
# ─────────────────────────────────────────────────────────────

class HungarianMatcher(nn.Module):
    """
    Optimal matching between predicted queries and ground-truth targets
    using the Hungarian algorithm.

    Cost = λ_cls * Cost_cls + λ_bbox * Cost_L1 + λ_giou * Cost_GIoU
    """

    def __init__(
        self,
        cls_weight:  float = 2.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
    ):
        super().__init__()
        self.cls_weight  = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Args:
            outputs: {
              'cls_logits': (B, Q, num_classes),
              'boxes':      (B, Q, 4),
            }
            targets: list of B dicts, each with:
              { 'labels': (N,), 'boxes': (N, 4) }

        Returns:
            list of (pred_idx, tgt_idx) tuples per batch item
        """
        B, Q, _ = outputs["cls_logits"].shape
        cls_logits = outputs["cls_logits"].flatten(0, 1).sigmoid()  # (B*Q, C)
        pred_boxes = outputs["boxes"].flatten(0, 1)                  # (B*Q, 4)

        indices = []
        for b in range(B):
            tgt = targets[b]
            if len(tgt["labels"]) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            tgt_labels = tgt["labels"].long()
            tgt_boxes  = tgt["boxes"].float()
            N = len(tgt_labels)

            # Classification cost
            cls_cost = -cls_logits[b * Q:(b + 1) * Q][:, tgt_labels]  # (Q, N)

            # L1 bbox cost
            pb = pred_boxes[b * Q:(b + 1) * Q]          # (Q, 4)
            l1_cost = torch.cdist(pb, tgt_boxes, p=1)   # (Q, N)

            # GIoU cost
            pb_xyxy  = box_cxcywh_to_xyxy(pb)
            tb_xyxy  = box_cxcywh_to_xyxy(tgt_boxes)
            giou_cost = -generalized_box_iou(pb_xyxy, tb_xyxy)  # (Q, N)

            cost = (
                self.cls_weight  * cls_cost +
                self.bbox_weight * l1_cost +
                self.giou_weight * giou_cost
            )

            pred_i, tgt_i = linear_sum_assignment(cost.cpu().detach().numpy())
            indices.append((
                torch.tensor(pred_i, dtype=torch.long),
                torch.tensor(tgt_i,  dtype=torch.long),
            ))

        return indices


# ─────────────────────────────────────────────────────────────
# Detection Loss (Set Prediction)
# ─────────────────────────────────────────────────────────────

class OliveDetectionLoss(nn.Module):
    """
    End-to-end detection loss combining:
      • Focal classification loss on matched pairs
      • L1 regression loss on matched boxes
      • GIoU loss on matched boxes
      • IoU-awareness auxiliary loss

    Computed with Hungarian matching (no NMS required).
    """

    def __init__(
        self,
        num_classes:  int   = 1,
        cls_weight:   float = 2.0,
        bbox_weight:  float = 5.0,
        giou_weight:  float = 2.0,
        iou_weight:   float = 1.0,
        focal_alpha:  float = 0.25,
        focal_gamma:  float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight  = cls_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.iou_weight  = iou_weight

        self.matcher    = HungarianMatcher(cls_weight, bbox_weight, giou_weight)
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma, reduction="sum")

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Args:
            outputs: dict from model with cls_logits, boxes, iou_pred
            targets: list of B target dicts

        Returns:
            dict with individual and total losses
        """
        indices = self.matcher(outputs, targets)

        B, Q, _ = outputs["cls_logits"].shape
        device   = outputs["cls_logits"].device

        # ── Classification Loss (Focal) ──────────────────────
        cls_logits = outputs["cls_logits"]   # (B, Q, C)
        tgt_cls = torch.zeros(B, Q, self.num_classes, device=device)

        num_matched = 0
        for b, (pred_i, tgt_i) in enumerate(indices):
            if len(pred_i) == 0:
                continue
            tgt_cls[b, pred_i, targets[b]["labels"][tgt_i].long()] = 1.0
            num_matched += len(pred_i)

        num_matched = max(num_matched, 1)
        loss_cls = self.focal_loss(
            cls_logits.flatten(0, 2),
            tgt_cls.flatten(0, 2)
        ) / num_matched

        # ── Regression Losses (L1 + GIoU) ───────────────────
        loss_l1   = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        loss_iou  = torch.tensor(0.0, device=device)

        pred_boxes = outputs["boxes"]
        iou_pred   = outputs.get("iou_pred")

        for b, (pred_i, tgt_i) in enumerate(indices):
            if len(pred_i) == 0:
                continue
            pb = pred_boxes[b][pred_i]                     # (n, 4)
            tb = targets[b]["boxes"][tgt_i].float()        # (n, 4)

            loss_l1 = loss_l1 + F.l1_loss(pb, tb, reduction="sum") / num_matched

            pb_xyxy = box_cxcywh_to_xyxy(pb)
            tb_xyxy = box_cxcywh_to_xyxy(tb)
            giou    = torch.diag(generalized_box_iou(pb_xyxy, tb_xyxy))
            loss_giou = loss_giou + (1 - giou).sum() / num_matched

            # IoU-awareness: predicted IoU should match actual IoU
            if iou_pred is not None:
                actual_iou = torch.diag(generalized_box_iou(pb_xyxy, tb_xyxy)).detach().clamp(0, 1)
                pred_iou   = iou_pred[b][pred_i].squeeze(-1)
                loss_iou   = loss_iou + F.binary_cross_entropy(
                    pred_iou.sigmoid(), actual_iou, reduction="sum"
                ) / num_matched

        total_loss = (
            self.cls_weight  * loss_cls  +
            self.bbox_weight * loss_l1   +
            self.giou_weight * loss_giou +
            self.iou_weight  * loss_iou
        )

        return {
            "loss_cls":   loss_cls,
            "loss_l1":    loss_l1,
            "loss_giou":  loss_giou,
            "loss_iou":   loss_iou,
            "total_loss": total_loss,
        }
