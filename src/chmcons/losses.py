"""
Loss functions: binary focal, Dice, differentiable pixel accuracy, and their weighted combination (Eq. 3).

All losses are computed on non-background pixels (label != -1). `pred` is (B, 1, H, W) in [0, 1], `target` (B, H, W).
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

import torch
import torch.nn.functional as F


def _masked(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.squeeze(1)
    mask = target != -1
    return pred[mask], target[mask]


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    p, t = _masked(pred, target)
    if t.numel() == 0:
        return pred.sum() * 0.0
    bce = F.binary_cross_entropy(p, t, reduction="none")
    return (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    p, t = _masked(pred, target)
    if t.numel() == 0:
        return pred.sum() * 0.0
    return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)


def pixel_accuracy_loss(pred: torch.Tensor, target: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
    """1 - accuracy, with the 0.5 threshold replaced by a sharp sigmoid to keep it differentiable."""
    p, t = _masked(pred, target)
    if t.numel() == 0:
        return pred.sum() * 0.0
    soft = torch.sigmoid((p - 0.5) * sharpness)
    return 1 - (soft * t + (1 - soft) * (1 - t)).sum() / t.numel()


def combined_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.4) -> torch.Tensor:
    """alpha * focal + beta * pixel accuracy + (1 - alpha - beta) * dice  (Eq. 3 in the paper)."""
    return (
        alpha * focal_loss(pred, target)
        + beta * pixel_accuracy_loss(pred, target)
        + (1 - alpha - beta) * dice_loss(pred, target)
    )
