"""
Evaluation metrics (accuracy, precision, recall, IoU, Dice) from pixel confusion counts
accumulated over non-background pixels.
"""

__author__ = "Marco L. Della Vedova"
__email__ = "marco.dellavedova@chalmers.se"

from dataclasses import dataclass

import torch


@dataclass
class Confusion:
    """Pixel confusion counts accumulated over non-background pixels, threshold 0.5."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        pred = pred.squeeze(1)
        mask = target != -1
        p = pred[mask] > threshold
        t = target[mask] == 1
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.tn += int((~p & ~t).sum())
        self.fn += int((~p & t).sum())

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def summary(self) -> dict[str, float]:
        """Metrics; 0.0 when undefined (e.g. precision when no pixel is predicted positive)."""
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        ratio = lambda a, b: a / b if b else 0.0
        return {
            "accuracy": ratio(tp + tn, self.total),
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "iou": ratio(tp, tp + fp + fn),
            "dice": ratio(2 * tp, 2 * tp + fp + fn),
        }
