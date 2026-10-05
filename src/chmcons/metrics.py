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
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        return {
            "accuracy": (tp + tn) / self.total,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
            "iou": tp / (tp + fp + fn),
            "dice": 2 * tp / (2 * tp + fp + fn),
        }
