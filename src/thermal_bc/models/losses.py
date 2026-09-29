"""Focal Tversky loss for imbalanced binary segmentation (Abraham & Khan, 2019)."""

import torch
import torch.nn as nn


class FocalTverskyLoss(nn.Module):
    """(1 - Tversky index) ** gamma on sigmoid probabilities.

    `alpha` weights false positives and `beta` false negatives; gamma > 1 focuses training on
    hard examples. Inputs are logits and binary targets of the same shape.
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-4, 1.0 - 1e-4)
        probs = probs.view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_index = torch.clamp(tversky_index, 1e-4, 1.0 - 1e-4)
        focal_tversky = torch.pow((1 - tversky_index), self.gamma)
        return focal_tversky.mean()
