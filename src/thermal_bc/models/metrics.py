"""Binary segmentation metrics (pixel-wise, computed over a whole batch or split)."""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)


def compute_dice(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def extract_boundary(mask, kernel_size=3):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    padding = kernel_size // 2

    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    eroded = F.conv2d(mask.float(), kernel, padding=padding) == kernel.numel()

    mask_bool = mask.bool()
    eroded_bool = eroded.bool()

    boundary = mask_bool & (~eroded_bool)
    return boundary.float()


def compute_boundary_iou(preds, targets, threshold=0.5, kernel_size=7, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = targets.float()

    preds_b = extract_boundary(preds, kernel_size)
    targets_b = extract_boundary(targets, kernel_size)

    intersection = (preds_b * targets_b).sum(dim=(1, 2, 3))
    union = ((preds_b + targets_b) > 0).float().sum(dim=(1, 2, 3))

    boundary_iou = (intersection + eps) / (union + eps)

    return boundary_iou.mean().item()


def get_metrics(preds, targets):
    iou = BinaryJaccardIndex(threshold=0.5)
    precision = BinaryPrecision(threshold=0.5)
    recall = BinaryRecall(threshold=0.5)
    f1 = BinaryF1Score(threshold=0.5)
    auc = BinaryAUROC()
    specificity = BinarySpecificity(threshold=0.5)
    probs = torch.sigmoid(preds)

    dice = compute_dice(probs, targets)

    preds_flat = probs.view(-1)
    targets_flat = targets.view(-1).float()

    return {
        "dice": dice,
        "iou": iou(probs, targets).item(),
        "precision": precision(probs, targets).item(),
        "recall (sensitivity)": recall(probs, targets).item(),
        "f1": f1(probs, targets).item(),
        "auc": auc(preds_flat, targets_flat.int()).item(),
        "boundary_iou": compute_boundary_iou(preds, targets),
        "specificity": specificity(probs, targets).item(),
    }


def confusion_metrics(preds, targets):
    """Pixel-wise metrics from binary prediction and ground-truth tensors of equal shape.

    Unlike `get_metrics`, `preds` are already thresholded, so AUC is computed on hard labels.
    """
    y_pred = preds.reshape(-1).cpu().numpy()
    y_true = targets.reshape(-1).cpu().numpy()

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")

    return {
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "IoU": tp / (tp + fp + fn + eps),
        "Precision": precision,
        "Recall": recall,
        "F1": 2 * precision * recall / (precision + recall + eps),
        "Specificity": tn / (tn + fp + eps),
        "AUC": auc,
    }


def breast_mask_from_multiclass(logits):
    """Binary breast mask (N, 1, H, W) from 3-class logits: pixels not predicted as background."""
    return (logits.argmax(dim=1, keepdim=True) > 0).float()
