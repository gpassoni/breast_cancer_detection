import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
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
