import torch
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)

def get_multiclass_metrics(preds, targets, num_classes=3):
    """
    preds: logits (B, C, H, W)
    targets: label map (B, H, W)
    """
    if preds.dim() != 4:
        raise ValueError(f"Expected preds to have shape [B, C, H, W], got {preds.shape}")
    if targets.dim() == 4 and targets.size(1) == 1:
        targets = targets.squeeze(1)  

    preds_class = torch.argmax(preds, dim=1)

    metrics = {}

    f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
    metrics["f1"] = f1_macro(preds_class, targets).item()

    metrics["iou_macro"] = MulticlassJaccardIndex(num_classes=num_classes, average="macro")(preds_class, targets).item()
    metrics["precision_macro"] = MulticlassPrecision(num_classes=num_classes, average="macro")(preds_class, targets).item()
    metrics["recall_macro"] = MulticlassRecall(num_classes=num_classes, average="macro")(preds_class, targets).item()

    f1_per_class = MulticlassF1Score(num_classes=num_classes, average="none")(preds_class, targets)
    recall_per_class = MulticlassRecall(num_classes=num_classes, average="none")(preds_class, targets)
    precision_per_class = MulticlassPrecision(num_classes=num_classes, average="none")(preds_class, targets)

    for i in range(num_classes):
        metrics[f"f1_class_{i}"] = f1_per_class[i].item()
        metrics[f"recall_class_{i}"] = recall_per_class[i].item()
        metrics[f"precision_class_{i}"] = precision_per_class[i].item()

    return metrics
