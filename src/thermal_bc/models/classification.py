"""Classification head on top of a frozen self-supervised ResNet-50 backbone."""

import random
import sys

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationHead(nn.Module):
    """ResNet-50 backbone (2048-d embedding) followed by a 2-layer MLP producing one logit."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.fc(self.backbone(x))


class EarlyStopping:
    """Stop when the validation loss has not improved by `delta` for `patience` epochs."""

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def shuffle_data(img_paths, labels):
    data = list(zip(img_paths, labels))
    random.shuffle(data)
    img_paths_shuffled, labels_shuffled = zip(*data)
    return list(img_paths_shuffled), list(labels_shuffled)


def load_vicreg_resnet50(checkpoint_path, vicreg_dir="vicreg"):
    """Load a ResNet-50 pretrained with VICReg.

    `vicreg_dir` must contain `resnet.py` from https://github.com/facebookresearch/vicreg,
    and `checkpoint_path` the backbone weights saved by its `main_vicreg.py`.
    """
    sys.path.insert(0, str(vicreg_dir))
    import resnet  # noqa: E402  (provided by the official VICReg repository)

    backbone, _ = resnet.__dict__["resnet50"](zero_init_residual=True)
    backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return backbone


def _binary_scores(labels, preds):
    return {
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "auc": roc_auc_score(labels, preds),
    }


def fit_classifier(
    model, train_loader, val_loader, criterion, optimizer, early_stopping, num_epochs, device
):
    """Train `model` and log validation metrics to W&B each epoch until early stopping."""
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1).float()
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                running_loss += loss.item()
                all_preds.extend((outputs > 0.0).float().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = running_loss / len(val_loader)
        scores = _binary_scores(all_labels, all_preds)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}, F1: {scores['f1']:.4f}, "
            f"Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, "
            f"AUC: {scores['auc']:.4f}"
        )
        wandb.log(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **{f"val_{k}": v for k, v in scores.items()},
                "epoch": epoch + 1,
            }
        )

        if early_stopping(avg_val_loss):
            print("Early stopping triggered. Stopping training.")
            break

    return model


def evaluate_classifier(model, test_loader, device):
    """Return F1, precision, recall and AUC of thresholded predictions on `test_loader`."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend((outputs.view(-1) > 0.0).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    scores = _binary_scores(all_labels, all_preds)
    print(
        f"Test F1: {scores['f1']:.4f}, Precision: {scores['precision']:.4f}, "
        f"Recall: {scores['recall']:.4f}, AUC: {scores['auc']:.4f}"
    )
    try:
        print(classification_report(all_labels, all_preds, target_names=["healthy", "cancer"]))
    except ValueError as e:
        print(f"Classification report unavailable (a class is missing from the test set): {e}")
    return scores
