"""Patient-level evaluation protocol for classifiers trained on frozen features.

* 5-fold stratified group k-fold: every patient's frames sit in exactly one test fold.
* For each fold and label budget n, n class-balanced images are sampled from the training
  fold (5 repeats with different samples); ``n=None`` uses the whole training fold.
* Heads are trained without any validation labels (fixed hyperparameters), so a budget of
  n images means exactly n labels.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def patient_folds(labels, groups, n_splits=5, seed=42):
    folds = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    for train_idx, test_idx in folds:
        assert not set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    return folds


def balanced_subsample(train_idx, labels, n, rng):
    """n indices from `train_idx`, half per class."""
    labels = np.asarray(labels)
    return np.concatenate(
        [rng.choice(train_idx[labels[train_idx] == c], n // 2, replace=False) for c in (0, 1)]
    )


def binary_scores(y_true, probs, threshold=0.5):
    preds = (np.asarray(probs) > threshold).astype(int)
    sensitivity = recall_score(y_true, preds, zero_division=0)
    specificity = recall_score(y_true, preds, pos_label=0, zero_division=0)
    return {
        "auc": roc_auc_score(y_true, probs),
        "f1": f1_score(y_true, preds, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


class LogisticHead:
    """Standardised features + L2-regularised logistic regression (scikit-learn defaults)."""

    def __init__(self, seed=0):
        self.seed = seed  # deterministic solver; kept for a uniform head interface

    def fit(self, x, y):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        self.model.fit(x, y)
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]


class MLPHead:
    """The project's classification head (2 hidden layers) trained on frozen features.

    Fixed recipe, no validation set: standardised inputs, AdamW (lr 1e-3, weight decay 1e-2),
    full-batch updates for a fixed number of epochs.
    """

    def __init__(self, epochs=300, lr=1e-3, weight_decay=1e-2, dropout=0.3, seed=0, device="cpu"):
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.dropout, self.seed, self.device = dropout, seed, device

    def _network(self, dim):
        return nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

    def fit(self, x, y):
        torch.manual_seed(self.seed)
        self.scaler = StandardScaler().fit(x)
        xt = torch.tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)[:, None]
        self.net = self._network(x.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        self.net.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss_fn(self.net(xt), yt).backward()
            optimizer.step()
        return self

    @torch.no_grad()
    def predict_proba(self, x):
        self.net.eval()
        xt = torch.tensor(self.scaler.transform(x), dtype=torch.float32, device=self.device)
        return torch.sigmoid(self.net(xt))[:, 0].cpu().numpy()


def run_protocol(
    features, labels, groups, make_head, sizes=(16, 32, 64, 128, 256, None), repeats=5, folds=None
):
    """Scores for every (fold, label budget, repeat); `features` may be a callable fold -> array."""
    labels = np.asarray(labels)
    folds = folds or patient_folds(labels, groups)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds):
        x = features(fold) if callable(features) else features
        for n in sizes:
            for repeat in range(repeats if n else 1):
                rng = np.random.default_rng(1000 * fold + repeat)
                idx = balanced_subsample(train_idx, labels, n, rng) if n else train_idx
                head = make_head(seed=repeat).fit(x[idx], labels[idx])
                scores = binary_scores(labels[test_idx], head.predict_proba(x[test_idx]))
                rows.append({"fold": fold, "n_labeled": n or "all", "repeat": repeat, **scores})
    return pd.DataFrame(rows)


def summarize(df, metrics=("auc", "f1", "balanced_accuracy"), by=("n_labeled",)):
    """Mean ± std across folds (repeats are averaged within each fold first)."""
    per_fold = df.groupby([*by, "fold"], sort=False)[list(metrics)].mean()
    return per_fold.groupby(list(by), sort=False).agg(["mean", "std"])


def out_of_fold_probabilities(features, labels, groups, make_head, folds=None):
    """Probability for every image from the model of the fold in which its patient is tested."""
    labels = np.asarray(labels)
    folds = folds or patient_folds(labels, groups)
    probs = np.full(len(labels), np.nan)
    for fold, (train_idx, test_idx) in enumerate(folds):
        x = features(fold) if callable(features) else features
        probs[test_idx] = (
            make_head().fit(x[train_idx], labels[train_idx]).predict_proba(x[test_idx])
        )
    return probs


def patient_level_scores(probs, labels, groups):
    """Average frame probabilities per patient, then score the patient-level predictions."""
    df = pd.DataFrame({"prob": probs, "label": labels, "patient": groups})
    per_patient = df.groupby("patient").agg(prob=("prob", "mean"), label=("label", "first"))
    return binary_scores(per_patient.label.values, per_patient.prob.values), per_patient
