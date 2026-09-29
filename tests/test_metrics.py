import numpy as np
import pytest
import torch
from sklearn import metrics as skm

from thermal_bc.evaluation import binary_scores
from thermal_bc.models.metrics import breast_mask_from_multiclass, confusion_metrics, get_metrics


@pytest.fixture
def masks():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 1, 24, 24, generator=g)
    target = (torch.rand(4, 1, 24, 24, generator=g) > 0.6).float()
    return logits, target


def test_get_metrics_matches_sklearn(masks):
    logits, target = masks
    ours = get_metrics(logits, target)
    y = target.flatten().numpy()
    pred = (logits.sigmoid() > 0.5).flatten().numpy()
    assert ours["iou"] == pytest.approx(skm.jaccard_score(y, pred), abs=1e-5)
    assert ours["precision"] == pytest.approx(skm.precision_score(y, pred), abs=1e-5)
    assert ours["recall (sensitivity)"] == pytest.approx(skm.recall_score(y, pred), abs=1e-5)
    assert ours["specificity"] == pytest.approx(skm.recall_score(y, pred, pos_label=0), abs=1e-5)
    assert ours["f1"] == pytest.approx(skm.f1_score(y, pred), abs=1e-5)
    probs = logits.sigmoid().flatten().numpy()
    assert ours["auc"] == pytest.approx(skm.roc_auc_score(y, probs), abs=1e-5)


def test_perfect_prediction_scores_one(masks):
    _, target = masks
    ours = get_metrics(target * 20 - 10, target)
    assert ours["dice"] == pytest.approx(1.0)
    assert ours["boundary_iou"] == pytest.approx(1.0)


def test_confusion_metrics_matches_sklearn(masks):
    logits, target = masks
    pred = (logits > 0).float()
    ours = confusion_metrics(pred, target)
    y, p = target.flatten().numpy(), pred.flatten().numpy()
    assert ours["F1"] == pytest.approx(skm.f1_score(y, p), abs=1e-6)
    assert ours["Dice"] == pytest.approx(ours["F1"], abs=1e-6)  # identical for binary masks
    assert ours["IoU"] == pytest.approx(skm.jaccard_score(y, p), abs=1e-6)


def test_multiclass_decoding_uses_argmax():
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 0, 0, 0] = 5.0  # background wins
    logits[0, 2, 1, 1] = 5.0  # right breast wins
    logits[0, 0, 0, 1], logits[0, 1, 0, 1] = 1.0, 2.0  # breast wins although its bg logit > 0
    mask = breast_mask_from_multiclass(logits)[0, 0]
    assert mask.tolist() == [[0.0, 1.0], [0.0, 1.0]]


def test_binary_scores():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.6, 0.4, 0.9])
    scores = binary_scores(y, probs)
    assert scores["auc"] == pytest.approx(0.75)
    assert scores["sensitivity"] == pytest.approx(0.5)
    assert scores["specificity"] == pytest.approx(0.5)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
