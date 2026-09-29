import numpy as np
import pytest

from thermal_bc.data.dmrir import patient_id_from_path
from thermal_bc.evaluation import balanced_subsample, patient_folds, patient_level_scores


@pytest.mark.parametrize(
    "name, expected",
    [
        ("198-1.txt.jpg", "198"),
        ("R198-10.txt.jpg", "198"),
        ("TFRON_V185_27-8-2013_0.jpg", "185"),
        ("RTFRON_V185_27-8-2013_12.txt.jpg", "185"),
        ("IR_2015-04-17_0007.jpg", "2015-04-17"),
    ],
)
def test_patient_id_parsing(name, expected):
    assert patient_id_from_path(f"data/abnormal/{name}") == expected


def test_unknown_naming_scheme_raises():
    with pytest.raises(ValueError):
        patient_id_from_path("image_001.png")


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i}" for i in range(30)], 20)
    labels = np.repeat(rng.integers(0, 2, 30), 20)
    return labels, groups


def test_patient_folds_are_disjoint_and_cover_everyone(cohort):
    labels, groups = cohort
    seen = []
    for train_idx, test_idx in patient_folds(labels, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])
        seen += list(test_idx)
    assert sorted(seen) == list(range(len(labels)))


def test_balanced_subsample(cohort):
    labels, groups = cohort
    train_idx, _ = patient_folds(labels, groups)[0]
    idx = balanced_subsample(train_idx, labels, 16, np.random.default_rng(0))
    assert len(idx) == 16 and labels[idx].sum() == 8 and set(idx) <= set(train_idx)


def test_patient_level_scores_average_frames():
    probs = np.array([0.9, 0.7, 0.2, 0.4, 0.6, 0.8])
    labels = np.array([1, 1, 0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    scores, per_patient = patient_level_scores(probs, labels, groups)
    assert per_patient.loc["a", "prob"] == pytest.approx(0.8)
    assert scores["auc"] == pytest.approx(1.0)
