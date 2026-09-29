"""Label-efficiency benchmark of frozen encoders on DMR-IR under patient-level evaluation.

Extracts (and caches) features for each encoder, then trains a classification head on
n = 16 ... all labeled images per fold of a 5-fold patient-grouped cross-validation.

Usage:
    python scripts/classification/benchmark.py \
        --encoder vicreg=weights/vicreg_resnet50.pth --encoder imagenet=imagenet \
        --input masked --layers layer3+4 --head logreg
    # One backbone per fold (e.g. adapted on that fold's training patients):
    python scripts/classification/benchmark.py \\
        --encoder adapted=weights/vicreg_adapted/fold{fold}.pth
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from thermal_bc.config import root_dir
from thermal_bc.data.dmrir import list_dmrir_images, patient_id_from_path
from thermal_bc.evaluation import (
    LogisticHead,
    MLPHead,
    out_of_fold_probabilities,
    patient_folds,
    patient_level_scores,
    run_protocol,
    summarize,
)
from thermal_bc.features import INPUTS, LAYERS, extract_features, predict_breast_masks

CACHE = Path("runs/cache")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir", default=root_dir() / "data" / "dmrir" / "lab_database" / "database"
    )
    parser.add_argument(
        "--encoder",
        action="append",
        required=True,
        metavar="NAME=SPEC",
        help="SPEC is 'imagenet', 'random' or a VICReg checkpoint (may contain {fold}).",
    )
    parser.add_argument("--input", choices=INPUTS, default="masked")
    parser.add_argument("--layers", choices=LAYERS, default="layer3+4")
    parser.add_argument("--head", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--seg-weights", default="weights/segmentation_r2attunet.pth")
    parser.add_argument("--seg-config", default="weights/segmentation_r2attunet.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = list_dmrir_images(args.data_dir)
    labels = np.array(labels)
    groups = np.array([patient_id_from_path(p) for p in paths])
    folds = patient_folds(labels, groups)
    masks = None
    if args.input != "full":
        masks = predict_breast_masks(
            paths, args.seg_weights, args.seg_config, device, CACHE / "breast_masks.npz"
        )

    def features_for(spec, fold=None):
        spec = spec.format(fold=fold) if fold is not None else spec
        return extract_features(paths, spec, args.input, args.layers, device, masks, CACHE)

    if args.head == "logreg":
        make_head = LogisticHead
    else:
        make_head = partial(MLPHead, device=device)

    results, patient_rows = [], []
    for item in args.encoder:
        name, spec = item.split("=", 1)
        features = partial(features_for, spec) if "{fold}" in spec else features_for(spec)
        df = run_protocol(features, labels, groups, make_head, folds=folds)
        df.insert(0, "encoder", name)
        results.append(df)
        print(f"\n{name} ({args.input}, {args.layers}, {args.head})")
        print(summarize(df).round(3).to_string())
        probs = out_of_fold_probabilities(features, labels, groups, make_head, folds=folds)
        scores, _ = patient_level_scores(probs, labels, groups)
        patient_rows.append({"encoder": name, "patients": len(set(groups)), **scores})
        print("Patient level, all labels:", {k: round(v, 3) for k, v in scores.items()})

    out = pd.concat(results, ignore_index=True)
    out.insert(1, "input", args.input)
    out.insert(2, "layers", args.layers)
    out.insert(3, "head", args.head)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    patient = pd.DataFrame(patient_rows)
    for col in ("head", "layers", "input"):
        patient.insert(1, col, getattr(args, col))
    patient.to_csv(
        Path(args.output).with_name(f"{Path(args.output).stem}_patients.csv"), index=False
    )


if __name__ == "__main__":
    main()
