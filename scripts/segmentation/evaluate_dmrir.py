"""Evaluate the breast segmentation model on the public DMR-IR breast masks.

The model was trained on private clinical FLIR images only, so this is an external test on a
different camera, site and patient population. DMR-IR masks cover both breasts and the
region between them as a single area, while the model segments each breast separately;
precision (predicted pixels inside the reference region) is therefore reported next to
Dice and recall.

Usage:
    python scripts/segmentation/evaluate_dmrir.py
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from thermal_bc.config import root_dir
from thermal_bc.data.dmrir import list_dmrir_images, patient_id_from_path
from thermal_bc.features import predict_breast_masks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=root_dir() / "data" / "dmrir" / "lab_database")
    parser.add_argument("--seg-weights", default="weights/segmentation_r2attunet.pth")
    parser.add_argument("--seg-config", default="weights/segmentation_r2attunet.json")
    parser.add_argument("--output", default="results/segmentation_dmrir.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    paths, labels = list_dmrir_images(data_dir / "database")
    predictions = predict_breast_masks(
        paths, args.seg_weights, args.seg_config, device, Path("runs/cache/breast_masks.npz")
    )

    rows = []
    for path, label, pred in zip(paths, labels, predictions):
        ref_path = data_dir / "labels" / Path(path).parent.name / Path(path).name
        reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE) > 127  # JPEG-compressed
        tp = (pred & reference).sum()
        rows.append(
            {
                "image": Path(path).name,
                "patient": patient_id_from_path(path),
                "label": label,
                "dice": 2 * tp / max(pred.sum() + reference.sum(), 1),
                "iou": tp / max((pred | reference).sum(), 1),
                "precision": tp / max(pred.sum(), 1),
                "recall": tp / max(reference.sum(), 1),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    metrics = ["dice", "iou", "precision", "recall"]
    per_patient = df.groupby("patient")[metrics].mean()
    print(f"{len(df)} images, {len(per_patient)} patients")
    print("Per image (mean):  ", df[metrics].mean().round(3).to_dict())
    print("Per patient (mean):", per_patient.mean().round(3).to_dict())
    print("Per image (median):", df[metrics].median().round(3).to_dict())


if __name__ == "__main__":
    main()
