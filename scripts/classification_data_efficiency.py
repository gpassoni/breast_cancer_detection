"""Label-efficiency study on DMR-IR: train the classification head with 64-512 labeled images.

For every training-set size, 5-fold cross-validation is run and each fold's model is
evaluated on the same held-out test set. Splits are grouped by patient, so images of one
patient never appear in more than one of train / validation / test.

Usage:
    python scripts/classification_data_efficiency.py --backbone vicreg/checkpoints/resnet50.pth
"""

import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from thermal_bc.config import root_dir
from thermal_bc.data.dmrir_dataset import (
    DMRIRDataset,
    grouped_split,
    list_dmrir_images,
    patient_id_from_path,
)
from thermal_bc.models.classification import (
    ClassificationHead,
    EarlyStopping,
    evaluate_classifier,
    fit_classifier,
    load_vicreg_resnet50,
    shuffle_data,
)

TRAIN_SIZES = [64, 128, 192, 256, 352, 416, 512]
CV_FOLDS = 5
MAX_VAL_SAMPLES = 100
LEARNING_RATE = 1e-5
NUM_EPOCHS = 300


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir",
        default=root_dir() / "data" / "dmrir" / "lab_database" / "database",
        help="Folder with normal/ and abnormal/ subfolders.",
    )
    parser.add_argument("--backbone", required=True, help="VICReg ResNet-50 checkpoint.")
    parser.add_argument("--vicreg-dir", default="vicreg", help="Clone of facebookresearch/vicreg.")
    parser.add_argument(
        "--patient-id-pattern",
        default=r"PAC_(\d+)_",
        help="Regex whose first group is the patient ID in each file name.",
    )
    parser.add_argument("--output", default="results/classification_data_efficiency.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths, labels = list_dmrir_images(args.data_dir)
    groups = [patient_id_from_path(p, args.patient_id_pattern) for p in paths]

    train_val_paths, train_val_labels, train_val_groups, test_paths, test_labels = grouped_split(
        paths, labels, groups, test_size=0.3, seed=42
    )

    results = pd.DataFrame(columns=["n_samples", "cv_fold", "f1", "precision", "recall", "auc"])
    skf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for n_samples in TRAIN_SIZES:
        folds = skf.split(train_val_paths, train_val_labels, train_val_groups)
        for fold, (train_index, val_index) in enumerate(folds):
            train_paths = [train_val_paths[i] for i in train_index]
            train_labels = [train_val_labels[i] for i in train_index]
            val_paths = [train_val_paths[i] for i in val_index][:MAX_VAL_SAMPLES]
            val_labels = [train_val_labels[i] for i in val_index][:MAX_VAL_SAMPLES]

            # Subsample the training fold, re-shuffling until both classes hold 20-80%.
            subset_paths, subset_labels = train_paths[:n_samples], train_labels[:n_samples]
            positive_ratio = sum(subset_labels) / len(subset_labels)
            while positive_ratio < 0.2 or positive_ratio > 0.8:
                train_paths, train_labels = shuffle_data(train_paths, train_labels)
                subset_paths, subset_labels = train_paths[:n_samples], train_labels[:n_samples]
                positive_ratio = sum(subset_labels) / len(subset_labels)

            print(
                f"n={n_samples} fold {fold + 1}/{CV_FOLDS} - train: {len(subset_paths)} "
                f"(positive ratio {positive_ratio:.2f}), val: {len(val_paths)}, "
                f"test: {len(test_paths)}"
            )

            batch_size = 64 if n_samples > 64 else n_samples
            train_loader = DataLoader(
                DMRIRDataset(subset_paths, subset_labels), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                DMRIRDataset(val_paths, val_labels), batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                DMRIRDataset(test_paths, test_labels), batch_size=batch_size, shuffle=False
            )

            backbone = load_vicreg_resnet50(args.backbone, args.vicreg_dir).to(device)
            backbone.eval()
            model = ClassificationHead(backbone).to(device)
            for param in model.backbone.parameters():
                param.requires_grad = False

            wandb.init(
                project="DownStreamDMRIR_data_study",
                name=f"DMRIR_Classification_{n_samples}_samples_cv_{fold}",
                config={
                    "learning_rate": LEARNING_RATE,
                    "batch_size": batch_size,
                    "epochs": NUM_EPOCHS,
                    "model": "ResNet50 with Classification Head",
                },
                reinit=True,
            )
            fit_classifier(
                model,
                train_loader,
                val_loader,
                criterion=nn.BCEWithLogitsLoss(),
                optimizer=optim.Adam(model.fc.parameters(), lr=LEARNING_RATE),
                early_stopping=EarlyStopping(patience=10, delta=0.001),
                num_epochs=NUM_EPOCHS,
                device=device,
            )
            scores = evaluate_classifier(model, test_loader, device)
            wandb.finish()

            row = pd.DataFrame([{"n_samples": n_samples, "cv_fold": fold, **scores}])
            results = pd.concat([results, row], ignore_index=True)
            results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
