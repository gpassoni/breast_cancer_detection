"""Train the classification head on the full DMR-IR dataset (70/15/15 patient-grouped split).

Usage:
    python scripts/classification_finetune.py --backbone vicreg/checkpoints/resnet50.pth
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
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
)

BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 200


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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    paths, labels = list_dmrir_images(args.data_dir)
    groups = [patient_id_from_path(p, args.patient_id_pattern) for p in paths]

    # 70% train, then the remaining 30% halved into validation and test.
    train_paths, train_labels, _, rest_paths, rest_labels = grouped_split(
        paths, labels, groups, test_size=0.3, seed=42
    )
    rest_groups = [patient_id_from_path(p, args.patient_id_pattern) for p in rest_paths]
    val_paths, val_labels, _, test_paths, test_labels = grouped_split(
        rest_paths, rest_labels, rest_groups, test_size=0.5, seed=42
    )
    for name, split_labels in (("Train", train_labels), ("Val", val_labels), ("Test", test_labels)):
        positives = sum(split_labels)
        healthy = len(split_labels) - positives
        print(f"{name}: {len(split_labels)} images ({positives} cancer, {healthy} healthy)")

    train_loader = DataLoader(
        DMRIRDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(DMRIRDataset(val_paths, val_labels), batch_size=BATCH_SIZE)
    test_loader = DataLoader(DMRIRDataset(test_paths, test_labels), batch_size=BATCH_SIZE)

    backbone = load_vicreg_resnet50(args.backbone, args.vicreg_dir).to(device)
    backbone.eval()
    model = ClassificationHead(backbone).to(device)
    for param in model.backbone.parameters():
        param.requires_grad = False

    wandb.init(
        project="DownStreamDMRIR",
        name="frozen_backbone_classification_head",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "model": "ResNet50 with Classification Head",
        },
    )
    fit_classifier(
        model,
        train_loader,
        val_loader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optim.Adam(model.fc.parameters(), lr=LEARNING_RATE),
        early_stopping=EarlyStopping(patience=5, delta=0.001),
        num_epochs=NUM_EPOCHS,
        device=device,
    )
    evaluate_classifier(model, test_loader, device)
    wandb.finish()


if __name__ == "__main__":
    main()
