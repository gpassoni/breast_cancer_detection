"""Augmentation ablation for breast segmentation.

Trains the multiclass R2AttU-Net 5 times under each of four augmentation settings
(none, geometric/structural, photometric degradation, both) and evaluates every model on
an external FLIR test set that was never used for training or model selection.

Expects `ROOT_DIR/data/images`, `ROOT_DIR/data/mask_merged` and the test set produced by
scripts/prepare_flir_test_set.py. Writes results/augmentation_ablation.csv.
"""

import time

import pandas as pd
import torch
import torch.optim as optim
import torchvision
import wandb
from torch.utils.data import Subset

from thermal_bc.config import root_dir
from thermal_bc.data.augmentations import (
    full_transform,
    geometric_structural_transform,
    global_transform,
    photometric_degradation_transform,
)
from thermal_bc.data.loaders import make_loader
from thermal_bc.data.thermal_dataset import ThermalDataset
from thermal_bc.models.losses import FocalTverskyLoss
from thermal_bc.models.metrics import breast_mask_from_multiclass, confusion_metrics
from thermal_bc.models.r2attunet import EarlyStopping, R2AttU_Net, init_weights
from thermal_bc.models.trainer import Trainer

N_RUNS = 5
HEIGHT, WIDTH = 256, 384
EXPERIMENTS = {
    "no_transform": None,
    "full_transform": full_transform(crop_size=(HEIGHT, WIDTH)),
    "geometric_structural_transform": geometric_structural_transform(crop_size=(HEIGHT, WIDTH)),
    "fotometric_degradation_transform": photometric_degradation_transform(),
}
CONFIG = {
    "val_ratio": 0.1,
    "batch_size": 4,
    "init_type": "kaiming",
    "loss": "focal_tversky",
    "lr": 0.0010076,
    "n_epochs": 200,
    "t": 4,
    "base_filters": 16,
    "depth": 5,
    "alpha": 0.56814,
    "gamma": 1.52575,
    "accum_steps": 4,
    "task_type": "multiclass",
}


def evaluate_on_test_set(model, test_loader, device):
    """Predict the external test set; return confusion metrics and one (image, gt, pred) example."""
    all_preds, all_gt, all_images = [], [], []
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images.to(device, non_blocking=True)).cpu()
            all_preds.append(breast_mask_from_multiclass(outputs))
            all_gt.append(masks.unsqueeze(1).float())
            all_images.append(images)

    preds, gt, images = torch.cat(all_preds), torch.cat(all_gt), torch.cat(all_images)
    return confusion_metrics(preds, gt), (images[0], gt[0], preds[0])


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = root_dir() / "data"

    test_dataset = ThermalDataset(
        images_dir=data_dir / "test_flir" / "images",
        masks_dir=data_dir / "test_flir" / "masks",
        height=HEIGHT,
        width=WIDTH,
    )
    test_loader = make_loader(test_dataset, CONFIG["batch_size"], train=False)

    rows = []
    for exp_name, heavy_transform in EXPERIMENTS.items():
        for run in range(N_RUNS):
            wandb.init(
                project="Augmentation Ablation Study",
                name=f"{exp_name} - run {run}",
                config=CONFIG,
            )
            config = wandb.config
            multiclass = config.task_type == "multiclass"
            start_time = time.time()
            print(f"Running experiment: {exp_name} (run {run + 1}/{N_RUNS})")

            dataset_kwargs = dict(
                images_dir=data_dir / "images",
                masks_dir=data_dir / "mask_merged",
                height=HEIGHT,
                width=WIDTH,
                multiclass=multiclass,
                global_transform=global_transform,
            )
            full_light = ThermalDataset(**dataset_kwargs, heavy_transform=None)
            full_heavy = ThermalDataset(**dataset_kwargs, heavy_transform=heavy_transform)

            indices = torch.randperm(len(full_light)).tolist()
            val_size = int(len(full_light) * config.val_ratio)
            train_dataset = Subset(full_heavy, indices[val_size:])
            val_dataset = Subset(full_light, indices[:val_size])

            model = R2AttU_Net(
                img_ch=1,
                output_ch=3 if multiclass else 1,
                t=config.t,
                base_filters=config.base_filters,
                depth=config.depth,
            ).to(device, non_blocking=True)
            init_weights(model, init_type=config.init_type)

            if multiclass:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = FocalTverskyLoss(
                    alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma
                )

            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optim.Adam(model.parameters(), lr=config.lr),
                train_loader=make_loader(train_dataset, config.batch_size, train=True),
                val_loader=make_loader(val_dataset, config.batch_size, train=False),
                config=config,
                device=device,
                early_stopping=EarlyStopping(patience=10, min_delta=0.001, verbose=True),
                task_type=config.task_type,
            )
            model, val_metrics = trainer.run()

            test_metrics, example = evaluate_on_test_set(model, test_loader, device)
            grid = torchvision.utils.make_grid(torch.stack(example), nrow=3, normalize=False)
            wandb.log({"Test Image": wandb.Image(grid, caption="Test [Input|GT|Pred]")})
            wandb.finish()

            rows.append(
                {
                    "experiment": exp_name,
                    "fold": run,
                    "val_f1": val_metrics["f1"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall (sensitivity)": val_metrics["recall (sensitivity)"],
                    "val_auc": val_metrics["auc"],
                    "val_specificity": val_metrics["specificity"],
                    "test_f1": test_metrics["F1"],
                    "test_iou": test_metrics["IoU"],
                    "test_precision": test_metrics["Precision"],
                    "test_recall (sensitivity)": test_metrics["Recall"],
                    "test_auc": test_metrics["AUC"],
                    "test_specificity": test_metrics["Specificity"],
                }
            )
            pd.DataFrame(rows).to_csv("results/augmentation_ablation.csv", index=False)
            print(f"Run completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
