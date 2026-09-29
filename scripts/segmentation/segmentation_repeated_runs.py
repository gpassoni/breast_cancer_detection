"""Train the tuned binary R2AttU-Net 5 times and report test metrics for each run.

Expects `ROOT_DIR/data/images_numpy` and `ROOT_DIR/data/mask_preprocessed`
(see scripts/segmentation/prepare_masks.py). Writes results/segmentation_repeated_runs.csv.
"""

import time

import pandas as pd
import torch
import torch.optim as optim
import wandb
from torch.utils.data import random_split

from thermal_bc.config import root_dir
from thermal_bc.data.loaders import make_loader
from thermal_bc.data.thermal_dataset import ThermalDataset
from thermal_bc.models.losses import FocalTverskyLoss
from thermal_bc.models.metrics import get_metrics
from thermal_bc.models.r2attunet import EarlyStopping, R2AttU_Net, init_weights
from thermal_bc.models.trainer import Trainer

N_RUNS = 5
CONFIG = {
    "val_ratio": 0.1,
    "test_ratio": 0.2,
    "batch_size": 4,
    "init_type": "kaiming",
    "loss": "focal_tversky",
    "lr": 0.00085939,
    "n_epochs": 100,
    "t": 4,
    "base_filters": 16,
    "depth": 7,
    "alpha": 0.56814,
    "gamma": 1.52575,
    "accum_steps": 4,
}


def main():
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = root_dir() / "data"
    dataset = ThermalDataset(
        images_dir=data_dir / "images_numpy",
        masks_dir=data_dir / "mask_preprocessed",
        height=256,
        width=384,
    )

    results = []
    for run in range(1, N_RUNS + 1):
        print(f"Starting run {run}/{N_RUNS}")
        wandb.init(project="segmentation_bcxtt", name=f"repeated_run_{run}", config=CONFIG)
        config = wandb.config

        val_size = int(len(dataset) * config.val_ratio)
        test_size = int(len(dataset) * config.test_ratio)
        train_size = len(dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        model = R2AttU_Net(
            img_ch=1,
            output_ch=1,
            t=config.t,
            base_filters=config.base_filters,
            depth=config.depth,
        ).to(device, non_blocking=True)
        init_weights(model, init_type=config.init_type)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")

        trainer = Trainer(
            model=model,
            criterion=FocalTverskyLoss(
                alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma
            ),
            optimizer=optim.Adam(model.parameters(), lr=config.lr),
            train_loader=make_loader(train_dataset, config.batch_size, train=True),
            val_loader=make_loader(val_dataset, config.batch_size, train=False),
            config=config,
            device=device,
            early_stopping=EarlyStopping(patience=5, min_delta=0.001, verbose=True),
        )
        model, _ = trainer.run()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in make_loader(test_dataset, config.batch_size, train=False):
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type):
                    outputs = model(images)
                all_preds.append(outputs.cpu())
                all_targets.append(masks)

        targets = torch.cat(all_targets).unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
        test_metrics = get_metrics(torch.cat(all_preds), targets)
        print(f"Run {run} test metrics: {test_metrics}")
        wandb.finish()

        results.append({"fold": run, **test_metrics})
        results_df = pd.DataFrame(results)
        results_df.to_csv("results/segmentation_repeated_runs.csv", index=False)

    print(results_df.agg(["mean", "std"]).T)
    print(f"Completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
