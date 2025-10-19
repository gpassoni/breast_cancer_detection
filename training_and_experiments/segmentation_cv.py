import wandb
import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import pandas as pd
import os
from dotenv import load_dotenv

from scripts.preprocessing.data_loader import ThermalDataset
from scripts.models.R2AU_dynamic import R2AttU_Net, init_weights, EarlyStopping
from scripts.models.losses import *
from scripts.models.metrics import *
from scripts.models.trainer import Trainer

# Load environment variables
load_dotenv()

start_time = time.time()
torch.backends.cudnn.benchmark = True
cv_folds = 5

cv_results = {
    "fold": [],
    "dice": [],
    "iou": [],
    "precision": [],
    "recall (sensitivity)": [],
    "f1": [],
    "auc": [],
    "boundary_iou": [],
    "specificity": [],
}

for fold in range(cv_folds):
    print(f"Starting fold {fold + 1}/{cv_folds}")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "segmentation_bcxtt"),
        entity=os.getenv("WANDB_ENTITY", None),
        name=f"cv_fold_{fold + 1}",
        config={
            "val_ratio": 0.1,
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
        },
    )
    config = wandb.config

    from pathlib import Path
    import os 
    from dotenv import load_dotenv

    load_dotenv()
    ROOT_DIR = Path(os.getenv("ROOT_DIR"))

    images_dir = ROOT_DIR / "data/images_numpy"
    masks_dir = ROOT_DIR / "data/mask_preprocessed"

    dataset = ThermalDataset(
        images_dir=images_dir, masks_dir=masks_dir, height=256, width=384
    )

    val_size = int(len(dataset) * config.val_ratio)
    test_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    criterion = FocalTverskyLoss(
        alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma
    )
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)
    accum_steps = config.accum_steps

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        early_stopping=early_stopping,
    )
    best_model, best_metrics_train = trainer.run()
    model = best_model

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                outputs = model(images)

            all_preds.append(outputs.cpu())
            all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    test_metrics = get_metrics(all_preds, all_targets)

    print(f"Fold {fold+1} Test Metrics:", test_metrics)

    cv_results["fold"].append(fold + 1)
    for k in test_metrics:
        cv_results[k].append(test_metrics[k])

    cv_df = pd.DataFrame(cv_results)
    print("\nCross-validation results:")
    print(cv_df)

    print("\nMean scores across folds:")
    print(cv_df.mean(numeric_only=True))

    print("\nStd dev of scores across folds:")
    print(cv_df.std(numeric_only=True))

cv_df.to_csv("cv_test_results.csv", index=False)
print(f"Training completed in {time.time() - start_time:.2f} seconds")
