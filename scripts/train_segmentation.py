"""Train a single R2AttU-Net segmentation model with light + heavy augmentation.

Expects `ROOT_DIR/data/images` and `ROOT_DIR/data/mask_merged` (see scripts/prepare_masks.py).
The best model across runs is kept in checkpoints/best/.
"""

import time

import torch
import torch.optim as optim
import wandb
from torch.utils.data import Subset

from thermal_bc.config import root_dir
from thermal_bc.data.augmentations import full_transform, global_transform
from thermal_bc.data.loaders import make_loader
from thermal_bc.data.thermal_dataset import ThermalDataset
from thermal_bc.models.losses import FocalTverskyLoss
from thermal_bc.models.r2attunet import EarlyStopping, R2AttU_Net, init_weights
from thermal_bc.models.trainer import Trainer

IMAGE_SIZE = (224, 224)
CONFIG = {
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
    "task_type": "binary",  # "binary" (breast vs background) or "multiclass" (left/right)
}


def main():
    wandb.init(project="Augmented Segmentation", config=CONFIG)
    config = wandb.config
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    multiclass = config.task_type == "multiclass"

    data_dir = root_dir() / "data"
    dataset_kwargs = dict(
        images_dir=data_dir / "images",
        masks_dir=data_dir / "mask_merged",
        height=IMAGE_SIZE[0],
        width=IMAGE_SIZE[1],
        multiclass=multiclass,
        global_transform=global_transform,
    )
    # Same images, two views: heavy augmentation for training, light for validation.
    full_light = ThermalDataset(**dataset_kwargs, heavy_transform=None)
    full_heavy = ThermalDataset(
        **dataset_kwargs,
        heavy_transform=full_transform(crop_size=IMAGE_SIZE, downscale_range=(0.5, 0.8)),
    )

    indices = torch.randperm(len(full_light)).tolist()
    val_size = int(len(full_light) * config.val_ratio)
    train_dataset = Subset(full_heavy, indices[val_size:])
    val_dataset = Subset(full_light, indices[:val_size])
    print(f"Train: {len(train_dataset)} images, validation: {len(val_dataset)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        criterion = FocalTverskyLoss(alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma)

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
    _, best_metrics = trainer.run()
    print(f"Best validation metrics: {best_metrics}")
    print(f"Training completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
