"""W&B sweep agent for R2AttU-Net hyperparameter search (Bayesian optimisation on validation F1).

Usage:
    wandb sweep configs/sweep_binary.yaml
    wandb agent <entity>/<project>/<sweep_id>
"""

import torch
import torch.optim as optim
import wandb
from torch.utils.data import random_split

from thermal_bc.config import root_dir
from thermal_bc.data.loaders import make_loader
from thermal_bc.data.thermal_dataset import ThermalDataset
from thermal_bc.models.losses import FocalTverskyLoss
from thermal_bc.models.r2attunet import EarlyStopping, R2AttU_Net, init_weights
from thermal_bc.models.trainer import Trainer

ERROR_LOG = "sweep_errors.txt"


def log_error_to_file(msg):
    with open(ERROR_LOG, "a") as f:
        f.write(msg + "\n")


def main():
    wandb.init()
    config = wandb.config
    config.val_ratio = 0.1
    config.init_type = "kaiming"
    config.loss = "focal_tversky"
    config.n_epochs = 300
    config.accum_steps = 4
    config.batch_size = 4

    torch.backends.cudnn.benchmark = True

    data_dir = root_dir() / "data"
    dataset = ThermalDataset(
        images_dir=data_dir / "images_numpy",
        masks_dir=data_dir / "mask_preprocessed",
        height=256,
        width=384,
    )
    val_size = int(len(dataset) * config.val_ratio)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    print(f"Train: {len(train_dataset)} images, validation: {len(val_dataset)} images")

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

    trainer = Trainer(
        model=model,
        criterion=FocalTverskyLoss(alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma),
        optimizer=optim.Adam(model.parameters(), lr=config.learning_rate),
        train_loader=make_loader(train_dataset, config.batch_size, train=True),
        val_loader=make_loader(val_dataset, config.batch_size, train=False),
        config=config,
        device=device,
        early_stopping=EarlyStopping(patience=10, min_delta=0.001, verbose=True),
    )
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        # Log out-of-memory configurations and let the sweep continue with the next trial.
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            cfg = wandb.config
            params = ["batch_size", "base_filters", "depth", "learning_rate", "t", "gamma", "alpha"]
            log_error_to_file(
                "[OOM] Params: " + ", ".join(f"{k}={getattr(cfg, k, 'NA')}" for k in params)
            )
        else:
            log_error_to_file(f"[RUNTIME ERROR] {e}")
            raise
    except Exception as e:
        log_error_to_file(f"[GENERAL ERROR] {e}")
        raise
    finally:
        wandb.finish(exit_code=1)
