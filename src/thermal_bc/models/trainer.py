"""Segmentation training loop: mixed precision, gradient accumulation and W&B logging."""

import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from .metrics import get_metrics


class Trainer:
    """Train a segmentation model and keep two checkpoints.

    * `run_dir/run_<timestamp>_best_model.pth`: best epoch (by validation F1) of this run.
    * `best_dir/best_model.pth` (+ `config.json`, `metric.txt`): best model across all runs.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        config,
        device,
        early_stopping=None,
        task_type="binary",
        run_dir="checkpoints/runs",
        best_dir="checkpoints/best",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.task_type = task_type
        self.early_stopping = early_stopping
        self.scaler = GradScaler()
        self.best_metrics = {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall (sensitivity)": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "boundary_iou": 0.0,
            "specificity": 0.0,
        }

        self.best_last_dir = Path(run_dir)
        self.best_last_dir.mkdir(parents=True, exist_ok=True)
        self.best_overall_dir = Path(best_dir)
        self.best_overall_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_model_path = self.best_last_dir / f"run_{self.run_id}_best_model.pth"

    def best_metrics_tracker(self, current_dict):
        if current_dict["f1"] > self.best_metrics["f1"]:
            self.best_metrics = current_dict
            torch.save(self.model.state_dict(), self.best_model_path)

    def update_best_overall(self):
        """Promote this run to `best_dir` if it beats the best F1 recorded in `metric.txt`."""
        metric_file = self.best_overall_dir / "metric.txt"
        best_overall = 0.0
        if metric_file.exists():
            try:
                best_overall = float(metric_file.read_text().strip())
            except (ValueError, OSError):
                print("Error reading the best overall metric file. Defaulting to 0.0.")

        current = self.best_metrics["f1"]
        if current > best_overall:
            torch.save(self.model.state_dict(), self.best_overall_dir / "best_model.pth")
            with open(self.best_overall_dir / "config.json", "w") as f:
                json.dump(dict(self.config), f, indent=4)
            metric_file.write_text(f"{current:.6f}")

    def _prediction_for_logging(self, output):
        if self.task_type == "binary":
            return torch.sigmoid(output.detach()).cpu()
        return torch.argmax(output.detach(), dim=0, keepdim=True).float().cpu()

    def run(self):
        self.optimizer.zero_grad()

        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()
            self.model.train()
            train_loss = 0.0

            for step, (images, masks) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks) / self.config.accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.config.accum_steps == 0 or (
                    step + 1 == len(self.train_loader)
                ):
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(self.train_loader.dataset)

            val_loss, metrics = self.validate()
            self.best_metrics_tracker(metrics)

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    **metrics,
                    "epoch": epoch,
                    "epoch_duration": (time.time() - epoch_start) / 60,
                }
            )

            if epoch % 2 == 0:
                with torch.no_grad():
                    mask = masks[0].cpu()
                    pred = self._prediction_for_logging(outputs[0])
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    comparison = torch.stack([images[0].cpu(), mask, pred], dim=0)
                    grid = torchvision.utils.make_grid(comparison, nrow=3, normalize=False)
                    wandb.log(
                        {
                            "Val Comparison [Ground Truth| Pred]": [
                                wandb.Image(grid, caption=f"Epoch {epoch}")
                            ]
                        }
                    )

            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered. Training stopped")
                    break

        self.update_best_overall()

        self.model.load_state_dict(torch.load(self.best_model_path))
        return self.model, self.best_metrics

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        log_images = []

        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                all_preds.append(outputs.cpu())
                all_targets.append(masks.cpu())

                if idx == 0:
                    mask = masks[0].cpu()
                    pred = self._prediction_for_logging(outputs[0])
                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)
                    comparison = torch.stack([images[0].cpu(), mask, pred], dim=0)
                    log_images.append(
                        wandb.Image(
                            torchvision.utils.make_grid(comparison, nrow=3, normalize=False),
                            caption="Validation [Input|GT|Pred]",
                        )
                    )

        val_loss /= len(self.val_loader.dataset)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        if all_targets.ndim == 3:
            all_targets = all_targets.unsqueeze(1)

        if self.task_type == "binary":
            metrics = get_metrics(all_preds, all_targets)
        else:
            # Multiclass (background / left / right breast): evaluate as breast vs background.
            preds = (torch.argmax(all_preds, dim=1) > 0).float().unsqueeze(1)
            targets = (all_targets > 0).float()
            metrics = get_metrics(preds, targets)

        wandb.log({"Validation Images [Input|GT|Pred]": log_images})

        return val_loss, metrics
