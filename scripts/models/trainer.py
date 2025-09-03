import torch
import torchvision
import wandb
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from .metrics import get_metrics
from .metrics_multiclass import get_multiclass_metrics
import time
import shutil
from pathlib import Path
from datetime import datetime
import json

def multiclass_to_binary(preds: torch.Tensor) -> torch.Tensor:
    if preds.ndim == 3:
        preds = preds.unsqueeze(1) 
    preds_bin = (preds > 0).float()
    return preds_bin


class Trainer:
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
            "specificity": 0.0}
            
        self.best_last_dir = Path("best_last_model")
        self.best_last_dir.mkdir(exist_ok=True)
        self.best_overall_dir = Path("best_model_saved_v2_inno")
        self.best_overall_dir.mkdir(exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_model_path = self.best_last_dir / f"run_{self.run_id}_best_model.pth"
        self.best_overall_metric = self.load_best_model_metric()

    def best_metrics_tracker(self, current_dict):
        if current_dict["f1"] > self.best_metrics["f1"]:
            self.best_metrics = current_dict
            torch.save(self.model.state_dict(), self.best_model_path)

    def load_best_model_metric(self):
        metric_file = self.best_overall_dir / "metric.txt"
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    return float(f.read().strip())
            except:
                pass
        return 0.0

    def update_best_overall(self):
        """Controlla metric.txt, crea o aggiorna la metrica e copia il .pth se la run corrente supera il best global."""
        metric_file = self.best_overall_dir / "metric.txt"
        best_overall = 0.0
        if metric_file.exists():
            try:
                with open(metric_file, "r") as f:
                    best_overall = float(f.read().strip())
            except (ValueError, IOError):
                print("Error reading the best overall metric file. Defaulting to 0.0.")
                best_overall = 0.0

        current = self.best_metrics["f1"]
        if current > best_overall:
            torch.save(
                self.model.state_dict(), self.best_overall_dir / "best_model.pth"
            )

            with open(self.best_overall_dir / "config.json", "w") as f:
                json.dump(dict(self.config), f, indent=4)

            with open(self.best_overall_dir / "metric.txt", "w") as f:
                f.write(f"{current:.6f}")

            with open(metric_file, "w") as f:
                f.write(f"{current:.6f}")

    def run(self):
        self.optimizer.zero_grad()

        for epoch in range(self.config.n_epochs):
            epoch_duration_start = time.time()
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
                    "epoch_duration": (time.time() - epoch_duration_start) / 60,
                }
            )

            if epoch % 2 == 0:
                with torch.no_grad():
                    mask = masks[0].cpu()
                    if self.task_type == "binary":
                        pred = torch.sigmoid(outputs[0].detach()).cpu()
                    else:
                        pred = torch.argmax(outputs[0].detach(), dim=0, keepdim=True).float().cpu()

                    if mask.ndim == 2:  
                        mask = mask.unsqueeze(0)

                    img = images[0].cpu()
                    comparison = torch.stack([img, mask, pred], dim=0)
                    grid = torchvision.utils.make_grid(
                        comparison, nrow=3, normalize=False
                    )
                    wandb.log(
                        {
                            "Val Comparison [Ground Truth| Pred]": [
                                wandb.Image(grid, caption=f"Epoch {epoch}")
                            ]
                        }
                    )

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered. Training stopped")
                    break

        self.update_best_overall()

        best_run_path = self.best_last_dir / f"run_{self.run_id}_best_model.pth"
        self.model.load_state_dict(torch.load(best_run_path))
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
                    img = images[0].cpu()
                    mask = masks[0].cpu()

                    if self.task_type == "binary":
                        pred = torch.sigmoid(outputs[0].detach()).cpu()
                    else:
                        pred = torch.argmax(outputs[0].detach(), dim=0, keepdim=True).float().cpu()

                    if mask.ndim == 2:
                        mask = mask.unsqueeze(0)

                    comparison = torch.stack([img, mask, pred], dim=0)
                    log_images.append(
                        wandb.Image(
                            torchvision.utils.make_grid(comparison, nrow=3, normalize=False),
                            caption="Validation [Input|GT|Pred]"
                        )
                    )

        val_loss /= len(self.val_loader.dataset)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if self.task_type == "binary":
            if all_targets.ndim == 3:
                all_targets = all_targets.unsqueeze(1)
            metrics = get_metrics(all_preds, all_targets)
        else:
            preds = torch.argmax(all_preds, dim=1)
            preds = (preds > 0).float().unsqueeze(1)

            if all_targets.ndim == 3:
                all_targets = all_targets.unsqueeze(1)
            targets = (all_targets > 0).float()
            metrics = get_metrics(preds, targets)

        wandb.log({
            "Validation Images [Input|GT|Pred]": log_images
        })

        return val_loss, metrics

