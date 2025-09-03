import wandb
import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import albumentations as A
from albumentations import CoarseDropout
import cv2
from torch.utils.data import Subset, DataLoader
import pandas as pd
import torchvision

from scripts.preprocessing.data_loader import ThermalDataset
from scripts.models.R2AU_dynamic import R2AttU_Net, init_weights, EarlyStopping
from scripts.models.losses import *
from scripts.models.metrics import *
from scripts.models.metrics_multiclass import get_multiclass_metrics
from scripts.models.metrics import get_metrics
from scripts.models.trainer import Trainer
import torch
from sklearn.metrics import roc_auc_score

import os 
from dotenv import load_dotenv
from pathlib import Path

def compute_metrics_torch(all_preds_tensor, all_gt_tensor):
    y_pred_flat = all_preds_tensor.view(-1).cpu().numpy()
    y_true_flat = all_gt_tensor.view(-1).cpu().numpy()

    TP = ((y_true_flat == 1) & (y_pred_flat == 1)).sum()
    TN = ((y_true_flat == 0) & (y_pred_flat == 0)).sum()
    FP = ((y_true_flat == 0) & (y_pred_flat == 1)).sum()
    FN = ((y_true_flat == 1) & (y_pred_flat == 0)).sum()

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    try:
        auc = roc_auc_score(y_true_flat, y_pred_flat)
    except:
        auc = float('nan')

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "AUC": auc
    }


global_transform = A.Compose([
    A.GaussianBlur(blur_limit=3, p=0.4),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.2,
        p=0.4),
    A.RandomGamma(gamma_limit=(80, 120), p=0.4),
    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
    A.OpticalDistortion(distort_limit=0.05, p=0.2),
    A.RandomToneCurve(scale=0.2, p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.01,
        scale_limit=0.01,
        rotate_limit=2,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.3),
], additional_targets={'mask': 'mask'})

full_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7),
    A.RandomResizedCrop(
        size=(256, 384),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=30.0, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.OpticalDistortion(distort_limit=0.1, p=0.3),
    A.CoarseDropout(num_holes_range=[1, 3], hole_height_range=[30, 50], 
                    hole_width_range=[30, 50], fill=0, p=0.3),
    A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.5),
    A.MotionBlur(blur_limit=(3, 5), p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.25,
        p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomToneCurve(scale=0.3, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    A.InvertImg(p=0.1), 
    A.Downscale(scale_range=(0.6, 0.8), p=0.3),
], additional_targets={'mask': 'mask'})

geometric_structural_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.05,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.7),
    A.RandomResizedCrop(
        size=(256, 384),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=30.0, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
    A.OpticalDistortion(distort_limit=0.1, p=0.3),
    A.CoarseDropout(num_holes_range=[1, 3], hole_height_range=[30, 50], 
                    hole_width_range=[30, 50], fill=0, p=0.3),
], additional_targets={'mask': 'mask'})

fotometric_degradation_transform = A.Compose([
    A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.5),
    A.MotionBlur(blur_limit=(3, 5), p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.25,
        p=0.7),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomToneCurve(scale=0.3, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
    A.InvertImg(p=0.1), 
    A.Downscale(scale_range=(0.6, 0.8), p=0.3),
], additional_targets={'mask': 'mask'})

load_dotenv()
root_dir = Path(os.getenv("ROOT_DIR"))

test_images_dir = root_dir / "data" / "test Flir" / "processed_numpy"
test_labels_dir = root_dir / "data" / "test Flir" / "labels" / "labels_merged_numpy"
test_dataset = ThermalDataset(images_dir=test_images_dir, masks_dir=test_labels_dir, height=256, width=384)
test_loader  = DataLoader(test_dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

exp_dict = {
    "no_transform": None,
    "full_transform": full_transform,
    "geometric_structural_transform": geometric_structural_transform,
    "fotometric_degradation_transform": fotometric_degradation_transform
}

# --- DATASET ---
images_dir = root_dir / "data" / "images"
masks_dir = root_dir / "data" / "mask_merged"

ablation_results = pd.DataFrame(columns=[
    "experiment", 
    "fold", 
    "val_f1", 
    "val_iou", 
    "val_precision", 
    "val_recall (sensitivity)", 
    "val_auc", 
    "val_specificity",
    "test_f1", 
    "test_iou", 
    "test_precision", 
    "test_recall (sensitivity)", 
    "test_auc", 
    "test_specificity"])

cv_folds = 5
for exp_name, exp in exp_dict.items():
    for fold in range(cv_folds):
        # --- WANDB SETUP ---
        wandb.init(
            project="Augmentation Ablation Study Test v3 best model",
            name=f"{exp_name} - fold {fold}",
            config={
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
            },
        )
        config = wandb.config

        start_time = time.time()
        torch.backends.cudnn.benchmark = True
        print(f"Running experiment: {exp_name}")

        full_global = ThermalDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,  
            height=256,
            width=384,
            multiclass=True if config.task_type == "multiclass" else False,
            global_transform=global_transform,
            heavy_transform=None  
        )
        full_heavy = ThermalDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,  
            height=256,
            width=384,
            multiclass=True if config.task_type == "multiclass" else False,
            global_transform=global_transform,
            heavy_transform=exp  
        )

        n_total = len(full_global)
        indices = torch.randperm(n_total).tolist()
        val_size = int(n_total * 0.1)   
        train_idx = indices[val_size:]
        val_idx   = indices[:val_size]

        train_dataset = Subset(full_heavy, train_idx)
        val_dataset   = Subset(full_global, val_idx)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

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

        # --- MODEL SETUP ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_ch = 3 if config.task_type == "multiclass" else 1

        model = R2AttU_Net(
            img_ch=1,
            output_ch=output_ch,
            t=config.t,
            base_filters=config.base_filters,
            depth=config.depth,
        ).to(device, non_blocking=True)
        init_weights(model, init_type=config.init_type)

        if config.task_type == "binary":
            criterion = FocalTverskyLoss(
                alpha=config.alpha, beta=1 - config.alpha, gamma=config.gamma
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

        # --- TRAINING ---
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            early_stopping=early_stopping,
            task_type=config.task_type,
        )

        best_model, best_metrics_train = trainer.run()

        # --- TESTING ---
        all_preds, all_gt, all_images = [], [], []
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                outputs = model(images)
                preds   = torch.sigmoid(outputs).cpu()
                preds   = (preds < 0.5).float()
                all_preds.append(preds)
                all_gt.append(masks.cpu())
                all_images.append(images.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_gt = torch.cat(all_gt, dim=0).numpy()
        all_images = torch.cat(all_images, dim=0).numpy()

        all_preds_fixed = all_preds[:, 0:1, :, :] 
        all_gt_fixed = all_gt[:, None, :, :] 

        # log the first test image to wandb 
        img_test = torch.tensor(all_images[0], dtype=torch.float32)
        gt_test = torch.tensor(all_gt_fixed[0], dtype=torch.float32)
        pred_test = torch.tensor(all_preds_fixed[0], dtype=torch.float32)
        print(f"img_test shape: {img_test.shape}, gt_test shape: {gt_test.shape}, pred_test shape: {pred_test.shape}")
        comparison_test = torch.stack([img_test, gt_test, pred_test], dim=0)
        grid_test = torchvision.utils.make_grid(
            comparison_test, nrow=3, normalize=False
        )
        wandb.log({
            "Test Image": wandb.Image(grid_test, caption="Test [Input|GT|Pred]")
        })

        wandb.finish()

        all_preds_tensor = torch.tensor(all_preds_fixed, dtype=torch.float32)
        all_gt_tensor = torch.tensor(all_gt_fixed, dtype=torch.float32)

        test_metrics = compute_metrics_torch(all_preds_tensor, all_gt_tensor)

        ablation_results = pd.concat([ablation_results, pd.DataFrame({
            "experiment": [exp_name],
            "fold": [fold],
            "val_f1": [best_metrics_train["f1"]],
            "val_iou": [best_metrics_train["iou"]],
            "val_precision": [best_metrics_train["precision"]],
            "val_recall (sensitivity)": [best_metrics_train["recall (sensitivity)"]],
            "val_auc": [best_metrics_train["auc"]],
            "val_specificity": [best_metrics_train["specificity"]],
            "test_f1": [test_metrics["F1"]],
            "test_iou": [test_metrics["IoU"]],
            "test_precision": [test_metrics["Precision"]],
            "test_recall (sensitivity)": [test_metrics["Recall"]],
            "test_auc": [test_metrics["AUC"]],
            "test_specificity": [test_metrics["Specificity"]]
        })], ignore_index=True)
        
        print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
        print(ablation_results)

    ablation_results.to_csv("ablation_results_test_v3_best_model.csv", index=False)
