import wandb
import time
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import albumentations as A
from albumentations import CoarseDropout
import cv2

from scripts.preprocessing.data_loader import ThermalDataset
from scripts.models.R2AU_dynamic import R2AttU_Net, init_weights, EarlyStopping
from scripts.models.losses import *
from scripts.models.metrics import *
from scripts.models.metrics_multiclass import get_multiclass_metrics
from scripts.models.metrics import get_metrics
from scripts.models.trainer import Trainer

from torch.utils.data import Subset, DataLoader

import os 
from pathlib import Path
from dotenv import load_dotenv

wandb.init(
    project="Augmented Segmentation",
    name="prova1",
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
        "task_type": "binary",  
    },
)
config = wandb.config

start_time = time.time()
torch.backends.cudnn.benchmark = True

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
        size=(224, 224),
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
    A.Downscale(scale_range=(0.5, 0.8), p=0.3),
], additional_targets={'mask': 'mask'})

load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR"))

images_dir = ROOT_DIR / "data" / "images"
masks_dir = ROOT_DIR / "data" / "mask_merged"

full_global = ThermalDataset(
    images_dir=images_dir, 
    masks_dir=masks_dir,   
    height=224,
    width=224,
    multiclass=True,
    global_transform=global_transform,
    heavy_transform=None   
)

full_heavy = ThermalDataset(
    images_dir=images_dir, 
    masks_dir=masks_dir,   
    height=224,
    width=224,
    multiclass=True,
    global_transform=global_transform,
    heavy_transform=full_transform  
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_ch = 3 if config.task_type == "multiclass" else 1

torch.cuda.empty_cache()   

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

print(f"Training completed in {time.time() - start_time:.2f} seconds")
