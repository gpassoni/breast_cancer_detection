"""In-domain adaptation: continue VICReg self-supervised training of the backbone on DMR-IR.

For each patient-level CV fold (the same folds as scripts/classification/benchmark.py), the
clinically pretrained backbone is trained further with the VICReg objective on breast-only
images of that fold's *training* patients; labels are never used and the fold's test patients
are never seen. Augmentations are chosen for thermography: geometric changes and blur are
strong, intensity changes are mild, because the temperature pattern is the diagnostic signal.

Usage:
    python scripts/ssl/vicreg_adapt.py --init weights/vicreg_resnet50.pth \\
        --out-dir weights/vicreg_adapted
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from thermal_bc.config import root_dir
from thermal_bc.data.dmrir import list_dmrir_images, patient_id_from_path
from thermal_bc.evaluation import patient_folds
from thermal_bc.features import INPUTS, encoder_input, predict_breast_masks
from thermal_bc.models.vicreg import load_vicreg_resnet50, projector, vicreg_loss
from thermal_bc.pipeline import EMB_SIZE, load_thermogram

CV_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-dir", default=root_dir() / "data" / "dmrir" / "lab_database" / "database"
    )
    parser.add_argument("--init", required=True, help="VICReg ResNet-50 checkpoint to adapt.")
    parser.add_argument("--vicreg-dir", default="vicreg")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input", choices=INPUTS, default="masked")
    parser.add_argument("--seg-weights", default="weights/segmentation_r2attunet.pth")
    parser.add_argument("--seg-config", default="weights/segmentation_r2attunet.json")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(CV_FOLDS)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sim-coeff", type=float, default=25.0)
    parser.add_argument("--std-coeff", type=float, default=25.0)
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    return parser.parse_args()


class TwoViews(Dataset):
    """Two independently augmented views of each frame, in the same [0, 1] format as inference."""

    def __init__(self, paths, masks, mode):
        self.images = [
            encoder_input(load_thermogram(p), None if masks is None else m, mode)
            for p, m in zip(paths, masks if masks is not None else [None] * len(paths))
        ]
        self.augment = v2.Compose(
            [
                v2.RandomResizedCrop(EMB_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
                v2.RandomHorizontalFlip(),
                v2.RandomApply([v2.RandomRotation(10)], p=0.5),
                v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.5),
                v2.RandomApply([v2.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i]
        return self.augment(x).clamp(0, 1), self.augment(x).clamp(0, 1)


def main():
    args = parse_args()
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = list_dmrir_images(args.data_dir)
    groups = [patient_id_from_path(p) for p in paths]
    folds = patient_folds(labels, groups, n_splits=CV_FOLDS)

    masks = None
    if args.input != "full":
        masks = predict_breast_masks(
            paths, args.seg_weights, args.seg_config, device, Path("runs/cache/breast_masks.npz")
        )

    for fold in args.folds:
        torch.manual_seed(fold)
        train_idx, _ = folds[fold]  # patient_folds guarantees no test patient is used here
        loader = DataLoader(
            TwoViews(
                [paths[i] for i in train_idx],
                None if masks is None else [masks[i] for i in train_idx],
                args.input,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        backbone = load_vicreg_resnet50(args.init, args.vicreg_dir).to(device)
        head = projector().to(device)
        params = list(backbone.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs * len(loader)
        )
        scaler = torch.amp.GradScaler()
        start = time.time()
        for epoch in range(args.epochs):
            backbone.train(), head.train()
            losses = []
            for x1, x2 in loader:
                x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    z1, z2 = head(backbone(x1)), head(backbone(x2))
                loss = vicreg_loss(
                    z1.float(), z2.float(), args.sim_coeff, args.std_coeff, args.cov_coeff
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(loss.item())
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                print(
                    f"fold {fold} epoch {epoch + 1}/{args.epochs} loss {np.mean(losses):.3f} "
                    f"({(time.time() - start) / 60:.1f} min)",
                    flush=True,
                )
        torch.save(backbone.state_dict(), out_dir / f"fold{fold}.pth")


if __name__ == "__main__":
    main()
