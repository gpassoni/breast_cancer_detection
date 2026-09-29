"""Frozen-encoder feature extraction for DMR-IR, with optional breast masking and caching.

Input variants (all in the [0, 1], 3-channel, 244x244 format used by the VICReg backbone):

* ``full``: the whole thermogram.
* ``masked``: pixels outside the R2AttU-Net breast mask set to 0.
* ``crop``: masked and cropped to the bounding box of both breasts.

Feature variants: global-average-pooled ``layer4`` (2048-d) or ``layer3+4`` (1024 + 2048-d).
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from thermal_bc.models.vicreg import load_vicreg_resnet50
from thermal_bc.pipeline import (
    EMB_SIZE,
    embedding_input,
    load_segmentation_model,
    load_thermogram,
    minmax,
    segmentation_input,
)

INPUTS = ("full", "masked", "crop")
LAYERS = ("layer4", "layer3+4")


def load_encoder(spec: str, vicreg_dir="vicreg"):
    """`imagenet`, `random` or a path to a VICReg ResNet-50 backbone checkpoint."""
    if spec in ("imagenet", "random"):
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2" if spec == "imagenet" else None)
        model.fc = torch.nn.Identity()
        return model
    return load_vicreg_resnet50(spec, vicreg_dir)


@torch.no_grad()
def predict_breast_masks(paths, seg_weights, seg_config, device, cache: Path | None = None):
    """Binary breast masks at each image's resolution, predicted by the segmentation model."""
    if cache is not None and cache.exists():
        cached = np.load(cache)
        if list(cached["paths"]) == [str(p) for p in paths]:
            return list(cached["masks"])
    model = load_segmentation_model(seg_weights, seg_config).to(device)
    masks = []
    for p in paths:
        image = load_thermogram(p)
        labels = model(segmentation_input(image).to(device)).argmax(1)[0].cpu().numpy()
        size = image.shape[::-1]
        masks.append(cv2.resize(labels.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, paths=[str(p) for p in paths], masks=np.stack(masks))
    return masks


def breast_crop(image, mask, margin=0.05):
    """Crop image and mask to the bounding box of the mask, plus a small margin."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return image, mask
    h, w = mask.shape
    dy, dx = int(margin * h), int(margin * w)
    y0, y1 = max(ys.min() - dy, 0), min(ys.max() + dy + 1, h)
    x0, x1 = max(xs.min() - dx, 0), min(xs.max() + dx + 1, w)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def encoder_input(image, mask, mode):
    """(3, 244, 244) float tensor in [0, 1] for one frame."""
    if mode == "full":
        return embedding_input(image)[0]
    if mode == "masked":
        return embedding_input(image, mask)[0]
    image, mask = breast_crop(image, mask)
    image = cv2.resize(minmax(image) * mask, EMB_SIZE[::-1], interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image)[None].expand(3, -1, -1).contiguous()


class EncoderInputs(Dataset):
    def __init__(self, paths, masks, mode):
        if mode not in INPUTS:
            raise ValueError(f"input must be one of {INPUTS}")
        if mode != "full" and masks is None:
            raise ValueError(f"input '{mode}' needs breast masks")
        self.paths, self.masks, self.mode = paths, masks, mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        mask = None if self.masks is None else self.masks[i]
        return encoder_input(load_thermogram(self.paths[i]), mask, self.mode)


@torch.no_grad()
def resnet_features(encoder, loader, device, layers="layer4"):
    """Global-average-pooled activations of a ResNet-50 (eval mode)."""
    encoder.eval().to(device)
    out = []
    for i, x in enumerate(loader):
        x = x.to(device)
        reference = encoder(x) if i == 0 else None
        if hasattr(encoder, "padding"):  # the VICReg ResNet pads the input before conv1
            x = encoder.padding(x)
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(x))))
        l3 = encoder.layer3(encoder.layer2(encoder.layer1(x)))
        l4 = encoder.layer4(l3)
        if reference is not None:  # the manual forward must reproduce the model's own forward
            assert torch.allclose(l4.mean((2, 3)), reference.flatten(1), atol=1e-4)
        pooled = [l4.mean((2, 3))] if layers == "layer4" else [l3.mean((2, 3)), l4.mean((2, 3))]
        out.append(torch.cat(pooled, 1).cpu().numpy())
    return np.concatenate(out)


def extract_features(paths, encoder_spec, mode, layers, device, masks=None, cache_dir=None):
    """Features for every path, cached on disk by (encoder, input, layers, image list)."""
    key = hashlib.sha1(
        "|".join([encoder_spec, mode, layers, *map(str, paths)]).encode()
    ).hexdigest()[:12]
    cache = (
        Path(cache_dir) / f"{Path(encoder_spec).stem}_{mode}_{layers}_{key}.npy"
        if cache_dir
        else None
    )
    if cache is not None and cache.exists():
        return np.load(cache)
    loader = DataLoader(EncoderInputs(paths, masks, mode), batch_size=64)
    features = resnet_features(load_encoder(encoder_spec), loader, device, layers)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, features)
    return features
