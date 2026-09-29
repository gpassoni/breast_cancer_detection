"""End-to-end inference pipeline: raw thermogram -> breast segmentation -> breast-only embedding.

The embedding is the input of the classification heads in `thermal_bc.evaluation`. Every stage
reproduces the preprocessing its model was trained with:

* Segmentation (R2AttU-Net): grayscale frame resized to 384x256 (INTER_AREA), then min-max
  normalised to [0, 1] (as in `ThermalDataset`). Output: 3-class logits (background, left
  breast, right breast), decoded with argmax.
* Embedding (VICReg ResNet-50): grayscale frame resized to 244x244 (INTER_AREA), min-max
  normalised to [0, 1] and replicated to 3 channels, with the background outside the predicted
  breast mask set to 0.
"""

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import torch

from thermal_bc.models.r2attunet import R2AttU_Net
from thermal_bc.models.vicreg import load_vicreg_resnet50

SEG_SIZE = (256, 384)  # (height, width) used to train the segmentation model
EMB_SIZE = (244, 244)  # (height, width) used for the classification experiments


def load_thermogram(path) -> np.ndarray:
    """Load a thermal frame as a 2-D array.

    Supports temperature matrices (.txt, decimal comma allowed) and NumPy arrays (.npy), returned
    as float32, and grayscale or gray-as-RGB images (.jpg/.png), returned as uint8 so that each
    stage can reproduce the exact dtype handling of its training dataset.
    """
    path = str(path)
    if path.endswith(".txt"):
        with open(path) as f:
            content = f.read().replace(",", ".")
        image = np.genfromtxt(StringIO(content))
    elif path.endswith(".npy"):
        image = np.load(path)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        return image
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel frame, got shape {image.shape} for {path}")
    return image


def minmax(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant image maps to zeros instead of NaN."""
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    return (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)


def resize(image: np.ndarray, size, interpolation=cv2.INTER_AREA) -> np.ndarray:
    height, width = size
    return cv2.resize(image, (width, height), interpolation=interpolation)


def segmentation_input(image: np.ndarray) -> torch.Tensor:
    """(1, 1, 256, 384) tensor in [0, 1], matching `ThermalDataset` (float, resize, normalise)."""
    return torch.from_numpy(minmax(resize(image.astype(np.float32), SEG_SIZE)))[None, None]


def embedding_input(image: np.ndarray, mask: np.ndarray | None = None) -> torch.Tensor:
    """(1, 3, 244, 244) tensor in [0, 1] (resized in the native dtype, then normalised), with
    the background zeroed if a mask is given."""
    x = minmax(resize(image, EMB_SIZE))
    if mask is not None:
        x = x * resize(mask.astype(np.uint8), EMB_SIZE, cv2.INTER_NEAREST)
    return torch.from_numpy(x)[None, None].expand(-1, 3, -1, -1).contiguous()


def load_segmentation_model(weights, config) -> R2AttU_Net:
    cfg = json.loads(Path(config).read_text())
    model = R2AttU_Net(
        img_ch=1, output_ch=3, t=cfg["t"], base_filters=cfg["base_filters"], depth=cfg["depth"]
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.eval()


@dataclass
class PipelineOutput:
    labels: np.ndarray  # (H, W) int: 0 background, 1 left breast, 2 right breast (input size)
    breast_mask: np.ndarray  # (H, W) bool
    embedding: np.ndarray  # (2048,)


class ThermalPipeline:
    """Segment the breasts in a thermogram and embed the breast-only image with the backbone.

    >>> pipe = ThermalPipeline(seg_weights, seg_config, backbone_weights)
    >>> out = pipe("frame.jpg")  # out.labels, out.breast_mask, out.embedding
    """

    def __init__(self, seg_weights, seg_config, backbone_weights, vicreg_dir="vicreg", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # TF32 convolutions change logits by up to ~0.07 on Ampere GPUs; evaluate in full FP32.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.segmenter = load_segmentation_model(seg_weights, seg_config).to(self.device)
        self.backbone = load_vicreg_resnet50(backbone_weights, vicreg_dir).eval().to(self.device)

    @torch.no_grad()
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Label map at the input resolution."""
        logits = self.segmenter(segmentation_input(image).to(self.device))
        labels = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return resize(labels, image.shape, cv2.INTER_NEAREST)

    @torch.no_grad()
    def embed(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.backbone(embedding_input(image, mask).to(self.device))[0].cpu().numpy()

    def __call__(self, path, mask_background=True) -> PipelineOutput:
        image = load_thermogram(path)
        labels = self.segment(image)
        breast = labels > 0
        embedding = self.embed(image, breast if mask_background else None)
        return PipelineOutput(labels=labels, breast_mask=breast, embedding=embedding)
