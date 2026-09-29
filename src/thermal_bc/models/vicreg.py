"""VICReg objective and projector (Bardes et al., ICLR 2022), following the official code."""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def projector(dims=(2048, 2048, 2048, 2048)):
    """MLP expander: Linear-BN-ReLU blocks followed by a final linear layer."""
    layers = []
    for d_in, d_out in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
    return nn.Sequential(*layers)


def off_diagonal(m):
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    """Invariance (MSE) + variance (hinge on per-dimension std) + covariance (decorrelation)."""
    repr_loss = F.mse_loss(z1, z2)
    z1, z2 = z1 - z1.mean(0), z2 - z2.mean(0)
    std1, std2 = torch.sqrt(z1.var(0) + 1e-4), torch.sqrt(z2.var(0) + 1e-4)
    std_loss = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean()) / 2
    n, d = z1.shape
    cov1, cov2 = (z1.T @ z1) / (n - 1), (z2.T @ z2) / (n - 1)
    cov_loss = (off_diagonal(cov1).pow(2).sum() + off_diagonal(cov2).pow(2).sum()) / d
    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def load_vicreg_resnet50(checkpoint_path, vicreg_dir="vicreg"):
    """Load a ResNet-50 pretrained with VICReg.

    `vicreg_dir` must contain `resnet.py` from https://github.com/facebookresearch/vicreg,
    and `checkpoint_path` the backbone weights saved by its `main_vicreg.py`.
    """
    sys.path.insert(0, str(vicreg_dir))
    import resnet  # noqa: E402  (provided by the official VICReg repository)

    backbone, _ = resnet.__dict__["resnet50"](zero_init_residual=True)
    backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return backbone
