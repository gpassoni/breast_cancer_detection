import pytest
import torch
import torchvision

from thermal_bc.features import resnet_features
from thermal_bc.models.r2attunet import R2AttU_Net
from thermal_bc.models.vicreg import projector, vicreg_loss


@pytest.mark.parametrize("depth", [3, 5, 6])
def test_r2attunet_output_shape(depth):
    model = R2AttU_Net(img_ch=1, output_ch=3, t=2, base_filters=8, depth=depth).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 64, 96))
    assert out.shape == (2, 3, 64, 96)


def test_vicreg_loss_rewards_invariance_and_penalises_collapse():
    torch.manual_seed(0)
    z = torch.randn(64, 32)
    same = vicreg_loss(z, z)
    assert same < vicreg_loss(z, torch.randn(64, 32))
    assert vicreg_loss(torch.zeros(64, 32), torch.zeros(64, 32)) > same


def test_focal_tversky_loss_prefers_correct_masks():
    from thermal_bc.models.losses import FocalTverskyLoss

    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    loss = FocalTverskyLoss(alpha=0.6, beta=0.4, gamma=1.5)
    perfect = loss(target * 20 - 10, target)
    inverted = loss(10 - target * 20, target)
    assert perfect < 0.01 < inverted


def test_projector_shapes():
    head = projector((16, 32, 32, 8)).eval()
    assert head(torch.randn(4, 16)).shape == (4, 8)


def test_resnet_features_match_forward():
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    x = torch.rand(2, 3, 64, 64)
    feats = resnet_features(model, [x], "cpu", layers="layer3+4")
    assert feats.shape == (2, 1024 + 2048)
    with torch.no_grad():
        reference = model.eval()(x).numpy()
    assert abs(feats[:, 1024:] - reference).max() < 1e-4
