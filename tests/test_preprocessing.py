import albumentations as A
import numpy as np
import torch

from thermal_bc.data.thermal_dataset import ThermalDataset, split_left_right_breasts
from thermal_bc.features import breast_crop, encoder_input
from thermal_bc.pipeline import EMB_SIZE, SEG_SIZE, embedding_input, minmax, segmentation_input


def test_minmax_handles_constant_images():
    out = minmax(np.full((8, 8), 3.0, np.float32))
    assert np.isfinite(out).all() and out.max() == 0


def test_model_inputs_have_expected_format():
    image = (np.random.default_rng(0).random((120, 160)) * 255).astype(np.uint8)
    seg = segmentation_input(image)
    emb = embedding_input(image)
    assert seg.shape == (1, 1, *SEG_SIZE) and seg.min() >= 0 and seg.max() <= 1
    assert emb.shape == (1, 3, *EMB_SIZE) and torch.equal(emb[0, 0], emb[0, 2])


def test_masked_input_is_zero_outside_the_mask():
    image = np.random.default_rng(0).random((100, 100)).astype(np.float32) + 0.1
    mask = np.zeros((100, 100), bool)
    mask[:, :50] = True
    x = encoder_input(image, mask, "masked")
    assert x.shape == (3, *EMB_SIZE)
    assert x[:, :, EMB_SIZE[1] // 2 + 2 :].abs().max() == 0
    assert x[:, :, : EMB_SIZE[1] // 2 - 2].min() >= 0


def test_breast_crop_uses_the_mask_bounding_box():
    image = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    mask = np.zeros((100, 100), bool)
    mask[40:60, 20:80] = True
    crop_img, crop_mask = breast_crop(image, mask, margin=0)
    assert crop_img.shape == (20, 60) and crop_mask.all()


def test_left_right_labels_follow_image_side():
    mask = np.zeros((20, 40), np.uint8)
    mask[5:15, 2:12] = 1
    mask[5:15, 25:38] = 1
    labels = split_left_right_breasts(mask)
    assert labels[10, 5] == 1 and labels[10, 30] == 2


def test_side_labels_are_assigned_after_flip(tmp_path):
    image = np.random.default_rng(0).random((20, 40)).astype(np.float32)
    mask = np.zeros((20, 40), np.float32)
    mask[5:15, 2:12] = 1
    mask[5:15, 25:38] = 1
    (tmp_path / "img").mkdir()
    (tmp_path / "mask").mkdir()
    np.save(tmp_path / "img" / "a.npy", image)
    np.save(tmp_path / "mask" / "a.npy", mask)
    flip = A.Compose([A.HorizontalFlip(p=1.0)], additional_targets={"mask": "mask"})
    dataset = ThermalDataset(
        tmp_path / "img",
        tmp_path / "mask",
        height=20,
        width=40,
        multiclass=True,
        global_transform=flip,
    )
    _, labels = dataset[0]
    left_x = torch.nonzero(labels == 1)[:, 1].float().mean()
    right_x = torch.nonzero(labels == 2)[:, 1].float().mean()
    assert left_x < right_x
