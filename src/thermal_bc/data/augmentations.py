"""Albumentations pipelines used for segmentation training and the augmentation ablation."""

import albumentations as A
import cv2

# Light augmentation applied to every training and validation sample.
global_transform = A.Compose(
    [
        A.GaussianBlur(blur_limit=3, p=0.4),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.4),
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
            p=0.3,
        ),
    ],
    additional_targets={"mask": "mask"},
)


def _geometric_structural(crop_size):
    return [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7,
        ),
        A.RandomResizedCrop(size=crop_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=30.0, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.OpticalDistortion(distort_limit=0.1, p=0.3),
        A.CoarseDropout(
            num_holes_range=[1, 3],
            hole_height_range=[30, 50],
            hole_width_range=[30, 50],
            fill=0,
            p=0.3,
        ),
    ]


def _photometric_degradation(downscale_range):
    return [
        A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.5),
        A.MotionBlur(blur_limit=(3, 5), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomToneCurve(scale=0.3, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
        A.InvertImg(p=0.1),
        A.Downscale(scale_range=downscale_range, p=0.3),
    ]


def geometric_structural_transform(crop_size=(256, 384)):
    """Flips, affine and elastic warps, random crops and cut-out."""
    return A.Compose(_geometric_structural(crop_size), additional_targets={"mask": "mask"})


def photometric_degradation_transform(downscale_range=(0.6, 0.8)):
    """Noise, blur, contrast/gamma changes, inversion and resolution loss."""
    return A.Compose(_photometric_degradation(downscale_range), additional_targets={"mask": "mask"})


def full_transform(crop_size=(256, 384), downscale_range=(0.6, 0.8)):
    """Geometric/structural followed by photometric degradation."""
    return A.Compose(
        _geometric_structural(crop_size) + _photometric_degradation(downscale_range),
        additional_targets={"mask": "mask"},
    )
