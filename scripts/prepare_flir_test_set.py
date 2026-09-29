"""Prepare the external FLIR test set used to evaluate segmentation models.

Input (under ROOT_DIR/data/test Flir/):
    processed/*.png                       thermal images
    labels/left_breast/*.tiff, labels/right_breast/*.tiff   per-breast masks
Output (under ROOT_DIR/data/test_flir/):
    images/<name>.npy, masks/<name>_merged.npy   (binary breast mask)
"""

import numpy as np
from PIL import Image

from thermal_bc.config import root_dir


def main():
    data_dir = root_dir() / "data"
    source = data_dir / "test Flir"
    images_out = data_dir / "test_flir" / "images"
    masks_out = data_dir / "test_flir" / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    for image_path in sorted((source / "processed").glob("*.png")):
        np.save(images_out / f"{image_path.stem}.npy", np.array(Image.open(image_path)))

    right_masks = sorted((source / "labels" / "right_breast").iterdir())
    left_masks = sorted((source / "labels" / "left_breast").iterdir())
    for right_path, left_path in zip(right_masks, left_masks):
        right_mask = np.array(Image.open(right_path))
        left_mask = np.array(Image.open(left_path))
        if right_mask.size != left_mask.size:
            print(f"Shapes do not match for {right_path.name} and {left_path.name}")
            continue

        merged_mask = np.zeros_like(right_mask, dtype=np.uint8)
        merged_mask[right_mask > 0] = 1
        merged_mask[left_mask > 0] = 1
        name = right_path.name.split(".")[0]
        np.save(masks_out / f"{name}_merged.npy", merged_mask)


if __name__ == "__main__":
    main()
