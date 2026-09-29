"""Prepare the segmentation training data.

1. Merge per-breast masks (data/masks/MD, data/masks/ME) into data/mask_merged/*.npy.
2. Save images and merged masks as consecutively numbered .npy files in
   data/images_numpy/img_<i>.npy and data/mask_preprocessed/mask_<i>.npy.
"""

import numpy as np
from PIL import Image

from thermal_bc.config import root_dir
from thermal_bc.utils import masks_merger


def main():
    data_path = root_dir() / "data"
    images_path = data_path / "images"
    masks_merged_path = data_path / "mask_merged"
    image_numpy_path = data_path / "images_numpy"
    masks_preprocessed_path = data_path / "mask_preprocessed"
    image_numpy_path.mkdir(exist_ok=True)
    masks_preprocessed_path.mkdir(exist_ok=True)

    masks_merger(data_path)

    image_files = sorted(images_path.iterdir())
    mask_files = sorted(masks_merged_path.iterdir())

    for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        new_image_path = image_numpy_path / f"img_{idx}.npy"
        new_mask_path = masks_preprocessed_path / f"mask_{idx}.npy"
        new_image_path.unlink(missing_ok=True)
        new_mask_path.unlink(missing_ok=True)

        if image_file.suffix == ".png":
            np.save(new_image_path, np.array(Image.open(image_file)))

        if mask_file.suffix == ".png":
            np.save(new_mask_path, np.array(Image.open(mask_file)))
        elif mask_file.suffix == ".npy":
            np.save(new_mask_path, np.load(mask_file))


if __name__ == "__main__":
    main()
