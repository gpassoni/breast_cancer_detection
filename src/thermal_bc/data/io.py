"""File I/O for the clinical FLIR data: per-breast mask merging and radiometric conversion."""

import os

import numpy as np
from PIL import Image


def load_tiff(file_path: str) -> np.ndarray:
    return np.array(Image.open(file_path))


def extract_base_mask_name(filename: str) -> str:
    """Strip all extensions and the `_ME` / `_MD` (left / right breast) suffix."""
    name = filename
    while "." in name:
        name = os.path.splitext(name)[0]
    if name.endswith(("_ME", "_MD")):
        name = name[:-3]
    return name


def masks_merger(path_data: str):
    """Merge per-breast masks (masks/MD = right, masks/ME = left) into data/mask_merged/*.npy."""
    masks_path = os.path.join(path_data, "masks")
    right_mask_path = os.path.join(masks_path, "MD")
    left_mask_path = os.path.join(masks_path, "ME")
    merged_masks_path = os.path.join(path_data, "mask_merged")
    os.makedirs(merged_masks_path, exist_ok=True)

    right_masks_files = sorted(os.listdir(right_mask_path))
    left_masks_files = sorted(os.listdir(left_mask_path))

    for right_mask_file, left_mask_file in zip(right_masks_files, left_masks_files):
        right_name = extract_base_mask_name(right_mask_file)
        left_name = extract_base_mask_name(left_mask_file)
        if right_name != left_name:
            print(f"Mismatch: {right_name} != {left_name}")
            continue

        right_mask = load_tiff(os.path.join(right_mask_path, right_mask_file))
        left_mask = load_tiff(os.path.join(left_mask_path, left_mask_file))
        np.save(os.path.join(merged_masks_path, f"{right_name}.npy"), right_mask + left_mask)
        print(f"Saved: {right_name}.npy")


def convert_raw_flir_to_thermal(raw_data_path):
    """Read a radiometric FLIR JPEG and return its temperature map in degrees Celsius."""
    import flyr

    try:
        image = flyr.unpack(raw_data_path).celsius
    except Exception as e:
        print(f"Error while converting {raw_data_path}: {e}")
        return np.array([])
    return np.array(image)


def convert_raw_flir_to_numpy(input_dir: str, output_dir: str):
    """Convert every radiometric FLIR JPEG in `input_dir` to a .npy temperature map."""
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".jpg"):
            continue
        img_path = os.path.join(input_dir, filename)
        temperatures = convert_raw_flir_to_thermal(img_path)
        if temperatures.size == 0:
            print(f"Empty image: {img_path}")
            continue
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".npy")
        np.save(output_path, temperatures)
        print(f"Saved: {output_path}")
