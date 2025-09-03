import os
import numpy as np
import cv2
from PIL import Image
from scripts.utils.utils import masks_merger
from pathlib import Path
from dotenv import load_dotenv
import os 

load_dotenv()
ROOT_DIR = Path(os.getenv("ROOT_DIR"))

data_path = ROOT_DIR / "data"
images_path = data_path / "images"
image_numpy_path = data_path / "images_numpy"
masks_merged_path = data_path / "mask_merged"
masks_splitted_path = data_path / "masks"
masks_preprocessed_path = data_path / "mask_preprocessed"

masks_merger(data_path)

images_files = sorted(os.listdir(images_path))
masks_files = sorted(os.listdir(masks_merged_path))

id_counter = 0
for image_file, mask_file in zip(images_files, masks_files):
    image_file_path = images_path / image_file
    mask_file_path = masks_merged_path / mask_file

    new_image_file_path = image_numpy_path / f"img_{id_counter}.npy"
    new_mask_file_path = masks_preprocessed_path / f"mask_{id_counter}.npy"

    if os.path.exists(new_image_file_path):
        os.remove(new_image_file_path)
    if os.path.exists(new_mask_file_path):
        os.remove(new_mask_file_path)

    id_counter += 1

    if image_file_path.endswith(".png"):
        image = Image.open(image_file_path) 
        if image is None:
            print(f"Error: {image_file_path} is None")
            continue

        image = np.array(image)
        np.save(new_image_file_path, image)

    if mask_file_path.endswith(".png"):
        mask = Image.open(mask_file_path)
        if mask is None:
            print(f"Error: {mask_file_path} is None")
            continue

        mask = np.array(mask)
        np.save(new_mask_file_path, mask)
    elif mask_file_path.endswith(".npy"):
        mask = np.load(mask_file_path)
        np.save(new_mask_file_path, mask)
