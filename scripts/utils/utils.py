from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.measure import label, regionprops
import flyr

def plot_image_grey_scale(image: np.ndarray):
    image = np.array(image)
    plt.imshow(image, cmap='gray')
    plt.show()

def plot_image_rgb(image: np.ndarray):
    image = np.array(image)
    plt.imshow(image, cmap='jet')
    plt.show()

def load_tiff(file_path: str) -> np.ndarray:
    mask = Image.open(file_path)
    return np.array(mask)

def load_png(file_path: str) -> np.ndarray:
    image = Image.open(file_path)
    return np.array(image)

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(image)
    image = image.resize((width, height))
    return np.array(image)

def crop_breasts(image, mask):
    bin_mask = (mask > 0).astype(np.uint8)
    labeled_mask = label(bin_mask)
    props = regionprops(labeled_mask)
    
    props_sorted = sorted(props, key=lambda region: region.bbox[1])
    
    cropped_images = []
    cropped_masks = []
    
    for region in props_sorted:
        minr, minc, maxr, maxc = region.bbox
        cropped_img = image[minr:maxr, minc:maxc]
        cropped_msk = mask[minr:maxr, minc:maxc]
        cropped_images.append(cropped_img)
        cropped_masks.append(cropped_msk)
    
    n_cropped_images = len(cropped_images)
    n_cropped_masks = len(cropped_masks)
    if n_cropped_images > 2:
        cropped_images = sorted(cropped_images, key=lambda x: x.size, reverse=True)[:2]
        cropped_masks = sorted(cropped_masks, key=lambda x: x.size, reverse=True)[:2]

    return cropped_images, cropped_masks

def extract_base_mask_name(filename: str) -> str:
    name_wo_ext = filename
    while '.' in name_wo_ext:
        name_wo_ext = os.path.splitext(name_wo_ext)[0]

    if name_wo_ext.endswith('_ME') or name_wo_ext.endswith('_MD'):
        name_wo_ext = name_wo_ext[:-3]

    return name_wo_ext

def masks_merger(path_data: str):
    masks_path = os.path.join(path_data, 'masks')
    right_mask_path = os.path.join(masks_path, 'MD')
    left_mask_path = os.path.join(masks_path, 'ME')
    merged_masks_path = os.path.join(path_data, 'mask_merged')

    if not os.path.exists(merged_masks_path):
        os.makedirs(merged_masks_path)

    right_masks_files = sorted(os.listdir(right_mask_path))
    left_masks_files = sorted(os.listdir(left_mask_path))

    for right_mask_file, left_mask_file in zip(right_masks_files, left_masks_files):
        right_name = extract_base_mask_name(right_mask_file)
        left_name = extract_base_mask_name(left_mask_file)

        if right_name != left_name:
            print(f"Mismatch: {right_name} ≠ {left_name}")
            continue

        right_mask = load_tiff(os.path.join(right_mask_path, right_mask_file))
        left_mask = load_tiff(os.path.join(left_mask_path, left_mask_file))
        merged_mask = right_mask + left_mask

        save_path = os.path.join(merged_masks_path, f"{right_name}.npy")
        np.save(save_path, merged_mask)
        print(f"✅ Salvata: {right_name}.npy")

def convert_jpg_to_npy(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            img_np = np.array(img)
            
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_dir, output_filename)
            
            np.save(output_path, img_np)
            print(f"Salvato: {output_path}")

def convert_raw_flir_to_thermal(raw_data_path):
    try:
        image = flyr.unpack(raw_data_path)
        image = image.celsius
    except Exception as e:
        print(f"Errore durante la conversione di {raw_data_path}: {e}")
        return np.array([])
    return np.array(image)

def convert_raw_flir_to_numpy(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img_np = convert_raw_flir_to_thermal(img_path)
            
            if img_np.size == 0:
                print(f"Immagine vuota: {img_path}")
                continue
            output_filename = os.path.splitext(filename)[0] + ".npy"
            output_path = os.path.join(output_dir, output_filename)
            
            np.save(output_path, img_np)
            print(f"Salvato: {output_path}")
