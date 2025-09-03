import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

def split_left_right_breasts(mask: np.ndarray) -> np.ndarray:
    mask = np.array(mask).astype(np.uint8)
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labeled = cv2.connectedComponents(mask_bin)
    n_comp = num_labels - 1

    out = np.zeros_like(labeled, dtype=np.int64)
    if n_comp < 2:
        out[labeled > 0] = 1
        return out

    areas = [(i, np.sum(labeled == i)) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    comp1, comp2 = areas[0][0], areas[1][0]

    def centroid_x(label_id):
        ys, xs = np.where(labeled == label_id)
        return xs.mean() if xs.size > 0 else 0

    c1, c2 = centroid_x(comp1), centroid_x(comp2)
    left_comp, right_comp = (comp1, comp2) if c1 < c2 else (comp2, comp1)

    out[labeled == left_comp] = 1
    out[labeled == right_comp] = 2
    return out


class ThermalDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str = None,
        height: int = 240,
        width: int = 320,
        multiclass: bool = False,
        global_transform=None,
        heavy_transform=None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.height = height
        self.width = width
        self.multiclass = multiclass
        self.global_transform = global_transform
        self.heavy_transform = heavy_transform

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))
        ])
        if masks_dir:
            self.mask_files = sorted([
                f for f in os.listdir(masks_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.npy'))
            ])
        else:
            self.mask_files = None

    def __len__(self):
        return len(self.image_files)

    def load_image(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npy'):
            arr = np.load(path)
        else:
            arr = np.array(Image.open(path).convert('L'))
        return arr.astype(np.float32)

    def __getitem__(self, idx: int):
        name = os.path.splitext(self.image_files[idx])[0]
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = self.load_image(img_path)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        img = np.array(img, dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        if self.mask_files:
            mask_path = os.path.join(self.masks_dir, self.mask_files[idx])
            mask = self.load_image(mask_path)
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
            if self.multiclass:
                mask = split_left_right_breasts(mask)
        else:
            mask = None

        if self.global_transform:
            img = img * 255.0
            img = img.astype(np.uint8)

            if mask is not None:
                out = self.global_transform(image=img, mask=mask)
                img, mask = out['image'], out['mask']
            else:
                img = self.global_transform(image=img)['image']

            if self.heavy_transform:
                if mask is not None:
                    out = self.heavy_transform(image=img, mask=mask)
                    img, mask = out['image'], out['mask']
                else:
                    img = self.heavy_transform(image=img)['image']
            
            img = img / 255.0
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = img.astype(np.float32)
            img = np.clip(img, 0, 1)

        img = np.expand_dims(img, axis=0)
        img_t = torch.from_numpy(img).float()
        if mask is not None:
            mask_t = torch.from_numpy(mask.astype(
                np.int64 if self.multiclass else np.float32
            ))
        else:
            mask_t = torch.zeros(
                (self.height, self.width),
                dtype=torch.float32
            )
        return img_t, mask_t
