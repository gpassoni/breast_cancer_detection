"""Image preprocessing utilities."""
import numpy as np
import cv2
from PIL import Image


def split_left_right_breasts(mask: np.ndarray) -> np.ndarray:
    """
    Split breast mask into left and right regions.
    
    Args:
        mask: Binary mask of breast regions
        
    Returns:
        Labeled mask where 1=left breast, 2=right breast
    """
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


def load_image(path: str) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        path: Path to image file (.png, .jpg, .jpeg, or .npy)
        
    Returns:
        Image as float32 numpy array
    """
    if path.lower().endswith('.npy'):
        arr = np.load(path)
    else:
        arr = np.array(Image.open(path).convert('L'))
    return arr.astype(np.float32)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    img = img.astype(np.float32)
    img_min, img_max = np.min(img), np.max(img)
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img


def preprocess_image(img_path: str, height: int = 224, width: int = 224) -> np.ndarray:
    """
    Load and preprocess a thermal image for model inference.
    
    Args:
        img_path: Path to image file
        height: Target height
        width: Target width
        
    Returns:
        Preprocessed image as numpy array with shape (1, height, width)
    """
    img = load_image(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = normalize_image(img)
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    return img
