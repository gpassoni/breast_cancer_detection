import cv2 
import numpy as np
from skimage.restoration import denoise_wavelet

def normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    return mask

def uint_to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def resize_array(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

def reshape_array(image: np.ndarray, height: int, width: int) -> np.ndarray:
    right_shape = (1, height, width)
    if image.shape == right_shape:
        return image

    resized = resize_array(image, width, height)

    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=0)
    
    if resized.shape == right_shape:
        return resized
    else:
        print(f'Error: the shape of the resized is {resized.shape} and not {right_shape}')
        return None

def normalize_image(image):
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def percentile_clipping_normalize(image, lower_percentile=5, upper_percentile=99):
    p1, p99 = np.percentile(image, [lower_percentile, upper_percentile])
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)  
    return image 

def remove_bad_pixels(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img.astype(np.float32), ksize)

def denoise_wavelet_light(img: np.ndarray,
                          wavelet: str = 'db1',
                          method: str = 'BayesShrink',
                          mode: str = 'hard',
                          rescale_sigma: bool = True) -> np.ndarray:
    den = denoise_wavelet(img,
                          wavelet=wavelet,
                          method=method,
                          mode=mode,
                          rescale_sigma=rescale_sigma)
    return den.astype(np.float32)

def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:

    orig_dtype = img.dtype
    if orig_dtype == np.float32 or orig_dtype == np.float64:
        mn, mx = img.min(), img.max()
        img_uint8 = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    else:
        img_uint8 = img.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out_uint8 = clahe.apply(img_uint8)

    if orig_dtype == np.float32 or orig_dtype == np.float64:
        out = out_uint8.astype(np.float32) / 255.0 * (mx - mn) + mn
    else:
        out = out_uint8

    return out

def global_dog_sharpen(img: np.ndarray,
                       k1: int = 5, sigma1: float = 1.0,
                       k2: int = 9, sigma2: float = 4.0,
                       amount: float = 2.0) -> np.ndarray:

    orig_dtype = img.dtype
    if orig_dtype != np.uint8:
        base = (img.astype(np.float32) * 255.0).copy()
    else:
        base = img.astype(np.float32).copy()

    g1 = cv2.getGaussianKernel(k1, sigma1)
    g2 = cv2.getGaussianKernel(k2, sigma2)
    G1 = g1 @ g1.T
    G2 = g2 @ g2.T

    blur1 = cv2.filter2D(base, -1, G1)
    blur2 = cv2.filter2D(base, -1, G2)
    dog   = blur1 - blur2

    out = base + amount * dog
    out = np.clip(out, 0, 255)

    if orig_dtype != np.uint8:
        out = (out / 255.0).astype(np.float32)
    else:
        out = out.round().astype(np.uint8)

    return out

def preprocess_raw_image(image):
    image = normalize_image(image)
    image = percentile_clipping_normalize(image)
    image = remove_bad_pixels(image)
    image = denoise_wavelet_light(image)
    image = apply_clahe(image)
    image = global_dog_sharpen(image)
    return image
