import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes


def load_image(path):
    """Load image from path and return as RGB array (uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def remove_background(img):
    """
    Remove light background and shadows using Otsu on grayscale plus HSV-based
    refinement. Otsu separates bean (darker) from light background. HSV is used
    to remove achromatic regions: bright low-saturation (background) and dark
    low-saturation (shadows). Hole-filling and morphology refine the mask.
    Expects RGB image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bean_mask = (otsu_mask > 0).copy()

    # Use HSV to expand background removal (achromatic regions only)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    
    # Bright background: high value + low saturation (white/light gray surface)
    bright_bg = (v > 175) & (s < 56)
    
    # Shadow: low-to-mid value + low saturation (gray beneath beans, light gray patches)
    shadow = (v < 170) & (s < 56)
    
    # Remove these from bean mask (conservative: only clear background/shadow)
    bean_mask = bean_mask & ~bright_bg & ~shadow

    # Fill holes (e.g. reflections on bean surface), then morphology
    bean_mask = binary_fill_holes(bean_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bean_mask = cv2.morphologyEx(bean_mask, cv2.MORPH_CLOSE, kernel)
    bean_mask = cv2.morphologyEx(bean_mask, cv2.MORPH_OPEN, kernel)

    result = img.copy()
    result[bean_mask == 0] = 0
    
    return result


def _bean_mask(img, bg_threshold = 5):
    """Return boolean mask of bean pixels (exclude black background)."""
    return img.sum(axis = 2) > bg_threshold


def extract_features(img):
    """
    Extract features from processed coffee bean image (RGB).
    Uses bean pixels only (excludes black background).
    Returns: 1D numpy array.
    """
    mask = _bean_mask(img)
    if not mask.any():
        return np.full(15, np.nan, dtype = float)

    eps = 1e-6
    r = img[:, :, 0][mask]
    g = img[:, :, 1][mask]
    b = img[:, :, 2][mask]
    
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    std_r, std_g, std_b = r.std(), g.std(), b.std()

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0][mask]
    s = hsv[:, :, 1][mask]
    v = hsv[:, :, 2][mask]
    
    mean_h, mean_s, mean_v = h.mean(), s.mean(), v.mean()
    std_h, std_s, std_v = h.std(), s.std(), v.std()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_std = float(gray[mask].std())

    ratio_rg = mean_r / (mean_g + eps)
    ratio_bg = mean_b / (mean_g + eps)

    features_array = np.array([
        mean_r, mean_g, mean_b,
        std_r, std_g, std_b,
        mean_h, mean_s, mean_v,
        std_h, std_s, std_v,
        ratio_rg, ratio_bg,
        gray_std,
    ], dtype = float)

    return features_array


FEATURE_NAMES = [
    "mean_R", "mean_G", "mean_B",
    "std_R", "std_G", "std_B",
    "mean_H", "mean_S", "mean_V",
    "std_H", "std_S", "std_V",
    "ratio_RG", "ratio_BG",
    "gray_std",
]
