import numpy as np
from numba import njit

@njit
def histogram_figure_numba(np_img):
    r_bars = np.zeros(256, dtype=np.int64)
    g_bars = np.zeros(256, dtype=np.int64)
    b_bars = np.zeros(256, dtype=np.int64)

    for h in range(np_img.shape[0]):
        for w in range(np_img.shape[1]):
            r_bars[np_img[h, w, 0]] += 1
            g_bars[np_img[h, w, 1]] += 1
            b_bars[np_img[h, w, 2]] += 1

    return r_bars, g_bars, b_bars


def compute_stats_and_entropy_from_hist(r_bars, g_bars, b_bars, total_pixels):
    pixel_values = np.arange(256)
    
    means = []
    modes = []
    stds = []
    entropies = []
    maxs = []
    mins = []
    
    channels_bars = [r_bars, g_bars, b_bars]
    
    for bars in channels_bars:
        modes.append(int(np.argmax(bars)))
        
        mean_val = np.sum(pixel_values * bars) / total_pixels
        means.append(float(mean_val))
        
        variance = np.sum(bars * (pixel_values - mean_val) ** 2) / total_pixels
        stds.append(float(np.sqrt(variance)))
        
        active_indices = np.where(bars > 0)[0]
        mins.append(int(active_indices[0]) if len(active_indices) > 0 else 0)
        maxs.append(int(active_indices[-1]) if len(active_indices) > 0 else 0)
        
        probs = bars / total_pixels
        nonzero = probs[probs > 0]
        entropy_val = -np.sum(nonzero * np.log2(nonzero))
        entropies.append(float(entropy_val))
        
    stats = {
        'mean': tuple(means),
        'mode': tuple(modes),
        'std':  tuple(stds),
        'max':  tuple(maxs),
        'min':  tuple(mins),
    }
    
    return stats, tuple(entropies)


def linear_transform(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    transformed = alpha * frame.astype(np.float64) + beta
    return np.clip(transformed, 0, 255).astype(np.uint8)


import cv2


def apply_filter(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (21, 21), sigmaX=1.0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


try:
    import mediapipe as mp

    _selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
        model_selection=0
    )
    _SEGMENTATION_AVAILABLE = True
except Exception:
    _SEGMENTATION_AVAILABLE = False


def segment_background(frame: np.ndarray, blur_ksize: int = 55) -> np.ndarray:
    if not _SEGMENTATION_AVAILABLE:
        return frame

    frame.flags.writeable = False
    results = _selfie_seg.process(frame)
    frame.flags.writeable = True

    prob_mask = results.segmentation_mask
    person_mask = (prob_mask > 0.5)

    person_mask_3ch = np.stack([person_mask] * 3, axis=-1)

    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred_background = cv2.GaussianBlur(frame, (ksize, ksize), sigmaX=0)

    output = np.where(person_mask_3ch, frame, blurred_background)

    return output.astype(np.uint8)


def histogram_equalization(frame: np.ndarray, r_bars, g_bars, b_bars) -> np.ndarray:
    total_pixels = frame.shape[0] * frame.shape[1]
    output = np.empty_like(frame)
    
    channels_bars = [r_bars, g_bars, b_bars]

    for ch in range(3):
        channel = frame[:, :, ch]
        hist = channels_bars[ch]

        cdf = hist.cumsum()

        cdf_min = int(cdf.min())
        denominator = total_pixels - cdf_min
        if denominator == 0:
            lut = np.zeros(256, dtype=np.uint8)
        else:
            lut = np.round((cdf - cdf_min) / denominator * 255).astype(np.uint8)

        output[:, :, ch] = lut[channel]

    return output
