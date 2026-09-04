import cv2
import numpy as np


def compute_blur_score(image_path: str) -> float:
    """
    Compute a sharpness score for an image using the variance of the
    Laplacian. Lower values indicate a blurrier image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return float(laplacian.var())


def is_sharp_enough(image_path: str, threshold: float = 100.0) -> bool:
    """
    Return True if the image's blur score meets or exceeds the given
    threshold, indicating the image is sharp enough to use as training
    data.
    """
    return compute_blur_score(image_path) >= threshold