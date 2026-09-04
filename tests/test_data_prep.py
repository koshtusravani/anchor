import numpy as np
import cv2
import pytest

from data_prep.filters import compute_blur_score, is_sharp_enough


@pytest.fixture
def sharp_image(tmp_path):
    """Create a synthetic sharp image with high-frequency detail."""
    path = tmp_path / "sharp.png"
    image = np.zeros((200, 200), dtype=np.uint8)
    image[::2, :] = 255  # alternating stripes create high-frequency edges
    cv2.imwrite(str(path), image)
    return str(path)


@pytest.fixture
def blurry_image(tmp_path):
    """Create a synthetic flat image with no edges, simulating blur."""
    path = tmp_path / "blurry.png"
    image = np.full((200, 200), 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def test_compute_blur_score_detects_sharp_image(sharp_image):
    score = compute_blur_score(sharp_image)
    assert score > 0


def test_compute_blur_score_flat_image_is_near_zero(blurry_image):
    score = compute_blur_score(blurry_image)
    assert score < 1.0


def test_is_sharp_enough_true_for_sharp_image(sharp_image):
    assert is_sharp_enough(sharp_image, threshold=100.0) is True


def test_is_sharp_enough_false_for_blurry_image(blurry_image):
    assert is_sharp_enough(blurry_image, threshold=100.0) is False