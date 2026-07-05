import cv2
import imagehash
import numpy as np
from PIL import Image  # pyright: ignore[reportMissingImports]
from typing import List


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR frame to PIL Image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def compute_phash(frame: np.ndarray, hash_size: int = 8) -> imagehash.ImageHash:
    """Compute perceptual hash for a single frame."""
    pil_image = frame_to_pil(frame)
    return imagehash.phash(pil_image, hash_size=hash_size)


def compute_hamming_distance(
    hash_a: imagehash.ImageHash, hash_b: imagehash.ImageHash
) -> int:
    """Hamming distance between two perceptual hashes (0 = identical)."""
    return hash_a - hash_b


def compute_consecutive_diffs(hashes: List[imagehash.ImageHash]) -> List[int]:
    """Compute Hamming distances between consecutive hashes."""
    if len(hashes) < 2:
        return []
    return [
        compute_hamming_distance(hashes[i], hashes[i + 1]) for i in range(len(hashes) - 1)
    ]
