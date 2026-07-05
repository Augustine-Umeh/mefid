from typing import List, Optional

import numpy as np


def compute_adaptive_threshold(
    diffs: List[int],
    multiplier: float = 1.5,
    min_samples: int = 7,
) -> Optional[float]:
    """
    Per-scene threshold: mean(diffs) + multiplier * std(diffs).

    Returns None when there are fewer than min_samples diffs; callers should
    keep every frame in that case rather than applying a fixed fallback.
    """
    if len(diffs) < min_samples:
        return None

    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    threshold = mean + (multiplier * std)
    return max(threshold, 2.0)
