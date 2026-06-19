import numpy as np
from typing import List


def compute_adaptive_threshold(
    diffs: List[int],
    multiplier: float = 1.5,
    fallback: int = 10,
    min_samples: int = 5,
) -> float:
    """
    Per-scene threshold: mean(diffs) + multiplier * std(diffs).

    Falls back to a fixed threshold when there are too few diffs for stats.
    """
    if len(diffs) < min_samples:
        return float(fallback)

    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    threshold = mean + (multiplier * std)
    return max(threshold, 2.0)
