"""Two-sample Kolmogorov–Smirnov statistic (numeric).

KS = max |F_left(x) - F_right(x)| over the pooled support. NaNs dropped.
"""

from __future__ import annotations

import numpy as np


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return 0.0
    x = np.sort(x)
    y = np.sort(y)
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))
