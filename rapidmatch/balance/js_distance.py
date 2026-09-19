"""Jensen–Shannon distance between two categorical samples.

JS divergence is 0.5 KL(P||M) + 0.5 KL(Q||M) with M = 0.5(P+Q).
We return the divergence itself (bounded [0, ln 2]), not the sqrt form.
Empty inputs yield 0.
"""

from __future__ import annotations

import numpy as np


def js_distance(left: np.ndarray, right: np.ndarray) -> float:
    """JS divergence of empirical category distributions of `left` vs `right`."""
    if len(left) == 0 or len(right) == 0:
        return 0.0
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    labels = np.unique(np.concatenate([left, right]))
    p = np.array([(left == lab).mean() for lab in labels], dtype=np.float64)
    q = np.array([(right == lab).mean() for lab in labels], dtype=np.float64)
    mid = 0.5 * (p + q)
    return 0.5 * _kl(p, mid) + 0.5 * _kl(q, mid)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))
