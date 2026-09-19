"""Module 7 — Pair scoring inside one stratum.

Steps, in order:
1. z-score numeric match_vars with GLOBAL target mean/std
2. multiply by user weights (default 1)
3. weighted Euclidean distance
4. match_strength = exp(-distance), always in (0, 1]

Missing-flag columns are intentionally absent from `numeric_vars`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rapidmatch.config import MatchConfig


def score_pairs(
    target_ids: np.ndarray,
    control_ids: np.ndarray,
    target_x: np.ndarray,
    control_x: np.ndarray,
    numeric_vars: Sequence[str],
    config: MatchConfig,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All target x control pairs in one stratum.

    Returns parallel arrays (target_id, control_id, strength).
    """
    if len(target_ids) == 0 or len(control_ids) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)

    weights = np.array([config.weight_for(v) for v in numeric_vars], dtype=np.float64)
    # Constant columns would divide by zero; treat them as already standardized.
    std = np.where(target_std == 0, 1.0, target_std)
    zt = (target_x - target_mean) / std
    zc = (control_x - target_mean) / std
    if weights.size:
        zt = zt * weights
        zc = zc * weights
        delta = zt[:, None, :] - zc[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
    else:
        # Categorical-only strata: every pair in the cell is equally close.
        dist = np.zeros((len(target_ids), len(control_ids)), dtype=np.float64)
    strength = np.exp(-dist)

    t_idx, c_idx = np.meshgrid(
        np.arange(len(target_ids)), np.arange(len(control_ids)), indexing="ij"
    )
    return (
        target_ids[t_idx.ravel()],
        control_ids[c_idx.ravel()],
        strength.ravel(),
    )


def target_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std of the target numeric matrix (axis=0)."""
    if values.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    return mean, std
