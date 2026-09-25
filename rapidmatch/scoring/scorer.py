"""Module 7 — Pair scoring inside one stratum.

Steps, in order:
1. z-score numeric match_vars with GLOBAL target mean/std
2. multiply by user weights (default 1)
3. weighted Euclidean distance
4. match_strength = exp(-distance), always in (0, 1]
5. optional per-target cap, keeping only the K closest controls

Step 5 is opt-in via `config.max_candidates_per_target`. It bounds retained
memory on very large datasets without changing which pair ranks first: a
target's single nearest control is always inside the cap.

Missing-flag columns are intentionally absent from `numeric_vars`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from rapidmatch.config import MatchConfig

_MAX_DISTANCE_CELLS = 16_000_000


def _prune_to_cap(
    strength_block: np.ndarray,
    target_block_ids: np.ndarray,
    control_ids: np.ndarray,
    cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the `cap` highest-strength controls for every row of the block.

    Falls through to the full cross-product when the block already has at
    most `cap` controls, so small strata keep their exact ordering.
    """
    n_rows, n_c = strength_block.shape
    if n_c <= cap:
        return (
            np.repeat(target_block_ids, n_c),
            np.tile(control_ids, n_rows),
            strength_block.ravel(),
        )
    keep = np.argpartition(strength_block, n_c - cap, axis=1)[:, n_c - cap :]
    flat_keep = keep.ravel()
    rows = np.repeat(np.arange(n_rows), cap)
    return (
        target_block_ids[rows],
        control_ids[flat_keep],
        strength_block[rows, flat_keep],
    )


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

    n_t = len(target_ids)
    n_c = len(control_ids)
    weights = np.array([config.weight_for(v) for v in numeric_vars], dtype=np.float64)
    # Constant columns would divide by zero; treat them as already standardized.
    std = np.where(target_std == 0, 1.0, target_std)
    zt = (target_x - target_mean) / std
    zc = (control_x - target_mean) / std
    cap = config.max_candidates_per_target
    if not weights.size:
        # Categorical-only strata: every pair in the cell is equally close.
        dist = np.zeros((n_t, n_c), dtype=np.float64)
        strength = np.exp(-dist)
        if cap is not None:
            return _prune_to_cap(strength, target_ids, control_ids, cap)
        return (
            np.repeat(target_ids, n_c),
            np.tile(control_ids, n_t),
            strength.ravel(),
        )

    zt = zt * weights
    zc = zc * weights
    n_dim = int(zt.shape[1])
    block_t = n_t
    if n_t * n_c * n_dim > _MAX_DISTANCE_CELLS:
        block_t = max(1, _MAX_DISTANCE_CELLS // (n_c * max(n_dim, 1)))

    if block_t >= n_t:
        delta = zt[:, None, :] - zc[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        strength = np.exp(-dist)
        if cap is not None:
            return _prune_to_cap(strength, target_ids, control_ids, cap)
        return (
            np.repeat(target_ids, n_c),
            np.tile(control_ids, n_t),
            strength.ravel(),
        )

    t_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    for start in range(0, n_t, block_t):
        end = min(start + block_t, n_t)
        zt_block = zt[start:end]
        delta = zt_block[:, None, :] - zc[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        strength = np.exp(-dist)
        if cap is not None:
            bt, bc, bs = _prune_to_cap(
                strength, target_ids[start:end], control_ids, cap
            )
            t_parts.append(bt)
            c_parts.append(bc)
            s_parts.append(bs)
            continue
        t_parts.append(np.repeat(target_ids[start:end], n_c))
        c_parts.append(np.tile(control_ids, end - start))
        s_parts.append(strength.ravel())
    return (
        np.concatenate(t_parts),
        np.concatenate(c_parts),
        np.concatenate(s_parts),
    )


def target_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population std of the target numeric matrix (axis=0)."""
    if values.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)
    return mean, std
