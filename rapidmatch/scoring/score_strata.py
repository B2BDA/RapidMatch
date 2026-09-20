"""Score every eligible stratum, optionally in parallel (opt-in).

Serial by default. When `n_workers >= 1` the strata are scored in a
`ThreadPoolExecutor`; `map` preserves input order, so results are identical
to serial regardless of worker count. This is pure CPU work released to the
GIL by NumPy, so threads are the right parallelism primitive here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Sequence

import numpy as np

from rapidmatch.config import MatchConfig
from rapidmatch.scoring.scorer import score_pairs


def score_all_strata(
    strata: Sequence[str],
    strata_labels: np.ndarray,
    ids: np.ndarray,
    treatment: np.ndarray,
    x: np.ndarray,
    numeric_vars: Sequence[str],
    config: MatchConfig,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    n_workers: Optional[int] = None,
    pbar: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score all `strata`; return concatenated (target_ids, control_ids, strengths).

    `strata_labels` aligns with `ids`/`treatment`/`x`; each label is a stratum
    key. Only rows whose mask matches `stratum in strata` are scored per unit.
    """
    if not strata:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)

    wanted = set(strata)
    target_idx: dict[str, list[int]] = {s: [] for s in wanted}
    control_idx: dict[str, list[int]] = {s: [] for s in wanted}
    for i, key in enumerate(strata_labels):
        if key not in wanted:
            continue
        if treatment[i] == 1:
            target_idx[key].append(i)
        elif treatment[i] == 0:
            control_idx[key].append(i)

    def _score(stratum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ti = np.asarray(target_idx[stratum], dtype=np.int64)
        ci = np.asarray(control_idx[stratum], dtype=np.int64)
        return score_pairs(
            ids[ti],
            ids[ci],
            x[ti],
            x[ci],
            numeric_vars,
            config,
            target_mean,
            target_std,
        )

    if n_workers is not None and n_workers > 1:
        with ThreadPoolExecutor(max_workers=int(n_workers)) as pool:
            results = list(pool.map(_score, strata))
    else:
        results = [_score(s) for s in strata]

    if pbar is not None:
        pbar.update(len(results))

    pair_t: list[np.ndarray] = []
    pair_c: list[np.ndarray] = []
    pair_s: list[np.ndarray] = []
    for t_ids, c_ids, strengths in results:
        if len(t_ids):
            pair_t.append(t_ids)
            pair_c.append(c_ids)
            pair_s.append(strengths)
    if not pair_t:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    return (
        np.concatenate(pair_t),
        np.concatenate(pair_c),
        np.concatenate(pair_s),
    )
