"""Module 9 — Global percentile cutoff on match_strength.

`tolerance` is a quantile of the *accepted* pair strengths.
0.0 keeps every assignment; 0.8 drops the weakest 80% of pairs.

A target whose every assignment is dropped becomes `below_tolerance`.
A target that still has at least one kept pair stays `matched`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_tolerance(
    assignments: Sequence[tuple[int, int, float, int]],
    tolerance: float,
) -> tuple[list[tuple[int, int, float, int]], set[int], float]:
    """Return (kept_assignments, uncovered_target_ids, cutoff_value)."""
    if not assignments:
        return [], set(), 0.0
    strengths = np.array([a[2] for a in assignments], dtype=np.float64)
    cutoff = float(np.quantile(strengths, tolerance))
    kept = [a for a in assignments if a[2] >= cutoff]
    kept_targets = {a[0] for a in kept}
    below = {a[0] for a in assignments if a[0] not in kept_targets}
    return kept, below, cutoff
