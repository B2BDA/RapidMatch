"""Module 9 — Global percentile cutoff on match_strength.

`tolerance` is a quantile of the *accepted* pair strengths.
0.0 keeps every assignment; 0.8 drops the weakest 80% of pairs, except that
the strongest assignment for a target is retained so quality filtering does not
silently remove that target from the full-target analysis population.

A target whose every assignment falls below the cutoff keeps its strongest
assignment. The caller can expose that assignment as low quality using its
strength relative to the returned cutoff.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def apply_tolerance(
    assignments: Sequence[tuple[int, int, float, int]],
    tolerance: float,
    preserve_primary: bool = True,
) -> tuple[list[tuple[int, int, float, int]], set[int], float]:
    """Return (kept_assignments, uncovered_target_ids, cutoff_value).

    When ``preserve_primary`` is true, every target that received an assignment
    keeps its strongest unique control even when that assignment is below the
    global cutoff. This preserves coverage without allowing control reuse.
    """
    if not assignments:
        return [], set(), 0.0
    strengths = np.array([a[2] for a in assignments], dtype=np.float64)
    cutoff = float(np.quantile(strengths, tolerance))
    kept = [a for a in assignments if a[2] >= cutoff]
    kept_targets = {a[0] for a in kept}
    if preserve_primary:
        best_by_target: dict[int, tuple[int, int, float, int]] = {}
        for assignment in assignments:
            target_id = assignment[0]
            current = best_by_target.get(target_id)
            if current is None or assignment[2] > current[2]:
                best_by_target[target_id] = assignment
        kept.extend(
            assignment
            for target_id, assignment in best_by_target.items()
            if target_id not in kept_targets
        )
        kept_targets = {a[0] for a in kept}
    below = {a[0] for a in assignments if a[0] not in kept_targets}
    return kept, below, cutoff
