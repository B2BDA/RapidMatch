"""Module 8 — Global greedy matching without replacement.

All candidate pairs (every eligible stratum) are pooled, sorted by
match_strength descending, then walked:

- skip if the control row is already used
- skip if the target already has `n` matches
- otherwise assign, rank 1 = strongest kept match for that target

No Hungarian algorithm, no ML. Deterministic given the same pair list
(mergesort keeps ties stable).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np


def greedy_match(
    target_ids: np.ndarray,
    control_ids: np.ndarray,
    strengths: np.ndarray,
    n: int = 1,
    pbar: Optional[Any] = None,
) -> list[tuple[int, int, float, int]]:
    """Return (target_id, control_id, strength, match_rank) assignments."""
    if len(target_ids) == 0:
        return []
    order = np.argsort(-strengths, kind="mergesort")
    used_control: set[int] = set()
    target_slots: dict[int, int] = defaultdict(int)
    assigned: list[tuple[int, int, float, int]] = []
    if pbar is not None:
        pbar.total = len(order)
    for batch_start in range(0, len(order), 4096):
        for i in order[batch_start : batch_start + 4096]:
            t = int(target_ids[i])
            c = int(control_ids[i])
            if c in used_control:
                continue
            if target_slots[t] >= n:
                continue
            rank = target_slots[t] + 1
            assigned.append((t, c, float(strengths[i]), rank))
            target_slots[t] = rank
            used_control.add(c)
        if pbar is not None:
            pbar.update(min(4096, len(order) - batch_start))
    return assigned
