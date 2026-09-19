"""Module 6 — Coverage flagging.

A target row whose stratum has zero controls cannot be matched
(`no_control_available`). A stratum with *some* controls but fewer than the
effective minimum is `thin_stratum`: still matched, flagged as lower confidence.

Effective minimum = max(min_control_pool_size,
                        ceil(min_control_ratio * n_target_in_stratum))
when a ratio is given, else just the flat floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from rapidmatch.binning.get_stratum_counts import StratumCount


@dataclass(frozen=True)
class Coverage:
    """Stratum keys grouped by how usable their control pool is."""

    no_control: frozenset[str]
    thin: frozenset[str]
    eligible: frozenset[str]


def effective_min(
    n_target: int,
    min_control_pool_size: int,
    min_control_ratio: Optional[float],
) -> int:
    floor = min_control_pool_size
    if min_control_ratio is None:
        return floor
    return max(floor, math.ceil(min_control_ratio * n_target))


def flag_coverage(
    counts: list[StratumCount],
    min_control_pool_size: int = 5,
    min_control_ratio: Optional[float] = None,
) -> Coverage:
    """Classify each target-bearing stratum. Thin stays in `eligible`."""
    no_control: set[str] = set()
    thin: set[str] = set()
    eligible: set[str] = set()
    for row in counts:
        if row.n_target == 0:
            continue
        if row.n_control == 0:
            no_control.add(row.stratum)
            continue
        needed = effective_min(row.n_target, min_control_pool_size, min_control_ratio)
        if row.n_control < needed:
            thin.add(row.stratum)
        eligible.add(row.stratum)
    return Coverage(
        no_control=frozenset(no_control),
        thin=frozenset(thin),
        eligible=frozenset(eligible),
    )
