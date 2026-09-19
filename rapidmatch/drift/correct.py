"""Module 12 — Drift diagnosis and trim-only correction.

For flagged *monitor* vars, find over-represented categories/bins in the
matched control, then drop the weakest-strength rows in that group until
the control share is back at the target's share.

Trim only — never swap in a replacement (see plan.md section 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa

from rapidmatch.balance.checker import BalanceRow
from rapidmatch.balance.columns import float_values, label_values
from rapidmatch.drift.diagnose import ExcessGroup, excess_groups


@dataclass
class TrimEvent:
    variable: str
    kind: str
    group: Any
    n_removed: int
    target_share: float
    control_share_before: float
    control_rm_ids: list[int]


@dataclass
class DriftCorrection:
    kept_control_ids: set[int]
    events: list[TrimEvent] = field(default_factory=list)


def correct_drift(
    target: pa.Table,
    matched_control: pa.Table,
    pair_strength: Mapping[int, float],
    flagged: Sequence[BalanceRow],
    n_bins: int = 4,
) -> DriftCorrection:
    """Trim excess control rows. Weakest match_strength dropped first."""
    remaining = set(int(i) for i in _ids(matched_control))
    events: list[TrimEvent] = []
    control = matched_control

    for row in flagged:
        if row.role != "monitor" or not row.flagged:
            continue
        if row.variable not in target.column_names or row.variable not in control.column_names:
            continue
        live = _filter_ids(control, remaining)
        if live.num_rows == 0:
            break
        groups = excess_groups(target, live, row.variable, row.kind, n_bins=n_bins)
        for group in groups:
            live = _filter_ids(control, remaining)
            if live.num_rows == 0:
                break
            mask = _members(live, target, group, n_bins)
            if not np.any(mask):
                continue
            ordered = sorted(
                int(i) for i in _ids(live)[mask]
            )
            ordered.sort(key=lambda i: pair_strength.get(i, 0.0))
            drop = ordered[: group.excess_rows]
            for i in drop:
                remaining.discard(i)
            events.append(
                TrimEvent(
                    variable=group.variable,
                    kind=group.kind,
                    group=group.group,
                    n_removed=len(drop),
                    target_share=group.target_share,
                    control_share_before=group.control_share,
                    control_rm_ids=drop,
                )
            )
    return DriftCorrection(kept_control_ids=remaining, events=events)


def _members(
    control: pa.Table,
    target: pa.Table,
    group: ExcessGroup,
    n_bins: int,
) -> np.ndarray:
    """Boolean mask over `control` rows belonging to `group`."""
    if group.kind == "js":
        return label_values(control, group.variable) == str(group.group)
    t_num = float_values(target, group.variable)
    t_num = t_num[np.isfinite(t_num)]
    probs = [i / n_bins for i in range(1, n_bins)]
    edges = np.unique(np.quantile(t_num, probs)) if len(t_num) else np.array([])
    bins = np.digitize(float_values(control, group.variable), edges, right=True)
    return bins == int(group.group)


def _ids(table: pa.Table) -> np.ndarray:
    return np.asarray(
        table.column("_rm_id").to_numpy(zero_copy_only=False), dtype=np.int64
    )


def _filter_ids(table: pa.Table, remaining: set[int]) -> pa.Table:
    if not remaining:
        return table.slice(0, 0)
    values = np.fromiter(remaining, dtype=np.int64, count=len(remaining))
    mask = np.isin(_ids(table), values)
    return table.filter(mask)