"""Module 12 — Drift diagnosis and trim-only correction.

For flagged *monitor* vars, find over-represented categories/bins in the
matched control, then drop the weakest-strength rows in that group until
the control share is back at the target's share.

Trim only — never swap in a replacement (see plan.md section 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rapidmatch.balance.checker import BalanceRow
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
    target: pd.DataFrame,
    matched_control: pd.DataFrame,
    pair_strength: dict[int, float],
    flagged: Sequence[BalanceRow],
    n_bins: int = 4,
) -> DriftCorrection:
    """Trim excess control rows. Weakest match_strength dropped first."""
    remaining = set(int(i) for i in matched_control["_rm_id"].tolist())
    events: list[TrimEvent] = []
    control = matched_control.copy()

    for row in flagged:
        if row.role != "monitor" or not row.flagged:
            continue
        if row.variable not in target.columns or row.variable not in control.columns:
            continue
        live = control[control["_rm_id"].isin(remaining)]
        if live.empty:
            break
        groups = excess_groups(target, live, row.variable, row.kind, n_bins=n_bins)
        for group in groups:
            live = control[control["_rm_id"].isin(remaining)]
            members = _members(live, target, group, n_bins)
            if members.empty:
                continue
            ordered = sorted(
                members["_rm_id"].astype(int).tolist(),
                key=lambda i: pair_strength.get(i, 0.0),
            )
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
    control: pd.DataFrame,
    target: pd.DataFrame,
    group: ExcessGroup,
    n_bins: int,
) -> pd.DataFrame:
    if group.kind == "js":
        mask = control[group.variable].astype(str) == str(group.group)
        return control.loc[mask]
    t_num = target[group.variable].to_numpy(dtype=np.float64)
    t_num = t_num[np.isfinite(t_num)]
    probs = [i / n_bins for i in range(1, n_bins)]
    edges = np.unique(np.quantile(t_num, probs)) if len(t_num) else np.array([])
    bins = np.digitize(
        control[group.variable].to_numpy(dtype=np.float64), edges, right=True
    )
    return control.loc[bins == int(group.group)]
