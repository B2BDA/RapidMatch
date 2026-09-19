"""Identify over-represented groups on a flagged monitor_var.

Numeric vars are split into quantile bins of the *target* so we can point
at a concrete bin rather than a continuous range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExcessGroup:
    variable: str
    kind: str
    group: Any
    target_share: float
    control_share: float
    excess_rows: int


def excess_groups(
    target: pd.DataFrame,
    control: pd.DataFrame,
    variable: str,
    kind: str,
    n_bins: int = 4,
) -> list[ExcessGroup]:
    """Groups in control whose share exceeds the target's share."""
    if kind == "js":
        t_vals = target[variable].astype(str)
        c_vals = control[variable].astype(str)
        labels = sorted(set(t_vals) | set(c_vals))
        t_share = t_vals.value_counts(normalize=True)
        c_share = c_vals.value_counts(normalize=True)
        n_control = len(c_vals)
        out = []
        for lab in labels:
            ts = float(t_share.get(lab, 0.0))
            cs = float(c_share.get(lab, 0.0))
            excess = int(np.floor(max(0.0, (cs - ts) * n_control)))
            if excess > 0:
                out.append(ExcessGroup(variable, kind, lab, ts, cs, excess))
        return sorted(out, key=lambda g: g.excess_rows, reverse=True)

    t_num = target[variable].to_numpy(dtype=np.float64)
    t_num = t_num[np.isfinite(t_num)]
    if len(t_num) == 0 or len(control) == 0:
        return []
    probs = [i / n_bins for i in range(1, n_bins)]
    edges = np.unique(np.quantile(t_num, probs))
    t_bins = np.digitize(target[variable].to_numpy(dtype=np.float64), edges, right=True)
    c_bins = np.digitize(control[variable].to_numpy(dtype=np.float64), edges, right=True)
    n_control = len(c_bins)
    n_obs_bins = int(c_bins.max()) + 1 if len(c_bins) else 0
    out = []
    for b in range(max(n_bins, n_obs_bins)):
        ts = float((t_bins == b).mean()) if len(t_bins) else 0.0
        cs = float((c_bins == b).mean()) if len(c_bins) else 0.0
        excess = int(np.floor(max(0.0, (cs - ts) * n_control)))
        if excess > 0:
            out.append(ExcessGroup(variable, kind, int(b), ts, cs, excess))
    return sorted(out, key=lambda g: g.excess_rows, reverse=True)
