"""Structural matching capacity, before candidates are scored or pruned."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from rapidmatch._sql import quote_ident
from rapidmatch.binning.get_stratum_counts import StratumCount
from rapidmatch.config import MatchConfig
from rapidmatch.coverage.flag_coverage import effective_min


class InsufficientStratumCapacityWarning(UserWarning):
    """The current grouping cannot supply the requested unique controls."""


@dataclass
class CapacityReport:
    summary: dict[str, Any]
    strata: pa.Table
    bin_edges: dict[str, list[float]]
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    trials: pa.Table = field(default_factory=lambda: pa.table({}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "strata": self.strata.to_pylist(),
            "bin_edges": self.bin_edges,
            "recommendations": self.recommendations,
            "trials": self.trials.to_pylist(),
        }


def build_capacity(
    counts: list[StratumCount],
    config: MatchConfig,
    bin_edges: dict[str, list[float]],
    *,
    max_report_strata: int = 1000,
    con=None,
    numeric: list[str] | None = None,
    emit_warning: bool = False,
) -> CapacityReport:
    """Counts give upper bounds, not promises of post-match coverage."""
    if isinstance(max_report_strata, bool) or not isinstance(max_report_strata, int) or max_report_strata < 1:
        raise ValueError("max_report_strata must be a positive integer")
    n_target = sum(r.n_target for r in counts)
    n_control = sum(r.n_control for r in counts)
    rows = []
    for r in counts:
        if not r.n_target:
            continue
        rows.append({
            "stratum": r.stratum,
            "n_target": r.n_target,
            "n_control": r.n_control,
            "requested_assignments": config.n * r.n_target,
            "assignment_capacity_ceiling": min(config.n * r.n_target, r.n_control),
            "assignment_shortfall": max(0, config.n * r.n_target - r.n_control),
            "targets_with_one_control_ceiling": min(r.n_target, r.n_control),
            "fully_supplied_targets_ceiling": min(r.n_target, r.n_control // config.n),
            "no_control": r.n_control == 0,
            "thin_stratum": 0 < r.n_control < effective_min(
                r.n_target, config.min_control_pool_size, config.min_control_ratio
            ),
            "group_values": "{}",
        })
    rows.sort(key=lambda r: (-r["assignment_shortfall"], r["stratum"]))
    keys = (
        "requested_assignments", "assignment_capacity_ceiling", "assignment_shortfall",
        "targets_with_one_control_ceiling", "fully_supplied_targets_ceiling",
    )
    summary = {k: sum(r[k] for r in rows) for k in keys}
    summary.update({
        "n_target": n_target, "n_control": n_control, "n": config.n,
        "n_bins": config.n_bins, "stratify_vars": list(config.grouping_vars),
        "n_target_strata": len(rows),
        "n_deficient_strata": sum(r["assignment_shortfall"] > 0 for r in rows),
        "n_targets_in_deficient_strata": sum(r["n_target"] for r in rows if r["assignment_shortfall"]),
        "global_assignment_shortfall": max(0, config.n * n_target - n_control),
        "coverage_ceiling": summary["targets_with_one_control_ceiling"] / n_target if n_target else 0.0,
        "full_ratio_coverage_ceiling": summary["fully_supplied_targets_ceiling"] / n_target if n_target else 0.0,
        "candidate_pairs": sum(r.n_target * r.n_control for r in counts),
        "largest_stratum_pairs": max((r.n_target * r.n_control for r in counts), default=0),
        "retained_candidate_pairs": sum(
            r.n_target * min(r.n_control, config.max_candidates_per_target)
            if config.max_candidates_per_target is not None else r.n_target * r.n_control
            for r in counts
        ),
        "strata_truncated": len(rows) > max_report_strata,
        "reported_strata": min(len(rows), max_report_strata),
    })
    detail = rows[:max_report_strata]
    if con is not None and detail:
        _describe_groups(con, detail, config, bin_edges, numeric or [])
    if con is not None:
        _group_diagnostics(con, summary, config, bin_edges, numeric or [])
    report = CapacityReport(summary, pa.Table.from_pylist(detail), bin_edges)
    report.recommendations = _recommendations(summary, config, bin_edges)
    if emit_warning and summary["assignment_shortfall"]:
        warnings.warn(
            f"{summary['n_deficient_strata']:,}/{len(rows):,} target strata cannot "
            f"supply 1:{config.n} unique controls. At most "
            f"{summary['fully_supplied_targets_ceiling']:,}/{n_target:,} targets "
            f"can receive the full ratio; assignment shortfall "
            f"{summary['assignment_shortfall']:,}. Inspect report.capacity. "
            "Consider fewer numeric bins or fewer hard grouping variables; use "
            "ControlMatcher.assess for measured capacity/work comparisons. "
            "A global shortage requires more eligible controls.",
            InsufficientStratumCapacityWarning,
            stacklevel=2,
        )
    return report


def _group_diagnostics(con, summary, config, edges, numeric):
    variables = list(config.grouping_vars)
    expressions = [
        f"COUNT(DISTINCT {quote_ident('_bin_' + v if v in edges else v)})"
        for v in variables
    ]
    expressions += [f"COUNT(*) FILTER (WHERE {quote_ident('is_missing_' + v)})" for v in numeric]
    values = con.execute("SELECT " + ", ".join(expressions) + " FROM v_stratified").fetchone() if expressions else []
    summary["grouping_cardinality"] = dict(zip(variables, map(int, values[:len(variables)])))
    summary["numeric_missing_counts"] = dict(zip(numeric, map(int, values[len(variables):])))


def _describe_groups(con, rows, config, edges, numeric):
    columns = [f"_bin_{v}" for v in edges]
    columns += [v for v in config.grouping_vars if v not in numeric]
    columns += [f"is_missing_{v}" for v in numeric]
    con.register("_rm_capacity_keys", pa.table({"_stratum": [r["stratum"] for r in rows]}))
    try:
        projections = ", ".join(f"ANY_VALUE({quote_ident(v)})" for v in columns)
        values = con.execute(
            "SELECT _stratum" + (", " + projections if projections else "")
            + " FROM v_stratified SEMI JOIN _rm_capacity_keys USING (_stratum) GROUP BY _stratum"
        ).fetchall()
    finally:
        con.unregister("_rm_capacity_keys")
    lookup = {}
    for key, *vals in values:
        groups = dict(zip(columns, vals))
        for var, cuts in edges.items():
            idx = int(groups.pop(f"_bin_{var}"))
            groups[var] = {
                "bin": idx,
                "lower_exclusive": cuts[idx - 1] if idx else None,
                "upper_inclusive": cuts[idx] if idx < len(cuts) else None,
            }
        lookup[key] = json.dumps(groups, default=str)
    for row in rows:
        row["group_values"] = lookup[row["stratum"]]


def _recommendations(summary, config, edges):
    out = []
    if summary["global_assignment_shortfall"]:
        out.append({
            "setting": "control_pool", "proposed_value": config.n * summary["n_target"],
            "reason": "The total eligible pool is smaller than the requested assignments.",
            "evidence": "measured_global_counts",
            "tradeoff": "More controls are necessary; feature overlap can still limit coverage.",
        })
    if summary["assignment_shortfall"] and edges and config.n_bins > 2:
        out.append({
            "setting": "n_bins", "proposed_value": max(2, config.n_bins // 2),
            "reason": "Coarser numeric grouping may reduce local capacity deficits; assess it first.",
            "evidence": "untested_suggestion",
            "tradeoff": "Larger strata can increase pairwise work and weaken feature balance.",
        })
    if summary["assignment_shortfall"]:
        out.append({
            "setting": "stratify_vars", "proposed_value": None,
            "reason": "Review optional numeric grouping constraints, categorical cardinality, and missingness patterns.",
            "evidence": "untested_suggestion",
            "tradeoff": "Numeric features can remain in distance scoring; categorical and missingness constraints remain hard.",
        })
    return out


def add_trials(report: CapacityReport, trials: list[dict[str, Any]]) -> None:
    report.trials = pa.Table.from_pylist(trials)
    current = report.summary
    better = [t for t in trials if t["full_ratio_coverage_ceiling"] > current["full_ratio_coverage_ceiling"]]
    if better:
        best = min(better, key=lambda t: (-t["full_ratio_coverage_ceiling"], t["candidate_pairs"]))
        report.recommendations.insert(0, {
            "setting": "n_bins", "proposed_value": best["n_bins"],
            "reason": "Highest full-ratio capacity among the tested settings (pair count breaks ties).",
            "evidence": "measured_preflight",
            "coverage_ceiling": best["full_ratio_coverage_ceiling"],
            "candidate_pairs": best["candidate_pairs"],
            "tradeoff": "Capacity is an upper bound; recheck final coverage and feature balance after matching.",
        })
    if trials and max(t["full_ratio_coverage_ceiling"] for t in trials) < 1:
        report.recommendations.append({
            "setting": "n_bins", "proposed_value": None,
            "reason": "No tested bin setting supports the full requested ratio.",
            "evidence": "measured_preflight",
            "tradeoff": "Consider additional eligible controls or a user-approved change to hard grouping.",
        })
