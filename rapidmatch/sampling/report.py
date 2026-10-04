"""Bounded sampling artifacts and full-population acceptance summaries."""

from dataclasses import dataclass
from typing import Any

import pyarrow as pa


@dataclass
class DownsampleReport:
    summary: dict[str, Any]
    classes: pa.Table
    strata: pa.Table
    balance: pa.Table
    bin_edges: dict[str, list[float]]
    recommendations: list[dict[str, Any]]

    def to_dict(self):
        return {
            "summary": self.summary, "classes": self.classes.to_pylist(),
            "strata": self.strata.to_pylist(), "balance": self.balance.to_pylist(),
            "bin_edges": self.bin_edges, "recommendations": self.recommendations,
        }


@dataclass
class DownsampleResult:
    sample: pa.Table
    report: DownsampleReport
    row_ids: pa.ChunkedArray


def build_sampling_report(con, config, n_population, edges):
    limit = config.max_report_rows
    statuses = dict(con.execute(
        "SELECT status, COUNT(*) FROM _rs_balance WHERE required GROUP BY status"
    ).fetchall())
    status = (
        "fail" if statuses.get("fail", 0) else
        "not_evaluated" if statuses.get("not_evaluated", 0) or not statuses else "pass"
    )
    n_classes, omitted_classes, omitted_class_rows = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE quota = 0),
            COALESCE(SUM(n_population) FILTER (WHERE quota = 0), 0) FROM _rs_classes
    """).fetchone()
    n_strata, omitted_strata, omitted_rows = con.execute("""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE quota = 0),
            COALESCE(SUM(n_population) FILTER (WHERE quota = 0), 0) FROM _rs_quotas
    """).fetchone()
    n_checks = int(con.execute("SELECT COUNT(*) FROM _rs_balance").fetchone()[0])
    classes = con.sql(f"""
        SELECT c._rs_label AS label, c.n_population, c.ideal_quota,
            c.quota AS allocated_rows, COALESCE(s.actual, 0)::BIGINT AS sampled_rows,
            c.n_population::DOUBLE / {n_population} AS population_share,
            COALESCE(s.actual, 0)::DOUBLE / {config.sample_size} AS sample_share,
            c.quota - c.ideal_quota AS rounding_deviation
        FROM _rs_classes c LEFT JOIN (
            SELECT _rs_label, COUNT(*) AS actual FROM _rs_sample GROUP BY _rs_label
        ) s USING (_rs_label)
        ORDER BY c.quota = 0 DESC, c.n_population DESC, c._rs_label LIMIT {limit}
    """).to_arrow_table()
    strata = con.sql(f"""
        SELECT q._rs_label AS label, q._rs_group AS feature_group,
            q.n_population, q.n_class, q.class_quota, q.ideal_quota,
            q.quota AS allocated_rows, COALESCE(s.actual, 0)::BIGINT AS sampled_rows,
            q.n_population::DOUBLE / {n_population} AS population_share,
            COALESCE(s.actual, 0)::DOUBLE / {config.sample_size} AS sample_share,
            q.n_population::DOUBLE / q.n_class AS within_class_population_share,
            COALESCE(s.actual, 0)::DOUBLE / NULLIF(q.class_quota, 0) AS within_class_sample_share
        FROM _rs_quotas q LEFT JOIN (
            SELECT _rs_label, _rs_group, COUNT(*) AS actual
            FROM _rs_sample GROUP BY _rs_label, _rs_group
        ) s USING (_rs_label, _rs_group)
        ORDER BY q.quota = 0 DESC, q.n_population DESC, q._rs_label, q._rs_group LIMIT {limit}
    """).to_arrow_table()
    balance = con.sql(f"""
        SELECT * FROM _rs_balance ORDER BY
            CASE status WHEN 'fail' THEN 0 WHEN 'not_evaluated' THEN 1 ELSE 2 END,
            required DESC, variable, scope, label LIMIT {limit}
    """).to_arrow_table()
    summary = {
        "n_population": n_population, "requested_rows": config.sample_size,
        "sampled_rows": config.sample_size, "count_invariants_passed": True,
        "class_budgets_passed": True if config.label_col else None,
        "random_state": config.random_state, "label_col": config.label_col,
        "stratify_vars": list(config.stratify_vars), "check_vars": list(config.check_vars),
        "checked_features": list(config.features), "check_by_label": config.check_by_label,
        "n_bins": config.n_bins, "ks_threshold": config.ks_threshold,
        "js_threshold": config.js_threshold, "balance_status": status,
        "feature_checks": sum(statuses.values()), "checks_passed": statuses.get("pass", 0),
        "checks_failed": statuses.get("fail", 0), "checks_not_evaluated": statuses.get("not_evaluated", 0),
        "n_classes": n_classes if config.label_col else 0,
        "omitted_classes": omitted_classes if config.label_col else 0,
        "omitted_class_population_share": omitted_class_rows / n_population if config.label_col else 0.0,
        "n_strata": n_strata, "omitted_strata": omitted_strata,
        "omitted_stratum_population_share": omitted_rows / n_population,
        "classes_truncated": n_classes > limit, "strata_truncated": n_strata > limit,
        "balance_truncated": n_checks > limit, "max_report_rows": limit,
    }
    recommendations = []
    if omitted_strata:
        recommendations.append({
            "setting": "sample_size", "proposed_value": None,
            "reason": f"{omitted_strata} feature groups received zero rows after proportional rounding.",
            "evidence": "measured_allocation",
            "tradeoff": "A larger user-requested sample can improve representation but needs more output memory.",
        })
    if config.label_col and omitted_classes:
        recommendations.append({
            "setting": "sample_size", "proposed_value": None,
            "reason": f"{omitted_classes} classes received zero outer budgets; feature-bin changes cannot add rows to them.",
            "evidence": "measured_allocation",
            "tradeoff": "Review a larger size; guaranteeing every rare class would be a different allocation policy.",
        })
    if omitted_strata and edges and config.n_bins > 2:
        recommendations.append({
            "setting": "n_bins", "proposed_value": max(2, config.n_bins // 2),
            "reason": "Coarser feature bins may reduce zero-allocation groups within fixed class budgets.",
            "evidence": "untested_suggestion",
            "tradeoff": "Coarser groups can hide distribution differences; recheck feature balance.",
        })
    if status != "pass":
        recommendations.append({
            "setting": "stratify_vars" if status == "fail" else "check_vars",
            "proposed_value": None,
            "reason": "Inspect failed/unavailable checks; review relevant grouping features or request more rows."
                      if statuses else "No feature checks were requested; select features to evaluate representation.",
            "evidence": "measured_balance",
            "tradeoff": "Any changed run must be rechecked. Size, class budgets, and thresholds were not altered.",
        })
    return DownsampleReport(summary, classes, strata, balance, edges, recommendations)
