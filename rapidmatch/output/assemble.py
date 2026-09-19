"""Module 10 — Output assembly.

Every target row appears in the result. Uncovered rows are not dropped;
they are labeled via `match_status`. `thin_stratum` is a separate boolean
and can co-occur with `matched`.

The heavy lifting (per-target roll-up) happens in DuckDB over `v_stratified`,
so a 7M-row run never materializes a Python-side target lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import duckdb
import pyarrow as pa

from rapidmatch._sql import quote_ident

_VIEWS = ("v_pairs_rm", "v_targets_rm")
_REGISTRY = ("_rm_pairs", "_rm_thin", "_rm_status")


@dataclass
class MatchResult:
    """What a caller inspects after `fit_match`.

    Attributes:
        pairs: One row per (target, control) match, plus one leftover row
            per unmatched target.
        targets: One row per target with rolled-up status.
        cutoff: Strength quantile that was actually applied.
        coverage_summary: Counts and percent matched.
        report: Module 13 artifacts (coverage / balance / drift log / profile).
    """

    pairs: pa.Table
    targets: pa.Table
    cutoff: float
    coverage_summary: dict[str, Any]
    report: Any = field(default=None)


def assemble(
    con: duckdb.DuckDBPyConnection,
    kept: Sequence[tuple[int, int, float, int]],
    below: set[int],
    no_control_ids: set[int],
    thin_ids: set[int],
    cutoff: float,
    id_col: Optional[str] = None,
) -> MatchResult:
    """Build `result.pairs` / `result.targets` from the kept assignments.

    `kept` are (target_rm_id, control_rm_id, match_strength, match_rank).
    `below` targets lost every slot to tolerance; `no_control_ids` targets had
    no eligible stratum; `thin_ids` targets sit in thin strata.
    """
    id_expr = quote_ident(id_col) if id_col else "_rm_id"
    null_id = _null_cast(con, id_col)

    pairs = _pairs_arrow(kept)
    thin = _thin_arrow(thin_ids)
    status = _status_arrow(below, no_control_ids)

    con.register("_rm_pairs", pairs)
    con.register("_rm_thin", thin)
    con.register("_rm_status", status)
    try:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW v_pairs_rm AS
            SELECT
                t.{id_expr} AS target_id,
                c.{id_expr} AS control_id,
                p.target_rm_id,
                p.control_rm_id,
                t._stratum AS stratum,
                p.match_strength,
                p.match_rank,
                'matched' AS match_status,
                (th.target_rm_id IS NOT NULL) AS thin_stratum
            FROM _rm_pairs p
            JOIN v_stratified t ON t._rm_id = p.target_rm_id
            JOIN v_stratified c ON c._rm_id = p.control_rm_id
            LEFT JOIN _rm_thin th ON th.target_rm_id = t._rm_id

            UNION ALL

            SELECT
                t.{id_expr} AS target_id,
                {null_id} AS control_id,
                t._rm_id AS target_rm_id,
                NULL::BIGINT AS control_rm_id,
                t._stratum AS stratum,
                NULL::DOUBLE AS match_strength,
                NULL::BIGINT AS match_rank,
                COALESCE(s.status, 'unmatched') AS match_status,
                (th.target_rm_id IS NOT NULL) AS thin_stratum
            FROM v_stratified t
            LEFT JOIN _rm_status s ON s.target_rm_id = t._rm_id
            LEFT JOIN _rm_thin th ON th.target_rm_id = t._rm_id
            WHERE t._treatment = 1
              AND t._rm_id NOT IN (SELECT target_rm_id FROM _rm_pairs)
            """
        )
        con.execute(
            """
            CREATE OR REPLACE VIEW v_targets_rm AS
            SELECT
                target_rm_id,
                ANY_VALUE(target_id) AS target_id,
                MIN(stratum) AS stratum,
                SUM(CASE WHEN control_rm_id IS NOT NULL THEN 1 ELSE 0 END)
                    AS n_matches,
                CASE MAX(
                    CASE match_status
                        WHEN 'matched' THEN 3
                        WHEN 'below_tolerance' THEN 2
                        WHEN 'no_control_available' THEN 1
                        ELSE 0
                    END
                )
                    WHEN 3 THEN 'matched'
                    WHEN 2 THEN 'below_tolerance'
                    WHEN 1 THEN 'no_control_available'
                    ELSE 'unmatched'
                END AS match_status,
                BOOL_OR(thin_stratum) AS thin_stratum,
                MAX(match_strength) AS best_strength
            FROM v_pairs_rm
            GROUP BY target_rm_id
            ORDER BY target_rm_id
            """
        )
        pairs_tbl = (
            con.sql(
                "SELECT * FROM v_pairs_rm ORDER BY target_rm_id, match_rank NULLS LAST"
            ).to_arrow_table()
        )
        targets_tbl = con.sql("SELECT * FROM v_targets_rm").to_arrow_table()
    finally:
        for name in _VIEWS:
            con.execute(f"DROP VIEW IF EXISTS {name}")
        for name in _REGISTRY:
            try:
                con.unregister(name)
            except Exception:
                pass

    summary = _coverage_summary(targets_tbl, cutoff)
    return MatchResult(
        pairs=pairs_tbl,
        targets=targets_tbl,
        cutoff=cutoff,
        coverage_summary=summary,
    )


def _pairs_arrow(kept: Sequence[tuple[int, int, float, int]]) -> pa.Table:
    if not kept:
        return pa.table(
            {
                "target_rm_id": pa.array([], type=pa.int64()),
                "control_rm_id": pa.array([], type=pa.int64()),
                "match_strength": pa.array([], type=pa.float64()),
                "match_rank": pa.array([], type=pa.int64()),
            }
        )
    return pa.table(
        {
            "target_rm_id": [int(t) for t, _, _, _ in kept],
            "control_rm_id": [int(c) for _, c, _, _ in kept],
            "match_strength": [float(s) for _, _, s, _ in kept],
            "match_rank": [int(r) for _, _, _, r in kept],
        }
    )


def _null_cast(con: duckdb.DuckDBPyConnection, id_col: Optional[str]) -> str:
    """`NULL::<type>` matching the id column, so UNION branches unify types."""
    if id_col is None:
        return "NULL::BIGINT"
    row = con.execute(
        """
        SELECT data_type FROM information_schema.columns
        WHERE table_name = ? AND column_name = ?
        """,
        ["v_stratified", id_col],
    ).fetchone()
    dtype = (row[0] if row else "BIGINT").upper()
    return f"NULL::{dtype}"


def _thin_arrow(thin_ids: set[int]) -> pa.Table:
    return pa.table(
        {"target_rm_id": [int(i) for i in sorted(thin_ids)]}
        if thin_ids
        else {"target_rm_id": pa.array([], type=pa.int64())}
    )


def _status_arrow(below: set[int], no_control_ids: set[int]) -> pa.Table:
    rows = [(int(i), "below_tolerance") for i in sorted(below)]
    rows.extend((int(i), "no_control_available") for i in sorted(no_control_ids))
    if not rows:
        return pa.table(
            {
                "target_rm_id": pa.array([], type=pa.int64()),
                "status": pa.array([], type=pa.string()),
            }
        )
    return pa.table(
        {
            "target_rm_id": [r[0] for r in rows],
            "status": [r[1] for r in rows],
        }
    )


def _coverage_summary(targets: pa.Table, cutoff: float) -> dict[str, Any]:
    statuses = targets["match_status"].to_pylist()
    n = len(statuses)
    n_matched = statuses.count("matched")
    n_no = statuses.count("no_control_available")
    n_below = statuses.count("below_tolerance")
    n_unmatched = statuses.count("unmatched")
    n_thin = targets["thin_stratum"].to_pylist().count(True)
    return {
        "n_target": n,
        "n_matched": n_matched,
        "n_no_control": n_no,
        "n_below_tolerance": n_below,
        "n_thin_stratum": n_thin,
        "tolerance_cutoff": cutoff,
        "pct_matched": (n_matched / n) if n else 0.0,
        "pct_no_control": (n_no / n) if n else 0.0,
        "pct_below_tolerance": (n_below / n) if n else 0.0,
        "n_unmatched": n_unmatched,
    }