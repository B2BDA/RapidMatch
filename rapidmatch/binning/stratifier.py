"""Module 5 orchestrator: edges -> buckets -> stratum key -> counts."""

from __future__ import annotations

from typing import Sequence

import duckdb

from rapidmatch.binning.apply_bucket import bucket_sql
from rapidmatch.binning.build_stratum_key import stratum_key_sql
from rapidmatch.binning.compute_bin_edges import compute_bin_edges
from rapidmatch.binning.get_stratum_counts import StratumCount, get_stratum_counts


class Stratifier:
    """Runs the four binning steps and stores the edges it used."""

    def __init__(self, n_bins: int = 4) -> None:
        self.n_bins = n_bins
        self.edges: dict[str, list[float]] = {}

    def run(
        self,
        con: duckdb.DuckDBPyConnection,
        numeric_vars: Sequence[str],
        categorical_vars: Sequence[str],
        source: str = "v_prepared",
        dest: str = "v_stratified",
    ) -> list[StratumCount]:
        self.edges = compute_bin_edges(con, numeric_vars, self.n_bins, source=source)
        bucket_parts = [
            bucket_sql(var, self.edges[var], self.n_bins) for var in numeric_vars
        ]
        inner_select = ", ".join(["*", *bucket_parts]) if bucket_parts else "*"
        key_sql = stratum_key_sql(numeric_vars, categorical_vars)
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {dest} AS
            SELECT binned.*, {key_sql}
            FROM (
                SELECT {inner_select}
                FROM {source}
            ) AS binned
            """
        )
        return get_stratum_counts(con, source=dest)
