"""Pipeline orchestrator — the table of contents for a match run.

`ControlMatcher.fit_match` is the public entry. It does no matching itself;
it calls one-job modules in the order locked in plan.md:

    ingest -> data report -> validate -> missingness -> bin/stratify
    -> coverage -> score -> greedy match -> tolerance -> assemble
    -> balance check -> drift trim -> report

Everything below `v_stratified` is pulled into PyArrow (never pandas):
rows are consumed with zero-copy NumPy views, so a 7M-row run keeps a flat,
streaming-friendly in-memory footprint.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyarrow as pa

from rapidmatch._progress import _is_tty, _track, manual_bar
from rapidmatch._sql import quote_ident
from rapidmatch.balance.checker import check_balance
from rapidmatch.binning.stratifier import Stratifier
from rapidmatch.config import MatchConfig
from rapidmatch.coverage.flag_coverage import flag_coverage
from rapidmatch.data_report.report import DataReport
from rapidmatch.drift.correct import correct_drift
from rapidmatch.ingestion.ingest import ingest
from rapidmatch.ingestion.validate import classify_match_vars, validate_schema
from rapidmatch.matching.greedy_match import greedy_match
from rapidmatch.missingness.handle_missing import handle_missing
from rapidmatch.output.assemble import MatchResult, assemble
from rapidmatch.reporting.report import build_report
from rapidmatch.scoring.score_strata import score_all_strata
from rapidmatch.scoring.scorer import target_moments
from rapidmatch.tolerance.apply_tolerance import apply_tolerance


class ControlMatcher:
    """Run the matching pipeline for one `MatchConfig`.

    The instance is cheap to construct. All work happens in `fit_match`.
    After a run, `bin_edges` and `cutoff` are kept for inspection.
    """

    def __init__(self, config: MatchConfig) -> None:
        self.config = config
        self.bin_edges: dict[str, list[float]] = {}
        self.cutoff: Optional[float] = None

    def fit_match(
        self,
        data: Any,
        work_dir: Optional[str] = None,
        keep_db: bool = False,
    ) -> MatchResult:
        """Ingest `data`, match, and always close the DuckDB session.

        Args:
            data: File path (csv/tsv/parquet/feather/excel) or in-memory
                pandas DataFrame / PyArrow Table.
            work_dir: Where the temp `.duckdb` file is written. Matters at
                large scale because DuckDB is disk-backed.
            keep_db: If True, leave the `.duckdb` file on disk after the run.
        """
        session = ingest(data, work_dir=work_dir, keep_db=keep_db)
        try:
            return self._run(session)
        finally:
            session.close()

    def _run(self, session) -> MatchResult:
        con = session.con
        cfg = self.config
        if cfg.duckdb_threads:
            con.execute(f"SET threads = {cfg.duckdb_threads}")
        # DuckDB's own progress bar prints ANSI to stderr; keep it terminal-only
        # (Jupyter gets our tqdm widgets instead, without escape-code noise).
        if cfg.progress and _is_tty():
            con.execute("PRAGMA enable_progress_bar")
            con.execute("PRAGMA progress_bar_time = 100")

        stages = manual_bar(
            total=10, desc="pipeline", enabled_flag=cfg.progress
        )
        stage = _track(stages)
        try:
            return self._run_stages(con, cfg, stage)
        finally:
            # Always close so the final 100% frame is force-rendered: without
            # this, tqdm's mininterval throttling can leave a fast run's bar
            # parked at an intermediate percentage (e.g. 90%).
            stages.close()

    def _run_stages(self, con, cfg, stage) -> MatchResult:
        # Lazy object; materialize here because the pipeline is a terminal point.
        profile = DataReport(con, cfg.treatment_col).summary()
        stage("profile")

        # Validation reuses the profile's treatment scan: one data pass total.
        types = validate_schema(
            con, cfg, treatment_values=profile["treatment_values"]
        )
        numeric, categorical = classify_match_vars(types, cfg.match_vars)
        stage("validate")

        handle_missing(con, cfg.treatment_col, numeric, categorical)
        stage("missing")

        stratifier = Stratifier(n_bins=cfg.n_bins)
        counts = stratifier.run(con, numeric, categorical)
        self.bin_edges = stratifier.edges
        coverage = flag_coverage(
            counts,
            min_control_pool_size=cfg.min_control_pool_size,
            min_control_ratio=cfg.min_control_ratio,
        )
        stage("stratify")

        # Projected pull: only the columns scoring/balance/report need.
        table = self._pull(con, numeric, categorical)

        ids = table["_rm_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        treatment = table["_treatment"].to_numpy(zero_copy_only=False).astype(np.int64)
        strata = table["_stratum"].to_numpy(zero_copy_only=False)
        if numeric:
            x = np.column_stack(
                [table[v].to_numpy(zero_copy_only=False).astype(np.float64) for v in numeric]
            )
        else:
            x = np.empty((len(ids), 0), dtype=np.float64)

        target_mask = treatment == 1
        mean, std = target_moments(x[target_mask])

        eligible = sorted(coverage.eligible)
        pbar = manual_bar(total=len(eligible), desc="score", enabled_flag=cfg.progress)
        try:
            all_t, all_c, all_s = score_all_strata(
                eligible,
                strata,
                ids,
                treatment,
                x,
                numeric,
                cfg,
                mean,
                std,
                n_workers=cfg.n_workers,
                pbar=pbar,
            )
        finally:
            pbar.close()
        stage("score")

        if len(all_t):
            gbar = manual_bar(total=len(all_t), desc="match", enabled_flag=cfg.progress)
            try:
                assignments = greedy_match(all_t, all_c, all_s, n=cfg.n, pbar=gbar)
            finally:
                gbar.close()
        else:
            assignments = []
        stage("match")

        kept, below, cutoff = apply_tolerance(assignments, cfg.tolerance)
        self.cutoff = cutoff
        stage("tolerance")

        no_control_ids = set(
            ids[target_mask & np.isin(strata, np.array(sorted(coverage.no_control)))]
        )
        thin_ids = set(
            ids[target_mask & np.isin(strata, np.array(sorted(coverage.thin)))]
        )
        target_frame = table.filter(target_mask)

        matched_c = {int(c) for _, c, _, _ in kept}
        pair_strength = {int(c): float(s) for _, c, s, _ in kept}
        matched_control = table.filter(
            np.isin(ids, np.fromiter(matched_c, dtype=np.int64, count=len(matched_c)))
        )
        before = check_balance(
            target_frame,
            matched_control,
            cfg.match_vars,
            cfg.monitor_vars,
            types,
            js_threshold=cfg.js_threshold,
            ks_threshold=cfg.ks_threshold,
        )
        flagged = [row for row in before if row.role == "monitor" and row.flagged]
        correction = correct_drift(
            target_frame,
            matched_control,
            pair_strength,
            flagged,
            n_bins=cfg.n_bins,
        )
        kept = [a for a in kept if int(a[1]) in correction.kept_control_ids]
        after_control = table.filter(
            np.isin(
                ids,
                np.fromiter(
                    correction.kept_control_ids,
                    dtype=np.int64,
                    count=len(correction.kept_control_ids),
                ),
            )
        )
        after = check_balance(
            target_frame,
            after_control,
            cfg.match_vars,
            cfg.monitor_vars,
            types,
            js_threshold=cfg.js_threshold,
            ks_threshold=cfg.ks_threshold,
        )
        stage("balance")

        result = assemble(
            con=con,
            kept=kept,
            below={int(i) for i in below},
            no_control_ids={int(i) for i in no_control_ids},
            thin_ids={int(i) for i in thin_ids},
            cutoff=cutoff,
            id_col=cfg.id_col,
        )
        stage("assemble")

        result.report = build_report(
            result.targets,
            cutoff,
            before,
            after,
            correction.events,
            profile,
        )
        result.coverage_summary = result.report.coverage
        stage("report")
        return result

    def _pull(self, con, numeric, categorical) -> pa.Table:
        cols = ["_rm_id", "_treatment", "_stratum"]
        for var in numeric:
            cols.append(var)
        for var in categorical:
            cols.append(var)
        for var in self.config.monitor_vars:
            cols.append(var)
        if self.config.id_col:
            cols.append(self.config.id_col)
        cols = list(dict.fromkeys(cols))
        selected = ", ".join(quote_ident(c) for c in cols)
        return con.sql(f"SELECT {selected} FROM v_stratified").to_arrow_table()