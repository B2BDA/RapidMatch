"""Pipeline orchestrator — the table of contents for a match run.

`ControlMatcher.fit_match` is the public entry. It does no matching itself;
it calls one-job modules in the order locked in plan.md:

    ingest -> data report -> validate -> missingness -> bin/stratify
    -> coverage -> score -> greedy match -> tolerance -> assemble
    -> balance check -> drift trim -> report

The projected `v_stratified` data is pulled into PyArrow (never pandas).
Pair storage depends on stratum sizes and the optional candidate cap; preflight
capacity reports expose that work before scoring.
"""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import Any, Optional

import numpy as np
import pyarrow as pa

from rapidmatch._progress import _is_tty, _track, manual_bar
from rapidmatch._sql import quote_ident
from rapidmatch._verbose import VerboseLog, rss_mb
from rapidmatch.balance.checker import check_balance
from rapidmatch.binning.stratifier import Stratifier
from rapidmatch.config import MatchConfig
from rapidmatch.coverage.flag_coverage import flag_coverage
from rapidmatch.coverage.capacity import CapacityReport, add_trials, build_capacity
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
        log = VerboseLog(enabled=cfg.verbose)
        try:
            return self._run_stages(con, cfg, stage, log)
        finally:
            # Always close so the final 100% frame is force-rendered: without
            # this, tqdm's mininterval throttling can leave a fast run's bar
            # parked at an intermediate percentage (e.g. 90%).
            stages.close()

    def assess(
        self, data: Any, *, n_bins_candidates=None,
        max_report_strata: int = 1000, work_dir: Optional[str] = None,
    ) -> CapacityReport:
        """Measure capacity and optional bin trials without scoring or matching.

        Uses one ingestion. Neither this matcher's config, bin_edges, nor cutoff
        is changed. Detail rows are bounded; summary/trial totals are complete.
        """
        candidates = list(n_bins_candidates or [])
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 2 for v in candidates):
            raise ValueError("n_bins_candidates must contain integers >= 2")
        with ingest(data, work_dir=work_dir) as session:
            con = session.con
            cfg = self.config
            if cfg.duckdb_threads:
                con.execute(f"SET threads = {cfg.duckdb_threads}")
            profile = DataReport(con, cfg.treatment_col).summary()
            types = validate_schema(con, cfg, treatment_values=profile["treatment_values"])
            numeric, categorical = classify_match_vars(types, cfg.match_vars)
            group_numeric = [v for v in numeric if v in cfg.grouping_vars]
            handle_missing(con, cfg.treatment_col, numeric, categorical)
            reports = []
            for bins in dict.fromkeys([cfg.n_bins, *candidates]):
                trial_cfg = replace(cfg, n_bins=int(bins))
                stratifier = Stratifier(n_bins=int(bins))
                counts = stratifier.run(con, group_numeric, categorical, missing_vars=numeric)
                reports.append(build_capacity(
                    counts, trial_cfg, stratifier.edges, con=con, numeric=numeric,
                    max_report_strata=max_report_strata,
                ))
            report = reports[0]
            if candidates:
                add_trials(report, [r.summary for r in reports])
            return report

    def _run_stages(self, con, cfg, stage, log) -> MatchResult:
        # Lazy object; materialize here because the pipeline is a terminal point.
        rss = rss_mb()
        log.emit(
            "start",
            n=cfg.n,
            tolerance=cfg.tolerance,
            n_bins=cfg.n_bins,
            n_workers=cfg.n_workers,
            max_candidates=cfg.max_candidates_per_target,
            rss_mb=None if rss is None else round(rss, 1),
        )
        profile = DataReport(con, cfg.treatment_col).summary()
        log.emit(
            "profile",
            rows=profile["n_rows"],
            target=profile["n_target"],
            untreated=profile["n_control"],
            rss_mb=None if rss_mb() is None else round(rss_mb(), 1),
        )
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
        group_numeric = [v for v in numeric if v in cfg.grouping_vars]
        counts = stratifier.run(con, group_numeric, categorical, missing_vars=numeric)
        self.bin_edges = stratifier.edges
        capacity = build_capacity(
            counts, cfg, stratifier.edges, con=con, numeric=numeric, emit_warning=True,
        )
        coverage = flag_coverage(
            counts,
            min_control_pool_size=cfg.min_control_pool_size,
            min_control_ratio=cfg.min_control_ratio,
        )
        log.emit(
            "stratify",
            strata=len(counts),
            eligible=len(coverage.eligible),
            no_control=len(coverage.no_control),
            thin=len(coverage.thin),
        )
        stage("stratify")

        # Projected pull: only the columns scoring/balance/report need.
        table = self._pull(con, numeric, categorical)
        rss = rss_mb()
        log.emit(
            "pull",
            rows=len(table),
            rss_mb=None if rss is None else round(rss, 1),
        )

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
        rss = rss_mb()
        log.emit(
            "score",
            pairs=len(all_t),
            eligible_strata=len(eligible),
            rss_mb=None if rss is None else round(rss, 1),
        )
        stage("score")

        def _match_tick(n_assigned: int, n_seen: int) -> None:
            log.rewrite("match", assigned=n_assigned, pairs_seen=n_seen)

        if len(all_t):
            gbar = manual_bar(total=len(all_t), desc="match", enabled_flag=cfg.progress)
            try:
                assignments = greedy_match(
                    all_t, all_c, all_s, n=cfg.n, pbar=gbar, on_progress=_match_tick
                )
            finally:
                gbar.close()
        else:
            assignments = []
        log.emit("match", assigned=len(assignments), pairs=len(all_t))
        capacity.summary["assignments_before_tolerance"] = len(assignments)
        if (cfg.max_candidates_per_target is not None
                and len(assignments) < capacity.summary["assignment_capacity_ceiling"]
                and capacity.summary["retained_candidate_pairs"] < capacity.summary["candidate_pairs"]):
            capacity.recommendations.append({
                "setting": "max_candidates_per_target",
                "proposed_value": cfg.max_candidates_per_target * 2,
                "reason": "The capped run assigned fewer controls than the unpruned structural ceiling; try a larger cap.",
                "evidence": "untested_suggestion",
                "tradeoff": "More retained pairs require more memory; this does not guarantee full coverage.",
            })
        stage("match")

        kept, below, cutoff = apply_tolerance(
            assignments,
            cfg.tolerance,
            preserve_primary=True,
        )
        self.cutoff = cutoff
        capacity.summary["assignments_after_tolerance"] = len(kept)
        log.emit(
            "tolerance",
            kept=len(kept),
            below=len(below),
            cutoff=round(float(cutoff), 4),
        )
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
        capacity.summary["assignments_after_drift"] = len(kept)
        capacity.summary["drift_removed"] = sum(e.n_removed for e in correction.events)
        if capacity.summary["drift_removed"]:
            capacity.recommendations.append({
                "setting": "monitor_vars", "proposed_value": None,
                "reason": "Inspect report.drift_log for the monitor groups whose correction removed controls.",
                "evidence": "measured_drift_removals",
                "tradeoff": "Coverage fell during balance correction; changing numeric bins alone is not a proven remedy.",
            })
        log.emit(
            "drift",
            kept=len(kept),
            trimmed=sum(e.n_removed for e in correction.events),
        )
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
        result.report.capacity = capacity
        result.coverage_summary = result.report.coverage
        stage("report")
        log.emit(
            "done",
            matched=result.coverage_summary.get("n_matched"),
            controls=len(kept),
            pct_matched=result.coverage_summary.get("pct_matched"),
        )
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
