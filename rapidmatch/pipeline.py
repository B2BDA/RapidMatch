"""Pipeline orchestrator — the table of contents for a match run.

`ControlMatcher.fit_match` is the public entry. It does no matching itself;
it calls one-job modules in the order locked in plan.md:

    ingest -> data report -> validate -> missingness -> bin/stratify
    -> coverage -> score -> greedy match -> tolerance -> assemble
    -> balance check -> drift trim -> report
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from rapidmatch.balance.checker import check_balance
from rapidmatch.binning.stratifier import Stratifier
from rapidmatch.config import MatchConfig
from rapidmatch.coverage.flag_coverage import flag_coverage
from rapidmatch.data_report.report import DataReport
from rapidmatch.drift.correct import correct_drift
from rapidmatch.ingestion.data_loader import UniversalDataLoader
from rapidmatch.ingestion.ingest import ingest
from rapidmatch.ingestion.validate import classify_match_vars, validate_schema
from rapidmatch.matching.greedy_match import greedy_match
from rapidmatch.missingness.handle_missing import handle_missing
from rapidmatch.output.assemble import MatchResult, assemble
from rapidmatch.reporting.report import build_report
from rapidmatch.scoring.scorer import score_pairs, target_moments
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
        types = validate_schema(con, self.config)
        numeric, categorical = classify_match_vars(types, self.config.match_vars)

        # Lazy object; materialize here because the pipeline is a terminal point.
        profile = DataReport(con, self.config.treatment_col).summary()

        handle_missing(
            con,
            self.config.treatment_col,
            numeric,
            categorical,
        )

        stratifier = Stratifier(n_bins=self.config.n_bins)
        counts = stratifier.run(con, numeric, categorical)
        self.bin_edges = stratifier.edges
        coverage = flag_coverage(
            counts,
            min_control_pool_size=self.config.min_control_pool_size,
            min_control_ratio=self.config.min_control_ratio,
        )

        table = UniversalDataLoader.duckdb_to_arrow(con, table_name="v_stratified")
        frame = table.to_pandas()
        ids = frame["_rm_id"].to_numpy(dtype=np.int64)
        treatment = frame["_treatment"].to_numpy(dtype=np.int64)
        strata = frame["_stratum"].astype(str).to_numpy()
        if numeric:
            x = np.column_stack(
                [frame[v].to_numpy(dtype=np.float64) for v in numeric]
            )
        else:
            x = np.empty((len(ids), 0), dtype=np.float64)

        target_mask = treatment == 1
        mean, std = target_moments(x[target_mask])

        pair_t: list[np.ndarray] = []
        pair_c: list[np.ndarray] = []
        pair_s: list[np.ndarray] = []
        for stratum in coverage.eligible:
            in_stratum = strata == stratum
            tm = in_stratum & (treatment == 1)
            cm = in_stratum & (treatment == 0)
            t_ids, c_ids, strengths = score_pairs(
                ids[tm],
                ids[cm],
                x[tm],
                x[cm],
                numeric,
                self.config,
                mean,
                std,
            )
            if len(t_ids):
                pair_t.append(t_ids)
                pair_c.append(c_ids)
                pair_s.append(strengths)

        if pair_t:
            all_t = np.concatenate(pair_t)
            all_c = np.concatenate(pair_c)
            all_s = np.concatenate(pair_s)
            assignments = greedy_match(all_t, all_c, all_s, n=self.config.n)
        else:
            assignments = []

        kept, below, cutoff = apply_tolerance(assignments, self.config.tolerance)
        self.cutoff = cutoff

        no_control_ids = set(ids[target_mask & np.isin(strata, list(coverage.no_control))])
        thin_ids = set(ids[target_mask & np.isin(strata, list(coverage.thin))])
        target_frame = frame.loc[frame["_treatment"] == 1].copy()
        control_frame = frame.loc[frame["_treatment"] == 0].copy()

        matched_c = {int(c) for _, c, _, _ in kept}
        pair_strength = {int(c): float(s) for _, c, s, _ in kept}
        matched_control = frame.loc[frame["_rm_id"].isin(matched_c)].copy()
        before = check_balance(
            target_frame,
            matched_control,
            self.config.match_vars,
            self.config.monitor_vars,
            types,
            js_threshold=self.config.js_threshold,
            ks_threshold=self.config.ks_threshold,
        )
        flagged = [row for row in before if row.role == "monitor" and row.flagged]
        correction = correct_drift(
            target_frame,
            matched_control,
            pair_strength,
            flagged,
            n_bins=self.config.n_bins,
        )
        kept = [a for a in kept if int(a[1]) in correction.kept_control_ids]
        after_control = frame.loc[
            frame["_rm_id"].isin(correction.kept_control_ids)
        ].copy()
        after = check_balance(
            target_frame,
            after_control,
            self.config.match_vars,
            self.config.monitor_vars,
            types,
            js_threshold=self.config.js_threshold,
            ks_threshold=self.config.ks_threshold,
        )

        result = assemble(
            target_frame=target_frame,
            control_frame=control_frame,
            kept=kept,
            below=below,
            no_control_ids={int(i) for i in no_control_ids},
            thin_ids={int(i) for i in thin_ids},
            cutoff=cutoff,
            id_col=self.config.id_col,
        )
        result.report = build_report(
            result.targets,
            cutoff,
            before,
            after,
            correction.events,
            profile,
        )
        result.coverage_summary = result.report.coverage
        return result
