"""Public exact-size, label-aware proportional stratified random downsampling."""

from rapidmatch._sql import quote_ident
from rapidmatch.ingestion.ingest import ingest
from rapidmatch.sampling.allocate import allocate, select_rows
from rapidmatch.sampling.balance import check_population_balance
from rapidmatch.sampling.config import SamplingConfig
from rapidmatch.sampling.prepare import prepare_population
from rapidmatch.sampling.report import DownsampleResult, build_sampling_report


def random_downsample(
    data, *, sample_size, label_col=None, stratify_vars=(), check_vars=(),
    n_bins=4, random_state=None, ks_threshold=0.05, js_threshold=0.10,
    check_by_label=False, duckdb_threads=None, max_report_rows=1000,
    work_dir=None, keep_db=False,
) -> DownsampleResult:
    """Select exactly ``sample_size`` distinct source rows, without replacement.

    Optional ``label_col`` preserves observed class proportions using rounded
    class budgets. Within each class, feature strata receive proportional slots.
    Numeric grouping uses full-population quantiles. The original ingested columns
    are returned as Arrow, with internal row ids available separately in row_ids.

    Feature checks compare the sample against the full input, optionally also
    within each label (``check_by_label=True``). A failed balance assessment does
    not shrink the sample or change label budgets; inspect report.summary and
    report.balance. Reports are bounded by max_report_rows, while aggregate
    counts and acceptance always include every group/check.

    Reproducibility: identical ingested data/order, seed, and library versions
    yield identical rows, independent of the requested DuckDB thread count.
    """
    cfg = SamplingConfig(
        sample_size=sample_size, label_col=label_col, stratify_vars=stratify_vars,
        check_vars=check_vars, n_bins=n_bins, random_state=random_state,
        ks_threshold=ks_threshold, js_threshold=js_threshold,
        check_by_label=check_by_label, duckdb_threads=duckdb_threads,
        max_report_rows=max_report_rows,
    )
    with ingest(data, work_dir=work_dir, keep_db=keep_db) as session:
        con = session.con
        types, n, edges = prepare_population(con, cfg)
        allocate(con, cfg, n)
        select_rows(con, cfg, n)
        check_population_balance(con, cfg, types)
        report = build_sampling_report(con, cfg, n, edges)
        columns = ", ".join(quote_ident(c) for c in types)
        selected = con.sql(f"SELECT _rs_id, {columns} FROM _rs_sample ORDER BY _rs_id").to_arrow_table()
        return DownsampleResult(selected.drop(["_rs_id"]), report, selected["_rs_id"])
