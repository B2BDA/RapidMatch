"""Population-backed grouping, with no synthetic treatment column."""

from rapidmatch._sql import quote_ident
from rapidmatch.ingestion.validate import column_types, is_numeric_dtype


def prepare_population(con, config):
    types = column_types(con, "udl_data")
    required = [*config.features, *([config.label_col] if config.label_col else [])]
    missing = sorted(set(required) - set(types))
    if missing:
        raise ValueError(f"columns not found in input: {missing}")
    if any(name.lower().startswith("_rs_") for name in types):
        raise ValueError("input column names beginning with _rs_ are reserved for sampling")

    # Pin row identity once, before any parallel grouping or selection. Reading
    # v_source repeatedly would re-evaluate its ROW_NUMBER window.
    old_threads = int(con.execute("SELECT current_setting('threads')").fetchone()[0])
    con.execute("SET threads = 1")
    try:
        con.execute("CREATE TABLE _rs_source AS SELECT ROW_NUMBER() OVER () AS _rs_id, * FROM udl_data")
    finally:
        con.execute(f"SET threads = {config.duckdb_threads or old_threads}")
    n = int(con.execute("SELECT COUNT(*) FROM _rs_source").fetchone()[0])
    if config.sample_size > n:
        raise ValueError(f"sample_size ({config.sample_size}) exceeds population size ({n})")

    numeric = [v for v in config.stratify_vars if is_numeric_dtype(types[v])]
    edges = {}
    if numeric:
        probs = "[" + ",".join(str(i / config.n_bins) for i in range(1, config.n_bins)) + "]"
        expressions = [
            f"quantile_cont({quote_ident(v)}, {probs}) FILTER (WHERE isfinite({quote_ident(v)}))"
            for v in numeric
        ]
        values = con.execute("SELECT " + ", ".join(expressions) + " FROM _rs_source").fetchone()
        edges = {v: list(cuts or []) for v, cuts in zip(numeric, values)}

    groups = []
    for var in config.stratify_vars:
        col = quote_ident(var)
        if var in edges:
            cases = " ".join(f"WHEN {col} <= {float(edge)!r} THEN {i}" for i, edge in enumerate(edges[var]))
            groups.append(
                f"CASE WHEN {col} IS NULL OR NOT isfinite({col}) THEN NULL "
                + cases + f" ELSE {len(edges[var])} END"
            )
        else:
            groups.append(col)
    label = quote_ident(config.label_col) if config.label_col else ""
    # JSON arrays preserve null, string, numeric, and delimiter identities.
    con.execute(
        "CREATE VIEW _rs_population AS SELECT *, "
        f"CAST(json_array({label}) AS VARCHAR) AS _rs_label, "
        f"CAST(json_array({', '.join(groups)}) AS VARCHAR) AS _rs_group FROM _rs_source"
    )
    return types, n, edges
