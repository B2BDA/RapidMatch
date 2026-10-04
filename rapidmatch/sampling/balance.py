"""Exact empirical KS/JS on DuckDB relations, without pulling population columns.

Frequency tables and cumulative windows are disk/spill-capable. Only bounded
report rows leave DuckDB; an empty numeric comparison is never called balanced.
"""

from rapidmatch._sql import quote_ident
from rapidmatch.ingestion.validate import is_numeric_dtype


def check_population_balance(con, config, types):
    con.execute("""
        CREATE TABLE _rs_balance (
            variable VARCHAR, role VARCHAR, scope VARCHAR, label VARCHAR,
            kind VARCHAR, statistic DOUBLE, threshold DOUBLE,
            n_population BIGINT, n_sample BIGINT,
            n_valid_population BIGINT, n_valid_sample BIGINT,
            population_missing_rate DOUBLE, sample_missing_rate DOUBLE,
            status VARCHAR, required BOOLEAN
        )
    """)
    for var in config.features:
        _check_variable(con, var, "feature", is_numeric_dtype(types[var]), config)
    if config.label_col:
        # Class allocation has its own exact invariant. JS is descriptive here,
        # not a substitute for (or a veto of) the rounded class budgets.
        _check_variable(con, config.label_col, "label", False, config)


def _check_variable(con, var, role, numeric, config):
    col = f"src.{quote_ident(var)}"
    value = f"CAST({col} AS DOUBLE)" if numeric else f"CAST(json_array({col}) AS VARCHAR)"
    valid = f"COALESCE(isfinite({col}), FALSE)" if numeric else "TRUE"
    missing = f"NOT ({valid})" if numeric else f"{col} IS NULL"
    scopes = "('overall', '')"
    if config.check_by_label and role == "feature":
        scopes += ", ('within_label', _rs_label)"
    combined = f"""
        SELECT d.scope, d.label, {value} AS value, {valid} AS valid,
            1 AS p, 0 AS s, CAST({missing} AS INTEGER) AS pm, 0 AS sm
        FROM _rs_population src, LATERAL (VALUES {scopes}) d(scope, label)
        UNION ALL
        SELECT d.scope, d.label, {value} AS value, {valid} AS valid,
            0 AS p, 1 AS s, 0 AS pm, CAST({missing} AS INTEGER) AS sm
        FROM _rs_sample src, LATERAL (VALUES {scopes}) d(scope, label)
    """
    if numeric:
        metric = """
            curves AS (
                SELECT f.scope, f.label,
                    SUM(pc) OVER (PARTITION BY f.scope, f.label ORDER BY value
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::DOUBLE
                        / NULLIF(t.nvp, 0) AS p,
                    SUM(sc) OVER (PARTITION BY f.scope, f.label ORDER BY value
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::DOUBLE
                        / NULLIF(t.nvs, 0) AS q
                FROM freq f JOIN totals t USING (scope, label) WHERE valid
            ), stats AS (
                SELECT scope, label, MAX(ABS(p - q)) AS statistic
                FROM curves GROUP BY scope, label
            )
        """
        threshold, kind = config.ks_threshold, "ks"
    else:
        metric = """
            probabilities AS (
                SELECT f.scope, f.label, pc::DOUBLE / NULLIF(t.nvp, 0) AS p,
                    sc::DOUBLE / NULLIF(t.nvs, 0) AS q
                FROM freq f JOIN totals t USING (scope, label)
            ), stats AS (
                SELECT scope, label,
                    CASE WHEN COUNT(p) = 0 OR COUNT(q) = 0 THEN NULL ELSE
                    GREATEST(0.0, SUM(
                        CASE WHEN p > 0 THEN 0.5 * p * LN(p / ((p + q) / 2)) ELSE 0 END
                        + CASE WHEN q > 0 THEN 0.5 * q * LN(q / ((p + q) / 2)) ELSE 0 END
                    )) END AS statistic
                FROM probabilities GROUP BY scope, label
            )
        """
        threshold, kind = config.js_threshold, "js"
    con.execute(f"""
        INSERT INTO _rs_balance
        WITH combined AS ({combined}), freq AS (
            SELECT scope, label, value, valid, SUM(p) AS pc, SUM(s) AS sc,
                SUM(pm) AS pm, SUM(sm) AS sm
            FROM combined GROUP BY scope, label, value, valid
        ), totals AS (
            SELECT scope, label, SUM(pc) AS np, SUM(sc) AS ns,
                SUM(CASE WHEN valid THEN pc ELSE 0 END) AS nvp,
                SUM(CASE WHEN valid THEN sc ELSE 0 END) AS nvs,
                SUM(pm)::DOUBLE / NULLIF(SUM(pc), 0) AS pm,
                SUM(sm)::DOUBLE / NULLIF(SUM(sc), 0) AS sm
            FROM freq GROUP BY scope, label
        ), {metric}
        SELECT ?, ?, t.scope, CASE WHEN t.scope = 'overall' THEN NULL ELSE t.label END,
            ?, s.statistic, ?, t.np, t.ns, t.nvp, t.nvs, t.pm, t.sm,
            CASE WHEN s.statistic IS NULL THEN 'not_evaluated'
                 WHEN s.statistic <= ? THEN 'pass' ELSE 'fail' END, ?
        FROM totals t LEFT JOIN stats s USING (scope, label)
    """, [var, role, kind, float(threshold), float(threshold), role == "feature"])
