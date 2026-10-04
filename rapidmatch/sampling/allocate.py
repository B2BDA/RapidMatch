"""Two-level largest-remainder apportionment, entirely in DuckDB."""


def allocate(con, config, population_size):
    m, n, seed = config.sample_size, population_size, config.random_state
    con.execute(f"""
        CREATE TABLE _rs_classes AS
        WITH counts AS (
            SELECT _rs_label, COUNT(*)::BIGINT AS n_population
            FROM _rs_population GROUP BY _rs_label
        ), weights AS (
            SELECT *, CAST(n_population AS HUGEINT) * {m} AS weight FROM counts
        ), base AS (
            SELECT *, weight // {n} AS base_quota, weight % {n} AS remainder FROM weights
        )
        SELECT _rs_label, n_population, weight::DOUBLE / {n} AS ideal_quota,
            CAST(base_quota + CASE WHEN ROW_NUMBER() OVER (
                ORDER BY remainder DESC, md5('{seed}:class:' || _rs_label), _rs_label
            ) <= {m} - SUM(base_quota) OVER () THEN 1 ELSE 0 END AS BIGINT) AS quota
        FROM base
    """)
    con.execute(f"""
        CREATE TABLE _rs_quotas AS
        WITH counts AS (
            SELECT _rs_label, _rs_group, COUNT(*)::BIGINT AS n_population
            FROM _rs_population GROUP BY _rs_label, _rs_group
        ), weights AS (
            SELECT g.*, c.n_population AS n_class, c.quota AS class_quota,
                CAST(g.n_population AS HUGEINT) * c.quota AS weight
            FROM counts g JOIN _rs_classes c USING (_rs_label)
        ), base AS (
            SELECT *, weight // n_class AS base_quota, weight % n_class AS remainder
            FROM weights
        )
        SELECT _rs_label, _rs_group, n_population, n_class, class_quota,
            weight::DOUBLE / n_class AS ideal_quota,
            CAST(base_quota + CASE WHEN ROW_NUMBER() OVER (
                PARTITION BY _rs_label ORDER BY remainder DESC,
                    md5('{seed}:group:' || _rs_label || ':' || _rs_group), _rs_group
            ) <= class_quota - SUM(base_quota) OVER (PARTITION BY _rs_label)
                THEN 1 ELSE 0 END AS BIGINT) AS quota
        FROM base
    """)
    invalid = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT q._rs_label FROM _rs_quotas q JOIN _rs_classes c USING (_rs_label)
            GROUP BY q._rs_label HAVING SUM(q.quota) <> MAX(c.quota)
                OR BOOL_OR(q.quota < 0 OR q.quota > q.n_population)
        )
    """).fetchone()[0]
    total = con.execute("SELECT SUM(quota) FROM _rs_classes").fetchone()[0]
    if invalid or total != m:
        raise RuntimeError("sampling allocation violated count or capacity invariants")


def select_rows(con, config, population_size):
    if config.sample_size == population_size:
        con.execute("CREATE TABLE _rs_selected AS SELECT _rs_id FROM _rs_source")
    else:
        seed = config.random_state
        con.execute(f"""
            CREATE TABLE _rs_selected AS
            SELECT _rs_id FROM (
                SELECT p._rs_id, q.quota, ROW_NUMBER() OVER (
                    PARTITION BY p._rs_label, p._rs_group
                    ORDER BY md5('{seed}:row:' || CAST(p._rs_id AS VARCHAR)), p._rs_id
                ) AS selection_rank
                FROM _rs_population p JOIN _rs_quotas q USING (_rs_label, _rs_group)
                WHERE q.quota > 0
            ) WHERE selection_rank <= quota
        """)
    con.execute("""
        CREATE VIEW _rs_sample AS
        SELECT p.* FROM _rs_population p SEMI JOIN _rs_selected s USING (_rs_id)
    """)
    count, unique = con.execute("SELECT COUNT(*), COUNT(DISTINCT _rs_id) FROM _rs_selected").fetchone()
    invalid = con.execute("""
        SELECT COUNT(*) FROM _rs_quotas q LEFT JOIN (
            SELECT _rs_label, _rs_group, COUNT(*) AS actual
            FROM _rs_sample GROUP BY _rs_label, _rs_group
        ) s USING (_rs_label, _rs_group) WHERE COALESCE(s.actual, 0) <> q.quota
    """).fetchone()[0]
    if count != config.sample_size or unique != count or invalid:
        raise RuntimeError("sampling selection violated size, uniqueness, or group quotas")
