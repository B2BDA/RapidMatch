import math

import duckdb
import pytest

from rapidmatch.sampling.balance import check_population_balance
from rapidmatch.sampling.config import SamplingConfig


def test_sql_statistics_match_hand_calculated_values():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE _rs_population AS SELECT * FROM (VALUES (0., 'a'), (1., 'b')) t(x, g)")
        con.execute("CREATE TABLE _rs_sample AS SELECT * FROM _rs_population WHERE x = 0")
        cfg = SamplingConfig(sample_size=1, check_vars=["x", "g"], ks_threshold=.5)
        check_population_balance(con, cfg, {"x": "DOUBLE", "g": "VARCHAR"})
        stats = {r[0]: (r[1], r[2]) for r in con.execute("SELECT variable, statistic, status FROM _rs_balance").fetchall()}
        assert stats["x"] == (.5, "pass")
        expected_js = .25 * math.log(2/3) + .25 * math.log(2) + .5 * math.log(4/3)
        assert stats["g"][0] == pytest.approx(expected_js)
        assert stats["g"][1] == "fail"


def test_within_label_checks_detect_imbalance_hidden_overall():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE _rs_population AS SELECT * FROM (VALUES "
                    "('A', 0.), ('A', 1.), ('B', 0.), ('B', 1.)) t(_rs_label, x)")
        con.execute("CREATE VIEW _rs_sample AS SELECT * FROM _rs_population "
                    "WHERE (_rs_label = 'A' AND x = 0) OR (_rs_label = 'B' AND x = 1)")
        cfg = SamplingConfig(sample_size=2, label_col="_rs_label", check_vars=["x"], check_by_label=True)
        check_population_balance(con, cfg, {"x": "DOUBLE", "_rs_label": "VARCHAR"})
        rows = con.execute("SELECT scope, statistic, status FROM _rs_balance WHERE role = 'feature' ORDER BY scope").fetchall()
        assert rows[0] == ("overall", 0., "pass")
        assert rows[1:] == [("within_label", .5, "fail")]*2


def test_all_missing_numeric_comparison_is_not_evaluated():
    with duckdb.connect() as con:
        con.execute("CREATE TABLE _rs_population AS SELECT NULL::DOUBLE AS x FROM range(5)")
        con.execute("CREATE VIEW _rs_sample AS SELECT * FROM _rs_population LIMIT 2")
        cfg = SamplingConfig(sample_size=2, check_vars=["x"])
        check_population_balance(con, cfg, {"x": "DOUBLE"})
        row = con.execute("SELECT statistic, status, population_missing_rate, sample_missing_rate FROM _rs_balance").fetchone()
        assert row == (None, "not_evaluated", 1., 1.)
