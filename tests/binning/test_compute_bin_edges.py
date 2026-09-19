import duckdb
import pytest

from rapidmatch.binning.compute_bin_edges import compute_bin_edges


def test_edges_come_from_target_only() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE v_prepared AS
        SELECT * FROM (VALUES
            (1, 1, 10.0),
            (2, 1, 20.0),
            (3, 1, 30.0),
            (4, 1, 40.0),
            (5, 0, 1000.0)
        ) AS t(_rm_id, _treatment, income)
        """
    )
    edges = compute_bin_edges(con, ["income"], n_bins=4)
    assert len(edges["income"]) == 3
    assert edges["income"] == sorted(edges["income"])
    assert edges["income"][0] >= 10.0
    assert edges["income"][-1] <= 40.0
    assert edges["income"][-1] < 1000.0
