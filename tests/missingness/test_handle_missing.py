import duckdb

from rapidmatch.missingness.handle_missing import handle_missing


def test_numeric_missing_gets_flag_and_sentinel() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE v_source AS
        SELECT * FROM (VALUES
            (1, 1, 10.0, 'N'),
            (2, 1, NULL, NULL)
        ) AS t(_rm_id, treat, income, region)
        """
    )
    handle_missing(con, "treat", ["income"], ["region"])
    rows = con.execute(
        "SELECT _rm_id, income, is_missing_income, region FROM v_prepared ORDER BY 1"
    ).fetchall()
    assert rows[0][1] == 10.0
    assert rows[0][2] is False
    assert rows[0][3] == "N"
    assert rows[1][1] == 0.0
    assert rows[1][2] is True
    assert rows[1][3] == "__MISSING__"
