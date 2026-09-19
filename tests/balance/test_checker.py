import pyarrow as pa

from rapidmatch.balance.checker import check_balance


def test_flags_monitor_var_over_threshold() -> None:
    target = pa.table({"income": [1.0, 2.0, 3.0], "seg": ["A", "A", "B"]})
    control = pa.table({"income": [1.1, 2.1, 3.1], "seg": ["A", "A", "A"]})
    rows = check_balance(
        target,
        control,
        match_vars=["income"],
        monitor_vars=["seg"],
        dtypes={"income": "DOUBLE", "seg": "VARCHAR"},
        js_threshold=0.05,
        ks_threshold=0.5,
    )
    by_var = {r.variable: r for r in rows}
    assert by_var["income"].role == "match"
    assert by_var["income"].kind == "ks"
    assert by_var["seg"].role == "monitor"
    assert by_var["seg"].kind == "js"
    assert by_var["seg"].flagged


def test_handles_null_category() -> None:
    target = pa.table({"seg": ["A", None, "B"]})
    control = pa.table({"seg": [None, "A", "B", "C"]})
    rows = check_balance(
        target,
        control,
        match_vars=[],
        monitor_vars=["seg"],
        dtypes={"seg": "VARCHAR"},
        js_threshold=0.05,
    )
    assert [r.variable for r in rows] == ["seg"]